# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma residual RMSNorm and training group32 QDQ with native mean ordering."""

from typing import TYPE_CHECKING, Literal, overload

import torch

from vllm.model_executor.layers.quantization.utils.sm80_group32_qdq import (
    _encode_e4m3,
    _round_e4m3,
)
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from .sm80_dense_fp8 import Group32FP8Input


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _norm(
    X,
    R,
    W,
    Y,
    Q,
    RES,
    SX: tl.constexpr,
    SR: tl.constexpr,
    C: tl.constexpr,
    BW: tl.constexpr,
    LOG_BW: tl.constexpr,
    EPS: tl.constexpr,
    HAS_R: tl.constexpr,
    FP8=None,
    SCALE=None,
    EMIT_FP8: tl.constexpr = False,
    EMIT_QDQ: tl.constexpr = True,
    OUTPUT_STRIDE: tl.constexpr = 0,
    EMIT_NORMALIZED: tl.constexpr = True,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BW)
    a0 = tl.full((BW,), 0, tl.float32)
    a1 = tl.full((BW,), 0, tl.float32)
    a2 = tl.full((BW,), 0, tl.float32)
    a3 = tl.full((BW,), 0, tl.float32)
    # Reduce.cuh vectorizes contiguous FP32 input into four serial accumulators.
    for step in range(C // (BW * 4)):
        col = (step * BW + lane) * 4
        x0 = tl.load(X + row * SX + col).to(tl.float32)
        x1 = tl.load(X + row * SX + col + 1).to(tl.float32)
        x2 = tl.load(X + row * SX + col + 2).to(tl.float32)
        x3 = tl.load(X + row * SX + col + 3).to(tl.float32)
        if HAS_R:
            x0 = x0 + tl.load(R + row * SR + col).to(tl.float32)
            x1 = x1 + tl.load(R + row * SR + col + 1).to(tl.float32)
            x2 = x2 + tl.load(R + row * SR + col + 2).to(tl.float32)
            x3 = x3 + tl.load(R + row * SR + col + 3).to(tl.float32)
        a0 = a0 + _mul(x0, x0)
        a1 = a1 + _mul(x1, x1)
        a2 = a2 + _mul(x2, x2)
        a3 = a3 + _mul(x3, x3)
    partial = ((a0 + a1) + a2) + a3
    # Native block_x_reduce combines equal lanes across warps first.
    for bit in tl.static_range(LOG_BW - 1, -1, -1):
        other = tl.gather(partial, (lane + (1 << bit)) % BW, axis=0)
        partial = partial + other
    total = tl.sum(tl.where(lane == 0, partial, 0.0), axis=0)
    variance = _mul(total, 1.0 / C)
    inv = tl.extra.cuda.libdevice.rsqrt(variance + EPS)
    col = tl.arange(0, C)
    x = tl.load(X + row * SX + col).to(tl.float32)
    if HAS_R:
        x = x + tl.load(R + row * SR + col).to(tl.float32)
    w = tl.load(W + col).to(tl.float32) + 1.0
    y = _mul(_mul(x, inv), w).to(tl.bfloat16)
    if EMIT_NORMALIZED:
        tl.store(Y + row * C + col, y)
        tl.store(RES + row * C + col, x.to(tl.bfloat16))
    groups = y.to(tl.float32).reshape((C // 32, 32))
    scales = tl.maximum(
        _mul(tl.max(tl.abs(groups), 1), 1.0 / 448.0), 1.1754943508222875e-38
    )
    code = _round_e4m3(tl.clamp(tl.div_rn(groups, scales[:, None]), -448.0, 448.0))
    if EMIT_QDQ:
        q = _mul(code, scales[:, None]).to(tl.bfloat16).reshape((C,))
        tl.store(Q + row * C + col, q)
    if EMIT_FP8:
        tl.store(FP8 + row * OUTPUT_STRIDE + col, _encode_e4m3(code).reshape((C,)))
        tl.store(SCALE + row * (OUTPUT_STRIDE // 32) + tl.arange(0, C // 32), scales)


@overload
def residual_norm_group32_qdq(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    *,
    emit_fp8: Literal[False] = False,
    emit_qdq: bool = True,
    fp8_output: tuple[torch.Tensor, torch.Tensor] | None = None,
    emit_normalized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def residual_norm_group32_qdq(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    *,
    emit_fp8: Literal[True],
    emit_qdq: bool = True,
    fp8_output: tuple[torch.Tensor, torch.Tensor] | None = None,
    emit_normalized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, "Group32FP8Input"]: ...


def residual_norm_group32_qdq(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    *,
    emit_fp8: bool = False,
    emit_qdq: bool = True,
    fp8_output: tuple[torch.Tensor, torch.Tensor] | None = None,
    emit_normalized: bool = True,
):
    """Return BF16 norm, prepared group32 input and updated residual.

    Uses the native contiguous FP32 mean reduction tree, including its
    row-count-dependent block width. Inputs are not modified.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.stride(1) != 1
        or x.shape[1] not in (256, 2048)
        or weight.device != x.device
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or weight.shape != (x.shape[1],)
        or not weight.is_contiguous()
    ):
        raise ValueError("Requires CUDA BF16 rows of width 256/2048 and norm weights")
    if residual is not None and (
        residual.device != x.device
        or residual.dtype != x.dtype
        or residual.shape != x.shape
        or residual.stride(1) != 1
    ):
        raise ValueError("Residual must match the CUDA BF16 input rows")
    rows, width = x.shape
    normalized = torch.empty(
        x.shape if emit_normalized else (0,), device=x.device, dtype=x.dtype
    )
    prepared = torch.empty(
        x.shape if emit_qdq else (0,), device=x.device, dtype=x.dtype
    )
    updated = torch.empty_like(normalized)
    codes = scales = None
    if emit_fp8:
        if fp8_output is None:
            codes = torch.empty(x.shape, device=x.device, dtype=torch.uint8)
            scales = torch.empty(
                (rows, width // 32), device=x.device, dtype=torch.float32
            )
        else:
            codes, scales = fp8_output
            if (
                codes.shape != x.shape
                or scales.shape != (rows, width // 32)
                or codes.dtype != torch.uint8
                or scales.dtype != torch.float32
                or codes.device != x.device
                or scales.device != x.device
                or codes.stride(1) != 1
                or scales.stride(1) != 1
                or codes.stride(0) != scales.stride(0) * 32
            ):
                raise ValueError(
                    "FP8 norm output buffers must match rows and G32 strides"
                )
    if rows:
        height = min(1 << (rows.bit_length() - 1), 16)
        block_width = min(width // 4, 512 // height)
        _norm[(rows,)](
            x,
            residual if residual is not None else x,
            weight,
            normalized,
            prepared,
            updated,
            x.stride(0),
            residual.stride(0) if residual is not None else 0,
            width,
            block_width,
            block_width.bit_length() - 1,
            epsilon,
            residual is not None,
            codes,
            scales,
            EMIT_FP8=emit_fp8,
            EMIT_QDQ=emit_qdq,
            OUTPUT_STRIDE=codes.stride(0) if codes is not None else width,
            EMIT_NORMALIZED=emit_normalized,
            num_warps=4,
            enable_fp_fusion=False,
        )
    if emit_fp8:
        from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
            Group32FP8Input,
        )

        assert codes is not None and scales is not None
        return (
            normalized,
            prepared,
            updated,
            Group32FP8Input(codes.view(torch.float8_e4m3fn), scales),
        )
    return normalized, prepared, updated


def norm_concat_fp8(x, y, x_weight, y_weight, epsilon):
    """MTP's two independent norms write directly into one FP8 input buffer."""
    from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
        Group32FP8Input,
    )

    if x.shape != y.shape or x.device != y.device:
        raise ValueError("MTP norm inputs must have matching shapes and devices")
    rows, width = x.shape
    codes = torch.empty((rows, 2 * width), device=x.device, dtype=torch.uint8)
    scales = torch.empty((rows, 2 * width // 32), device=x.device, dtype=torch.float32)
    for i, (value, weight) in enumerate(((x, x_weight), (y, y_weight))):
        residual_norm_group32_qdq(
            value,
            None,
            weight,
            epsilon,
            emit_fp8=True,
            emit_qdq=False,
            emit_normalized=False,
            fp8_output=(
                codes[:, i * width : (i + 1) * width],
                scales[:, i * (width // 32) : (i + 1) * (width // 32)],
            ),
        )
    return Group32FP8Input(codes.view(torch.float8_e4m3fn), scales)


@triton.jit
def _gated_norm_fp8(
    X,
    Z,
    W,
    Q,
    S,
    SX,
    SZ,
    ROWS,
    VALID_ROWS,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Same row reduction, sigmoid and multiplication ordering as FLA RMSNormGated.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, WIDTH)
    mask = rows[:, None] < VALID_ROWS
    x = tl.load(X + rows[:, None] * SX + cols[None, :], mask, 0).to(tl.float32)
    z = tl.load(Z + rows[:, None] * SZ + cols[None, :], mask, 0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / WIDTH + EPS)
    y = (x * inv[:, None]) * w[None, :]
    gate = tl.sigmoid(z)
    if ACTIVATION != "sigmoid":
        gate = z * gate
    y = (y * gate).to(tl.bfloat16).to(tl.float32)
    groups = y.reshape((BLOCK_ROWS, WIDTH // 32, 32))
    scale = tl.maximum(
        tl.max(tl.abs(groups), 2) * (1.0 / 448.0), 1.1754943508222875e-38
    )
    code = _round_e4m3(tl.clamp(tl.div_rn(groups, scale[:, :, None]), -448.0, 448.0))
    tl.store(
        Q + rows[:, None] * WIDTH + cols[None, :],
        _encode_e4m3(code).reshape((BLOCK_ROWS, WIDTH)),
        rows[:, None] < ROWS,
    )
    tl.store(
        S + rows[:, None] * (WIDTH // 32) + tl.arange(0, WIDTH // 32)[None, :],
        scale,
        rows[:, None] < ROWS,
    )


def gated_norm_fp8(x, z, weight, epsilon, activation="silu", *, valid_rows=None):
    from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
        Group32FP8Input,
    )
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        calc_rows_per_block,
    )

    if (
        x.ndim != 2
        or x.shape != z.shape
        or x.dtype != torch.bfloat16
        or z.dtype != torch.bfloat16
        or not x.is_cuda
        or z.device != x.device
        or x.shape[1] not in (128, 256)
        or weight.shape != (x.shape[1],)
        or weight.device != x.device
        or not weight.is_contiguous()
        or x.stride(1) != 1
        or z.stride(1) != 1
        or activation not in ("silu", "swish", "sigmoid")
    ):
        raise ValueError("Unsupported GDN norm/FP8 producer configuration")
    rows, width = x.shape
    codes = torch.empty(x.shape, device=x.device, dtype=torch.uint8)
    scales = torch.empty((rows, width // 32), device=x.device, dtype=torch.float32)
    if rows:
        block_rows = calc_rows_per_block(rows, x.device)
        _gated_norm_fp8[(triton.cdiv(rows, block_rows),)](
            x,
            z,
            weight,
            codes,
            scales,
            x.stride(0),
            z.stride(0),
            rows,
            rows if valid_rows is None else valid_rows,
            width,
            epsilon,
            block_rows,
            activation,
            num_warps=max(width // 256, 1),
        )
    return Group32FP8Input(codes.view(torch.float8_e4m3fn), scales)
