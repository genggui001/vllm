# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma residual RMSNorm and training group32 QDQ with native mean ordering."""

import torch

from vllm.model_executor.layers.quantization.utils.sm80_group32_qdq import _round_e4m3
from vllm.triton_utils import tl, triton


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
    tl.store(Y + row * C + col, y)
    tl.store(RES + row * C + col, x.to(tl.bfloat16))
    groups = y.to(tl.float32).reshape((C // 32, 32))
    scales = tl.maximum(
        _mul(tl.max(tl.abs(groups), 1), 1.0 / 448.0), 1.1754943508222875e-38
    )
    code = _round_e4m3(tl.clamp(tl.div_rn(groups, scales[:, None]), -448.0, 448.0))
    q = _mul(code, scales[:, None]).to(tl.bfloat16).reshape((C,))
    tl.store(Q + row * C + col, q)


def residual_norm_group32_qdq(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    normalized, prepared, updated = (
        torch.empty(x.shape, device=x.device, dtype=x.dtype) for _ in range(3)
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
            num_warps=4,
            enable_fp_fusion=False,
        )
    return normalized, prepared, updated
