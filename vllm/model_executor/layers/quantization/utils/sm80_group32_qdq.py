# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 group32 activation QDQ with the training BF16 rounding boundaries."""

import threading

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton

_silu_tables: dict[torch.device, torch.Tensor] = {}
_silu_tables_lock = threading.Lock()


def _training_silu_table(x: torch.Tensor) -> torch.Tensor | None:
    table = _silu_tables.get(x.device)
    if table is not None:
        return table
    with torch.accelerator.device_index(x.device.index):
        if torch.cuda.is_current_stream_capturing():
            return None
        with _silu_tables_lock:
            table = _silu_tables.get(x.device)
            if table is None:
                codes = torch.arange(65536, device=x.device, dtype=torch.int32)
                table = F.silu(codes.to(torch.int16).view(torch.bfloat16))
                torch.cuda.current_stream().synchronize()
                _silu_tables[x.device] = table
    return table


@triton.jit
def _round_e4m3(values):
    # Software E4M3 RNE, as in the existing SM80 token QDQ backend. Only
    # the conversion primitive is shared; group scaling has different semantics.
    magnitude = tl.abs(values)
    bits = magnitude.to(tl.int32, bitcast=True)
    rounded = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -0x100000
    normal = rounded.to(tl.float32, bitcast=True)
    subnormal = tl.extra.cuda.libdevice.rint(magnitude * 512.0) * 0.001953125
    quantized = tl.minimum(tl.where(magnitude < 0.015625, subnormal, normal), 448.0)
    sign = values.to(tl.int32, bitcast=True) & -2147483648
    result = (quantized.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    return tl.where(values == values, result, float("nan"))


@triton.jit
def _encode_e4m3(rounded):
    """Encode already-rounded finite E4M3 values without SM89 conversion."""
    bits = rounded.to(tl.int32, bitcast=True)
    magnitude = tl.abs(rounded)
    normal = ((bits & 0x7FFFFFFF) >> 20) - 960
    subnormal = (magnitude * 512.0).to(tl.int32)
    code = tl.where(magnitude < 0.015625, subnormal, normal)
    code = tl.where(rounded == rounded, code, 127)
    return (code | ((bits >> 24) & 128)).to(tl.uint8)


@triton.jit(do_not_specialize=["rows"])
def _group32_qdq_kernel(
    X,
    Y,
    P,
    Order,
    Table,
    rows,
    input_row_stride,
    input_col_stride,
    prob_stride,
    order_stride,
    WIDTH: tl.constexpr,
    SWIGLU: tl.constexpr,
    PROBABILITY: tl.constexpr,
    ROUTED: tl.constexpr,
    LOOKUP: tl.constexpr,
    GROUPS: tl.constexpr,
    Q=None,
    S=None,
    WRITE_FP8: tl.constexpr = False,
    WRITE_BF16: tl.constexpr = True,
):
    groups_per_row: tl.constexpr = WIDTH // 32
    group = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = group // groups_per_row
    col = (group % groups_per_row)[:, None] * 32 + tl.arange(0, 32)[None, :]
    offsets = row[:, None].to(tl.int64) * input_row_stride + col * input_col_stride
    value = tl.load(X + offsets, row[:, None] < rows, 0)
    if SWIGLU:
        up = tl.load(X + offsets + WIDTH * input_col_stride, row[:, None] < rows, 0).to(
            tl.float32
        )
        # Preserve training's explicit up + (+0), including negative zero.
        up = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[up, 0.0],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        if LOOKUP:
            code = value.to(tl.uint16, bitcast=True).to(tl.int32)
            silu = tl.load(Table + code).to(tl.float32)
        else:
            gate = value.to(tl.float32)
            silu = tl.div_rn(gate, 1.0 + tl.extra.cuda.libdevice.exp(-gate))
            silu = silu.to(tl.bfloat16).to(tl.float32)
        value = (silu * up).to(tl.bfloat16)
    value = value.to(tl.float32)
    absmax = tl.max(tl.abs(value), axis=1)
    # PyTorch's CUDA tensor/scalar division multiplies by the rounded FP32
    # reciprocal. div.rn(absmax, 448) differs at FP8 and BF16 halfway values.
    scale = tl.maximum(absmax * (1.0 / 448.0), 1.1754943508222875e-38)
    normalized = tl.clamp(tl.div_rn(value, scale[:, None]), -448.0, 448.0)
    rounded = _round_e4m3(normalized)
    if WRITE_FP8:
        tl.store(
            Q + row[:, None].to(tl.int64) * WIDTH + col,
            _encode_e4m3(rounded),
            row[:, None] < rows,
        )
        tl.store(S + group, scale, row < rows)
    result = (rounded * scale[:, None]).to(tl.bfloat16)
    if PROBABILITY:
        if ROUTED:
            slot = tl.load(Order + row * order_stride, row < rows, 0).to(tl.int64)
        else:
            slot = row.to(tl.int64)
        prob = tl.load(P + slot * prob_stride, row < rows, 0)
        # FC2 consumes BF16(QDQ(activation) * FP32(router_probability)).
        result = (result.to(tl.float32) * prob[:, None]).to(tl.bfloat16)
    if WRITE_BF16:
        tl.store(
            Y + row[:, None].to(tl.int64) * WIDTH + col, result, row[:, None] < rows
        )


def group32_fp8_quant(
    x: torch.Tensor, *, swiglu: bool = False, return_qdq: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Produce Dense FP8 input and optionally the existing MoE BF16 input."""
    width = _validate_input(x, swiglu=swiglu)
    rows = x.shape[0]
    codes = torch.empty((rows, width), dtype=torch.uint8, device=x.device)
    scales = torch.empty((rows, width // 32), dtype=torch.float32, device=x.device)
    qdq = (
        torch.empty((rows, width), dtype=x.dtype, device=x.device)
        if return_qdq
        else None
    )
    groups, warps = 8, 4
    if swiglu and width == 256:
        groups = 16 if rows >= 256 else 8
        warps = 2 if rows >= 8192 else 4
    table = (
        _training_silu_table(x) if swiglu and (width != 256 or rows >= 2048) else None
    )
    if rows:
        _group32_qdq_kernel[(triton.cdiv(rows * width // 32, groups),)](
            x,
            qdq if qdq is not None else x,
            x,
            x,
            table if table is not None else x,
            rows,
            x.stride(0),
            x.stride(1),
            0,
            0,
            WIDTH=width,
            SWIGLU=swiglu,
            PROBABILITY=False,
            ROUTED=False,
            LOOKUP=table is not None,
            GROUPS=groups,
            Q=codes,
            S=scales,
            WRITE_FP8=True,
            WRITE_BF16=return_qdq,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return codes.view(torch.float8_e4m3fn), scales, qdq


def _validate_input(x: torch.Tensor, *, swiglu: bool) -> int:
    divisor = 64 if swiglu else 32
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] == 0
        or x.shape[1] % divisor
    ):
        raise ValueError(
            f"Requires CUDA BF16 [M, K] with positive K divisible by {divisor}"
        )
    return x.shape[1] // (2 if swiglu else 1)


def _output(x: torch.Tensor, width: int, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return torch.empty((x.shape[0], width), device=x.device, dtype=x.dtype)
    if (
        out.device != x.device
        or out.dtype != x.dtype
        or out.shape != (x.shape[0], width)
        or not out.is_contiguous()
    ):
        raise ValueError("Output must be contiguous BF16 [M, K] on the input device")
    return out


def group32_fp8_qdq_fused(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Return training group32 QDQ, accepting strided two-dimensional inputs."""
    width = _validate_input(x, swiglu=False)
    out = _output(x, width, out)
    if not x.numel():
        return out
    groups = 8
    if width == 2048:
        groups = 32 if x.shape[0] >= 128 else (16 if x.shape[0] >= 16 else 8)
    _group32_qdq_kernel[(triton.cdiv(x.numel() // 32, groups),)](
        x,
        out,
        x,
        x,
        x,
        x.shape[0],
        x.stride(0),
        x.stride(1),
        0,
        0,
        WIDTH=width,
        SWIGLU=False,
        PROBABILITY=False,
        ROUTED=False,
        LOOKUP=False,
        GROUPS=groups,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def silu_mul_group32_fp8_qdq_fused(
    x: torch.Tensor,
    probabilities: torch.Tensor | None = None,
    routed_slots: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse training SwiGLU, group32 QDQ, and optional pre-FC2 probability.

    Args:
        x: BF16 [M, 2K] with gate followed by up.
        probabilities: FP32 vector in row order, or flattened top-k slot order.
        routed_slots: Integer [M] mapping expert-major rows to top-k slots.
        out: Optional contiguous BF16 [M, K] output.
    """
    width = _validate_input(x, swiglu=True)
    if probabilities is not None and (
        probabilities.device != x.device
        or probabilities.dtype != torch.float32
        or probabilities.ndim != 1
        or (routed_slots is None and probabilities.numel() != x.shape[0])
    ):
        raise ValueError("Probabilities must be an FP32 vector on the input device")
    if routed_slots is not None and (
        probabilities is None
        or routed_slots.device != x.device
        or routed_slots.dtype not in (torch.int32, torch.int64)
        or routed_slots.shape != (x.shape[0],)
    ):
        raise ValueError("Routed slots must map rows into the probability vector")
    out = _output(x, width, out)
    if not x.numel():
        return out
    rows = x.shape[0]
    groups, warps = 8, 4
    lookup = width != 256 or rows >= 2048
    if width == 256:
        groups = 16 if rows >= 256 else 8
        warps = 2 if rows >= 8192 else 4
    table = _training_silu_table(x) if lookup else None
    _group32_qdq_kernel[(triton.cdiv(x.shape[0] * width // 32, groups),)](
        x,
        out,
        probabilities if probabilities is not None else x,
        routed_slots if routed_slots is not None else x,
        table if table is not None else x,
        x.shape[0],
        x.stride(0),
        x.stride(1),
        probabilities.stride(0) if probabilities is not None else 0,
        routed_slots.stride(0) if routed_slots is not None else 0,
        WIDTH=width,
        SWIGLU=True,
        PROBABILITY=probabilities is not None,
        ROUTED=routed_slots is not None,
        LOOKUP=table is not None,
        GROUPS=groups,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out
