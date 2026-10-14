# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Group32 QK norm/MRoPE preserving native reduction and rotary rounding."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _native_variance(Src, BW: tl.constexpr, LOG_BW: tl.constexpr):
    lane = tl.arange(0, BW)
    a0 = tl.full((BW,), 0, tl.float32)
    a1 = tl.full((BW,), 0, tl.float32)
    a2 = tl.full((BW,), 0, tl.float32)
    a3 = tl.full((BW,), 0, tl.float32)
    for step in range(256 // (BW * 4)):
        col = (step * BW + lane) * 4
        x0 = tl.load(Src + col).to(tl.float32)
        x1 = tl.load(Src + col + 1).to(tl.float32)
        x2 = tl.load(Src + col + 2).to(tl.float32)
        x3 = tl.load(Src + col + 3).to(tl.float32)
        a0 = a0 + x0 * x0
        a1 = a1 + x1 * x1
        a2 = a2 + x2 * x2
        a3 = a3 + x3 * x3
    partial = ((a0 + a1) + a2) + a3
    for bit in tl.static_range(LOG_BW - 1, -1, -1):
        other = tl.gather(partial, (lane + (1 << bit)) % BW, axis=0)
        partial = partial + other
    total = tl.sum(tl.where(lane == 0, partial, 0.0), axis=0)
    return total * (1.0 / 256.0)


@triton.jit
def _bf16_fma(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.bf16 $0, $1, $2, $3;",
        constraints="=h,h,h,h",
        args=[a.to(tl.bfloat16), b.to(tl.bfloat16), c.to(tl.bfloat16)],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _qknorm_rope(
    X,
    QWeight,
    KWeight,
    Cache,
    Positions,
    Q,
    K,
    Gate,
    XS,
    PS,
    CS,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    THREE_D: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    EPS: tl.constexpr,
    ROUND_PRODUCTS: tl.constexpr,
    FMA: tl.constexpr,
    QBW: tl.constexpr,
    KBW: tl.constexpr,
    QL: tl.constexpr,
    KL: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= HQ
    h = tl.where(is_k, head - HQ, head)
    if is_k:
        src = X + row * XS + 2 * HQ * D + h * D
        w = KWeight
        variance = _native_variance(src, KBW, KL)
        dst = K + (row * HK + h) * D
    else:
        src = X + row * XS + h * 2 * D
        w = QWeight
        variance = _native_variance(src, QBW, QL)
        dst = Q + (row * HQ + h) * D
    inv = tl.extra.cuda.libdevice.rsqrt(variance + EPS)
    col = tl.arange(0, D)
    x = tl.load(src + col).to(tl.float32)
    weight = tl.load(w + col).to(tl.float32) + 1.0
    norm = (x * inv) * weight
    norm = norm.to(tl.bfloat16).to(tl.float32)
    half = RD // 2
    rc = col % half
    if THREE_D:
        if INTERLEAVED:
            h_axis = (rc % 3 == 1) & (rc < 3 * SH)
            w_axis = (rc % 3 == 2) & (rc < 3 * SW)
            axis = tl.where(w_axis, 2, tl.where(h_axis, 1, 0))
        else:
            axis = tl.where(rc < ST, 0, tl.where(rc < ST + SH, 1, 2))
        pos = tl.load(Positions + axis * PS + row).to(tl.int64)
    else:
        pos = tl.load(Positions + row).to(tl.int64)
    cos = tl.load(Cache + pos * CS + rc).to(tl.float32)
    sin = tl.load(Cache + pos * CS + half + rc).to(tl.float32)
    pc = tl.where(col < half, col + half, col - half)
    partner = tl.load(src + pc, col < RD, 0).to(tl.float32)
    pw = tl.load(w + pc, col < RD, 0).to(tl.float32) + 1.0
    pn = (partner * inv) * pw
    pn = pn.to(tl.bfloat16).to(tl.float32)
    a = norm * cos
    b = pn * sin
    if ROUND_PRODUCTS:
        a = a.to(tl.bfloat16).to(tl.float32)
        b = b.to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(col < half, a - b, a + b)
    if FMA:
        rotated = tl.where(
            col < half,
            _bf16_fma(norm, cos, -b).to(tl.float32),
            _bf16_fma(pn, sin, a).to(tl.float32),
        )
    result = tl.where(col < RD, rotated, norm)
    tl.store(dst + col, result)
    if not is_k:
        tl.store(Gate + (row * HQ + h) * D + col, tl.load(src + D + col))


def qknorm_mrope_native_order(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    *,
    eps: float,
    cuda_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw Q/K, the V view, and the unsigmoided gate for SM80 TP2.

    Q/K retain the existing BF16 norm boundary. Native rotary rounds both
    products; CUDA 3D rotary uses BF16 FMA with different product ordering in
    each half. No query QDQ is performed here.
    """
    if (
        not qkv.is_cuda
        or qkv.dtype != torch.bfloat16
        or qkv.ndim != 2
        or qkv.shape[1] != 6144
        or qkv.stride(1) != 1
        or positions.device != qkv.device
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.ndim not in (1, 2)
        or positions.shape[-1] != qkv.shape[0]
        or positions.stride(-1) != 1
        or (positions.ndim == 2 and positions.shape[0] != 3)
        or cache.device != qkv.device
        or cache.dtype != torch.bfloat16
        or cache.ndim != 2
        or cache.shape[1] != 64
        or cache.stride(1) != 1
    ):
        raise ValueError("Expected SM80 BF16 gated QKV and NeoX MRoPE inputs")
    for weight in (q_weight, k_weight):
        if (
            weight.device != qkv.device
            or weight.dtype not in (torch.bfloat16, torch.float32)
            or weight.shape != (256,)
            or not weight.is_contiguous()
        ):
            raise ValueError("Expected contiguous Q/K norm weights of length 256")
    rows = qkv.shape[0]
    query = torch.empty((rows, 2048), device=qkv.device, dtype=qkv.dtype)
    key = torch.empty((rows, 1024), device=qkv.device, dtype=qkv.dtype)
    gate = torch.empty_like(query)
    value = qkv[:, 5120:]
    if rows:
        # Reduce.cuh vectorizes the contiguous FP32 square into four serial
        # accumulators. Its block width depends on the total Q or K head rows.
        q_height = min(1 << ((rows * 8).bit_length() - 1), 16)
        k_height = min(1 << ((rows * 4).bit_length() - 1), 16)
        q_width, k_width = min(64, 512 // q_height), min(64, 512 // k_height)
        is_3d = positions.ndim == 2
        _qknorm_rope[(rows, 12)](
            qkv,
            q_weight,
            k_weight,
            cache,
            positions,
            query,
            key,
            gate,
            qkv.stride(0),
            positions.stride(0) if is_3d else 0,
            cache.stride(0),
            8,
            4,
            256,
            64,
            is_3d,
            True,
            11,
            11,
            10,
            eps,
            (is_3d if cuda_rope else True),
            cuda_rope and is_3d,
            q_width,
            k_width,
            q_width.bit_length() - 1,
            k_width.bit_length() - 1,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return query, key, value, gate
