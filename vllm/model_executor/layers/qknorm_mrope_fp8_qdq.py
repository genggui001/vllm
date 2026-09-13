# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 norm/RoPE/QDQ epilogue with native FP32 intermediates."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _static_qdq(value, scale):
    value = value.to(tl.bfloat16).to(tl.float32)
    divided = tl.div_rn(value, scale)
    normalized = tl.inline_asm_elementwise(
        "max.f32 $0, $1, 0fC3E00000; min.f32 $0, $0, 0f43E00000;",
        constraints="=f,f",
        args=[divided],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    magnitude = tl.abs(normalized)
    bits = magnitude.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFFF + ((bits >> 20) & 1)) & 0xFFF00000
    quantized = tl.where(
        magnitude < 0.015625,
        tl.extra.cuda.libdevice.nearbyint(magnitude * 512.0) * 0.001953125,
        rounded.to(tl.float32, bitcast=True),
    )
    return (
        tl.extra.cuda.libdevice.copysign(tl.minimum(quantized, 448.0), normalized)
        * scale
    )


@triton.jit
def _qknorm_mrope_qdq_epilogue(
    qkv,
    q_sum,
    k_sum,
    q_weight,
    k_weight,
    cache,
    positions,
    q_scale,
    q_out,
    k_out,
    gate_out,
    qkv_stride,
    pos_stride,
    cache_stride,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    THREE_D: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= HQ
    local_head = tl.where(is_k, head - HQ, head)
    if is_k:
        src = qkv + token * qkv_stride + 2 * HQ * D + local_head * D
        weight = k_weight
        total = tl.load(k_sum + token * HK + local_head)
        dst = k_out + (token * HK + local_head) * D
    else:
        src = qkv + token * qkv_stride + local_head * 2 * D
        weight = q_weight
        total = tl.load(q_sum + token * HQ + local_head)
        dst = q_out + (token * HQ + local_head) * D
    inv = tl.extra.cuda.libdevice.rsqrt(total * (1.0 / D) + EPS)
    col = tl.arange(0, D)
    value = tl.load(src + col).to(tl.float32)
    effective_weight = tl.load(weight + col).to(tl.float32) + 1.0
    norm = (value * inv) * effective_weight

    half = RD // 2
    rot_col = col % half
    if THREE_D:
        if INTERLEAVED:
            is_h = (rot_col % 3 == 1) & (rot_col < 3 * SECTION_H)
            is_w = (rot_col % 3 == 2) & (rot_col < 3 * SECTION_W)
            axis = tl.where(is_w, 2, tl.where(is_h, 1, 0))
        else:
            axis = tl.where(
                rot_col < SECTION_T, 0, tl.where(rot_col < SECTION_T + SECTION_H, 1, 2)
            )
        pos = tl.load(positions + axis * pos_stride + token).to(tl.int64)
    else:
        pos = tl.load(positions + token).to(tl.int64)
    cos = tl.load(cache + pos * cache_stride + rot_col).to(tl.float32)
    sin = tl.load(cache + pos * cache_stride + half + rot_col).to(tl.float32)
    partner_col = tl.where(col < half, col + half, col - half)
    partner = tl.load(src + partner_col, col < RD, other=0).to(tl.float32)
    partner_w = tl.load(weight + partner_col, col < RD, other=0).to(tl.float32) + 1.0
    partner_norm = (partner * inv) * partner_w
    rotary = tl.where(
        col < half, norm * cos - partner_norm * sin, norm * cos + partner_norm * sin
    )
    result = tl.where(col < RD, rotary, norm)
    if not is_k:
        result = _static_qdq(result, tl.load(q_scale))
        gate = tl.load(src + D + col)
        tl.store(gate_out + (token * HQ + local_head) * D + col, gate)
    tl.store(dst + col, result)


def qknorm_mrope_qdq(
    qkv,
    q_weight,
    k_weight,
    positions,
    cache,
    q_scale,
    eps=1.0e-6,
    num_q_heads=32,
    num_kv_heads=4,
    head_dim=256,
    rotary_dim=64,
    sections=(11, 11, 10),
    interleaved=True,
):
    if qkv.dtype != torch.bfloat16 or head_dim != 256:
        raise TypeError("SM80 norm/RoPE/QDQ requires BF16 input and head_dim 256")
    m = qkv.shape[0]
    qg, k, v = qkv.split(
        [2 * num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim],
        dim=-1,
    )
    q, gate = qg.view(m, num_q_heads, 2 * head_dim).chunk(2, dim=-1)
    q = q.reshape(m, num_q_heads * head_dim)
    # Match the independent reduction used by the native compiled chain.
    q_sum = q.view(m, num_q_heads, head_dim).float().square().sum(-1)
    k_sum = k.view(m, num_kv_heads, head_dim).float().square().sum(-1)
    q_out = torch.empty((m, num_q_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(
        (m, num_kv_heads * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    gate_out = torch.empty_like(q_out)
    if m:
        _qknorm_mrope_qdq_epilogue[(m, num_q_heads + num_kv_heads)](
            qkv,
            q_sum,
            k_sum,
            q_weight,
            k_weight,
            cache,
            positions,
            q_scale,
            q_out,
            k_out,
            gate_out,
            qkv.stride(0),
            positions.stride(0) if positions.ndim == 2 else 0,
            cache.stride(0),
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            positions.ndim == 2,
            interleaved,
            *sections,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, k_out, v, gate_out
