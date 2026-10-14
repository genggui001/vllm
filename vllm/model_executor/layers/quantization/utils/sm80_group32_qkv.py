# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static QKV QDQ with the saved FP32 scales and reference BF16 rounding."""

import torch

from vllm.model_executor.layers.quantization.utils.sm80_group32_qdq import _round_e4m3
from vllm.triton_utils import tl, triton


@triton.jit
def _qkv_qdq_kernel(
    Q,
    K,
    V,
    QS,
    KS,
    VS,
    OQ,
    OK,
    OV,
    qrs,
    qcs,
    krs,
    kcs,
    vrs,
    vcs,
    QWIDTH: tl.constexpr,
    KWIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    if part == 0:
        src, dst, width, rs, cs, scale = Q, OQ, QWIDTH, qrs, qcs, tl.load(QS)
    elif part == 1:
        src, dst, width, rs, cs, scale = K, OK, KWIDTH, krs, kcs, tl.load(KS)
    else:
        src, dst, width, rs, cs, scale = V, OV, KWIDTH, vrs, vcs, tl.load(VS)
    x = tl.load(src + row * rs + col * cs, col < width, 0).to(tl.float32)
    divided = tl.clamp(tl.div_rn(x, scale), -448.0, 448.0)
    result = (_round_e4m3(divided) * scale).to(tl.bfloat16)
    tl.store(dst + row * width + col, result, col < width)


def qkv_fp8_qdq_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply three saved scales in one launch, accepting strided Q/K/V views."""
    tensors = (q, k, v)
    scales = (q_scale, k_scale, v_scale)
    if (
        not q.is_cuda
        or any(x.ndim != 2 or x.dtype != torch.bfloat16 for x in tensors)
        or any(x.device != q.device or x.shape[1] == 0 for x in tensors)
        or not (q.shape[0] == k.shape[0] == v.shape[0] and k.shape == v.shape)
    ):
        raise ValueError("QKV requires CUDA BF16 matrices with matching token counts")
    if any(
        s.dtype != torch.float32 or s.numel() != 1 or s.device != q.device
        for s in scales
    ):
        raise ValueError("QKV requires one saved FP32 scale per tensor on its device")
    out = tuple(torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in tensors)
    if q.shape[0]:
        _qkv_qdq_kernel[(q.shape[0], 3)](
            q,
            k,
            v,
            *scales,
            *out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            q.shape[1],
            k.shape[1],
            triton.next_power_of_2(max(q.shape[1], k.shape[1])),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
