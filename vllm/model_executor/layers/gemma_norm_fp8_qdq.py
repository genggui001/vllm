# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 fusion of Gemma residual RMSNorm and routed-expert FP8 QDQ."""

import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_fp8_qdq_fused_moe import (
    _e4m3fn_qdq_software,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _mul_rn_ieee(a, b):
    # Explicit rounding blocks contraction while preserving subnormal inputs.
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gemma_norm_qdq_direct_ieee(
    X, R, W, NORM, QDQ, C: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + column, column < C, 0).to(tl.float32)
    residual = tl.load(R + row * C + column, column < C, 0).to(tl.float32)
    weight = tl.load(W + column, column < C, 0).to(tl.float32) + 1.0
    summed = x + residual
    variance = _mul_rn_ieee(tl.sum(_mul_rn_ieee(summed, summed), 0), 1.0 / C)
    normalized = _mul_rn_ieee(summed, tl.extra.cuda.libdevice.rsqrt(variance + EPS))
    normalized = _mul_rn_ieee(normalized, weight).to(NORM.dtype.element_ty)
    tl.store(NORM + row * C + column, normalized, column < C)
    # This BF16 rounding boundary is part of the accepted QDQ contract.
    values = normalized.to(tl.float32)
    values = tl.where(column < C, values, 0.0)
    scale = tl.maximum(
        _mul_rn_ieee(tl.max(tl.abs(values), 0), 1.0 / 448.0), 1.1754943508222875e-38
    )
    quantized = _mul_rn_ieee(
        _e4m3fn_qdq_software(tl.clamp(tl.div_rn(values, scale), -448.0, 448.0)), scale
    )
    tl.store(QDQ + row * C + column, quantized, column < C)


def norm_qdq_deferred_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 norm and routed-expert QDQ, leaving the residual untouched."""
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
    assert x.shape == residual.shape and x.dtype == residual.dtype
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    assert weight.ndim == 1 and weight.numel() == x.shape[1]
    assert x.device == residual.device == weight.device
    norm, qdq = torch.empty_like(x), torch.empty_like(x)
    if x.shape[0]:
        _gemma_norm_qdq_direct_ieee[(x.shape[0],)](
            x,
            residual,
            weight,
            norm,
            qdq,
            x.shape[1],
            epsilon,
            triton.next_power_of_2(x.shape[1]),
            num_warps=8,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return norm, qdq
