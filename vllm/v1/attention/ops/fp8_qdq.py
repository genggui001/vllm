# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference FP8 quantize-dequantize operations.

These operations intentionally materialize the quantized values. They are a
numerical reference for hardware that cannot execute FP8 attention natively;
performance-oriented backends should fuse the conversion with their producer
or consumer kernels.
"""

import torch

FP8_E4M3_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = torch.finfo(FP8_E4M3_DTYPE).max


def scaled_fp8_e4m3_qdq(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply saturating, per-tensor FP8 E4M3 QDQ with a static scale.

    The implementation matches the QAT path used by the target model:
    ``BF16 -> FP32 divide -> clamp[-448, 448] -> E4M3FN -> FP32 multiply
    -> BF16``.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Expected bfloat16 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("FP8 E4M3 QDQ requires a CUDA input")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("scale must be a scalar float32 tensor")
    if scale.device != x.device:
        raise ValueError("scale must be on the same device as x")

    compute_scale = scale.reshape(1)
    scaled = x / compute_scale
    quantized = scaled.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX).to(FP8_E4M3_DTYPE)
    return (quantized.float() * compute_scale).to(x.dtype)


__all__ = ["scaled_fp8_e4m3_qdq"]
