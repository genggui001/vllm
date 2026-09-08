# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software E4M3FN decode helpers for the opt-in SM80 KV-cache backend.

Ampere can store E4M3FN bytes but has no native FP8 tensor-core attention.
The attention kernel therefore expands each cache tile to FP32/BF16 as it is
loaded. The regular Triton attention path never selects this conversion.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVQuantMode


@triton.jit
def e4m3fn_uint8_to_float32(data):
    """Decode raw OCP E4M3FN bytes without native FP8 instructions."""
    bits = data.to(tl.uint8)
    magnitude_bits = bits & 0x7F
    exponent = (magnitude_bits >> 3) & 0x0F
    mantissa = magnitude_bits & 0x07

    mantissa_f32 = mantissa.to(tl.float32)
    subnormal = mantissa_f32 * 0.001953125  # mantissa * 2**-9
    normal = (1.0 + mantissa_f32 * 0.125) * tl.exp2(
        exponent.to(tl.float32) - 7.0
    )
    magnitude = tl.where(exponent == 0, subnormal, normal)
    magnitude = tl.where(magnitude_bits == 0x7F, float("nan"), magnitude)
    sign = tl.where((bits & 0x80) == 0, 1.0, -1.0)
    return sign * magnitude


@triton.jit
def scaled_e4m3fn_qdq_float32(values, scale):
    """Apply static-scale E4M3FN QDQ without FP8 instructions."""
    normalized = tl.clamp(
        tl.div_rn(values.to(tl.float32), scale),
        -448.0,
        448.0,
    )
    magnitude = tl.abs(normalized)
    normal_magnitude = tl.maximum(magnitude, 0.015625)
    exponent = tl.floor(tl.log2(normal_magnitude))
    exponent = tl.maximum(tl.minimum(exponent, 8.0), -6.0)
    normal_step = tl.exp2(exponent - 3.0)
    step = tl.where(magnitude < 0.015625, 0.001953125, normal_step)
    quantized_magnitude = tl.extra.cuda.libdevice.rint(magnitude / step) * step
    quantized_magnitude = tl.minimum(quantized_magnitude, 448.0)
    quantized = tl.where(normalized < 0.0, -quantized_magnitude, quantized_magnitude)
    return quantized * scale


@triton.jit
def _scaled_e4m3fn_qdq_inplace_kernel(
    tensor_ptr,
    scale_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    values = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
    qdq = scaled_e4m3fn_qdq_float32(values, tl.load(scale_ptr))
    tl.store(tensor_ptr + offsets, qdq, mask=mask)


def scaled_e4m3fn_qdq_inplace(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    *,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """Apply one-pass static-scale E4M3FN QDQ to a temporary BF16 buffer."""
    if tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise TypeError("tensor must be a CUDA bfloat16 tensor")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("scale must be a scalar float32 tensor")
    if scale.device != tensor.device:
        raise ValueError("scale must be on the same device as tensor")
    if num_tokens is None:
        numel = tensor.numel()
    else:
        if tensor.ndim < 1 or not 0 <= num_tokens <= tensor.shape[0]:
            raise ValueError(
                f"invalid num_tokens={num_tokens} for shape {tensor.shape}"
            )
        numel = num_tokens * tensor.stride(0)

    block_size = 256
    _scaled_e4m3fn_qdq_inplace_kernel[(triton.cdiv(numel, block_size),)](
        tensor,
        scale,
        numel,
        BLOCK_SIZE=block_size,
    )
    return tensor


@triton.jit
def _decode_e4m3fn_kernel(
    src_ptr,
    dst_ptr,
    scale_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    encoded = tl.load(src_ptr + offsets, mask=mask, other=0)
    decoded = e4m3fn_uint8_to_float32(encoded) * tl.load(scale_ptr)
    tl.store(dst_ptr + offsets, decoded, mask=mask)


def decode_e4m3fn_uint8(
    encoded: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Materialize decoded values for tests and offline diagnostics."""
    if encoded.dtype != torch.uint8 or not encoded.is_cuda:
        raise TypeError("encoded must be a CUDA uint8 tensor")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("scale must be a scalar float32 tensor")
    if scale.device != encoded.device:
        raise ValueError("scale must be on the same device as encoded")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported output dtype: {dtype}")

    output = torch.empty_like(encoded, dtype=dtype)
    block_size = 256
    grid = (triton.cdiv(encoded.numel(), block_size),)
    _decode_e4m3fn_kernel[grid](
        encoded,
        output,
        scale,
        encoded.numel(),
        BLOCK_SIZE=block_size,
    )
    return output


def sm80_fp8_unified_attention(**kwargs) -> None:
    """Dedicated entry point for BF16 Q with raw E4M3FN K/V pages."""
    query = kwargs["q"]
    key_cache = kwargs["k"]
    value_cache = kwargs["v"]
    if query.dtype != torch.bfloat16:
        raise TypeError(f"SM80 FP8 attention requires BF16 query, got {query.dtype}")
    if key_cache.dtype != torch.uint8 or value_cache.dtype != torch.uint8:
        raise TypeError("SM80 FP8 attention requires uint8 K/V cache storage")
    if kwargs.get("q_descale") is not None:
        raise ValueError("SM80 FP8 attention currently keeps Q in BF16")
    if kwargs.get("k_descale") is None or kwargs.get("v_descale") is None:
        raise ValueError("SM80 FP8 attention requires static K/V scales")

    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    kwargs["kv_quant_mode"] = KVQuantMode.SM80_FP8_PER_TENSOR
    unified_attention(**kwargs)


__all__ = [
    "decode_e4m3fn_uint8",
    "e4m3fn_uint8_to_float32",
    "scaled_e4m3fn_qdq_float32",
    "scaled_e4m3fn_qdq_inplace",
    "sm80_fp8_unified_attention",
]
