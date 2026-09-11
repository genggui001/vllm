# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80-friendly fused FP8 QDQ helpers for Marlin W4A16 MoE.

The Marlin FP8 activation GEMM uses FP8 tensor-core instructions and therefore
cannot run on SM80.  This opt-in backend keeps the proven W4A16 Marlin GEMMs,
but folds the QAT-compatible dynamic per-token FP8 round trip into one Triton
kernel.  The FC2 path also folds SwiGLU into that same kernel.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.triton_utils import tl, triton


@triton.jit
def _e4m3fn_qdq_software(values):
    """Round FP32 values to finite E4M3 and decode, without FP8 hardware."""
    magnitude = tl.abs(values)
    bits = magnitude.to(tl.int32, bitcast=True)
    # Keep three fraction bits. The retained LSB supplies the ties-to-even bit.
    rounded_bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -0x100000
    normal = rounded_bits.to(tl.float32, bitcast=True)
    subnormal = tl.extra.cuda.libdevice.rint(magnitude * 512.0) * 0.001953125
    quantized_magnitude = tl.where(magnitude < 0.015625, subnormal, normal)
    quantized_magnitude = tl.minimum(quantized_magnitude, 448.0)
    # Preserve the E4M3 sign bit, including an input negative zero.
    sign = values.to(tl.int32, bitcast=True) & -2147483648
    magnitude_bits = quantized_magnitude.to(tl.int32, bitcast=True)
    return (magnitude_bits | sign).to(tl.float32, bitcast=True)


@triton.jit
def _fp8_e4m3_per_token_qdq_kernel(
    input_ptr,
    output_ptr,
    input_stride,
    output_stride,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    values = tl.load(input_ptr + row * input_stride + offsets, mask=mask, other=0.0)
    values = values.to(tl.float32)

    absmax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(absmax * (1.0 / 448.0), 1.1754943508222875e-38)
    normalized = tl.clamp(tl.div_rn(values, scale), -448.0, 448.0)
    dequantized = _e4m3fn_qdq_software(normalized) * scale

    tl.store(
        output_ptr + row * output_stride + offsets,
        dequantized,
        mask=mask,
    )


@triton.jit
def _silu_mul_fp8_e4m3_per_token_qdq_kernel(
    input_ptr,
    output_ptr,
    input_stride,
    output_stride,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    input_row = input_ptr + row * input_stride

    gate = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(input_row + n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + tl.extra.cuda.libdevice.exp(-gate))
    # The packed CUDA activation rounds SiLU to the input dtype before the
    # multiply, then rounds the product again when it stores the activation.
    silu = silu.to(input_ptr.dtype.element_ty).to(tl.float32)
    activated = (silu * up).to(input_ptr.dtype.element_ty).to(tl.float32)
    absmax = tl.max(tl.abs(activated), axis=0)
    scale = tl.maximum(absmax * (1.0 / 448.0), 1.1754943508222875e-38)
    normalized = tl.clamp(tl.div_rn(activated, scale), -448.0, 448.0)
    dequantized = _e4m3fn_qdq_software(normalized) * scale

    tl.store(
        output_ptr + row * output_stride + offsets,
        dequantized,
        mask=mask,
    )


def _launch_config(n_cols: int) -> tuple[int, int]:
    if n_cols <= 0:
        raise ValueError("FP8 per-token QDQ requires a positive row width.")
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError(f"FP8 per-token QDQ row is too wide: {n_cols}.")
    num_warps = 4 if block_size <= 2048 else 8
    return block_size, num_warps


def fp8_e4m3_per_token_qdq_fused(
    x: torch.Tensor, output: torch.Tensor | None = None
) -> torch.Tensor:
    """Run dynamic per-token E4M3 QDQ in one Triton kernel."""
    if x.ndim != 2:
        raise ValueError(f"FP8 per-token QDQ expects a 2D tensor, got {x.shape}.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"FP8 per-token QDQ expects float16 or bfloat16 input, got {x.dtype}."
        )
    if not x.is_contiguous():
        raise ValueError("FP8 per-token QDQ expects contiguous input.")
    if not x.is_cuda:
        raise ValueError("FP8 per-token QDQ expects a CUDA input.")
    if output is None:
        output = torch.empty_like(x)
    if output.shape != x.shape or output.dtype != x.dtype or not output.is_contiguous():
        raise ValueError("FP8 per-token QDQ output must match the contiguous input.")
    if output.device != x.device:
        raise ValueError("FP8 per-token QDQ tensors must be on the same CUDA device.")

    rows, n_cols = x.shape
    block_size, num_warps = _launch_config(n_cols)
    if rows == 0:
        return output
    _fp8_e4m3_per_token_qdq_kernel[(rows,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_cols=n_cols,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def silu_mul_fp8_e4m3_per_token_qdq_fused(
    input: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Run SwiGLU and dynamic per-token E4M3 QDQ in one Triton kernel."""
    if input.ndim != 2 or output.ndim != 2:
        raise ValueError("Fused SwiGLU FP8 QDQ expects 2D tensors.")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported input dtype {input.dtype}.")
    if not input.is_cuda or output.device != input.device:
        raise ValueError(
            "Fused SwiGLU FP8 QDQ tensors must be on the same CUDA device."
        )
    if input.shape[0] != output.shape[0] or input.shape[1] != output.shape[1] * 2:
        raise ValueError(
            f"Expected input [M, 2N] and output [M, N], got "
            f"{input.shape} and {output.shape}."
        )
    if (
        input.dtype != output.dtype
        or not input.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("Fused SwiGLU FP8 QDQ expects matching contiguous tensors.")

    rows, n_cols = output.shape
    block_size, num_warps = _launch_config(n_cols)
    if rows == 0:
        return output
    _silu_mul_fp8_e4m3_per_token_qdq_kernel[(rows,)](
        input,
        output,
        input.stride(0),
        output.stride(0),
        n_cols=n_cols,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


class MarlinFp8QdqFusedExperts(MarlinExperts):
    """Opt-in Marlin W4A16 experts with fused QAT-compatible FP8 QDQ."""

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        if activation != MoEActivation.SILU:
            raise ValueError(
                "marlin_fp8_qdq_fused currently supports only the SILU/SwiGLU "
                f"activation, got {activation.value}."
            )
        # Plain SwiGLU is expert-independent. Routing/remapping is handled by
        # the surrounding Marlin GEMMs exactly as in the original backend.
        silu_mul_fp8_e4m3_per_token_qdq_fused(input, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self.input_dtype is not None:
            raise ValueError(
                "marlin_fp8_qdq_fused keeps Marlin GEMMs W4A16. Unset "
                "VLLM_MARLIN_INPUT_DTYPE."
            )
        if apply_router_weight_on_input:
            raise ValueError(
                "marlin_fp8_qdq_fused requires router probabilities after FC2."
            )

        qdq_hidden_states = torch.empty_like(hidden_states)
        fp8_e4m3_per_token_qdq_fused(hidden_states, qdq_hidden_states)
        super().apply(
            output=output,
            hidden_states=qdq_hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=False,
        )
