# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin WNA16 experts with QAT-compatible FP8 activation QDQ."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts


def fp8_e4m3_per_token_qdq(x: torch.Tensor) -> torch.Tensor:
    """Apply the training-time dynamic per-token FP8 E4M3 fake quantizer.

    The returned tensor retains the input dtype, so the existing W4A16 Marlin
    GEMM remains unchanged. This is intentionally a reference implementation;
    a fused CUDA implementation can replace this helper without changing the
    separately selected experts backend.
    """
    if x.ndim != 2:
        raise ValueError(f"FP8 per-token QDQ expects a 2D tensor, got {x.shape}.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"FP8 per-token QDQ expects float16 or bfloat16 input, got {x.dtype}."
        )

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scales = x.abs().amax(dim=1, keepdim=True).float() / fp8_max
    scales = scales.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (x / scales).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return (quantized.float() * scales).to(x.dtype)


class MarlinFp8QdqExperts(MarlinExperts):
    """Opt-in Marlin W4A16 experts with per-token FP8 activation QDQ.

    FC1 input and the post-activation FC2 input are fake-quantized using the
    same FP8 E4M3 formula as QAT. Router probabilities remain on Marlin's
    default post-FC2 path.
    """

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        super().activation(
            activation,
            output,
            input,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )
        output.copy_(fp8_e4m3_per_token_qdq(output))

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
                "marlin_fp8_qdq keeps Marlin GEMMs W4A16. Unset "
                "VLLM_MARLIN_INPUT_DTYPE."
            )
        if apply_router_weight_on_input:
            raise ValueError(
                "marlin_fp8_qdq requires router probabilities to be applied "
                "after FC2. Set apply_router_weight_on_input=False."
            )

        super().apply(
            output=output,
            hidden_states=fp8_e4m3_per_token_qdq(hidden_states),
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
