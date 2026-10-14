# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training group32 activation QDQ around the native Marlin expert pipeline."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_fp8_qdq_fused_moe import (
    fp8_e4m3_qdq_align_prepared,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.quantization.utils.sm80_group32_qdq import (
    group32_fp8_qdq_fused,
    silu_mul_group32_fp8_qdq_fused,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _mask_out_of_range_routes(
    IDs,
    Safe,
    N,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(IDs + (i // K) * S0 + (i % K) * S1, i < N, -1)
    tl.store(Safe + i, tl.where((value >= 0) & (value < E), value, -1), i < N)


@triton.jit
def _group32_masked_moe_sum(
    Input,
    Output,
    IDs,
    E: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    # Match native moe_sum's ordered FP32 accumulation. Invalid slots may
    # contain stale values or NaNs and must not be read.
    for k in tl.static_range(8):
        expert = tl.load(IDs + row * I0 + k * I1)
        valid = (expert >= 0) & (expert < E)
        value = tl.load(
            Input + row * S0 + k * S1 + col * S2,
            valid & (col < 2048),
            other=0,
        ).to(tl.float32)
        acc = tl.where(valid, acc + value, acc)
    tl.store(Output + row * 2048 + col, acc, col < 2048)


class MarlinGroup32Fp8QdqExperts(MarlinExperts):
    """BF16 Marlin inputs, group32 QDQ, and native post-FC2 weighting."""

    input_prequantized: bool = False

    def moe_sum(self, input, output, topk_ids, expert_map):
        if (
            expert_map is None
            and input.dtype == output.dtype == torch.bfloat16
            and input.ndim == 3
            and input.shape[1:] == (8, 2048)
            and output.shape == (input.shape[0], 2048)
            and output.is_contiguous()
            and topk_ids.shape == input.shape[:2]
            and topk_ids.dtype in (torch.int32, torch.int64)
        ):
            rows = input.shape[0]
            if rows:
                block = 256 if rows <= 16 else (512 if rows < 2048 else 1024)
                _group32_masked_moe_sum[(rows, triton.cdiv(2048, block))](
                    input,
                    output,
                    topk_ids,
                    self.moe_config.num_experts,
                    *input.stride(),
                    *topk_ids.stride(),
                    block,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            return
        # Skipped GEMM slots alias reused FC1/FC2 workspace and are not zeroed.
        # Native pad-aware moe_sum must not read those undefined values.
        safe_ids = torch.empty_like(topk_ids, memory_format=torch.contiguous_format)
        n = topk_ids.numel()
        if n:
            _mask_out_of_range_routes[(triton.cdiv(n, 256),)](
                topk_ids,
                safe_ids,
                n,
                topk_ids.shape[1],
                topk_ids.stride(0),
                topk_ids.stride(1),
                self.moe_config.num_experts,
                256,
                num_warps=4,
            )
        ops.moe_sum(input, output, safe_ids, expert_map)

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and current_platform.is_device_capability(80)

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.input_prequantized:
            hidden_states = group32_fp8_qdq_fused(hidden_states)
        # This entry and all its fallbacks only prepare integer routing metadata.
        return fp8_e4m3_qdq_align_prepared(
            hidden_states, topk_ids, block_size, num_experts, expert_map
        )

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
        valid_rows: torch.Tensor | None = None,
    ) -> None:
        if activation != MoEActivation.SILU:
            raise ValueError("marlin_group32_fp8_qdq requires SILU/SwiGLU")
        if self.activation_config.clamp_limit is not None:
            raise ValueError("marlin_group32_fp8_qdq rejects clamped SwiGLU")
        silu_mul_group32_fp8_qdq_fused(input, out=output)

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
                "marlin_group32_fp8_qdq requires BF16 Marlin inputs; "
                "VLLM_MARLIN_INPUT_DTYPE must be unset"
            )
        if apply_router_weight_on_input:
            raise ValueError("marlin_group32_fp8_qdq requires post-FC2 router weights")
        if self._lora_context is not None:
            raise ValueError("marlin_group32_fp8_qdq does not support LoRA")
        super().apply(
            output=output,
            hidden_states=hidden_states,
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
