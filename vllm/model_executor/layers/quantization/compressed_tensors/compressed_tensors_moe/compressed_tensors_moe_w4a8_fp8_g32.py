# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM90 grouped INT4/FP8 G32 experts, preserving BF16 weight scales."""

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
    moe_permute,
    moe_unpermute,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.sm90_g32 import (
    require_sm90,
    validate_partition,
)

from .compressed_tensors_moe_w4a8_fp8 import CompressedTensorsW4A8Fp8MoEMethod

logger = init_logger(__name__)


class CompressedTensorsW4A8Fp8G32MoEMethod(CompressedTensorsW4A8Fp8MoEMethod):
    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        FusedMoEMethodBase.__init__(self, moe)
        require_sm90()
        self.layer_name = layer_name or "MoE"
        self.weight_quant, self.input_quant = weight_quant, input_quant
        self.group_size, self.num_bits, self.packed_factor = 32, 4, 8
        logger.info_once(
            "Using native SM90 WINT4/AFP8 G32 MoE with original BF16 G32 scales"
        )
        self._scratch = None
        backend = get_current_vllm_config().kernel_config.moe_backend
        if backend not in ("auto", "cutlass", "cutlass_g32"):
            raise ValueError(
                f"{self.layer_name}: W4A8 G32 requires cutlass_g32, got {backend}"
            )
        if weight_quant.scale_dtype not in (None, torch.bfloat16):
            raise ValueError("SM90 WINT4 G32 requires BF16 checkpoint weight scales")
        if moe.ep_size != 1:
            raise ValueError(
                "SM90 G32 experts currently support tensor parallelism, "
                "not expert parallelism"
            )

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        rank = get_tensor_model_parallel_rank()
        validate_partition(hidden_size, hidden_size, rank, self.layer_name + ".fc1")
        validate_partition(
            self.moe.intermediate_size,
            intermediate_size_per_partition,
            rank,
            self.layer_name + ".fc2",
        )
        if params_dtype != torch.bfloat16 or num_experts > 256:
            raise ValueError(
                "SM90 G32 experts require BF16 scales and at most 256 experts"
            )
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **attrs,
        )

    def process_weights_after_loading(self, layer):
        for prefix in ("w13", "w2"):
            original = getattr(layer, prefix + "_weight_packed")
            scales = getattr(layer, prefix + "_weight_scale")
            if (
                scales.dtype != torch.bfloat16
                or not torch.isfinite(scales).all()
                or (scales <= 0).any()
            ):
                raise ValueError(
                    f"{self.layer_name}: expected positive original BF16 G32 scales"
                )
            # compressed-tensors uint4 uses an offset of 8; kernels use signed INT4.
            signed = torch.bitwise_xor(original, -2004318072).contiguous()
            packed, layout = (
                torch.ops._moe_C_gemmfollowup.h20_w4a8_cutlass_groupwise_pack(signed)
            )
            setattr(
                layer, prefix + "_weight_packed", Parameter(packed, requires_grad=False)
            )
            setattr(
                layer,
                prefix + "_weight_scale",
                Parameter(scales.transpose(1, 2).contiguous(), requires_grad=False),
            )
            layer.register_buffer(prefix + "_g32_layout", layout)
            # The small FC2 kernel has a separate register-friendly immutable layout.
            if prefix == "w2":
                e, n, kp = signed.shape
                native = torch.empty(
                    (e, (n + 63) // 64, kp // 4, 128, 2),
                    device=signed.device,
                    dtype=torch.int32,
                )
                native_scale = torch.empty(
                    (e, (n + 63) // 64, kp // 4, 64),
                    device=signed.device,
                    dtype=torch.float32,
                )
                torch.ops._moe_C.h20_w4a8_groupwise_pack(
                    signed, scales.float(), native, native_scale, 32
                )
                layer.register_buffer("w2_g32_native", native)
                layer.register_buffer("w2_g32_native_scale", native_scale)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def get_fused_moe_quant_config(self, layer):
        a = FusedMoEQuantDesc(dtype=torch.float8_e4m3fn, shape=GroupShape(1, 32))
        return FusedMoEQuantConfig(
            _a1=a,
            _a2=a,
            _w1=FusedMoEQuantDesc(
                dtype="int4", shape=GroupShape(1, 32), scale=layer.w13_weight_scale
            ),
            _w2=FusedMoEQuantDesc(
                dtype="int4", shape=GroupShape(1, 32), scale=layer.w2_weight_scale
            ),
            is_scale_swizzled=False,
        )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        if layer.apply_router_weight_on_input:
            raise ValueError("SM90 G32 requires router weights on expert outputs")
        if str(layer.activation.value) != "silu":
            raise ValueError("SM90 G32 experts require SiLU gate/up")
        m, k = x.shape
        if not m:
            return torch.empty_like(x)
        x = x.contiguous()
        if self._scratch is None:
            self._scratch = MoEPermuteScratch(
                max_num_tokens=self.moe.max_num_tokens,
                topk=self.moe.experts_per_token,
                num_experts=layer.num_experts,
                num_local_experts=layer.num_experts,
                device=x.device,
            )
        # CUB sorts only the expert-id bit range. Map graph padding (-1)
        # to its documented end sentinel instead of letting signed IDs wrap.
        topk_ids = torch.where(topk_ids < 0, layer.num_experts, topk_ids)
        perm = torch.empty((m * topk_ids.shape[1], k), device=x.device, dtype=x.dtype)
        perm, _, offsets64, inverse, _ = moe_permute(
            x,
            None,
            topk_ids,
            layer.num_experts,
            layer.num_experts,
            permuted_hidden_states=perm,
            scratch=self._scratch,
        )
        offsets = offsets64.to(torch.int32)
        r = perm.shape[0]
        n = layer.w13_weight_packed.shape[1]
        q1 = torch.empty_like(perm, dtype=torch.float8_e4m3fn)
        s1 = torch.empty((r, k // 32), device=x.device, dtype=torch.float32)
        # The grouped GEMM skips sentinel routes; SiLU quantization reads all
        # allocated rows, so initialize those inactive rows deterministically.
        h = torch.zeros((r, n), device=x.device, dtype=x.dtype)
        q2 = torch.empty((r, n // 2), device=x.device, dtype=torch.float8_e4m3fn)
        s2 = torch.empty((r, n // 64), device=x.device, dtype=torch.float32)
        y = torch.empty((r, k), device=x.device, dtype=x.dtype)
        tile = 32 if r <= 8192 else 64
        group_ops = (
            torch.ops._moe_C_exact if r <= 8192 else torch.ops._moe_C_gemmfollowup
        )
        torch.ops._moe_C.h20_fp8_group_quant(perm, q1, s1, 32)
        group_ops.h20_w4a8_cutlass_groupwise_mm(
            q1,
            layer.w13_weight_packed,
            layer.w13_g32_layout,
            s1,
            layer.w13_weight_scale,
            offsets,
            h,
            32,
            32,
            tile,
            True,
        )
        torch.ops._moe_C.h20_silu_fp8_group_quant(h, q2, s2, 32, None)
        if r < 512:
            prefix = torch.empty_like(offsets)
            torch.ops._moe_C.h20_w4a8_groupwise_mm(
                q2,
                layer.w2_g32_native,
                s2,
                layer.w2_g32_native_scale,
                offsets,
                prefix,
                y,
                32,
                32,
                tile,
                None,
            )
        else:
            group_ops.h20_w4a8_cutlass_groupwise_mm(
                q2,
                layer.w2_weight_packed,
                layer.w2_g32_layout,
                s2,
                layer.w2_weight_scale,
                offsets,
                y,
                32,
                32,
                tile,
                True,
            )
        out = torch.empty_like(x)
        moe_unpermute(out, y, topk_weights, inverse, offsets64)
        return out
