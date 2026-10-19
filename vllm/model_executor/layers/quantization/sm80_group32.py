# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed INT4/group32 FP8 QDQ backend for the validated SM80 export."""

from functools import partial
from typing import cast

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.marlin_group32_fp8_qdq import (
    MarlinGroup32Fp8QdqExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    make_wna16_moe_kernel,
    make_wna16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.sm80_qdq_reference import (
    DenseGroup32QDQReference,
    MoEGroup32QDQReference,
    SM80QDQReferenceConfig,
    _validate_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_moe_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
    Group32FP8Input,
    dense_fp8,
    prepare_weight,
    quantize_input,
    validate_dense_shape,
)
from vllm.model_executor.layers.quantization.utils.sm80_group32_config import (
    has_static_fp8_qkv,
    is_group32_export,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


class SM80Group32Config(SM80QDQReferenceConfig):
    """Select saved group32 scales and the native SM80 Marlin inference path."""

    @classmethod
    def get_name(cls):
        return "sm80_group32"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if user_quant == cls.get_name():
            return super().override_quantization_method(
                hf_quant_cfg, user_quant, hf_config
            )
        if (
            user_quant is not None
            or not current_platform.is_cuda()
            or not current_platform.is_device_capability(80)
            or not is_group32_export(hf_quant_cfg, hf_config)
        ):
            return None
        logger.info_once(
            "Automatically selecting sm80_group32 for the Qwen3.5 MoE "
            "INT4/FP8 group32 export with saved FP32 Dense and QKV scales."
        )
        return cls.get_name()

    def uses_sm80_fp8_qkv(self) -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability(80)
            and has_static_fp8_qkv(self.kv_cache_scheme)
        )

    def _check_runtime(self, *, allow_graph: bool = True) -> None:
        # Routing and reduction are GPU-only; the reference retains its eager guard.
        super()._check_runtime(allow_graph=allow_graph)

    def _check_attention_runtime(self, config) -> None:
        model_config = getattr(config, "model_config", None)
        mm_config = getattr(model_config, "multimodal_config", None)
        if (
            mm_config is not None
            and not mm_config.language_model_only
            and mm_config.mm_encoder_attn_dtype is not None
        ):
            raise ValueError(
                "sm80_group32 requires an unquantized BF16 vision encoder; "
                "omit --mm-encoder-attn-dtype."
            )
        backend = config.attention_config.backend
        per_kind = getattr(config.attention_config, "backend_per_kind", None)
        if per_kind:
            backend = per_kind.get("full_attention", backend)
        implicit_sm80 = (
            backend is None
            and current_platform.is_cuda()
            and current_platform.is_device_capability(80)
        )
        cache_dtype = config.cache_config.cache_dtype
        if cache_dtype == "auto" and self.kv_cache_scheme is not None:
            cache_dtype = "fp8"
        if cache_dtype in ("fp8", "fp8_e4m3"):
            if (
                implicit_sm80
                or backend == AttentionBackendEnum.FLASH_ATTN_QKV_FP8_SM80_FUSED
            ):
                return
        elif cache_dtype in ("auto", "bfloat16") and (
            implicit_sm80 or backend == AttentionBackendEnum.FLASH_ATTN
        ):
            return
        raise ValueError(
            "sm80_group32 requires BF16 KV with FLASH_ATTN or FP8 E4M3 KV "
            "with FLASH_ATTN_QKV_FP8_SM80_FUSED; automatic attention "
            "selection is supported on SM80."
        )

    def _get_scheme_from_parts(
        self, weight_quant, input_quant, output_quant=None, format=None, layer_name=None
    ):
        super()._get_scheme_from_parts(
            weight_quant, input_quant, output_quant, format, layer_name
        )
        return DenseGroup32Fused()

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, MoEGroup32QDQReference):
            return PackedGroup32MoE(layer.moe_config)
        return method


class DenseGroup32Fused(DenseGroup32QDQReference):
    """Load-time validated FP8-only Dense backend; dispatch lives in CUDA."""

    def process_weights_after_loading(self, layer):
        from vllm.config import get_current_vllm_config

        if getattr(layer, "bias", None) is not None:
            raise ValueError(
                "SM80 Dense FP8 currently requires bias-free Linear layers"
            )
        config = get_current_vllm_config()
        self.n, self.k = layer.weight.shape
        max_tokens = max(
            config.scheduler_config.max_num_batched_tokens,
            config.scheduler_config.max_num_seqs,
            config.compilation_config.max_cudagraph_capture_size or 0,
        )
        validate_dense_shape(self.n, self.k, max_tokens)
        weight, scale, auxiliary = prepare_weight(layer.weight, layer.weight_scale)
        for name, value in (
            ("weight", weight),
            ("weight_scale", scale),
            ("weight_auxiliary", auxiliary),
        ):
            layer.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )

    def apply_weights(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("Dense FP8 bias must be rejected during model loading")
        if isinstance(x, Group32FP8Input):
            shape = x.codes.shape
            prepared = x
        else:
            shape = x.shape
            prepared = quantize_input(x.reshape(-1, shape[-1]))
        out = dense_fp8(
            prepared, layer.weight, layer.weight_scale, layer.weight_auxiliary, self.n
        )
        return out.reshape(*shape[:-1], self.n)


def _load_group32_expert_weight(
    param,
    loaded_weight,
    weight_name,
    shard_id,
    expert_id,
    return_success=False,
    *,
    loader,
):
    # Stage each contiguous expert before the TP column slice; copying a
    # strided CPU shard can dominate loading with many host threads.
    if (
        shard_id == "w2"
        and loaded_weight.device.type == "cpu"
        and param.device.type == "cuda"
        and loaded_weight.ndim == 2
        and param.ndim == 3
        and loaded_weight.is_contiguous()
        and loaded_weight.dtype == param.dtype
        and loaded_weight.dtype in (torch.int32, torch.bfloat16)
        and loaded_weight.shape[0] == param.shape[1]
        and loaded_weight.shape[1] == 2 * param.shape[2]
        and loaded_weight.numel() * loaded_weight.element_size() <= 1024 * 1024
    ):
        loaded_weight = loaded_weight.to(param.device)
    return loader(
        param=param,
        loaded_weight=loaded_weight,
        weight_name=weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=return_success,
    )


class PackedGroup32MoE(MoEGroup32QDQReference):
    """Load original INT4 codes and BF16 scales without expanding expert weights."""

    def __init__(self, moe):
        super().__init__(moe)
        if moe.moe_backend not in ("auto", "marlin_group32_fp8_qdq"):
            raise ValueError(
                "SM80 group32 activations require moe_backend=auto or "
                "marlin_group32_fp8_qdq to preserve activation QDQ"
            )
        self.experts_cls = MarlinGroup32Fp8QdqExperts

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if layer.moe_config.moe_parallel_config.tp_size == 2:
            extra_weight_attrs["weight_loader"] = partial(
                _load_group32_expert_weight,
                loader=extra_weight_attrs["weight_loader"],
            )
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer):
        for stem in ("w13", "w2"):
            packed = getattr(layer, stem + "_weight_packed")
            scales = getattr(layer, stem + "_weight_scale")
            _validate_scales(scales, stem + "_weight_scale")
            experts, n, k_packed = packed.shape
            k = k_packed * 8
            if n % 64 or k % 128:
                raise ValueError(
                    "Group32 Marlin requires N divisible by 64 and K by 128"
                )
            permutation = torch.empty(
                (experts, 0), device=packed.device, dtype=torch.int32
            )
            weight = ops.gptq_marlin_moe_repack(
                packed.transpose(1, 2).contiguous(), permutation, k, n, 4, False
            )
            scale = marlin_moe_permute_scales(
                scales.transpose(1, 2).contiguous(), k, n, 32
            )
            layer.register_parameter(
                stem + "_weight", torch.nn.Parameter(weight, requires_grad=False)
            )
            layer.register_parameter(
                stem + "_scale", torch.nn.Parameter(scale, requires_grad=False)
            )
            delattr(layer, stem + "_weight_packed")
            delattr(layer, stem + "_weight_scale")
            delattr(layer, stem + "_weight_shape")
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=WNA16MoEBackend.MARLIN_GROUP32_FP8_QDQ,
            is_k_full=True,
            routing_tables=layer._expert_routing_tables(),
        )
        experts = cast(MarlinGroup32Fp8QdqExperts, self.moe_kernel.fused_experts)
        if experts.input_dtype is not None:
            raise ValueError("Group32 QDQ requires VLLM_MARLIN_INPUT_DTYPE unset")
        experts.input_prequantized = getattr(layer, "_sm80_group32_prepared", False)

    def get_fused_moe_quant_config(self, layer):
        w1_scale = (
            layer.w13_scale if hasattr(layer, "w13_scale") else layer.w13_weight_scale
        )
        w2_scale = (
            layer.w2_scale if hasattr(layer, "w2_scale") else layer.w2_weight_scale
        )
        return make_wna16_moe_quant_config(
            w1_scale=w1_scale, w2_scale=w2_scale, group_size=32, num_bits=4
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
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
