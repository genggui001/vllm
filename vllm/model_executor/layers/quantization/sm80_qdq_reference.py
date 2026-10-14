# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, eager BF16 reference for the training group32 QDQ contract.

Checkpoint weights are already quantized: only dequantize them, once at load.
Activation QDQ deliberately uses separate PyTorch operations to expose every
rounding boundary. This is an accuracy oracle, not a production MoE backend.
"""

import os
from typing import Literal

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.v1.attention.backends.registry import AttentionBackendEnum

GROUP_SIZE = 32
FP8_MAX = 448.0


def group32_fp8_qdq(x: torch.Tensor) -> torch.Tensor:
    """Training per-token group32 FP8 QDQ, including FP32 tiny and signed zero."""
    if x.dtype != torch.bfloat16 or x.ndim < 2 or x.shape[-1] % GROUP_SIZE:
        raise ValueError("group32 QDQ requires BF16 [..., K] with K divisible by 32")
    if x.numel() == 0:
        return x
    groups = x.unflatten(-1, (x.shape[-1] // GROUP_SIZE, GROUP_SIZE))
    scales = groups.abs().amax(-1).float() / FP8_MAX
    scales = scales.clamp_min(torch.finfo(torch.float32).tiny)
    codes = (groups / scales.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    codes = codes.to(torch.float8_e4m3fn)
    return (codes.float() * scales.unsqueeze(-1)).flatten(-2).to(x.dtype)


def tensor_fp8_qdq(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply a validated checkpoint FP32 scalar without recalibrating it."""
    if x.dtype != torch.bfloat16 or scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("QKV QDQ requires BF16 input and one FP32 checkpoint scale")
    # Length-one scale is essential: a 0D scale would use scalar promotion.
    s = scale.reshape(1)
    codes = (x / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (codes.float() * s).to(x.dtype)


def training_swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    # Training rounds SiLU and the explicit (up + 0) before the product.
    return F.silu(gate) * (up + 0.0)


@torch.jit.script
def _grouped_reference_linear(
    x: torch.Tensor, weights: torch.Tensor, expert_ends_cpu: torch.Tensor
) -> torch.Tensor:
    """Keep per-expert cuBLAS shapes while dispatching outside Python."""
    output = torch.empty((x.size(0), weights.size(1)), dtype=x.dtype, device=x.device)
    start = 0
    for expert in range(weights.size(0)):
        end = int(expert_ends_cpu[expert])
        if end > start:
            torch.mm(x[start:end], weights[expert].t(), out=output[start:end])
        start = end
    return output


def dequant_fp8_group32(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        raise ValueError("Dense weights require E4M3 codes and FP32 scales")
    groups = codes.unflatten(-1, (codes.shape[-1] // GROUP_SIZE, GROUP_SIZE))
    if groups.shape[:-1] != scales.shape:
        raise ValueError("Dense FP8 weight/scale group shape mismatch")
    return (groups.float() * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def dequant_int4_group32(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int32 or scales.dtype != torch.bfloat16:
        raise ValueError("MoE weights require packed INT32 codes and BF16 scales")
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    # compressed-tensors uses unsigned nibbles with an offset of 8.
    codes = ((packed.unsqueeze(-1) >> shifts) & 15).sub(8).flatten(-2)
    groups = codes.unflatten(-1, (codes.shape[-1] // GROUP_SIZE, GROUP_SIZE))
    if groups.shape[:-1] != scales.shape:
        raise ValueError("MoE INT4 weight/scale group shape mismatch")
    return (groups.to(torch.bfloat16) * scales.unsqueeze(-1)).flatten(-2)


def _validate_scales(scales: torch.Tensor, label: str) -> None:
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError(f"{label}: missing, nonfinite or nonpositive checkpoint scale")


def _check_group32(args, *, bits: int, kind: str, dynamic: bool) -> None:
    if not (
        args is not None
        and args.num_bits == bits
        and args.type == kind
        and args.strategy == "group"
        and args.group_size == GROUP_SIZE
        and args.symmetric
        and args.dynamic == dynamic
        and args.actorder is None
    ):
        raise ValueError(f"Unsupported QDQ reference quantization scheme: {args}")


class SM80QDQReferenceConfig(CompressedTensorsConfig):
    @classmethod
    def get_name(cls) -> Literal["sm80_qdq_reference"]:
        return "sm80_qdq_reference"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if user_quant == cls.get_name():
            if hf_quant_cfg.get("quant_method") != "compressed-tensors":
                raise ValueError(
                    "sm80_qdq_reference requires a compressed-tensors model"
                )
            return cls.get_name()
        return None

    def _check_runtime(self, *, allow_graph: bool = False) -> None:
        config = get_current_vllm_config()
        additional_config = config.additional_config
        if not isinstance(additional_config, dict):
            raise ValueError("sm80_qdq_reference requires dictionary additional_config")
        deterministic = additional_config.get("sm80_qdq_reference_deterministic", False)
        if not isinstance(deterministic, bool):
            raise ValueError("sm80_qdq_reference_deterministic must be a boolean")
        if deterministic:
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                raise ValueError(
                    "Deterministic QDQ reference requires "
                    "CUBLAS_WORKSPACE_CONFIG=:4096:8 before process startup"
                )
            torch.use_deterministic_algorithms(True)
        if not allow_graph and not config.model_config.enforce_eager:
            raise ValueError("sm80_qdq_reference requires --enforce-eager")
        if config.model_config.dtype != torch.bfloat16:
            raise ValueError("sm80_qdq_reference requires --dtype bfloat16")
        if config.model_config.hf_text_config.model_type != "qwen3_5_moe_text":
            raise ValueError("sm80_qdq_reference currently supports Qwen3.5 MoE only")
        self._check_attention_runtime(config)
        if config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("sm80_qdq_reference currently supports PP=1")
        if config.parallel_config.enable_expert_parallel:
            raise ValueError("sm80_qdq_reference currently supports TP, not EP")
        if config.parallel_config.use_sequence_parallel_moe:
            raise ValueError(
                "sm80_qdq_reference requires sequence parallel MoE disabled"
            )
        if config.lora_config is not None:
            raise ValueError("sm80_qdq_reference does not support LoRA")

    def _check_attention_runtime(self, config) -> None:
        if config.cache_config.cache_dtype != "bfloat16":
            raise ValueError("sm80_qdq_reference requires --kv-cache-dtype bfloat16")
        if config.attention_config.backend != AttentionBackendEnum.FLASH_ATTN:
            raise ValueError(
                "sm80_qdq_reference requires --attention-backend FLASH_ATTN"
            )

    def _get_scheme_from_parts(
        self, weight_quant, input_quant, output_quant=None, format=None, layer_name=None
    ):
        _check_group32(weight_quant, bits=8, kind="float", dynamic=False)
        _check_group32(input_quant, bits=8, kind="float", dynamic=True)
        if output_quant is not None or format != "float-quantized":
            raise ValueError(f"Unsupported Dense QDQ reference format in {layer_name}")
        return DenseGroup32QDQReference()

    def get_quant_method(self, layer, prefix):
        self._check_runtime()
        if isinstance(layer, Attention):
            return SavedQKVScaleQDQReference(self)
        if isinstance(layer, RoutedExperts):
            if layer.apply_router_weight_on_input:
                raise ValueError(
                    "Reference MoE applies router weight after FC2-input QDQ"
                )
            schemes = [
                self.get_scheme_dict(layer, f"{prefix}.0.{proj}_proj")
                for proj in ("gate", "up", "down")
            ]
            scheme = schemes[0]
            if scheme is None or any(s != scheme for s in schemes):
                raise ValueError(f"All reference experts must use one scheme: {prefix}")
            _check_group32(scheme["weights"], bits=4, kind="int", dynamic=False)
            _check_group32(
                scheme["input_activations"], bits=8, kind="float", dynamic=True
            )
            if scheme["format"] != "pack-quantized":
                raise ValueError("Reference MoE requires pack-quantized weights")
            return MoEGroup32QDQReference(layer.moe_config)
        return super().get_quant_method(layer, prefix)


class DenseGroup32QDQReference(CompressedTensorsScheme):
    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        weight_loader,
        **kwargs,
    ):
        if params_dtype != torch.bfloat16 or input_size_per_partition % GROUP_SIZE:
            raise ValueError(
                "Reference Dense TP partitions must be BF16 and group aligned"
            )
        n, k = sum(output_partition_sizes), input_size_per_partition
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(n, k, dtype=torch.float8_e4m3fn),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=torch.full(
                    (n, k // GROUP_SIZE), float("nan"), dtype=torch.float32
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer):
        _validate_scales(layer.weight_scale, "Dense weight_scale")
        weight = dequant_fp8_group32(layer.weight, layer.weight_scale)
        replace_parameter(layer, "weight", weight)
        del layer.weight_scale

    def apply_weights(self, layer, x, bias=None):
        return F.linear(group32_fp8_qdq(x), layer.weight, bias)


class SavedQKVScaleQDQReference(CompressedTensorsKVCacheMethod):
    quant_config: SM80QDQReferenceConfig

    def create_weights(self, layer):
        scheme = self.quant_config.kv_cache_scheme
        if not scheme or scheme.get("strategy") != "tensor":
            raise ValueError("Reference QKV requires saved per-tensor scales")
        super().create_weights(layer)
        for name in ("q_scale", "k_scale", "v_scale"):
            getattr(layer, name).data.fill_(float("nan"))

    def process_weights_after_loading(self, layer):
        for name in ("q_scale", "k_scale", "v_scale"):
            scale = getattr(layer, name)
            if scale.dtype != torch.float32 or scale.numel() != 1:
                raise ValueError(f"{layer.layer_name}.{name} must be one FP32 scale")
            _validate_scales(scale, f"{layer.layer_name}.{name}")
        # This CT override preserves checkpoint scales even with BF16 KV storage.
        super().process_weights_after_loading(layer)


class MoEGroup32QDQReference(FusedMoEMethodBase):
    def __init__(self, moe):
        super().__init__(moe)
        if str(moe.activation) not in ("silu", "MoEActivation.SILU"):
            raise ValueError("Reference MoE supports SwiGLU only")
        if moe.has_bias or moe.swiglu_limit is not None:
            raise ValueError("Reference MoE requires bias-free, unclamped SwiGLU")

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        h, i = hidden_size, intermediate_size_per_partition
        if params_dtype != torch.bfloat16 or h % GROUP_SIZE or i % GROUP_SIZE:
            raise ValueError(
                "Reference MoE TP partitions must be BF16 and group aligned"
            )
        attrs = {
            "weight_loader": extra_weight_attrs["weight_loader"],
            "is_transposed": False,
            "quant_method": "group",
        }
        shapes = {
            "w13_weight_packed": ((num_experts, 2 * i, h // 8), torch.int32),
            "w2_weight_packed": ((num_experts, h, i // 8), torch.int32),
            "w13_weight_scale": ((num_experts, 2 * i, h // GROUP_SIZE), params_dtype),
            "w2_weight_scale": ((num_experts, h, i // GROUP_SIZE), params_dtype),
            "w13_weight_shape": ((num_experts, 2), torch.int64),
            "w2_weight_shape": ((num_experts, 2), torch.int64),
        }
        for name, (shape, dtype) in shapes.items():
            value = torch.empty(shape, dtype=dtype)
            if "scale" in name:
                value.fill_(float("nan"))
            param = torch.nn.Parameter(value, requires_grad=False)
            set_weight_attrs(param, attrs)
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer):
        for stem in ("w13", "w2"):
            packed = getattr(layer, stem + "_weight_packed")
            scales = getattr(layer, stem + "_weight_scale")
            _validate_scales(scales, stem + "_weight_scale")
            # Bound load-time temporary memory; a whole layer can be >1 GiB.
            weight = torch.empty(
                *packed.shape[:-1],
                packed.shape[-1] * 8,
                dtype=torch.bfloat16,
                device=packed.device,
            )
            for e in range(packed.shape[0]):
                weight[e].copy_(dequant_int4_group32(packed[e], scales[e]))
            layer.register_parameter(
                stem + "_weight",
                torch.nn.Parameter(
                    weight,
                    requires_grad=False,
                ),
            )
            delattr(layer, stem + "_weight_packed")
            delattr(layer, stem + "_weight_scale")
            delattr(layer, stem + "_weight_shape")

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        # The runner computes shared experts separately (NO_OVERLAP).
        if x.numel() == 0:
            return torch.empty_like(x)
        # Stable expert-major ordering matches the training permutation.
        flat_ids = topk_ids.reshape(-1)
        order = flat_ids.argsort(stable=True)
        rows = order // topk_ids.size(1)
        expert_limits = torch.arange(
            1,
            layer.w13_weight.size(0) + 1,
            device=topk_ids.device,
            dtype=topk_ids.dtype,
        )
        # Share one host transfer between FC1 and FC2. Per-expert torch.where
        # and tolist would otherwise synchronize repeatedly. Keeping exact
        # cuBLAS GEMM shapes still requires these boundaries on the host.
        expert_ends_cpu = torch.searchsorted(
            flat_ids[order], expert_limits, out_int32=True
        ).cpu()
        fc1 = _grouped_reference_linear(
            group32_fp8_qdq(x)[rows], layer.w13_weight, expert_ends_cpu
        )
        fc2_input = group32_fp8_qdq(training_swiglu(fc1))
        fc2_input = (fc2_input * topk_weights.reshape(-1)[order, None]).to(x.dtype)
        permuted = _grouped_reference_linear(
            fc2_input, layer.w2_weight, expert_ends_cpu
        )
        output = torch.zeros_like(x)
        # Megatron's non-fused unpermute uses BF16 scatter_add, not an FP32
        # top-k sum. Preserve that rounding boundary; CUDA atomics may vary
        # in order just as they do in the training implementation.
        if torch.are_deterministic_algorithms_enabled():
            output.index_add_(0, rows, permuted)
        else:
            output.scatter_add_(0, rows[:, None].expand_as(permuted), permuted)
        return output
