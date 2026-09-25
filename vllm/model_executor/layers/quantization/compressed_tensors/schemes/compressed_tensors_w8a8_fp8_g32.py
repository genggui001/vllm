# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.sm90_g32 import (
    prepare_linear_stream,
    require_sm90,
    validate_partition,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter

logger = init_logger(__name__)


class CompressedTensorsW8A8Fp8G32(CompressedTensorsScheme):
    def __init__(self, layer_name: str | None):
        self.layer_name = layer_name or "Linear"
        require_sm90()
        logger.info_once(
            "Using native SM90 WFP8/AFP8 G32 Linear with original FP32 G32 scales"
        )
        backend = get_current_vllm_config().kernel_config.linear_backend
        if backend not in ("auto", "cutlass", "cutlass_g32"):
            raise ValueError(
                f"{self.layer_name}: W8A8 G32 requires cutlass_g32, got {backend}"
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

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
        rank = get_tensor_model_parallel_rank()
        k = input_size_per_partition
        n = sum(output_partition_sizes)
        validate_partition(input_size, k, rank, self.layer_name)
        if n % 8 or any(width <= 0 for width in output_partition_sizes):
            raise ValueError(
                f"{self.layer_name}: unsupported output partitions "
                f"{output_partition_sizes}"
            )
        if params_dtype != torch.bfloat16:
            raise ValueError(
                f"{self.layer_name}: SM90 G32 currently requires dtype=bfloat16"
            )

        def checked_loader(param, loaded, *args, **kwargs):
            if loaded.dtype != param.dtype:
                raise ValueError(
                    f"{self.layer_name}: checkpoint dtype {loaded.dtype} "
                    f"does not match G32 parameter dtype {param.dtype}"
                )
            return weight_loader(param, loaded, *args, **kwargs)

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(n, k, dtype=torch.float8_e4m3fn),
                input_dim=1,
                output_dim=0,
                weight_loader=checked_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=torch.empty(n, k // 32, dtype=torch.float32),
                input_dim=1,
                output_dim=0,
                weight_loader=checked_loader,
            ),
        )
        layer.orig_dtype = params_dtype
        layer.logical_widths = output_partition_sizes
        layer.input_scale = None
        layer.weight_block_size = None
        layer.g32_global_k = input_size
        layer.g32_local_k = k
        layer.g32_k_offset = 0 if input_size == k else rank * k

    def process_weights_after_loading(self, layer):
        if layer.weight.is_cuda:
            prepare_linear_stream(layer.weight.device)
        if layer.weight_scale.dtype != torch.float32:
            raise ValueError(
                f"{self.layer_name}: original G32 weight scales must be FP32"
            )
        if (
            not torch.isfinite(layer.weight_scale).all()
            or (layer.weight_scale <= 0).any()
        ):
            raise ValueError(f"{self.layer_name}: invalid G32 weight scale values")
        layer.weight = Parameter(layer.weight.data.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data.t().contiguous().t(), requires_grad=False
        )

    def apply_weights(self, layer, x, bias=None):
        out = torch.ops.vllm.sm90_g32_linear(x, layer.weight, layer.weight_scale)
        return out if bias is None else out + bias
