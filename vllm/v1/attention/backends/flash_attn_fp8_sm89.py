# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM89 QKV-FP8 backend; FA2 tiling with FA3 numerical conventions."""

from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionImpl, AttentionType
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.flash_attn_fp8_sm89 import sm89_fp8_paged_attention


class FlashAttentionFP8SM89Backend(FlashAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["fp8", "fp8_e4m3"]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_FP8_SM89"

    @staticmethod
    # Reuse FA2 cache/metadata layout with an independent FP8 implementation.
    def get_impl_cls() -> type["FlashAttentionFP8SM89Impl"]:  # type: ignore[override]
        return FlashAttentionFP8SM89Impl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionFP8SM89MetadataBuilder"]:
        return FlashAttentionFP8SM89MetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(8, 9)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (64, 128, 256)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if use_mla or has_sink or use_sparse or use_mm_prefix:
            return "SM89 FP8 supports dense causal decoder attention"
        if not cls.supports_compute_capability(device_capability):
            return "Native SM89 FP8 attention requires compute capability 8.9"
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            return "SM89 FP8 attention requires an E4M3 KV cache"
        return None


class FlashAttentionFP8SM89MetadataBuilder(FlashAttentionMetadataBuilder):
    def _get_scheduler_metadata(self, **kwargs) -> None:
        return None

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class FlashAttentionFP8SM89Impl(AttentionImpl):
    can_return_lse_for_decode = False
    supports_dcp = False
    enforce_cuda_query_quant = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if (
            attn_type != AttentionType.DECODER
            or alibi_slopes is not None
            or sliding_window is not None
            or logits_soft_cap not in (None, 0)
            or sinks is not None
        ):
            raise NotImplementedError("SM89 FP8 supports dense causal attention")
        if kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            raise ValueError("SM89 FP8 attention requires an E4M3 KV cache")
        config = get_current_vllm_config_or_none()
        if config is not None and (
            config.parallel_config.decode_context_parallel_size != 1
            or config.parallel_config.prefill_context_parallel_size != 1
        ):
            raise NotImplementedError("SM89 FP8 context parallelism is not supported")
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.supports_quant_query_input = True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("SM89 FP8 output quantization is not supported")
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.use_cascade or attn_metadata.causal is not True:
            raise NotImplementedError("SM89 FP8 requires causal non-cascade metadata")
        count = attn_metadata.num_actual_tokens
        sm89_fp8_paged_attention(
            query[:count],
            kv_cache.view(torch.float8_e4m3fn),
            output[:count],
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            attn_metadata.block_table,
            attn_metadata.max_query_len,
            self.scale,
            layer._q_scale,
            layer._k_scale,
            layer._v_scale,
            max_seq_len=attn_metadata.max_seq_len,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
