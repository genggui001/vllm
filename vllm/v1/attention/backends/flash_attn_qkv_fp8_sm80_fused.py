# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA2-derived SM80 attention with fused static-scale QKV FP8 QDQ."""

import importlib
from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

try:
    importlib.import_module("vllm.vllm_flash_attn._vllm_fa2_sm80_fp8_C")

    _EXTENSION_ERROR: str | None = None
except ImportError as exc:
    _EXTENSION_ERROR = str(exc)

logger = init_logger(__name__)


class FlashAttentionQkvFp8Sm80FusedMetadataBuilder(FlashAttentionMetadataBuilder):
    """Use the stock FA2 cascade heuristic and metadata representation."""


class FlashAttentionQkvFp8Sm80FusedBackend(FlashAttentionBackend):
    """Publish a byte-packed cache without changing the stock FA2 backend."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

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
        # The custom loader handles FP8; the inherited FA2 math consumes BF16.
        return super().supports_combination(
            head_size,
            dtype,
            "bfloat16",
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        spec = super().customize_spec(spec)
        return replace(
            spec,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.SM80_FP8_PER_TENSOR,
        )

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_QKV_FP8_SM80_FUSED"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionQkvFp8Sm80FusedImpl"]:
        return FlashAttentionQkvFp8Sm80FusedImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return FlashAttentionQkvFp8Sm80FusedMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 16

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 256

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(8, 0)

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class FlashAttentionQkvFp8Sm80FusedImpl(FlashAttentionImpl):
    """Run BF16 tensor-core FA2 while decoding FP8 K/V tiles in the loader."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        *args,
        **kwargs,
    ) -> None:
        if kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "SM80 fused attention supports E4M3 KV cache, "
                f"got kv_cache_dtype={kv_cache_dtype}."
            )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            "bfloat16",
            *args,
            **kwargs,
        )
        self.kv_cache_dtype = kv_cache_dtype
        # Q quantization is performed exactly once by the custom CUDA kernel.
        self.supports_quant_query_input = False
        if _EXTENSION_ERROR is not None:
            raise ImportError(
                "_vllm_fa2_sm80_fp8_C is not installed: " + _EXTENSION_ERROR
            )
        capability = current_platform.get_device_capability()
        if capability != DeviceCapability(8, 0):
            raise NotImplementedError(
                "FLASH_ATTN_QKV_FP8_SM80_FUSED requires compute capability "
                f"8.0, got {capability}."
            )
        if self.vllm_flash_attn_version != 2:
            raise NotImplementedError(
                "FLASH_ATTN_QKV_FP8_SM80_FUSED requires FlashAttention 2, got "
                f"version {self.vllm_flash_attn_version}."
            )
        if self.head_size != 256:
            raise NotImplementedError(
                f"The v1 fused kernel requires head_dim=256, got {self.head_size}."
            )
        if self.alibi_slopes is not None:
            raise NotImplementedError("The v1 fused kernel does not support ALiBi.")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError(
                "The v1 fused kernel does not support sliding-window attention."
            )
        if self.logits_soft_cap != 0:
            raise NotImplementedError("The v1 fused kernel does not support softcap.")
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "The v1 fused kernel only supports decoder attention."
            )
        if self.dcp_world_size != 1:
            raise NotImplementedError("The v1 fused kernel does not support DCP.")

        logger.info_once(
            "Using independent FA2-derived SM80 fused Q QDQ + uint8 E4M3FN "
            "paged K/V decode (static per-layer scales; cascade supported)"
        )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if kv_cache.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 KV cache, got {kv_cache.dtype}")
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "fp8_e4m3",
            layer._k_scale,
            layer._v_scale,
        )

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
            raise NotImplementedError("Fused output quantization is not supported.")
        if attn_metadata is None:
            return output.fill_(0)
        if isinstance(attn_metadata.causal, torch.Tensor):
            raise NotImplementedError(
                "Per-sequence dynamic causal masks are unsupported."
            )
        if query.dtype != torch.bfloat16:
            raise TypeError(f"Expected BF16 query, got {query.dtype}")
        if kv_cache.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 KV cache, got {kv_cache.dtype}")

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens < output.shape[0]:
            output[num_actual_tokens:].zero_()
        # Packed single-token GQA writes every physical row, including the
        # zero-KV dummy sequences used by full Decode graphs. Other captured
        # layouts need initialization because device query lengths can shrink.
        packed_decode = (
            not attn_metadata.use_cascade
            and attn_metadata.max_query_len == 1
            and num_actual_tokens == attn_metadata.query_start_loc.numel() - 1
            and query.shape[1] > self.num_kv_heads
        )
        if not packed_decode and torch.cuda.is_current_stream_capturing():
            output[:num_actual_tokens].zero_()
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)

        prequantized_q = getattr(layer, "_qrope_prequantized_q", False)
        if prequantized_q:
            fwd = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_prequantized_q
            fwd_lse = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_lse_prequantized_q
        else:
            fwd = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd
            fwd_lse = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_lse

        query = query[:num_actual_tokens]
        actual_output = output[:num_actual_tokens]

        if not attn_metadata.use_cascade:
            fwd(
                query,
                key_cache,
                value_cache,
                actual_output,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                attn_metadata.block_table,
                layer._q_scale,
                layer._k_scale,
                layer._v_scale,
                attn_metadata.max_query_len,
                attn_metadata.max_seq_len,
                self.scale,
                bool(attn_metadata.causal),
            )
            return output

        cu_prefix_query_lens = attn_metadata.cu_prefix_query_lens
        prefix_kv_lens = attn_metadata.prefix_kv_lens
        suffix_kv_lens = attn_metadata.suffix_kv_lens
        if (
            cu_prefix_query_lens is None
            or prefix_kv_lens is None
            or suffix_kv_lens is None
        ):
            raise RuntimeError("Cascade metadata is incomplete.")
        page_size = key_cache.shape[1]
        if attn_metadata.common_prefix_len % page_size != 0:
            raise RuntimeError(
                "Cascade common prefix must be page aligned, got "
                f"{attn_metadata.common_prefix_len} tokens for page size "
                f"{page_size}."
            )
        num_common_blocks = attn_metadata.common_prefix_len // page_size
        if num_common_blocks <= 0:
            raise RuntimeError("Cascade requires a non-empty common prefix.")

        logger.info_once(
            "Executing FA2-derived SM80 FP8 cascade attention with a "
            "%d-token shared prefix",
            attn_metadata.common_prefix_len,
        )
        prefix_output = torch.empty_like(query)
        suffix_output = torch.empty_like(query)
        prefix_output, prefix_lse = fwd_lse(
            query,
            key_cache,
            value_cache,
            prefix_output,
            cu_prefix_query_lens,
            prefix_kv_lens,
            attn_metadata.block_table[:1],
            layer._q_scale,
            layer._k_scale,
            layer._v_scale,
            num_actual_tokens,
            attn_metadata.common_prefix_len,
            self.scale,
            False,
        )
        suffix_output, suffix_lse = fwd_lse(
            query,
            key_cache,
            value_cache,
            suffix_output,
            attn_metadata.query_start_loc,
            suffix_kv_lens,
            attn_metadata.block_table[:, num_common_blocks:],
            layer._q_scale,
            layer._k_scale,
            layer._v_scale,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len - attn_metadata.common_prefix_len,
            self.scale,
            True,
        )
        merge_attn_states(
            actual_output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
        )
        return output


__all__ = [
    "FlashAttentionQkvFp8Sm80FusedBackend",
    "FlashAttentionQkvFp8Sm80FusedImpl",
    "FlashAttentionQkvFp8Sm80FusedMetadataBuilder",
]
