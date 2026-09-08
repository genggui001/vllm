# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in paged attention with byte-packed E4M3FN KV cache on SM80."""

from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_sm80_fp8 import (
    scaled_e4m3fn_qdq_inplace,
    sm80_fp8_unified_attention,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

logger = init_logger(__name__)


class TritonFp8Sm80AttentionBackend(TritonAttentionBackend):
    """Publish uint8 cache pages while retaining an explicit backend switch."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

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
        return "TRITON_ATTN_FP8_SM80"

    @staticmethod
    def get_impl_cls() -> type["TritonFp8Sm80AttentionImpl"]:
        return TritonFp8Sm80AttentionImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(8, 0)


class TritonFp8Sm80AttentionImpl(TritonAttentionImpl):
    """Store static-scale E4M3FN bytes and decode each paged-attention tile."""

    attention_fn = staticmethod(sm80_fp8_unified_attention)
    quantize_bf16_query: ClassVar[bool] = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        capability = current_platform.get_device_capability()
        if capability != DeviceCapability(8, 0):
            raise NotImplementedError(
                "TRITON_ATTN_FP8_SM80 requires compute capability 8.0, got "
                f"{capability}."
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "TRITON_ATTN_FP8_SM80 uses its backend selection as the cache "
                f"switch and requires external dtype auto/bfloat16, got "
                f"{self.kv_cache_dtype}."
            )
        self._kv_quant_mode = KVQuantMode.SM80_FP8_PER_TENSOR
        query_mode = (
            "one-pass in-place E4M3FN Q QDQ"
            if self.quantize_bf16_query
            else "BF16 query"
        )
        logger.info_once(
            "Using SM80 software-decoded E4M3FN uint8 KV cache with %s", query_mode
        )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        if kv_cache.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 KV cache, got {kv_cache.dtype}")

        key_cache, value_cache = kv_cache.transpose(1, 2).split(
            self.head_size, dim=-1
        )
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


class TritonQkvFp8Sm80AttentionBackend(TritonFp8Sm80AttentionBackend):
    """SM80 uint8 K/V cache plus one-pass static-scale E4M3FN Q QDQ."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_QKV_FP8_SM80"

    @staticmethod
    def get_impl_cls() -> type["TritonQkvFp8Sm80AttentionImpl"]:
        return TritonQkvFp8Sm80AttentionImpl


class TritonQkvFp8Sm80AttentionImpl(TritonFp8Sm80AttentionImpl):
    """Match Q/K/V FP8 QAT while computing attention on SM80 BF16 cores."""

    quantize_bf16_query: ClassVar[bool] = True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is not None:
            scaled_e4m3fn_qdq_inplace(
                query,
                layer._q_scale,
                num_tokens=attn_metadata.num_actual_tokens,
            )
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


__all__ = [
    "TritonFp8Sm80AttentionBackend",
    "TritonFp8Sm80AttentionImpl",
    "TritonQkvFp8Sm80AttentionBackend",
    "TritonQkvFp8Sm80AttentionImpl",
]
