# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QAT-aligned FP8 QDQ followed by FlashAttention 2 on SM80.

This opt-in backend is deliberately separate from the regular FlashAttention
backend. It uses the checkpoint's static Q/K/V scales, materializes BF16 QDQ
values, stores K/V in a BF16 cache, and delegates attention computation to FA2.
It is the numerical reference for a future fused SM80 FP8-cache kernel.
"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.ops.fp8_qdq import scaled_fp8_e4m3_qdq

logger = init_logger(__name__)


class FlashAttentionFp8QdqSm80Backend(FlashAttentionBackend):
    """Opt-in SM80 numerical-reference backend for QAT FP8 Q/K/V."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_FP8_QDQ_SM80"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionFp8QdqSm80Impl"]:
        return FlashAttentionFp8QdqSm80Impl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(8, 0)


class FlashAttentionFp8QdqSm80Impl(FlashAttentionImpl):
    """QDQ Q/K/V to BF16, then use the unmodified FA2 implementation."""

    quantize_query: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.vllm_flash_attn_version != 2:
            raise NotImplementedError(
                "FLASH_ATTN_FP8_QDQ_SM80 requires FlashAttention 2, got "
                f"version {self.vllm_flash_attn_version}."
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "The reference backend currently requires a BF16 KV cache; "
                f"got {self.kv_cache_dtype}."
            )
        components = "Q/K/V" if self.quantize_query else "K/V"
        logger.info_once(
            "Using SM80 QAT FP8 %s QDQ with a BF16 KV cache and FlashAttention 2",
            components,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        num_actual_tokens = slot_mapping.shape[0]
        key = scaled_fp8_e4m3_qdq(key[:num_actual_tokens], layer._k_scale)
        value = scaled_fp8_e4m3_qdq(value[:num_actual_tokens], layer._v_scale)
        super().do_kv_cache_update(
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
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
        if attn_metadata is None:
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

        num_actual_tokens = attn_metadata.num_actual_tokens
        if self.quantize_query:
            query = scaled_fp8_e4m3_qdq(
                query[:num_actual_tokens],
                layer._q_scale,
            )

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            key = scaled_fp8_e4m3_qdq(key[:num_actual_tokens], layer._k_scale)
            value = scaled_fp8_e4m3_qdq(value[:num_actual_tokens], layer._v_scale)

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


class FlashAttentionKvFp8QdqSm80Backend(FlashAttentionFp8QdqSm80Backend):
    """SM80 numerical-reference backend that quantizes K/V but leaves Q BF16."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_KV_FP8_QDQ_SM80"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionKvFp8QdqSm80Impl"]:
        return FlashAttentionKvFp8QdqSm80Impl


class FlashAttentionKvFp8QdqSm80Impl(FlashAttentionFp8QdqSm80Impl):
    """QDQ K/V to BF16, then use BF16 Q with unmodified FA2."""

    quantize_query: ClassVar[bool] = False


__all__ = [
    "FlashAttentionFp8QdqSm80Backend",
    "FlashAttentionFp8QdqSm80Impl",
    "FlashAttentionKvFp8QdqSm80Backend",
    "FlashAttentionKvFp8QdqSm80Impl",
]
