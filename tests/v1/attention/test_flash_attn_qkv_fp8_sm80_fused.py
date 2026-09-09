# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import math

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused import (
    FlashAttentionQkvFp8Sm80FusedBackend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode


def _is_sm80() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def test_backend_registration_spec_and_guards() -> None:
    assert (
        AttentionBackendEnum.FLASH_ATTN_QKV_FP8_SM80_FUSED.get_class()
        is FlashAttentionQkvFp8Sm80FusedBackend
    )
    assert FlashAttentionQkvFp8Sm80FusedBackend.supports_compute_capability(
        DeviceCapability(8, 0)
    )
    assert not FlashAttentionQkvFp8Sm80FusedBackend.supports_compute_capability(
        DeviceCapability(8, 9)
    )
    assert FlashAttentionQkvFp8Sm80FusedBackend.get_supported_kernel_block_sizes() == [
        16
    ]

    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=256,
        dtype=torch.bfloat16,
    )
    customized = FlashAttentionQkvFp8Sm80FusedBackend.customize_spec(spec)
    assert customized.dtype == torch.uint8
    assert customized.kv_quant_mode == KVQuantMode.SM80_FP8_PER_TENSOR


def _qdq(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return (
        (x.float() / scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .float()
        .mul(scale)
        .bfloat16()
    )


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize(
    ("query_lens", "kv_lens", "causal"),
    [
        ([1, 1], [17, 31], True),  # zero-copy decode GQA packing
        ([1, 3], [17, 31], True),
        ([7, 19], [33, 61], True),
        ([5, 11], [23, 45], False),
    ],
)
def test_cuda_op_matches_materialized_fa2_qdq(
    query_lens: list[int], kv_lens: list[int], causal: bool
) -> None:
    importlib.import_module("vllm.vllm_flash_attn._vllm_fa2_C")
    importlib.import_module("vllm.vllm_flash_attn._vllm_fa2_sm80_fp8_C")

    torch.manual_seed(1234 + int(causal))
    device = torch.device("cuda")
    num_query_heads, num_kv_heads, head_size, page_size = 8, 4, 256, 16
    batch_size = len(query_lens)
    pages_per_sequence = max(math.ceil(length / page_size) for length in kv_lens)
    num_pages = batch_size * pages_per_sequence
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(batch_size, pages_per_sequence)
    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    q_scale = torch.tensor(0.0137, dtype=torch.float32, device=device)
    k_scale = torch.tensor(0.0113, dtype=torch.float32, device=device)
    v_scale = torch.tensor(0.0151, dtype=torch.float32, device=device)
    key_bytes = (
        (key.float() / k_scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    value_bytes = (
        (value.float() / v_scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    query_reference = _qdq(query, q_scale)
    key_reference = key_bytes.view(torch.float8_e4m3fn).float().mul(k_scale).bfloat16()
    value_reference = (
        value_bytes.view(torch.float8_e4m3fn).float().mul(v_scale).bfloat16()
    )
    cu_query_lens = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    output = torch.empty_like(query)
    reference = torch.empty_like(query)
    softmax_scale = head_size**-0.5

    torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd(
        query,
        key_bytes,
        value_bytes,
        output,
        cu_query_lens,
        kv_lens_tensor,
        block_table,
        q_scale,
        k_scale,
        v_scale,
        max(query_lens),
        max(kv_lens),
        softmax_scale,
        causal,
    )
    dummy_cu_k = torch.zeros_like(cu_query_lens)
    torch.ops._vllm_fa2_C.varlen_fwd(
        query_reference,
        key_reference,
        value_reference,
        reference,
        cu_query_lens,
        dummy_cu_k,
        kv_lens_tensor,
        None,
        block_table,
        None,
        max(query_lens),
        max(kv_lens),
        0.0,
        softmax_scale,
        False,
        causal,
        -1,
        0 if causal else -1,
        0.0,
        False,
        0,
        None,
    )

    assert torch.equal(output, reference)
