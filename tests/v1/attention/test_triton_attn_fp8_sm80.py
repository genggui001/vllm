# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn_fp8_sm80 import (
    TritonFp8Sm80AttentionBackend,
    TritonQkvFp8Sm80AttentionBackend,
)
from vllm.v1.attention.ops.fp8_qdq import scaled_fp8_e4m3_qdq
from vllm.v1.attention.ops.triton_sm80_fp8 import (
    decode_e4m3fn_uint8,
    scaled_e4m3fn_qdq_inplace,
    sm80_fp8_unified_attention,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode


def test_backend_registration_spec_and_sm80_guard() -> None:
    assert (
        AttentionBackendEnum.TRITON_ATTN_FP8_SM80.get_class()
        is TritonFp8Sm80AttentionBackend
    )
    assert (
        AttentionBackendEnum.TRITON_ATTN_QKV_FP8_SM80.get_class()
        is TritonQkvFp8Sm80AttentionBackend
    )
    assert TritonFp8Sm80AttentionBackend.supports_compute_capability(
        DeviceCapability(8, 0)
    )
    assert not TritonFp8Sm80AttentionBackend.supports_compute_capability(
        DeviceCapability(8, 9)
    )
    assert TritonQkvFp8Sm80AttentionBackend.supports_compute_capability(
        DeviceCapability(8, 0)
    )

    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
    )
    customized = TritonFp8Sm80AttentionBackend.customize_spec(spec)
    assert customized.dtype == torch.uint8
    assert customized.kv_quant_mode == KVQuantMode.SM80_FP8_PER_TENSOR


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_software_e4m3fn_decode_matches_torch() -> None:
    encoded = torch.arange(256, dtype=torch.uint8, device="cuda")
    scale = torch.tensor([0.125], dtype=torch.float32, device="cuda")

    actual = decode_e4m3fn_uint8(encoded, scale, dtype=torch.float32)
    reference = encoded.view(torch.float8_e4m3fn).float() * scale
    finite = reference.isfinite()

    torch.testing.assert_close(actual[finite], reference[finite], rtol=0, atol=0)
    assert torch.equal(actual.isnan(), reference.isnan())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inplace_query_qdq_matches_materialized_reference() -> None:
    torch.manual_seed(1)
    query = torch.randn(37, 8, 256, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    num_tokens = 29
    expected = query.clone()
    expected[:num_tokens] = scaled_fp8_e4m3_qdq(expected[:num_tokens], scale)

    actual = query.clone()
    scaled_e4m3fn_qdq_inplace(actual, scale, num_tokens=num_tokens)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_cache_scatter_writes_e4m3fn_bytes_on_sm80() -> None:
    torch.manual_seed(0)
    key = torch.randn(4, 2, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    k_scale = torch.tensor([0.015625], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([0.03125], dtype=torch.float32, device="cuda")
    cache = torch.zeros(4, 2, 16, 256, dtype=torch.uint8, device="cuda")
    key_cache, value_cache = cache.transpose(1, 2).split(128, dim=-1)
    slot_mapping = torch.tensor([0, 17, 34, 51], dtype=torch.int64, device="cuda")

    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "fp8_e4m3",
        k_scale,
        v_scale,
    )

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    key_reference = (
        (key.float() / k_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    ).view(torch.uint8)
    value_reference = (
        (value.float() / v_scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    ).view(torch.uint8)
    blocks = torch.div(slot_mapping, 16, rounding_mode="floor")
    offsets = slot_mapping % 16

    torch.testing.assert_close(key_cache[blocks, offsets], key_reference)
    torch.testing.assert_close(value_cache[blocks, offsets], value_reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm80_fp8_attention_matches_materialized_bf16_qdq() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 128
    block_size = 16
    num_blocks = 8
    query_lens = [1, 3]
    kv_lens = [17, 21]

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    value = torch.randn_like(key)
    k_scale = torch.tensor([0.015625], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.03125], dtype=torch.float32, device=device)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    key_bytes = (
        (key.float() / k_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    ).view(torch.uint8)
    value_bytes = (
        (value.float() / v_scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    ).view(torch.uint8)
    key_qdq = (key_bytes.view(torch.float8_e4m3fn).float() * k_scale).to(dtype)
    value_qdq = (value_bytes.view(torch.float8_e4m3fn).float() * v_scale).to(dtype)

    cu_query_lens = torch.tensor(
        [0, query_lens[0], sum(query_lens)], dtype=torch.int32, device=device
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    scale_shape = (len(query_lens), num_kv_heads)
    k_descale = k_scale.expand(scale_shape)
    v_descale = v_scale.expand(scale_shape)
    output = torch.empty_like(query)
    reference = torch.empty_like(query)

    common = {
        "cu_seqlens_q": cu_query_lens,
        "max_seqlen_q": max(query_lens),
        "seqused_k": kv_lens_tensor,
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": head_size**-0.5,
        "causal": True,
        "window_size": (-1, -1),
        "block_table": block_tables,
        "softcap": 0,
        "seq_threshold_3D": 0,
    }
    sm80_fp8_unified_attention(
        q=query,
        k=key_bytes,
        v=value_bytes,
        out=output,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        **common,
    )
    unified_attention(
        q=query,
        k=key_qdq,
        v=value_qdq,
        out=reference,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.NONE,
        **common,
    )

    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
