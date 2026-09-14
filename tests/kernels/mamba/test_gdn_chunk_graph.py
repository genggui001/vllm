# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short GDN graphs must preserve bits, output lifetime and bounded capture."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.ops import gdn_chunk_graph as graph_ops
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(89),
    reason="Short-prefill CUDA Graph specialization targets SM89.",
)


class _Owner:
    pass


def _inputs(lengths, state_dtype):
    total = sum(lengths)
    q = torch.nn.functional.normalize(
        torch.randn((1, total, 8, 128), device="cuda"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    starts, offsets = [0], [0]
    indices: list[tuple[int, int]] = []
    for sequence, length in enumerate(lengths):
        starts.append(starts[-1] + length)
        chunks = (length + 63) // 64
        offsets.append(offsets[-1] + chunks)
        indices.extend((sequence, chunk) for chunk in range(chunks))
    return dict(
        q=q,
        k=k,
        v=torch.randn((1, total, 16, 128), device="cuda", dtype=torch.bfloat16),
        g=-torch.rand((1, total, 16), device="cuda"),
        beta=torch.rand((1, total, 16), device="cuda"),
        initial_state=torch.randn(
            (len(lengths), 16, 128, 128), device="cuda", dtype=state_dtype
        ),
        cu_seqlens=torch.tensor(starts, device="cuda", dtype=torch.int32),
        chunk_indices=torch.tensor(indices, device="cuda", dtype=torch.int32),
        chunk_offsets=torch.tensor(offsets, device="cuda", dtype=torch.int64),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )


def _assert_bits(actual, expected):
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


@pytest.mark.parametrize(
    "option,expected",
    [
        ("auto", "triton_graph"),
        ("triton_graph", "triton_graph"),
        ("triton", "triton"),
        ("w4a16", "triton"),
        ("other_attention", "triton"),
        ("eager", "triton"),
        ("sleep", "triton"),
        ("other_shape", "triton"),
        ("dynamic_kv", "triton"),
    ],
)
def test_backend_respects_recipe_explicit_choice_and_graph_constraints(
    option, expected
):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _resolve_gdn_prefill_backend,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    scheme = dict(num_bits=8, type="float", strategy="tensor", symmetric=True)
    config = SimpleNamespace(
        additional_config={"gdn_prefill_backend": option},
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=option == "eager",
            enable_sleep_mode=option == "sleep",
            hf_text_config=SimpleNamespace(
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_num_key_heads=16,
                linear_num_value_heads=16 if option == "other_shape" else 32,
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        quant_config=SimpleNamespace(
            get_name=lambda: "compressed-tensors", kv_cache_scheme=scheme
        ),
        attention_config=SimpleNamespace(backend=None),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
    )
    if option not in ("auto", "triton", "triton_graph"):
        config.additional_config = {}
    if option == "w4a16":
        config.quant_config.kv_cache_scheme = None
        config.cache_config.cache_dtype = "bfloat16"
    elif option == "other_attention":
        config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    elif option == "dynamic_kv":
        scheme["dynamic"] = True
    assert _resolve_gdn_prefill_backend(cast(VllmConfig, config))[1] == expected


@pytest.fixture(scope="module", params=[torch.float32, torch.bfloat16])
def cache(request):
    owner = _Owner()
    result = graph_ops.get_short_prefill_graph_cache(owner)
    with torch.inference_mode():
        result.warmup(torch.device("cuda", 0), request.param)
    assert len(result.entries) == 8
    assert 0 < result.charged_bytes <= result.max_bytes
    yield result, request.param


@pytest.mark.parametrize(
    "total", [1, 63, 64, 65, 97, 127, 128, 129, 191, 257, 511, 512]
)
@torch.inference_mode()
def test_bucket_replay_preserves_bits_without_runtime_capture(cache, total):
    graph_cache, dtype = cache
    first = _inputs([total], dtype)
    expected = chunk_gated_delta_rule(**first)
    charged = graph_cache.charged_bytes
    with (
        patch.object(
            torch.cuda, "CUDAGraph", side_effect=AssertionError("Runtime capture")
        ),
        patch.object(
            graph_ops,
            "chunk_gated_delta_rule",
            side_effect=AssertionError("Unexpected fallback"),
        ),
    ):
        actual = graph_cache(**first)
    _assert_bits(actual, expected)
    second = _inputs([total], dtype)
    next_expected = chunk_gated_delta_rule(**second)
    _assert_bits(graph_cache(**second), next_expected)
    _assert_bits(actual, expected)
    assert graph_cache.charged_bytes == charged


@pytest.mark.parametrize(
    "lengths,scale", [([513], None), ([63, 65], None), ([65], 0.0)]
)
@torch.inference_mode()
def test_unsupported_shape_or_scale_falls_back_without_capture(cache, lengths, scale):
    graph_cache, dtype = cache
    arguments = _inputs(lengths, dtype)
    arguments["scale"] = scale
    expected = chunk_gated_delta_rule(**arguments)
    with (
        patch.object(
            torch.cuda, "CUDAGraph", side_effect=AssertionError("Runtime capture")
        ),
        patch.object(
            graph_ops, "chunk_gated_delta_rule", wraps=chunk_gated_delta_rule
        ) as fallback,
    ):
        actual = graph_cache(**arguments)
    assert fallback.call_count == 1
    _assert_bits(actual, expected)


@torch.inference_mode()
def test_other_stream_uses_native_without_reusing_live_graph_buffers(cache):
    graph_cache, dtype = cache
    arguments = _inputs([128], dtype)
    first = graph_cache(**arguments)
    expected = chunk_gated_delta_rule(**arguments)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.cuda.stream(stream),
        patch.object(
            graph_ops, "chunk_gated_delta_rule", wraps=chunk_gated_delta_rule
        ) as fallback,
    ):
        actual = graph_cache(**arguments)
    torch.cuda.current_stream().wait_stream(stream)
    assert fallback.call_count == 1
    _assert_bits(actual, expected)
    _assert_bits(first, expected)


@torch.inference_mode()
def test_zero_cache_budget_preserves_native_result():
    owner = _Owner()
    cache = graph_ops.ShortPrefillGraphCache(owner, max_bytes=0)
    with patch.object(
        torch.cuda, "CUDAGraph", side_effect=AssertionError("Budget exceeded")
    ):
        cache.warmup(torch.device("cuda", 0), torch.float32)
        arguments = _inputs([65], torch.float32)
        actual = cache(**arguments)
    assert not cache.entries and cache.charged_bytes == 0
    _assert_bits(actual, chunk_gated_delta_rule(**arguments))
