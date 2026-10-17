# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical contracts for the eager training QDQ precision reference."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.sm80_qdq_reference import (
    MoEGroup32QDQReference,
    SavedQKVScaleQDQReference,
    SM80QDQReferenceConfig,
    dequant_fp8_group32,
    dequant_int4_group32,
    group32_fp8_qdq,
    tensor_fp8_qdq,
    training_swiglu,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "kind, expected",
    [
        ("group32", "211bcb93510af86ca87d586cae540eda0af7360cdf412b5b6211f83bb9b12fc3"),
        ("qkv", "bbea023dce78f5a989ff2f7c133ec0569b3f0fbb914904a6a16ebbd04c963372"),
    ],
)
def test_all_finite_bf16_against_training_goldens(kind, expected):
    """Goldens from qdq.fp8_e4m3, source SHA ba625b64022eee0fa79bfe92dc6b08c5.

    All 65,280 finite BF16 encodings, including both signed zeros and subnormals.
    Catch changed clipping, scale rounding/floor, or scalar promotion rules.
    """
    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = raw[raw.isfinite()].reshape(-1, 32).cuda()
    if kind == "group32":
        actual = group32_fp8_qdq(x)
    else:
        scale = torch.tensor([0.0350167416036129], dtype=torch.float32, device="cuda")
        actual = tensor_fp8_qdq(x, scale)
    digest = hashlib.sha256(
        actual.view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()
    assert digest == expected


def test_int4_unsigned_nibble_offset_and_bf16_scale():
    """0x0 is -8, 0x8 is zero, and 0xf is +7 in this checkpoint format."""
    packed = torch.tensor([[0x76543210, -19088744] * 2], dtype=torch.int32)
    scales = torch.tensor([[0.25]], dtype=torch.bfloat16)
    expected = (torch.arange(-8, 8).repeat(2).reshape(1, 32) * 0.25).bfloat16()
    assert torch.equal(dequant_int4_group32(packed, scales), expected)


def test_dense_scale_dtype_cannot_be_silently_demoted():
    codes = torch.ones(1, 32).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="FP32 scales"):
        dequant_fp8_group32(codes, torch.ones(1, 1, dtype=torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_router_probability_rounds_before_fc2_against_training_golden():
    """The old post-FC2 router multiplication fails this training-generated hash."""
    x = ((torch.arange(128, device="cuda").reshape(2, 64) % 17) - 8).bfloat16() / 8
    w1 = ((torch.arange(4096, device="cuda").reshape(64, 64) % 23) - 11).bfloat16() / 64
    w2 = ((torch.arange(2048, device="cuda").reshape(64, 32) % 19) - 9).bfloat16() / 32
    probs = torch.tensor([[0.371], [0.619]], device="cuda", dtype=torch.float32)
    ids = torch.zeros(2, 1, device="cuda", dtype=torch.int64)
    method = MoEGroup32QDQReference(
        SimpleNamespace(
            activation=MoEActivation.SILU,
            has_bias=False,
            swiglu_limit=None,
        )
    )
    layer = SimpleNamespace(w13_weight=w1[None], w2_weight=w2[None])
    actual = method.apply(layer, x, probs, ids)
    digest = hashlib.sha256(
        actual.view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()
    assert digest == "a6e4c6985873ae7fb735fabefb476356d159acd965b057af7c98f59b55091a14"


@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), 0.0, -0.1])
def test_invalid_or_missing_qkv_checkpoint_scale_fails_loading(bad_scale):
    config = SimpleNamespace(
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "symmetric": True,
        }
    )
    method = SavedQKVScaleQDQReference(config)
    layer = torch.nn.Module()
    layer.num_kv_heads = 4
    layer.layer_name = "test.attn"
    method.create_weights(layer)
    layer.q_scale.data.fill_(bad_scale)
    layer.k_scale.data.fill_(0.03)
    layer.v_scale.data.fill_(0.02)
    with pytest.raises(ValueError, match="checkpoint scale"):
        method.process_weights_after_loading(layer)


def test_reference_is_never_an_automatic_override():
    checkpoint = {"quant_method": "compressed-tensors"}
    assert SM80QDQReferenceConfig.override_quantization_method(checkpoint, None) is None
    assert (
        SM80QDQReferenceConfig.override_quantization_method(
            checkpoint, "compressed-tensors"
        )
        is None
    )
    assert (
        SM80QDQReferenceConfig.override_quantization_method(
            checkpoint, "sm80_qdq_reference"
        )
        == "sm80_qdq_reference"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "tokens,skewed,strided",
    [
        (0, False, False),
        (1, True, False),
        (17, False, True),
        (65, True, False),
        (257, False, False),
    ],
)
def test_batched_reference_moe_matches_expert_loop(tokens, skewed, strided):
    """Guard routing order, inactive experts and GEMM/activation rounding."""
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        torch.manual_seed(20260927)
        h, i, experts, top_k = 2048, 256, 16, 4
        x = torch.randn(
            tokens, h * (2 if strided else 1), device="cuda", dtype=torch.bfloat16
        )
        if strided:
            x = x[:, ::2]
        layer = SimpleNamespace(
            w13_weight=torch.randn(experts, 2 * i, h, device="cuda", dtype=x.dtype)
            * 0.03,
            w2_weight=torch.randn(experts, h, i, device="cuda", dtype=x.dtype) * 0.03,
        )
        p, ids = torch.randn(tokens, experts, device="cuda").softmax(-1).topk(top_k, -1)
        if skewed:
            ids = torch.arange(top_k, device="cuda").expand(tokens, -1)
        p = p / p.sum(-1, keepdim=True)
        if strided:
            p = p.t().contiguous().t()
        method = MoEGroup32QDQReference(
            SimpleNamespace(
                activation=MoEActivation.SILU, has_bias=False, swiglu_limit=None
            )
        )
        expected = torch.zeros_like(x)
        if tokens:
            xq = group32_fp8_qdq(x)
            values, all_rows = [], []
            for expert in torch.unique(ids).tolist():
                rows, slots = torch.where(ids == expert)
                fc1 = F.linear(xq[rows], layer.w13_weight[expert])
                mid = group32_fp8_qdq(training_swiglu(fc1))
                mid = (mid * p[rows, slots, None]).to(x.dtype)
                values.append(F.linear(mid, layer.w2_weight[expert]))
                all_rows.append(rows)
            expected.index_add_(0, torch.cat(all_rows), torch.cat(values))
        actual = method.apply(layer, x, p, ids)
        assert torch.equal(
            actual.contiguous().view(torch.int16),
            expected.contiguous().view(torch.int16),
        )
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            old_reduction
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,width",
    [
        (0, 32),
        (1, 2048),
        (17, 256),
        (257, 512),
        (15, 1024),
        (16, 1024),
        (127, 1024),
        (128, 1024),
        (255, 256),
        (256, 256),
        (2047, 256),
        (2048, 256),
        (8191, 256),
        (8192, 256),
    ],
)
def test_fused_group32_preserves_training_bits_for_strided_inputs(rows, width):
    """Catch group boundaries, FP32-tiny scale floor and dropped BF16 roundings."""
    from vllm.model_executor.layers.quantization.utils.sm80_group32_qdq import (
        group32_fp8_qdq_fused,
        silu_mul_group32_fp8_qdq_fused,
    )

    torch.manual_seed(20260929)
    x = torch.randn(rows, width * 4, device="cuda", dtype=torch.bfloat16)[:, ::2]
    if rows:
        x[0, :32] *= torch.finfo(torch.bfloat16).tiny
        x[0, 32:64] = -0.0
    expected = group32_fp8_qdq(x)
    actual = group32_fp8_qdq_fused(x)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    # Intentionally strided probabilities and reversed expert-major slot order.
    probabilities = torch.rand(rows * 2, device="cuda")[::2]
    order = torch.arange(rows - 1, -1, -1, device="cuda", dtype=torch.int32)
    expected = (
        group32_fp8_qdq(training_swiglu(x)) * probabilities[order.long(), None]
    ).bfloat16()
    actual = silu_mul_group32_fp8_qdq_fused(x, probabilities, order)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("arithmetic_silu", [False, True])
def test_fused_group32_all_finite_bf16_bits(monkeypatch, arithmetic_silu):
    """Exhaustive BF16 codes cover signed zero, subnormals and FP8 ties."""
    from vllm.model_executor.layers.quantization.utils import sm80_group32_qdq as fused

    if arithmetic_silu:
        monkeypatch.setattr(fused, "_training_silu_table", lambda x: None)
    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = raw[raw.isfinite()].reshape(-1, 32).cuda()
    expected = group32_fp8_qdq(x)
    actual = fused.group32_fp8_qdq_fused(x)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    gate_up = torch.cat((x, torch.ones_like(x)), -1)
    expected = group32_fp8_qdq(training_swiglu(gate_up))
    actual = fused.silu_mul_group32_fp8_qdq_fused(gate_up)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    wide = x.reshape(-1, 256)
    shared_gate_up = torch.cat((wide, torch.ones_like(wide)), -1)
    shared_expected = group32_fp8_qdq(training_swiglu(shared_gate_up))
    shared_actual = fused.silu_mul_group32_fp8_qdq_fused(shared_gate_up)
    assert torch.equal(
        shared_actual.view(torch.int16), shared_expected.view(torch.int16)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_group32_graph_replays_new_values_and_router_order():
    """Graph replay must read fresh activation, FP32 probabilities and row mapping."""
    from vllm.model_executor.layers.quantization.utils.sm80_group32_qdq import (
        silu_mul_group32_fp8_qdq_fused,
    )

    x = torch.randn(37, 512, device="cuda", dtype=torch.bfloat16)
    p = torch.rand(37, device="cuda")
    order = torch.arange(37, device="cuda", dtype=torch.int64)
    out = torch.empty(37, 256, device="cuda", dtype=x.dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        silu_mul_group32_fp8_qdq_fused(x, p, order, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        silu_mul_group32_fp8_qdq_fused(x, p, order, out)
    for _ in range(8):
        x.normal_()
        p.uniform_()
        order.copy_(torch.randperm(37, device="cuda"))
        expected = (group32_fp8_qdq(training_swiglu(x)) * p[order, None]).bfloat16()
        graph.replay()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("leading", [(0,), (3,), (2, 5)])
def test_group32_dense_fp8_preserves_strided_and_empty_shapes(
    leading, default_vllm_config
):
    """The FP8 core handles empty/batched inputs and cannot use BF16 Linear."""
    from vllm.model_executor.layers.quantization.sm80_group32 import DenseGroup32Fused

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM80 kernel")
    x = torch.randn(*leading, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    layer, reference_weight = _make_dense_fp8_test_layer(128, 128)
    scheme = DenseGroup32Fused()
    scheme.process_weights_after_loading(layer)
    expected = F.linear(group32_fp8_qdq(x), reference_weight)
    actual = scheme.apply_weights(layer, x)
    assert actual.shape == (*leading, 128)
    assert layer.weight.dtype == torch.int32
    assert layer.weight_scale.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.002)


def _make_dense_fp8_test_layer(n, k):
    layer = torch.nn.Module()
    codes = (torch.randn(n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    scales = torch.rand(n, k // 32, device="cuda") * 0.002 + 0.001
    layer.weight = torch.nn.Parameter(codes, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    return layer, dequant_fp8_group32(codes, scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("top_k", [2, 8])
def test_group32_packed_moe_graph_refreshes_routing_and_ignores_invalid_slots(
    deterministic, top_k, default_vllm_config, workspace_init
):
    """Native Marlin replay uses current routes and masks invalid expert slots."""

    torch.manual_seed(20260930)
    method, layer = _make_group32_marlin_test_layer(8, 128, 128, top_k)
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    p = torch.rand(17, top_k, device="cuda")
    ids = torch.zeros(17, top_k, device="cuda", dtype=torch.int64)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            method.apply(layer, x, p, ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method.apply(layer, x, p, ids)
        for mode in ("mixed", "mixed", "negative", "positive"):
            x.normal_()
            p.uniform_()
            ids.random_(-1, 9)
            if mode == "negative":
                ids.fill_(-1)
            elif mode == "positive":
                ids.fill_(8)
            torch.use_deterministic_algorithms(True)
            expected = method.apply(layer, x, p, ids)
            torch.use_deterministic_algorithms(deterministic)
            graph.replay()
            assert torch.isfinite(captured).all()
            assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
            if mode in ("negative", "positive"):
                assert torch.count_nonzero(captured) == 0
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("language_model_only", [None, True, False])
def test_group32_graph_capability_keeps_reference_eager_guard(
    monkeypatch, language_model_only
):
    """Graph support must not imply support for unvalidated multimodal inputs."""
    from vllm.config.multimodal import MultiModalConfig
    from vllm.model_executor.layers.quantization import sm80_qdq_reference as ref
    from vllm.model_executor.layers.quantization.sm80_group32 import SM80Group32Config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            enforce_eager=False,
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
            multimodal_config=(
                None
                if language_model_only is None
                else MultiModalConfig(language_model_only=language_model_only)
            ),
        ),
        cache_config=SimpleNamespace(cache_dtype="bfloat16"),
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            use_sequence_parallel_moe=False,
        ),
        lora_config=None,
    )
    monkeypatch.setattr(ref, "get_current_vllm_config", lambda: config)
    reference = object.__new__(SM80QDQReferenceConfig)
    packed = object.__new__(SM80Group32Config)
    with pytest.raises(ValueError, match="enforce-eager"):
        reference._check_runtime()
    if language_model_only is False:
        with pytest.raises(ValueError, match="--language-model-only"):
            packed._check_runtime()
    else:
        packed._check_runtime()
    config.model_config.enforce_eager = True
    reference._check_runtime()
    config.model_config.multimodal_config = None
    config.parallel_config.enable_expert_parallel = True
    with pytest.raises(ValueError, match="not EP"):
        packed._check_runtime()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 17, 129])
def test_saved_qkv_fusion_preserves_independent_fp32_scales(rows):
    """Catch scale narrowing, noncontiguous views and signed-zero changes."""
    from vllm.model_executor.layers.quantization.sm80_qdq_reference import (
        tensor_fp8_qdq,
    )
    from vllm.model_executor.layers.quantization.utils.sm80_group32_qkv import (
        qkv_fp8_qdq_fused,
    )

    torch.manual_seed(20260929)
    inputs = [
        torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
        for width in (2048, 1024, 1024)
    ]
    if rows:
        for value in inputs:
            value[0, 0] = -0.0
            value[0, 1] = torch.finfo(torch.bfloat16).tiny
            value[0, 2] = torch.finfo(torch.bfloat16).max
    scales = [
        torch.tensor([scale], device="cuda", dtype=torch.float32)
        for scale in (0.0137, 0.0113, 0.0151)
    ]
    expected = [tensor_fp8_qdq(x, scale) for x, scale in zip(inputs, scales)]
    actual = qkv_fp8_qdq_fused(*inputs, *scales)
    for result, reference in zip(actual, expected):
        assert torch.equal(result.view(torch.int16), reference.view(torch.int16))


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_group32_fp8_cache_requires_matching_backend_and_keeps_reference_guard(
    cache_dtype,
):
    """Reject reference/FA2 mismatches using the actual v0.29 CacheConfig schema."""
    from vllm.config.cache import CacheConfig
    from vllm.model_executor.layers.quantization.sm80_group32 import SM80Group32Config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    reference = object.__new__(SM80QDQReferenceConfig)
    packed = object.__new__(SM80Group32Config)
    config = SimpleNamespace(
        cache_config=CacheConfig(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASH_ATTN_QKV_FP8_SM80_FUSED
        ),
    )
    packed._check_attention_runtime(config)
    with pytest.raises(ValueError, match="kv-cache-dtype bfloat16"):
        reference._check_attention_runtime(config)
    config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    with pytest.raises(ValueError, match="requires BF16 KV"):
        packed._check_attention_runtime(config)
    config.cache_config.cache_dtype = "bfloat16"
    reference._check_attention_runtime(config)
    packed._check_attention_runtime(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 3, 4, 16, 129])
@pytest.mark.parametrize("with_residual", [False, True])
def test_group32_residual_norm_matches_native_shape_dependent_rounding(
    rows, with_residual
):
    """Preserve native mean ordering, BF16 residual/norm and group32 boundaries."""
    from vllm.model_executor.layers.quantization.utils.sm80_group32_norm import (
        residual_norm_group32_qdq,
    )

    torch.manual_seed(20260930)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)[:, :2048]
    residual = torch.randn_like(x) if with_residual else None
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16) * 0.02
    before = x.clone()
    summed = x.float() if residual is None else x.float() + residual.float()
    expected = (
        (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6))
        * (weight.float() + 1)
    ).bfloat16()
    normalized, prepared, updated = residual_norm_group32_qdq(x, residual, weight, 1e-6)
    for actual, target in (
        (normalized, expected),
        (prepared, group32_fp8_qdq(expected)),
        (updated, summed.bfloat16()),
        (x, before),
    ):
        assert torch.equal(actual.view(torch.int16), target.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 17, 257])
def test_group32_prepared_dense_fp8_reuses_codes_without_requantization(
    rows, default_vllm_config
):
    """Both GDN consumers receive identical prepared FP8 bytes and FP32 scales."""
    from vllm.model_executor.layers.quantization.sm80_group32 import DenseGroup32Fused
    from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
        quantize_input,
    )

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM80 kernel")
    torch.manual_seed(20260930)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)[:, ::2]
    prepared = quantize_input(x)
    for width in (32, 6144):
        layer, reference_weight = _make_dense_fp8_test_layer(width, 2048)
        scheme = DenseGroup32Fused()
        scheme.process_weights_after_loading(layer)
        expected = F.linear(group32_fp8_qdq(x), reference_weight)
        actual = scheme.apply_weights(layer, prepared)
        separate = scheme.apply_weights(layer, x)
        assert torch.equal(actual.view(torch.int16), separate.view(torch.int16))
        # Split reduction is not required to reproduce cuBLAS's accumulation tree.
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.002)


@pytest.mark.parametrize(
    "n,k,m", [(0, 128, 1), (128, 64, 1), (128, 192, 1), (6144, 2048, 524280)]
)
def test_dense_fp8_unsupported_static_shapes_fail_without_rollback(n, k, m):
    from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
        validate_dense_shape,
    )

    with pytest.raises(ValueError, match="BF16 rollback is disabled"):
        validate_dense_shape(n, k, m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_fp8_encoding_preserves_all_finite_bf16_qdq_bits():
    from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
        quantize_input,
    )

    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = raw[raw.isfinite()].reshape(-1, 32).cuda()
    prepared = quantize_input(x)
    actual = dequant_fp8_group32(prepared.codes, prepared.scales)
    expected = group32_fp8_qdq(x)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 17, 257])
def test_norm_fp8_outputs_keep_qdq_and_residual_bits(rows):
    from vllm.model_executor.layers.quantization.utils.sm80_group32_norm import (
        residual_norm_group32_qdq,
    )

    x = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16) * 0.02
    old = residual_norm_group32_qdq(x, residual, weight, 1e-6)
    new = residual_norm_group32_qdq(x, residual, weight, 1e-6, emit_fp8=True)
    for a, b in zip(old, new[:3]):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    decoded = dequant_fp8_group32(new[3].codes, new[3].scales)
    assert torch.equal(decoded.view(torch.int16), old[1].view(torch.int16))


@pytest.fixture
def group32_export_config():
    import json
    from pathlib import Path

    return json.loads(
        (Path(__file__).parent / "fixtures" / "sm80_group32_export.json").read_text()
    )


@pytest.mark.parametrize(
    "user_quant", [None, "sm80_qdq_reference", "compressed-tensors", "sm80_group32"]
)
@pytest.mark.parametrize("sm80", [True, False])
def test_group32_auto_preserves_explicit_quantization(
    group32_export_config, monkeypatch, user_quant, sm80
):
    from vllm.model_executor.layers.quantization import sm80_group32 as module

    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True, is_device_capability=lambda capability: sm80
        ),
    )
    config = group32_export_config
    hf = SimpleNamespace(to_dict=lambda: config.copy())
    result = module.SM80Group32Config.override_quantization_method(
        config["quantization_config"], user_quant, hf
    )
    expected = (
        "sm80_group32"
        if user_quant == "sm80_group32" or (user_quant is None and sm80)
        else None
    )
    assert result == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "token_input",
        "group128",
        "w4a16",
        "dense_bf16_scales",
        "qkv_dynamic",
        "wrong_targets",
        "ignore_all",
        "wrong_family",
    ],
)
def test_group32_auto_rejects_other_quantization_contracts(
    group32_export_config, mutation
):
    from vllm.model_executor.layers.quantization.utils.sm80_group32_config import (
        is_group32_export,
    )

    config = group32_export_config
    quant = config["quantization_config"]
    moe, dense = quant["config_groups"].values()
    if mutation == "token_input":
        moe["input_activations"].update(strategy="token", group_size=None)
    elif mutation == "group128":
        moe["weights"]["group_size"] = 128
    elif mutation == "w4a16":
        moe["input_activations"] = None
    elif mutation == "dense_bf16_scales":
        dense["weights"]["scale_dtype"] = "torch.bfloat16"
    elif mutation == "qkv_dynamic":
        quant["kv_cache_scheme"]["dynamic"] = True
    elif mutation == "wrong_targets":
        dense["targets"] = ["Linear"]
    elif mutation == "ignore_all":
        quant["ignore"] = ["re:.*"]
    else:
        config["model_type"] = "qwen3_next"
    assert not is_group32_export(quant, SimpleNamespace(to_dict=lambda: config.copy()))


@pytest.mark.parametrize("cache", ["fp8", "fp8_e4m3", "bfloat16", "auto"])
def test_group32_auto_attention_supports_implicit_and_per_kind_choice(
    monkeypatch, cache
):
    from vllm.model_executor.layers.quantization import sm80_group32 as module
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as Backend

    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True, is_device_capability=lambda capability: True
        ),
    )
    method = object.__new__(module.SM80Group32Config)
    method.kv_cache_scheme = {"type": "float", "num_bits": 8}
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache),
        attention_config=SimpleNamespace(backend=None, backend_per_kind=None),
    )
    method._check_attention_runtime(config)
    expected = (
        Backend.FLASH_ATTN
        if cache == "bfloat16"
        else Backend.FLASH_ATTN_QKV_FP8_SM80_FUSED
    )
    config.attention_config.backend = Backend.FLASHINFER
    config.attention_config.backend_per_kind = {"full_attention": expected}
    method._check_attention_runtime(config)
    config.attention_config.backend_per_kind = {"full_attention": Backend.FLASHINFER}
    with pytest.raises(ValueError, match="requires BF16 KV"):
        method._check_attention_runtime(config)


@pytest.mark.parametrize("cache", ["fp8_e4m3", "bfloat16", "auto"])
@pytest.mark.parametrize("explicit", [None, "FLASH_ATTN"])
def test_group32_auto_uses_actual_attention_selector(
    group32_export_config, monkeypatch, cache, explicit
):
    from vllm.config import (
        AttentionConfig,
        CacheConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.quantization.sm80_group32 import SM80Group32Config
    from vllm.v1.attention import selector
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as Backend

    quant = SM80Group32Config.from_config(group32_export_config["quantization_config"])
    monkeypatch.setattr(quant, "uses_sm80_fp8_qkv", lambda: True)
    monkeypatch.setattr(
        selector, "_cached_get_attn_backend", lambda **kw: kw["backend"]
    )
    config = VllmConfig(
        cache_config=CacheConfig(cache_dtype=cache),
        attention_config=AttentionConfig(backend=explicit),
    )
    config.quant_config = quant
    with set_current_vllm_config(config):
        selected = selector.get_attn_backend(256, torch.bfloat16, cache)
    expected = (
        Backend[explicit]
        if explicit
        else (Backend.FLASH_ATTN_QKV_FP8_SM80_FUSED if cache == "fp8_e4m3" else None)
    )
    assert selected == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 2, 3, 16, 129, 4096])
@pytest.mark.parametrize("three_d", [False, True])
@pytest.mark.parametrize("cuda_rope", [False, True])
def test_group32_qknorm_rope_preserves_actual_dispatch_bits(rows, three_d, cuda_rope):
    """Catch native reduction dispatch and distinct CUDA/native RoPE rounding."""
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.quantization.utils.sm80_group32_qknorm_rope import (
        qknorm_mrope_native_order,
    )
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=0, custom_ops=["all" if cuda_rope else "none"]
        )
    )
    torch.manual_seed(20260930)
    with set_current_vllm_config(config):
        qn = GemmaRMSNorm(256, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
        kn = GemmaRMSNorm(256, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            qn.weight.normal_(0, 0.02)
            kn.weight.normal_(0, 0.02)
        rope = MRotaryEmbedding(
            256,
            64,
            512,
            1000000.0,
            True,
            torch.bfloat16,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        ).cuda()
        x = torch.randn(rows * 2, 6144, device="cuda", dtype=torch.bfloat16)[::2]
        x[::17, ::11] *= 1e-7
        pos = torch.randint(0, 512, (3, rows) if three_d else (rows,), device="cuda")
        before = x.clone()
        actual = qknorm_mrope_native_order(
            x,
            qn.weight,
            kn.weight,
            pos,
            rope.cos_sin_cache,
            eps=1e-6,
            cuda_rope=cuda_rope,
        )
        if rows:
            qg, k, v = x.split((4096, 1024, 1024), -1)
            q, gate = qg.view(rows, 8, 512).chunk(2, -1)
            q = qn(q.reshape(rows, 2048).view(rows, 8, 256)).view(rows, 2048)
            k = kn(k.view(rows, 4, 256)).view(rows, 1024)
            q, k = rope(pos, q, k)
            for result, expected in zip(actual, (q, k, v, gate.reshape(rows, 2048))):
                assert torch.equal(
                    result.contiguous().view(torch.int16),
                    expected.contiguous().view(torch.int16),
                )
        else:
            assert [tuple(v.shape) for v in actual] == [
                (0, 2048),
                (0, 1024),
                (0, 1024),
                (0, 2048),
            ]
        assert torch.equal(x, before)
        assert actual[2].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cuda_rope", [False, True])
def test_group32_qknorm_rope_graph_reads_new_input_and_positions(cuda_rope):
    """Replay must consume changed data and positions, not a captured norm/cache."""
    from vllm.model_executor.layers.quantization.utils.sm80_group32_qknorm_rope import (
        qknorm_mrope_native_order,
    )

    x = torch.randn(3, 6144, device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(2)
    ]
    positions = torch.randint(0, 512, (3, 3), device="cuda")
    cache = torch.randn(512, 64, device="cuda", dtype=torch.bfloat16)

    def project():
        return qknorm_mrope_native_order(
            x, *weights, positions, cache, eps=1e-6, cuda_rope=cuda_rope
        )

    for _ in range(3):
        project()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = project()
    for _ in range(3):
        x.normal_()
        positions.random_(0, 512)
        cache.normal_()
        for weight in weights:
            weight.normal_(0, 0.02)
        graph.replay()
        for result, expected in zip(actual, project()):
            assert torch.equal(
                result.contiguous().view(torch.int16),
                expected.contiguous().view(torch.int16),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_group32_gdn_state_write_preserves_unselected_storage(
    dtype, transpose, index_dtype
):
    """Only selected rows change, even with padded strides and repeated indices."""
    from vllm.model_executor.layers.quantization.utils.sm80_group32_state import (
        store_group32_gdn_state,
    )

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        backing = torch.randn((7, 2 * 17 * 19 + 128), device="cuda", dtype=dtype)
        expected, actual = backing.clone(), backing.clone()
        reference = expected[:, : 2 * 17 * 19].view(7, 2, 17, 19)
        state = actual[:, : 2 * 17 * 19].view(7, 2, 17, 19)
        values = torch.randn((4, 2, 17, 19, 2), device="cuda", dtype=dtype)[..., 0]
        values[1, 0, 0, :3] = torch.tensor(
            [-0.0, float("nan"), float("inf")], device="cuda", dtype=dtype
        )
        if transpose:
            reference, state, values = (
                tensor.transpose(-1, -2) for tensor in (reference, state, values)
            )
        indices = torch.tensor([5, 0, -1, 5], device="cuda", dtype=index_dtype)
        reference[indices] = values
        store_group32_gdn_state(state, indices, values)
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_group32_gdn_state_write_matches_native_special_encodings(dtype):
    """Deterministic index_put canonicalizes BF16 NaNs; other codes stay exact."""
    from vllm.model_executor.layers.quantization.utils.sm80_group32_state import (
        store_group32_gdn_state,
    )

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        if dtype == torch.bfloat16:
            raw = torch.arange(65536, device="cuda").to(torch.int16)
        else:
            raw = torch.randint(
                -(2**31), 2**31 - 1, (65536,), device="cuda", dtype=torch.int32
            )
        values = raw.view(dtype).reshape(1, 1, 256, 256)
        expected = torch.zeros((3, 73728), device="cuda", dtype=dtype)
        actual = expected.clone()
        reference = expected[:, :65536].view(3, 1, 256, 256)
        state = actual[:, :65536].view(3, 1, 256, 256)
        indices = torch.tensor([1], device="cuda")
        reference[indices] = values
        store_group32_gdn_state(state, indices, values)
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_group32_gdn_state_graph_uses_current_rows_and_values(dtype):
    """Replayed writes follow new cache indices and preserve earlier other rows."""
    from vllm.model_executor.layers.quantization.utils.sm80_group32_state import (
        store_group32_gdn_state,
    )

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        expected = torch.zeros((7, 2 * 17 * 19 + 128), device="cuda", dtype=dtype)
        actual = expected.clone()
        reference = expected[:, : 2 * 17 * 19].view(7, 2, 17, 19)
        state = actual[:, : 2 * 17 * 19].view(7, 2, 17, 19)
        values = torch.zeros((3, 2, 17, 19), device="cuda", dtype=dtype)
        indices = torch.tensor([0, 1, 2], device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            store_group32_gdn_state(state, indices, values)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            store_group32_gdn_state(state, indices, values)
        for destinations in ([3, 1, -1], [2, 2, 4], [0, 5, 6]):
            indices.copy_(torch.tensor(destinations, device="cuda"))
            values.normal_()
            reference[indices] = values
            graph.replay()
            assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.bfloat16])
@pytest.mark.parametrize("rank", [0, 1])
def test_group32_staged_loading_preserves_expert_tp_slices(dtype, rank):
    """A staged CPU expert retains exact bytes, slot selection and loader return."""
    from vllm.model_executor.layers.quantization.sm80_group32 import (
        _load_group32_expert_weight,
    )

    raw = torch.randint(-32768, 32767, (128, 64), dtype=torch.int16)
    source = raw.view(torch.bfloat16) if dtype == torch.bfloat16 else raw.int()
    param = torch.zeros((3, 128, 32), dtype=dtype, device="cuda")

    def loader(**kw):
        assert kw["param"] is param and kw["shard_id"] == "w2"
        assert kw["loaded_weight"].device == param.device
        param[kw["expert_id"]].copy_(
            kw["loaded_weight"][:, rank * 32 : (rank + 1) * 32]
        )
        return True if kw["return_success"] else None

    for expert in (0, 2):
        result = _load_group32_expert_weight(
            param, source, "w2_weight", "w2", expert, bool(expert), loader=loader
        )
        assert result is (True if expert else None)
        expected = source[:, rank * 32 : (rank + 1) * 32].contiguous()
        assert torch.equal(
            param[expert].cpu().view(torch.uint8), expected.view(torch.uint8)
        )
    assert torch.count_nonzero(param[1]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case", ["w1", "w3", "shape", "strided", "gpu", "dtype", "size", "shard"]
)
def test_group32_staged_loading_preserves_unsupported_loader_inputs(case):
    """Uncovered layouts/dtypes retain the original input and error behavior."""
    from vllm.model_executor.layers.quantization.sm80_group32 import (
        _load_group32_expert_weight,
    )

    source = torch.ones((128, 64), dtype=torch.int32)
    param = torch.empty((3, 128, 32), device="cuda", dtype=torch.int32)
    shard = case if case in ("w1", "w3") else "w2"
    if case == "shape":
        source = torch.tensor([128, 64])
    elif case == "strided":
        source = torch.ones((128, 128), dtype=torch.int32)[:, ::2]
    elif case == "gpu":
        source = source.cuda()
    elif case == "dtype":
        source = source.float()
    elif case == "size":
        source = torch.ones((2048, 256), dtype=torch.int32)
        param = torch.empty((1, 2048, 128), device="cuda", dtype=torch.int32)
    elif case == "shard":
        param = torch.empty((3, 128, 16), device="cuda", dtype=torch.int32)
    marker = object()

    def loader(**kw):
        assert kw["loaded_weight"] is source
        assert kw["weight_name"] == "unchanged" and kw["expert_id"] == 2
        return marker

    assert (
        _load_group32_expert_weight(param, source, "unchanged", shard, 2, loader=loader)
        is marker
    )


@pytest.mark.parametrize(
    "method_name,tp_size", [("reference", 2), ("packed", 1), ("packed", 4)]
)
def test_group32_other_loading_paths_keep_the_original_callback(method_name, tp_size):
    """The optimization is scoped to packed group32 TP2, not its reference."""
    from vllm.model_executor.layers.quantization.sm80_group32 import PackedGroup32MoE

    config = SimpleNamespace(
        activation=MoEActivation.SILU,
        has_bias=False,
        swiglu_limit=None,
        moe_backend="auto",
    )
    method = (PackedGroup32MoE if method_name == "packed" else MoEGroup32QDQReference)(
        config
    )
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=tp_size)
    )

    def loader(*args, **kwargs):
        return None

    method.create_weights(layer, 1, 128, 128, torch.bfloat16, weight_loader=loader)
    assert all(param.weight_loader is loader for param in layer.parameters())


def _make_group32_marlin_test_layer(experts, hidden, intermediate, top_k):
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.quantization.sm80_group32 import PackedGroup32MoE

    config = FusedMoEConfig(
        num_experts=experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=experts,
        num_logical_experts=experts,
        activation=MoEActivation.SILU,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.Renormalize,
        max_num_tokens=8192,
    )
    method = PackedGroup32MoE(config)
    layer = torch.nn.Module()
    layer.moe_config = config
    layer.activation = MoEActivation.SILU
    layer.global_num_experts = experts
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer._expert_routing_tables = lambda: None
    for stem, n, k in [("w13", 2 * intermediate, hidden), ("w2", hidden, intermediate)]:
        packed = torch.empty(experts, n, k // 8, device="cuda", dtype=torch.int32)
        packed.random_(-(2**31), 2**31 - 1)
        scales = torch.empty(experts, n, k // 32, device="cuda")
        scales.uniform_(0.003, 0.02)
        layer.register_parameter(
            stem + "_weight_packed", torch.nn.Parameter(packed, False)
        )
        layer.register_parameter(
            stem + "_weight_scale", torch.nn.Parameter(scales.bfloat16(), False)
        )
        layer.register_buffer(
            stem + "_weight_shape", torch.empty(experts, 2, device="cuda")
        )
    method.process_weights_after_loading(layer)
    return method, layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 17, 128, 513, 2048, 2049])
@pytest.mark.parametrize("prepared", [False, True])
def test_group32_native_marlin_matches_materialized_qdq_and_post_fc2_weighting(
    rows, prepared, default_vllm_config, workspace_init, monkeypatch
):
    """Check real MK, independent QDQ, native FC2 weighting, and ops.moe_sum bits.

    Freeze only integer route metadata for the reference: CUDA route order can
    vary, changing Marlin partitions independently of the activation fusion.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    torch.manual_seed(20261013)
    e, h, i, top_k = 256, 2048, 256, 8
    method, layer = _make_group32_marlin_test_layer(e, h, i, top_k)
    experts = method.moe_kernel.fused_experts
    experts.input_prequantized = prepared
    actual_prepare = experts.prepare_inputs
    observed: list[torch.Tensor] = []

    def capture_preparation(*args):
        result = actual_prepare(*args)
        observed[:] = result
        return result

    monkeypatch.setattr(experts, "prepare_inputs", capture_preparation)
    x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16)
    if prepared:
        x = group32_fp8_qdq(x)
    probabilities = torch.rand(rows, top_k, device="cuda")
    ids = torch.randint(e, (rows, top_k), device="cuda", dtype=torch.int32)

    def activation_reference(activation, output, input, **kwargs):
        assert activation == MoEActivation.SILU
        output.copy_(group32_fp8_qdq(training_swiglu(input)))

    def compare(actual):
        reference_input = x if prepared else group32_fp8_qdq(x)
        if rows == 0:
            assert actual.shape == (0, h)
            return
        assert torch.equal(
            observed[0].view(torch.int16), reference_input.view(torch.int16)
        )
        slots = observed[1][observed[1] < ids.numel()].long()
        assert torch.equal(
            slots.sort().values, torch.arange(ids.numel(), device="cuda")
        )
        expected = torch.empty_like(x)
        fused_marlin_moe(
            hidden_states=reference_input,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            bias1=None,
            bias2=None,
            w1_scale=layer.w13_scale,
            w2_scale=layer.w2_scale,
            topk_weights=probabilities,
            topk_ids=ids,
            quant_type_id=scalar_types.uint4b8.id,
            apply_router_weight_on_input=False,
            global_num_experts=e,
            activation=MoEActivation.SILU,
            activation_func=activation_reference,
            input_preparation=lambda a, *_: (a, *observed[1:]),
            moe_sum=lambda a, out, *_: ops.moe_sum(a, out),
            is_k_full=True,
            output=expected,
        )
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    actual = method.apply(layer, x, probabilities, ids)
    compare(actual)
    if rows == 0:
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        method.apply(layer, x, probabilities, ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x, probabilities, ids)
    for _ in range(3):
        x.normal_()
        probabilities.uniform_()
        ids.random_(0, e)
        graph.replay()
        compare(captured)


@pytest.mark.parametrize("backend", ["marlin", "marlin_fp8_qdq_fused", "triton"])
def test_group32_native_marlin_rejects_backend_that_changes_activation_contract(
    backend,
):
    from vllm.model_executor.layers.quantization.sm80_group32 import PackedGroup32MoE

    with pytest.raises(ValueError, match="preserve activation QDQ"):
        PackedGroup32MoE(
            SimpleNamespace(
                activation=MoEActivation.SILU,
                has_bias=False,
                swiglu_limit=None,
                moe_backend=backend,
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 17, 256, 257, 2048])
@pytest.mark.parametrize("prepared", [False, True])
def test_shared_group32_mlp_preserves_gate_and_prepared_fp8(
    default_vllm_config, monkeypatch, rows, prepared
):
    """Prepared FP8 and raw inputs agree; the shared gate still reads raw BF16."""
    from vllm.model_executor.layers.quantization.sm80_group32 import DenseGroup32Fused
    from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
        quantize_input,
    )
    from vllm.model_executor.models import qwen2_moe

    class Projection(torch.nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            output_size = (
                sum(output_size) if isinstance(output_size, list) else output_size
            )
            layer, reference = _make_dense_fp8_test_layer(output_size, input_size)
            self.weight = layer.weight
            self.weight_scale = layer.weight_scale
            self.register_buffer("reference_weight", reference)
            self.scheme = DenseGroup32Fused()
            self.scheme.process_weights_after_loading(self)

        def forward(self, x):
            return self.scheme.apply_weights(self, x), None

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16) / 32

        def forward(self, x):
            return F.linear(x, self.weight), None

    monkeypatch.setattr(qwen2_moe, "MergedColumnParallelLinear", Projection)
    monkeypatch.setattr(qwen2_moe, "RowParallelLinear", Projection)
    gate = Gate()
    mlp = qwen2_moe.Qwen2MoeMLP(
        2048,
        256,
        "silu",
        SimpleNamespace(get_name=lambda: "sm80_group32"),
        expert_gate=gate,
    )
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)[:, ::2]
    if rows:
        x[0, :32] = -0.0
        x[0, 32:64] *= torch.finfo(torch.bfloat16).tiny
    # Use the same GEMM for both paths: this test isolates fusion semantics
    # from the GEMM's documented accumulation error against BF16 cuBLAS.
    gate_up = mlp.gate_up_proj(quantize_input(x))[0]
    expected = mlp.down_proj(quantize_input(training_swiglu(gate_up)))[0]
    expected = expected * F.sigmoid(F.linear(x, gate.weight))
    actual = mlp(x, prepared_input=quantize_input(x) if prepared else None)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    # The same instance must also support a later call without a prepared tensor.
    if prepared:
        separate = mlp(x)
        assert torch.equal(actual.view(torch.int16), separate.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_group32_fused_activation_graph_uses_new_inputs():
    """Shared expert shapes preserve bits across Graph replay, including zeros."""
    from vllm.model_executor.layers.quantization.utils.sm80_group32_qdq import (
        silu_mul_group32_fp8_qdq_fused,
    )

    x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            silu_mul_group32_fp8_qdq_fused(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = silu_mul_group32_fp8_qdq_fused(x)
    for i in range(8):
        x.normal_()
        if i == 7:
            x.zero_()
        expected = group32_fp8_qdq(training_swiglu(x))
        graph.replay()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 17, 256, 2048])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_group32_masked_combine_preserves_native_sum_bits(rows, strided, ids_dtype):
    """Preserve native sum order and ignore poisoned invalid GEMM slots."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.experts.marlin_group32_fp8_qdq import (
        MarlinGroup32Fp8QdqExperts,
    )

    expert = SimpleNamespace(moe_config=SimpleNamespace(num_experts=128))
    x = torch.randn(
        rows, 8, 4096 if strided else 2048, device="cuda", dtype=torch.bfloat16
    )
    if strided:
        x = x[..., ::2]
    ids = torch.randint(
        -3, 132, (rows, 16 if strided else 8), device="cuda", dtype=ids_dtype
    )
    if strided:
        ids = ids[:, ::2]
    output = torch.empty(rows, 2048, device="cuda", dtype=x.dtype)
    expected = torch.empty_like(output)
    for all_invalid in (False, True):
        if all_invalid:
            ids.fill_(128)
        valid = (ids >= 0) & (ids < 128)
        x[~valid] = float("nan")
        if rows:
            ops.moe_sum(x, expected, torch.where(valid, ids, -1).contiguous())
        MarlinGroup32Fp8QdqExperts.moe_sum(expert, x, output, ids, None)
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_group32_masked_combine_graph_refreshes_invalid_routes():
    """A replay must not reuse prior valid IDs or poisoned FC2 workspace."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.experts.marlin_group32_fp8_qdq import (
        MarlinGroup32Fp8QdqExperts,
    )

    expert = SimpleNamespace(moe_config=SimpleNamespace(num_experts=128))
    x = torch.randn(17, 8, 2048, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 128, (17, 8), device="cuda", dtype=torch.int64)
    output = torch.empty(17, 2048, device="cuda", dtype=x.dtype)
    expected = torch.empty_like(output)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            MarlinGroup32Fp8QdqExperts.moe_sum(expert, x, output, ids, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        MarlinGroup32Fp8QdqExperts.moe_sum(expert, x, output, ids, None)
    for turn in range(8):
        x.normal_()
        ids.random_(-3, 132)
        if turn == 7:
            ids.fill_(-1)
        valid = (ids >= 0) & (ids < 128)
        x[~valid] = float("nan")
        ops.moe_sum(x, expected, torch.where(valid, ids, -1))
        graph.replay()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_fp8_rejects_bias_and_invalid_scales_during_loading(default_vllm_config):
    """Invalid static Dense contracts fail before the first forward call."""
    from vllm.model_executor.layers.quantization.sm80_group32 import DenseGroup32Fused

    layer, _ = _make_dense_fp8_test_layer(128, 128)
    layer.bias = torch.nn.Parameter(torch.zeros(128, device="cuda"))
    with pytest.raises(ValueError, match="bias-free"):
        DenseGroup32Fused().process_weights_after_loading(layer)
    layer.bias = None
    layer.weight_scale.data[0, 0] = 0
    with pytest.raises(ValueError, match="positive finite"):
        DenseGroup32Fused().process_weights_after_loading(layer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape",
    [
        (8, 6144, 2048),
        (16, 6144, 2048),
        (16, 2048, 2048),
        (16, 512, 2048),
        (16, 2048, 256),
        (16, 32, 2048),
        (17, 6144, 2048),
        (129, 2048, 2048),
        (256, 32, 2048),
        (1024, 1024, 4096),
    ],
)
def test_dense_fp8_graph_refresh_and_concurrent_workspaces(shape, default_vllm_config):
    """Split partials/locks are private to each call and read fresh FP8 inputs."""
    from vllm.model_executor.layers.quantization.sm80_group32 import DenseGroup32Fused
    from vllm.model_executor.layers.quantization.utils.sm80_dense_fp8 import (
        quantize_input,
    )

    m, n, k = shape
    layer, _ = _make_dense_fp8_test_layer(n, k)
    method = DenseGroup32Fused()
    method.process_weights_after_loading(layer)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    prepared = quantize_input(x)
    expected = method.apply_weights(layer, prepared)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply_weights(layer, prepared)
    for _ in range(3):
        new = quantize_input(torch.randn_like(x))
        prepared.codes.copy_(new.codes)
        prepared.scales.copy_(new.scales)
        expected = method.apply_weights(layer, prepared)
        graph.replay()
        assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
    inputs = [quantize_input(torch.randn_like(x)) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    outputs = []
    for stream, value in zip(streams, inputs):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs.append(method.apply_weights(layer, value))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for value, output in zip(inputs, outputs):
        expected = method.apply_weights(layer, value)
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
