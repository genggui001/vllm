# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_fp8_qdq_fused_moe import (
    fp8_e4m3_per_token_qdq_fused,
    silu_mul_fp8_e4m3_per_token_qdq_fused,
)


def fp8_e4m3_per_token_qdq(x: torch.Tensor) -> torch.Tensor:
    """Independent materialized QAT formula, used only as a test oracle."""
    scales = x.abs().amax(dim=1, keepdim=True).float() / 448.0
    scales = scales.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (x / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (quantized.float() * scales).to(x.dtype)


def _require_sm80() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("This backend targets SM80")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 512), (37, 2560), (128, 896)])
def test_fused_per_token_qdq_matches_reference(dtype: torch.dtype, shape):
    _require_sm80()
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    row_scales = torch.logspace(-4, 3, shape[0], device="cuda").unsqueeze(1)
    x = (x.float() * row_scales).to(dtype)
    x[0].zero_()

    expected = fp8_e4m3_per_token_qdq(x)
    actual = fp8_e4m3_per_token_qdq_fused(x)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 127, 896, 2560, 8192, 32768, 65536])
def test_fused_qdq_extreme_widths_and_zero_rows(dtype, width):
    _require_sm80()
    torch.manual_seed(718)
    x = torch.randn((3, width), device="cuda", dtype=dtype)
    x[0].zero_()
    expected = fp8_e4m3_per_token_qdq(x)
    actual = fp8_e4m3_per_token_qdq_fused(x)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    empty = x[:0]
    assert fp8_e4m3_per_token_qdq_fused(empty).shape == (0, width)
    gate = torch.randn((3, width * 2), device="cuda", dtype=dtype)
    activation = torch.empty_like(x)
    torch.ops._C.silu_and_mul(activation, gate)
    silu_mul_fp8_e4m3_per_token_qdq_fused(gate, actual)
    assert torch.equal(
        actual.view(torch.int16), fp8_e4m3_per_token_qdq(activation).view(torch.int16)
    )
    assert silu_mul_fp8_e4m3_per_token_qdq_fused(gate[:0], actual[:0]).shape == (
        0,
        width,
    )


def test_fused_qdq_rejects_cpu_and_invalid_width():
    with pytest.raises(ValueError, match="CUDA"):
        fp8_e4m3_per_token_qdq_fused(torch.ones((1, 8), dtype=torch.bfloat16))
    _require_sm80()
    for width in (0, 65537):
        with pytest.raises(ValueError):
            fp8_e4m3_per_token_qdq_fused(
                torch.empty((1, width), device="cuda", dtype=torch.bfloat16)
            )


def test_fused_qdq_rejects_output_on_another_device():
    _require_sm80()
    x = torch.ones((1, 8), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same CUDA device"):
        fp8_e4m3_per_token_qdq_fused(x, torch.empty_like(x, device="cpu"))
    with pytest.raises(ValueError, match="same CUDA device"):
        silu_mul_fp8_e4m3_per_token_qdq_fused(
            torch.cat([x, x], dim=1), torch.empty_like(x, device="cpu")
        )


@pytest.mark.parametrize("width", [896, 2560])
def test_fused_qdq_graph_replay_matches_fresh_reference(width):
    _require_sm80()
    torch.manual_seed(719)
    x = torch.randn((257, width * 2), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((257, width), device="cuda", dtype=torch.bfloat16)
    activation = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        silu_mul_fp8_e4m3_per_token_qdq_fused(x, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        silu_mul_fp8_e4m3_per_token_qdq_fused(x, out)
    for _ in range(100):
        x.normal_()
        torch.ops._C.silu_and_mul(activation, x)
        expected = fp8_e4m3_per_token_qdq(activation)
        graph.replay()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_qdq_all_finite_input_codes(dtype: torch.dtype) -> None:
    _require_sm80()
    values = torch.arange(65536, device="cuda", dtype=torch.int32)
    values = values.to(torch.int16).view(dtype).float()
    values = values[torch.isfinite(values)].clamp(-448.0, 448.0).to(dtype)
    # A 448 sentinel fixes each row's dynamic scale to one. Exhaustive input
    # codes then exercise E4M3 subnormals, ties, and exponent transitions.
    padding = (-values.numel()) % 511
    values = torch.cat([values, values.new_zeros(padding)]).reshape(-1, 511)
    x = torch.cat([values, values.new_full((values.shape[0], 1), 448.0)], dim=1)

    expected = fp8_e4m3_per_token_qdq(x)
    actual = fp8_e4m3_per_token_qdq_fused(x)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 512), (37, 2560), (128, 896)])
def test_fused_swiglu_qdq_matches_reference(dtype: torch.dtype, shape):
    _require_sm80()
    torch.manual_seed(1)
    rows, n_cols = shape
    x = torch.randn((rows, n_cols * 2), device="cuda", dtype=dtype)
    expected_activation = torch.empty(shape, device="cuda", dtype=dtype)
    torch.ops._C.silu_and_mul(expected_activation, x)
    expected = fp8_e4m3_per_token_qdq(expected_activation)

    actual = torch.empty_like(expected)
    silu_mul_fp8_e4m3_per_token_qdq_fused(x, actual)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_qdq_preserves_negative_zero(dtype):
    _require_sm80()
    x = torch.tensor([[-0.0, 0.0, 448.0, -448.0]], device="cuda", dtype=dtype)
    actual = fp8_e4m3_per_token_qdq_fused(x)
    expected = fp8_e4m3_per_token_qdq(x)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
