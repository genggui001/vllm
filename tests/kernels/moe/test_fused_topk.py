# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE fused topk kernel

Run `pytest tests/kernels/moe/test_fused_topk.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform


def torch_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.float(), dim=-1)
    else:
        assert scoring_func == "sigmoid"
        scores = torch.sigmoid(gating_output.float())

    if e_score_correction_bias is not None:
        num_experts = gating_output.shape[-1]
        scores_for_choice = scores.view(
            -1, num_experts
        ) + e_score_correction_bias.unsqueeze(0)
        _, topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_weights = scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_tokens", [1, 33, 56])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [6, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    topk_weights_ref, topk_ids_ref = torch_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    torch.testing.assert_close(
        topk_weights_ref.to(torch.float32), topk_weights, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(topk_ids_ref.to(torch.int32), topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_tokens", [1, 33, 56])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [6, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_bias(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = torch_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        scoring_func=scoring_func,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    torch.testing.assert_close(
        topk_weights_ref.to(torch.float32), topk_weights, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(topk_ids_ref.to(torch.int32), topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [6, 8, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_nan_inf_clamp(
    num_experts: int,
    topk: int,
    scoring_func: str,
    bad_value: float,
    dtype: torch.dtype,
):
    """Regression test for the NaN/Inf clamp in topk_softmax_kernels.cu.

    Degenerate hidden states (e.g., from CUDA graph padding) can produce
    NaN/Inf gating logits. Without the clamp, softmax/sigmoid outputs are
    NaN and the argmax loop picks expert 0 for every top-k slot (since
    "NaN > NaN" is false per IEEE 754), yielding duplicate expert IDs that
    crash downstream MoE sort kernels. The fix clamps NaN/Inf to 0 before
    argmax so index tie-breaking selects unique experts [0, 1, ..., k-1].
    """
    torch.manual_seed(0)
    num_tokens = 4
    hidden_size = 1024
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")

    # Row 0: all normal. Rows 1-3: fully poisoned with NaN or Inf.
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    gating_output[1:, :] = bad_value

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=False,
        scoring_func=scoring_func,
    )

    # Normal row must still match the torch reference.
    ref_weights, ref_ids = torch_topk(
        gating_output=gating_output[:1],
        topk=topk,
        renormalize=False,
        scoring_func=scoring_func,
    )
    torch.testing.assert_close(
        ref_weights.to(torch.float32), topk_weights[:1], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(ref_ids.to(torch.int32), topk_ids[:1], atol=0, rtol=0)

    # Poisoned rows: IDs must be unique (no duplicates) and weights must be
    # finite (no NaN/Inf propagation into downstream MoE kernels).
    for row in range(1, num_tokens):
        row_ids = topk_ids[row]
        assert row_ids.unique().numel() == topk, (
            f"Row {row} has duplicate expert IDs {row_ids.tolist()} "
            f"(bad_value={bad_value}, scoring_func={scoring_func})"
        )
        assert torch.isfinite(topk_weights[row]).all(), (
            f"Row {row} has non-finite weights {topk_weights[row].tolist()} "
            f"(bad_value={bad_value}, scoring_func={scoring_func})"
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [6, 8, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_bias_nan_inf_clamp(
    num_experts: int,
    topk: int,
    scoring_func: str,
    bad_value: float,
    dtype: torch.dtype,
):
    """Regression test: NaN/Inf in gating logits must not produce duplicate
    expert IDs or non-finite weights when e_score_correction_bias is present.

    Same scenario as test_fused_topk_nan_inf_clamp but exercising the bias
    path (fused_topk_bias) so the fix in topk_softmax_kernels.cu is covered
    for that entry point as well.
    """
    torch.manual_seed(0)
    num_tokens = 4
    hidden_size = 1024
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    gating_output[1:, :] = bad_value

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=False,
        scoring_func=scoring_func,
    )

    # Normal row must still match the torch reference.
    ref_weights, ref_ids = torch_topk(
        gating_output=gating_output[:1],
        topk=topk,
        renormalize=False,
        e_score_correction_bias=e_score_correction_bias,
        scoring_func=scoring_func,
    )
    torch.testing.assert_close(
        ref_weights.to(torch.float32), topk_weights[:1], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(ref_ids.to(torch.int32), topk_ids[:1], atol=0, rtol=0)

    # Poisoned rows: IDs must be unique (no duplicates) and weights must be
    # finite (no NaN/Inf propagation into downstream MoE kernels).
    for row in range(1, num_tokens):
        row_ids = topk_ids[row]
        assert row_ids.unique().numel() == topk, (
            f"Row {row} has duplicate expert IDs {row_ids.tolist()} "
            f"(bad_value={bad_value}, scoring_func={scoring_func})"
        )
        assert torch.isfinite(topk_weights[row]).all(), (
            f"Row {row} has non-finite weights {topk_weights[row].tolist()} "
            f"(bad_value={bad_value}, scoring_func={scoring_func})"
        )


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or "H20" not in current_platform.get_device_name().split(),
    reason="Requires the native H20 top-k kernel.",
)
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 33, 128, 256])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("renormalize", [False, True])
@torch.inference_mode()
def test_h20_topk_padding_graph_replay(
    monkeypatch, num_tokens: int, index_dtype: torch.dtype, renormalize: bool
):
    """V2 padding must retain reference weights and -1 IDs after graph replay."""
    from vllm import _custom_ops as ops

    if not hasattr(torch.ops.h20_topk, "run"):
        pytest.skip("H20 extension is not built")
    monkeypatch.setattr(ops, "_H20_NATIVE_TOPK_ENABLED", True)
    torch.manual_seed(7)
    scores = torch.randn(num_tokens, 256, device="cuda", dtype=torch.bfloat16)
    padding = torch.zeros(num_tokens, device="cuda", dtype=torch.bool)

    def outputs():
        return (
            torch.empty(num_tokens, 8, device="cuda", dtype=torch.float32),
            torch.empty(num_tokens, 8, device="cuda", dtype=index_dtype),
            torch.empty(num_tokens, 8, device="cuda", dtype=torch.int32),
        )

    actual, expected = outputs(), outputs()

    def run():
        ops.topk_softmax(*actual, scores, renormalize, is_padding=padding)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for mode in ("all", "alternating", "none"):
        scores.normal_()
        padding.fill_(mode == "all")
        if mode == "alternating":
            padding[::2] = True
        torch.ops._moe_C.topk_softmax(*expected, scores, renormalize, None, padding)
        graph.replay()
        torch.accelerator.synchronize()
        for reference, result in zip(expected, actual):
            assert torch.equal(reference.view(torch.uint8), result.view(torch.uint8))
        assert (actual[1][padding] == -1).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("num_tokens", [129, 257, 512, 8192])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_h20_fallback_padding_ids_graph_replay(num_tokens, index_dtype):
    """Large-batch padding sorts after valid experts without mutating routing."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.h20_fp8_prepare import (
        normalize_padding_ids,
    )

    ids = (
        torch.arange(num_tokens * 8, device="cuda", dtype=index_dtype).view(-1, 8) % 256
    )
    scratch = SimpleNamespace(
        topk_ids_for_sort=torch.empty_like(ids, dtype=torch.int32).flatten()
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        normalize_padding_ids(ids, scratch)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = normalize_padding_ids(ids, scratch)
    for mode in ("all", "alternating", "none"):
        ids.copy_(torch.arange(ids.numel(), device="cuda").view_as(ids) % 256)
        if mode == "all":
            ids.fill_(-1)
        elif mode == "alternating":
            ids[::2] = -1
            ids[1::2, ::2] = 256
        before = ids.clone()
        expected = torch.where(ids < 0, 256, ids).to(torch.int32)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(result, expected)
        assert torch.equal(ids, before)
