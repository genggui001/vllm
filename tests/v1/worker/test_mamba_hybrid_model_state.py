# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted


@pytest.mark.parametrize(
    "case",
    [
        "enabled",
        "prefill_cap",
        "computed_cap",
        "no_history",
        "no_prefix",
        "capture",
        "full_graph",
        "piecewise",
        "disabled",
        "prefix_disabled",
        "reference",
        "legacy",
        "dcp",
        "pcp",
        "alibi",
        "sliding",
        "local",
        "native_backend",
    ],
)
def test_group32_cascade_uses_only_confirmed_shared_history(monkeypatch, case):
    """Avoid unmasked speculative suffixes and keep old/Graph paths untouched."""
    import numpy as np

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    from vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused import (
        FlashAttentionQkvFp8Sm80FusedMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.gpu.model_states import mamba_hybrid

    monkeypatch.setattr(mamba_hybrid, "num_compute_units", lambda _: 108)
    state = object.__new__(MambaHybridModelState)
    state.device = torch.device("cpu")
    state.model_config = SimpleNamespace(
        quantization={
            "reference": "sm80_qdq_reference",
            "legacy": "compressed-tensors",
        }.get(case, "sm80_group32"),
        disable_cascade_attn=case == "disabled",
        uses_alibi=case == "alibi",
    )
    state.cache_config = SimpleNamespace(
        enable_prefix_caching=case != "prefix_disabled"
    )
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2 if case == "pcp" else 1,
            decode_context_parallel_size=2 if case == "dcp" else 1,
        )
    )
    cls = (
        FlashAttentionMetadataBuilder
        if case == "native_backend"
        else FlashAttentionQkvFp8Sm80FusedMetadataBuilder
    )
    builder = object.__new__(cls)
    builder.num_heads_q, builder.dcp_world_size, builder.fa4_hd256 = 8, 1, False
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        sliding_window=4096 if case == "sliding" else None,
        attention_chunk_size=4096 if case == "local" else None,
    )
    group = SimpleNamespace(kv_cache_spec=spec, get_metadata_builder=lambda _: builder)
    batch = SimpleNamespace(
        num_reqs=16,
        num_common_prefix_blocks=None if case == "no_prefix" else [256],
        num_computed_tokens_np=np.full(16, 3071 if case == "computed_cap" else 5000),
        num_computed_prefill_tokens_np=np.full(
            16, 0 if case == "no_history" else (3071 if case == "prefill_cap" else 4096)
        ),
        num_scheduled_tokens=np.full(16, 4),
    )
    mode = {"full_graph": CUDAGraphMode.FULL, "piecewise": CUDAGraphMode.PIECEWISE}.get(
        case, CUDAGraphMode.NONE
    )
    actual = state._group32_cascade_prefixes(batch, mode, [[group]], case == "capture")
    expected = (
        {id(builder): 3056 if case != "enabled" else 4096}
        if case in ("enabled", "prefill_cap", "computed_cap")
        else None
    )
    assert actual == expected
    metadata = mamba_hybrid.MambaHybridAttnMetadata(
        is_prefilling=torch.zeros(16, dtype=torch.bool), common_prefix_lens=actual
    )
    assert metadata.get_extra_attn_kwargs(builder, 16) == (
        {"common_prefix_len": expected[id(builder)]} if expected else {}
    )
