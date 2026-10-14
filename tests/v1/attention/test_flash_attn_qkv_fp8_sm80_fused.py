# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused import (
    FlashAttentionQkvFp8Sm80FusedBackend,
    FlashAttentionQkvFp8Sm80FusedMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
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

    builder = object.__new__(FlashAttentionQkvFp8Sm80FusedMetadataBuilder)
    builder.fa4_hd256 = False
    cascade_args = {
        "query_lens": np.ones(64, dtype=np.int32),
        "num_query_heads": 8,
        "num_kv_heads": 4,
        "use_alibi": False,
        "use_sliding_window": False,
        "use_local_attention": False,
        "num_sms": 108,
        "dcp_world_size": 1,
    }
    assert builder.use_cascade_attention(common_prefix_len=1024, **cascade_args)
    assert not builder.use_cascade_attention(common_prefix_len=128, **cascade_args)


@pytest.mark.parametrize(
    "tokens,context,sequences,expected_budget",
    [(16, 64, 2, 0), (32, 64, 2, 656640), (128, 1024, 4, 4 * 1024**2)],
)
def test_group32_prefill_reserves_shared_capacity_before_kv_sizing(
    monkeypatch, tokens, context, sequences, expected_budget
):
    """Ten attention layers must share one reservation per execution slot."""
    import vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused as backend
    import vllm.v1.worker.workspace as workspace

    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=context),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=tokens, max_num_seqs=sequences
        ),
    )
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(backend, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(backend, "_GROUP32_PREFILL_WORKSPACE_LIMIT", 4 * 1024**2)
    pointers = None
    for _ in range(10):
        impl = object.__new__(backend.FlashAttentionQkvFp8Sm80FusedImpl)
        impl.num_heads, impl.num_kv_heads, impl.head_size = 8, 4, 256
        impl._group32_prefill_budget = 0
        impl.prepare_group32_prefill()
        assert impl._group32_prefill_budget == expected_budget
        current = tuple(
            None if ws is None else (ws.data_ptr(), ws.numel())
            for ws in manager._current_workspaces
        )
        if pointers is not None:
            assert current == pointers
        pointers = current
    assert manager._current_workspaces[1::2] == [None, None]
    allocated = sum(ws.numel() for ws in manager._current_workspaces if ws is not None)
    assert allocated == 2 * expected_budget


@pytest.mark.parametrize(
    "total,batch,max_query,max_kv,budget,expected",
    [
        (128, 1, 128, 128, 1536 * 1024**2, False),
        (511, 1, 511, 511, 1536 * 1024**2, False),
        (513, 1, 513, 513, 1536 * 1024**2, False),
        (8192, 1, 8192, 49152, 1536 * 1024**2, False),
        (8192, 1, 8192, 57344, 1536 * 1024**2, True),
        (8192, 4, 8192, 32768, 1536 * 1024**2, True),
        (8192, 1, 8192, 262080, 1536 * 1024**2, True),
        (8192, 16, 8192, 262080, 1536 * 1024**2, False),
        (64, 1, 16, 131072, 1536 * 1024**2, False),
        (64, 1, 17, 131072, 1536 * 1024**2, True),
        (8192, 1, 8192, 57344, 256 * 1024**2, False),
        (8192, 1, 8192, 57344, 268711935, False),
        (8192, 1, 8192, 57344, 268711936, True),
    ],
)
def test_group32_prefill_expands_only_between_staging_limits(
    total, batch, max_query, max_kv, budget, expected
):
    """Preserve the small path and direct fallback; cover exact capacity edges."""
    from vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused import (
        FlashAttentionQkvFp8Sm80FusedImpl,
    )

    impl = object.__new__(FlashAttentionQkvFp8Sm80FusedImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 8, 4, 256
    impl._group32_prefill_budget = budget
    assert (
        impl._needs_expanded_group32_prefill(total, batch, max_query, max_kv)
        is expected
    )


@pytest.mark.parametrize(
    "max_kv,group32,prequantized,expected",
    [
        (128, True, False, "original"),
        (131072, True, False, "workspace"),
        (524288, True, False, "original"),
        (131072, False, False, "original"),
        (131072, False, True, "prequantized"),
        (131072, True, True, "prequantized"),
    ],
)
def test_group32_prefill_dispatch_preserves_original_short_and_legacy_calls(
    monkeypatch, max_kv, group32, prequantized, expected
):
    """Only eligible expanded prefill may request a workspace or its operator."""
    import vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused as backend

    impl = object.__new__(backend.FlashAttentionQkvFp8Sm80FusedImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 8, 4, 256
    impl.scale = 0.0625
    impl._group32_prefill_budget = 1536 * 1024**2
    calls = []
    scratch = torch.empty(1, dtype=torch.uint8)

    def get_workspace(*args):
        calls.append("get_workspace")
        return [scratch]

    monkeypatch.setattr(
        backend,
        "current_workspace_manager",
        lambda: SimpleNamespace(get_simultaneous=get_workspace),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ops = torch.ops._vllm_fa2_sm80_fp8_C
    for name, label in [
        ("varlen_fwd", "original"),
        ("varlen_fwd_lse", "original_lse"),
        ("varlen_fwd_prequantized_q", "prequantized"),
        ("varlen_fwd_group32_prefill_workspace_lse", "workspace"),
    ]:
        monkeypatch.setattr(ops, name, lambda *args, label=label: calls.append(label))
    query = torch.empty((128, 8, 256), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    cache = torch.empty((1, 4, 16, 512), dtype=torch.uint8)
    scale = torch.ones(1, dtype=torch.float32)
    layer = SimpleNamespace(
        _q_scale=scale,
        _k_scale=scale,
        _v_scale=scale,
        _sm80_group32_decode=group32,
        _qrope_prequantized_q=prequantized,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=128,
        use_cascade=False,
        max_query_len=128,
        max_seq_len=max_kv,
        causal=True,
        query_start_loc=torch.tensor([0, 128], dtype=torch.int32),
        seq_lens=torch.tensor([max_kv], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    actual = impl.forward(layer, query, None, None, cache, metadata, output)
    assert actual is output
    assert calls == (
        ["get_workspace", "workspace"] if expected == "workspace" else [expected]
    )


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
def test_group32_prefill_runtime_reuses_workspace_and_preserves_lse(monkeypatch):
    """The backend must use the reserved slot and preserve cascade intermediates."""
    import vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused as backend
    import vllm.v1.worker.workspace as workspace

    manager = workspace.WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: manager)
    impl = object.__new__(backend.FlashAttentionQkvFp8Sm80FusedImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 8, 4, 256
    impl._group32_prefill_budget = 8 * 1024**2
    monkeypatch.setattr(backend, "_GROUP32_PREFILL_DEFAULT_BUDGET", 0)
    manager.reserve_for_all_ubatches(impl._group32_prefill_budget)
    manager.lock()
    args = _small_op_inputs((64, 65), (257, 513))
    output, lse = impl._forward_group32_prefill(*args)
    saved_output, saved_lse = output.clone(), lse.clone()
    other = _small_op_inputs((128,), (2049,))
    impl._forward_group32_prefill(*other)
    impl._forward_group32_prefill(*_small_op_inputs((1,), (17,)))
    assert torch.equal(output.view(torch.int16), saved_output.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), saved_lse.view(torch.int32))
    reference = list(args)
    reference[3] = torch.empty_like(output)
    expected, expected_lse = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_lse(*reference)
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), expected_lse.view(torch.int32))


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
@pytest.mark.parametrize("cache_padding", [0, 8, -1])
@pytest.mark.parametrize(
    ("query_lens", "kv_lens", "causal"),
    [
        ([1, 1], [17, 31], True),  # zero-copy decode GQA packing
        ([1, 1], [127, 512], True),  # multiple async tiles and a partial page
        ([1, 1], [1025, 2048], True),  # long decode with repeated buffer reuse
        pytest.param([1] * 63, [257] * 63, True, id="decode-grid-below"),
        pytest.param([1] * 65, [257] * 65, True, id="decode-grid-above"),
        pytest.param([17], [8193], True, id="prefill-staging-first-row"),
        pytest.param([18], [28690], True, id="prefill-long-tail"),
        ([1] * 64, [129 + 3 * i for i in range(64)], True),  # compact grid
        ([16, 16], [511, 2048], True),  # all rows of the small-query tile
        ([1, 3], [17, 31], True),
        ([7, 19], [33, 61], True),
        ([5, 11], [23, 45], False),
        pytest.param([63, 64], [2049, 4097], True, id="prefill-edge"),
        pytest.param([64], [8192], True, id="prefill-context-boundary-old"),
        pytest.param([64], [8193], True, id="prefill-context-boundary-staged"),
        pytest.param([2048], [16385], True, id="prefill-chunk"),
        pytest.param([128] * 8, [2049] * 8, False, id="prefill-batch"),
        pytest.param([64] * 32, [8193] * 32, True, id="prefill-budget-fallback"),
    ],
)
def test_cuda_op_matches_materialized_fa2_qdq(
    query_lens: list[int],
    kv_lens: list[int],
    causal: bool,
    cache_padding: int,
    num_query_heads: int = 8,
    num_kv_heads: int = 4,
    group32_decode: bool = False,
    num_splits: int = 0,
    max_length_bound: int | None = None,
    stage_budget_bytes: int | None = None,
) -> None:
    importlib.import_module("vllm.vllm_flash_attn._vllm_fa2_C")
    importlib.import_module("vllm.vllm_flash_attn._vllm_fa2_sm80_fp8_C")

    torch.manual_seed(1234 + int(causal))
    device = torch.device("cuda")
    head_size, page_size = 256, 16
    batch_size = len(query_lens)
    max_length = max(kv_lens) if max_length_bound is None else max_length_bound
    pages_per_sequence = math.ceil(max_length / page_size)
    num_pages = batch_size * pages_per_sequence
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        batch_size, pages_per_sequence
    )
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
    if cache_padding == -1:
        # Production cache packs K/V together with the head before the page row.
        packed = key_bytes.new_empty(
            (num_pages, num_kv_heads, page_size, head_size * 2)
        )
        packed_key, packed_value = packed.transpose(1, 2).split(head_size, dim=-1)
        packed_key.copy_(key_bytes)
        packed_value.copy_(value_bytes)
        key_bytes, value_bytes = packed_key, packed_value
    elif cache_padding:
        # Eight-byte-aligned views remain valid for the original loader, but
        # cannot use the sixteen-byte asynchronous copy path.
        key_storage = key_bytes.new_empty((*key_bytes.shape[:-1], head_size + 8))
        value_storage = torch.empty_like(key_storage)
        key_storage[..., 8:].copy_(key_bytes)
        value_storage[..., 8:].copy_(value_bytes)
        key_bytes = key_storage[..., 8:]
        value_bytes = value_storage[..., 8:]
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

    workspace = None
    if stage_budget_bytes is not None:
        workspace = torch.empty(stage_budget_bytes, dtype=torch.uint8, device=device)
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    allocated = torch.accelerator.memory_allocated()
    ops = torch.ops._vllm_fa2_sm80_fp8_C
    fwd = ops.varlen_fwd_group32_lse if group32_decode else ops.varlen_fwd_lse
    extra: dict[str, int | torch.Tensor] = (
        {"num_splits": num_splits} if group32_decode else {}
    )
    if stage_budget_bytes is not None:
        assert not group32_decode
        assert workspace is not None
        fwd = ops.varlen_fwd_group32_prefill_workspace_lse
        extra = {"stage_budget_bytes": stage_budget_bytes, "workspace": workspace}
    _, output_lse = fwd(
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
        max_length,
        softmax_scale,
        causal,
        **extra,
    )
    torch.accelerator.synchronize()
    temporary_bytes = torch.accelerator.max_memory_allocated() - allocated
    workspace_limit = 256 * 1024 * 1024
    if stage_budget_bytes is not None:
        lse_bytes = sum(query_lens) * num_query_heads * 4
        workspace_limit = lse_bytes + 512
    assert temporary_bytes <= workspace_limit
    dummy_cu_k = torch.zeros_like(cu_query_lens)
    _, reference_lse, *_ = torch.ops._vllm_fa2_C.varlen_fwd(
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
        max_length,
        0.0,
        softmax_scale,
        False,
        causal,
        -1,
        -1 if group32_decode else (0 if causal else -1),
        0.0,
        False,
        num_splits,
        None,
    )

    assert torch.equal(output.view(torch.int16), reference.view(torch.int16))
    if max(query_lens) >= 64:
        assert torch.equal(
            output_lse.view(torch.int32), reference_lse.view(torch.int32)
        )
    torch.testing.assert_close(output_lse, reference_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
def test_cuda_op_cascade_matches_single_attention() -> None:
    importlib.import_module("vllm.vllm_flash_attn._vllm_fa2_sm80_fp8_C")

    torch.manual_seed(4321)
    device = torch.device("cuda")
    batch_size = 8
    num_query_heads, num_kv_heads, head_size, page_size = 8, 4, 256, 16
    common_prefix_len = 64
    common_blocks = common_prefix_len // page_size
    suffix_lens = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    suffix_blocks = 1
    num_pages = common_blocks + batch_size * suffix_blocks
    block_table = torch.empty(
        (batch_size, common_blocks + suffix_blocks),
        dtype=torch.int32,
        device=device,
    )
    block_table[:, :common_blocks] = torch.arange(
        common_blocks, dtype=torch.int32, device=device
    )
    block_table[:, common_blocks] = torch.arange(
        common_blocks, num_pages, dtype=torch.int32, device=device
    )
    query = torch.randn(
        batch_size,
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
    cu_query_lens = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    full_kv_lens = suffix_lens + common_prefix_len
    softmax_scale = head_size**-0.5
    reference = torch.empty_like(query)
    torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd(
        query,
        key_bytes,
        value_bytes,
        reference,
        cu_query_lens,
        full_kv_lens,
        block_table,
        q_scale,
        k_scale,
        v_scale,
        1,
        int(full_kv_lens.max()),
        softmax_scale,
        True,
    )

    prefix_output = torch.empty_like(query)
    prefix_cu_query_lens = torch.tensor(
        [0, batch_size], dtype=torch.int32, device=device
    )
    prefix_kv_lens = torch.tensor([common_prefix_len], dtype=torch.int32, device=device)
    prefix_output, prefix_lse = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_lse(
        query,
        key_bytes,
        value_bytes,
        prefix_output,
        prefix_cu_query_lens,
        prefix_kv_lens,
        block_table[:1],
        q_scale,
        k_scale,
        v_scale,
        batch_size,
        common_prefix_len,
        softmax_scale,
        False,
    )
    suffix_output = torch.empty_like(query)
    suffix_output, suffix_lse = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_lse(
        query,
        key_bytes,
        value_bytes,
        suffix_output,
        cu_query_lens,
        suffix_lens,
        block_table[:, common_blocks:],
        q_scale,
        k_scale,
        v_scale,
        1,
        int(suffix_lens.max()),
        softmax_scale,
        True,
    )
    merged = torch.empty_like(query)
    merge_attn_states(
        merged,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
    )

    torch.testing.assert_close(merged, reference, rtol=1e-2, atol=1e-2)


def _small_op_inputs(query_lens=(1,), kv_lens=(16,), heads=8, kv_heads=4):
    """Valid small paged inputs for graph and public-API failure tests."""
    batch = len(query_lens)
    pages = max(1, math.ceil(max(kv_lens) / 16))
    q = torch.randn((sum(query_lens), heads, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randint(
        0, 127, (batch * pages, 16, kv_heads, 256), device="cuda", dtype=torch.uint8
    )
    v = torch.randint_like(k, 0, 127)
    cu = torch.tensor(
        [0, *np.cumsum(query_lens).tolist()], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    table = torch.arange(batch * pages, device="cuda", dtype=torch.int32).reshape(
        batch, pages
    )
    scale = torch.tensor(0.0137, device="cuda", dtype=torch.float32)
    return [
        q,
        k,
        v,
        torch.empty_like(q),
        cu,
        lengths,
        table,
        scale,
        scale,
        scale,
        max(1, max(query_lens)),
        max(1, max(kv_lens)),
        0.0625,
        True,
    ]


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize(
    "case",
    [
        "zero_kv_heads",
        "zero_query_heads",
        "bad_rank",
        "bad_metadata_rank",
        "short_table",
        "int64_length",
        "bad_alignment",
        "cpu_scale",
    ],
)
def test_cuda_op_rejects_invalid_metadata_before_launch(case):
    args = _small_op_inputs()
    if case == "zero_kv_heads":
        args[1] = args[1][:, :, :0]
        args[2] = args[2][:, :, :0]
    elif case == "zero_query_heads":
        args[0] = args[0][:, :0]
        args[3] = args[3][:, :0]
    elif case == "bad_rank":
        args[0] = args[0].flatten()
    elif case == "bad_metadata_rank":
        args[6] = args[6].flatten()
    elif case == "short_table":
        args[11] = 17
    elif case == "int64_length":
        args[11] = 2**40
    elif case == "bad_alignment":
        args[1] = torch.empty((1, 16, 4, 257), device="cuda", dtype=torch.uint8)[
            ..., 1:
        ]
    elif case == "cpu_scale":
        args[7] = args[7].cpu()
    with pytest.raises(RuntimeError):
        torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd(*args)
    # A rejected call must leave the CUDA context usable.
    torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd(*_small_op_inputs())
    torch.accelerator.synchronize()


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
def test_cuda_op_rejects_cross_device_tensors():
    if torch.accelerator.device_count() < 2:
        pytest.skip("Two visible SM80 devices are required")
    for index in [1, 2, 3, 4, 5, 6, 7]:
        args = _small_op_inputs()
        args[index] = args[index].to("cuda:1")
        with pytest.raises(RuntimeError, match="same device"):
            torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd(*args)


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
def test_cuda_op_empty_query_does_not_launch_zero_grid():
    args = _small_op_inputs(query_lens=(0, 0), kv_lens=(0, 16))
    args[10] = 64  # Staging dispatch must also handle an entirely empty batch.
    out, lse = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_lse(*args)
    torch.accelerator.synchronize()
    assert out.shape == (0, 8, 256)
    assert lse.shape == (8, 0)


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize(
    "query_lens,kv_lens,heads,kv_heads",
    [
        ([1], [262144], 8, 4),
        ([1] * 256, [513] * 256, 8, 4),
        ([2048], [262144], 8, 4),
        ([64], [32704], 8, 4),
        ([64], [32768], 8, 4),
        ([16384], [16384], 8, 4),
        ([0, 1, 63, 64, 65, 2048], [0, 17, 63, 8193, 32769, 2048], 8, 4),
        ([1, 1], [1, 8193], 32, 1),
        ([64, 65], [8193, 127], 8, 1),
        ([1, 16, 65], [0, 1, 33], 1, 1),
    ],
)
def test_cuda_op_extreme_shapes_match_materialized_fa2(
    query_lens, kv_lens, heads, kv_heads
):
    test_cuda_op_matches_materialized_fa2_qdq(
        query_lens, kv_lens, True, -1, heads, kv_heads
    )


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize(
    "query_lens,kv_lens", [((1,) * 64, (129,) * 64), ((64, 65), (2049, 8193))]
)
def test_cuda_graph_replay_refreshes_queries_pages_and_lengths(query_lens, kv_lens):
    args = _small_op_inputs(query_lens, kv_lens)
    op = torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    expected = torch.empty_like(args[3])
    for step in range(100):
        args[0].normal_()
        args[5].copy_(
            torch.tensor(
                [max(q, k - step % 17) for q, k in zip(query_lens, kv_lens)],
                device="cuda",
                dtype=torch.int32,
            )
        )
        args[6].copy_(args[6].roll(1, dims=1))
        eager = list(args)
        eager[3] = expected
        op(*eager)
        graph.replay()
        assert torch.equal(args[3].view(torch.int16), expected.view(torch.int16))
    torch.accelerator.synchronize()


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_staged_empty_kv_defines_every_packed_lse_row(prequantized, causal):
    """A graph must refresh empty-KV LSE without writing past its allocation."""
    args = _small_op_inputs((128,) * 4, (512,) * 4)
    args[10] = 256  # Deliberately larger than each packed query segment.
    args[13] = causal
    if prequantized:
        args[0] = _qdq(args[0], args[7])
    ops = torch.ops._vllm_fa2_sm80_fp8_C
    op = ops.varlen_fwd_lse_prequantized_q if prequantized else ops.varlen_fwd_lse
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, lse = op(*args)
    for lengths in [(0, 0, 0, 0), (0, 1, 128, 512), (512,) * 4]:
        args[5].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        lse.fill_(12345.0)
        args[3].fill_(13)
        graph.replay()
        assert not bool((lse == 12345).any())
        assert not bool(torch.isnan(lse).any() | torch.isneginf(lse).any())
        masked = sum(max(128 - n, 0) if causal else 128 * (n == 0) for n in lengths)
        assert int(torch.isposinf(lse).sum()) == 8 * masked
        if not any(lengths):
            assert not bool(args[3].any())


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize(
    "query_lens,kv_lens",
    [((1,) * 32, (129,) * 32), ((17, 33), (128, 256)), ((64, 65), (257, 513))],
)
def test_prequantized_query_matches_public_qdq_bits(query_lens, kv_lens):
    args = _small_op_inputs(query_lens, kv_lens)
    ops = torch.ops._vllm_fa2_sm80_fp8_C
    ops.varlen_fwd(*args)
    reference = args[3].clone()
    args[0] = _qdq(args[0], args[7])
    ops.varlen_fwd_prequantized_q(*args)
    assert torch.equal(reference.view(torch.int16), args[3].view(torch.int16))


def _sm80_fp8_checkpoint_config():
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

    return CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "group",
                        "group_size": 128,
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "token",
                        "dynamic": True,
                        "symmetric": True,
                    },
                }
            },
            "kv_cache_scheme": {
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "symmetric": True,
                "dynamic": False,
            },
        }
    )


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_sm80_original_fp8_config_selects_both_backends(cache_dtype, monkeypatch):
    """Unmodified token-FP8 metadata must enable QDQ without global config mutation."""
    from unittest.mock import Mock

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.experts.marlin_fp8_qdq_fused_moe import (
        MarlinFp8QdqFusedExperts,
    )
    from vllm.model_executor.layers.fused_moe.layer import RoutedExperts
    from vllm.v1.attention.selector import get_attn_backend

    monkeypatch.delenv("VLLM_MARLIN_INPUT_DTYPE", raising=False)
    quant = _sm80_fp8_checkpoint_config()
    activation = quant.target_scheme_map["Linear"]["input_activations"]
    before = activation.model_dump()
    config = VllmConfig(
        quant_config=quant, cache_config=CacheConfig(cache_dtype=cache_dtype)
    )
    with set_current_vllm_config(config):
        moe = make_dummy_moe_config(
            num_experts=256, experts_per_token=8, hidden_dim=2048, intermediate_size=256
        )
        layer = Mock(spec=RoutedExperts)
        layer.moe_config = moe
        method = quant.get_quant_method(layer, "model.layers.0.mlp.experts")
        assert method.experts_cls is MarlinFp8QdqFusedExperts
        assert method.input_quant is activation
        assert activation.model_dump() == before
        assert moe.moe_backend == "auto"
        assert config.kernel_config.moe_backend == "auto"
        assert (
            get_attn_backend(256, torch.bfloat16, cache_dtype)
            is FlashAttentionQkvFp8Sm80FusedBackend
        )
    assert config.attention_config.backend is None


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_sm80_fp8_cache_keeps_bf16_queries_and_byte_storage(cache_dtype):
    """FP8 configuration must avoid native FA2 rejection and double Q quantization."""
    from vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused import (
        FlashAttentionQkvFp8Sm80FusedImpl,
    )

    impl = FlashAttentionQkvFp8Sm80FusedImpl(8, 256, 0.0625, 4, None, None, cache_dtype)
    assert impl.kv_cache_dtype == cache_dtype
    assert impl.supports_quant_query_input is False
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    actual = FlashAttentionQkvFp8Sm80FusedBackend.customize_spec(spec)
    assert actual.dtype == torch.uint8
    assert actual.kv_quant_mode == KVQuantMode.SM80_FP8_PER_TENSOR


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
def test_sm80_auto_selection_preserves_explicit_attention_backend():
    from vllm.config import (
        AttentionConfig,
        CacheConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.selector import get_attn_backend

    config = VllmConfig(
        quant_config=_sm80_fp8_checkpoint_config(),
        cache_config=CacheConfig(cache_dtype="bfloat16"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
    )
    with set_current_vllm_config(config):
        assert (
            get_attn_backend(256, torch.bfloat16, "bfloat16") is FlashAttentionBackend
        )


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
def test_sm80_auto_moe_does_not_silently_drop_fp8_for_explicit_marlin(monkeypatch):
    from unittest.mock import Mock

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.layer import RoutedExperts

    monkeypatch.delenv("VLLM_MARLIN_INPUT_DTYPE", raising=False)
    quant = _sm80_fp8_checkpoint_config()
    config = VllmConfig(
        quant_config=quant, kernel_config=KernelConfig(moe_backend="marlin")
    )
    with set_current_vllm_config(config):
        layer = Mock(spec=RoutedExperts)
        layer.moe_config = make_dummy_moe_config(
            num_experts=256, experts_per_token=8, hidden_dim=2048, intermediate_size=256
        )
        layer.moe_config.moe_backend = config.kernel_config.moe_backend
        with pytest.raises(ValueError, match="explicitly requested"):
            quant.get_quant_method(layer, "model.layers.0.mlp.experts")


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize(
    "change",
    ["int8", "static", "asymmetric", "no_activations", "group32", "per_head_kv"],
)
def test_sm80_auto_qkv_does_not_claim_other_quantization_schemes(change):
    quant = _sm80_fp8_checkpoint_config()
    scheme = quant.target_scheme_map["Linear"]
    if change == "int8":
        scheme["input_activations"].type = "int"
    elif change == "static":
        scheme["input_activations"].dynamic = False
    elif change == "asymmetric":
        scheme["input_activations"].symmetric = False
    elif change == "no_activations":
        scheme["input_activations"] = None
    elif change == "group32":
        scheme["weights"].group_size = 32
    else:
        quant.kv_cache_scheme["strategy"] = "attn_head"
    assert not quant.uses_sm80_fp8_qkv()


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (9, 0), (10, 0)])
def test_sm80_auto_qkv_does_not_replace_other_gpu_backends(capability, monkeypatch):
    from vllm.platforms import current_platform

    quant = _sm80_fp8_checkpoint_config()
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda *args, **kwargs: DeviceCapability(*capability),
    )
    assert not quant.uses_sm80_fp8_qkv()


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize(
    ("batch", "length", "bound", "splits"),
    [
        (1, 129, 2048, 0),
        (4, 513, 513, 0),
        (43, 129, 2048, 0),
        (44, 129, 2048, 0),
        (65, 257, 257, 0),
        (1, 129, 2048, 1),
        (4, 129, 2048, 3),
        (1, 17, 2048, 128),
    ],
)
@pytest.mark.parametrize("cache_padding", [-1, 8])
def test_group32_decode_matches_native_automatic_split_bits(
    batch, length, bound, splits, cache_padding
):
    """Production no-window decode must keep native split and BF16 rounding."""
    test_cuda_op_matches_materialized_fa2_qdq(
        [1] * batch,
        [length] * batch,
        True,
        cache_padding,
        group32_decode=True,
        num_splits=splits,
        max_length_bound=bound,
    )


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize("cache_padding", [-1, 8])
@pytest.mark.parametrize(
    ("query_lens", "kv_lens", "causal", "budget_mib"),
    [
        pytest.param([64], [257], True, 0, id="explicit-direct"),
        pytest.param([8192], [49152], True, 256, id="original-budget-fit"),
        pytest.param([8192], [57344], True, 256, id="original-budget-fallback"),
        pytest.param([8192], [57344], True, 768, id="first-expanded-chunk"),
        pytest.param([8192], [131072], True, 768, id="long-chunk"),
        pytest.param([8192], [262080], True, 1536, id="near-model-limit"),
        pytest.param([1, 8191], [32769, 8191], True, 768, id="mixed-batch2"),
        pytest.param([1, 1, 8190], [32770, 32769, 8190], True, 768, id="mixed-batch3"),
        pytest.param(
            [1, 1, 1, 8189], [32771, 32770, 32769, 8189], True, 768, id="mixed-batch4"
        ),
        pytest.param([64, 64], [65537, 131071], False, 1536, id="noncausal-ragged"),
    ],
)
def test_group32_prefill_budget_preserves_materialized_fa2_bits(
    query_lens, kv_lens, causal, budget_mib, cache_padding
):
    test_cuda_op_matches_materialized_fa2_qdq(
        query_lens,
        kv_lens,
        causal,
        cache_padding,
        stage_budget_bytes=budget_mib * 1024 * 1024,
    )


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize(
    "case",
    ["small", "dtype", "rank", "stride", "alignment", "cpu", "device", "alias"],
)
def test_group32_prefill_rejects_invalid_workspace_before_launch(case):
    """Reject unsafe scratch storage without damaging the CUDA context."""
    args = _small_op_inputs((64,), (257,))
    budget = 256
    workspace = torch.empty(budget, dtype=torch.uint8, device="cuda")
    if case == "small":
        workspace = workspace[:-1]
    elif case == "dtype":
        workspace = workspace.float()
    elif case == "rank":
        workspace = workspace.reshape(16, 16)
    elif case == "stride":
        workspace = torch.empty(budget * 2, dtype=torch.uint8, device="cuda")[::2]
    elif case == "alignment":
        workspace = torch.empty(budget + 1, dtype=torch.uint8, device="cuda")[1:]
    elif case == "cpu":
        workspace = workspace.cpu()
    elif case == "device":
        if torch.accelerator.device_count() < 2:
            pytest.skip("Two visible SM80 devices are required")
        workspace = workspace.to("cuda:1")
    elif case == "alias":
        workspace = args[1].flatten()
    with pytest.raises(RuntimeError):
        torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_group32_prefill_workspace_lse(
            *args, budget, workspace
        )
    torch.ops._vllm_fa2_sm80_fp8_C.varlen_fwd_lse(*args)
    torch.accelerator.synchronize()


@pytest.mark.skipif(not _is_sm80(), reason="An SM80 GPU is required")
@pytest.mark.parametrize("budget_delta", [0, -1])
def test_group32_prefill_workspace_reuse_preserves_results_and_graph(budget_delta):
    """Test the exact staging boundary, canaries, LSE ownership and graph replay."""
    args = _small_op_inputs((64, 65), (257, 513))
    q, k = args[:2]
    pages = math.ceil(args[11] / 16)
    budget = q.numel() * 2 + q.shape[0] * q.shape[1] * 4
    budget += 2 * pages * (4 * 16 * k.shape[2] * 256 + 4)
    budget += budget_delta
    storage = torch.full((budget + 512,), 173, dtype=torch.uint8, device="cuda")
    workspace = storage[256 : 256 + budget]
    ops = torch.ops._vllm_fa2_sm80_fp8_C
    op = ops.varlen_fwd_group32_prefill_workspace_lse
    result, lse = op(*args, budget, workspace)
    saved_result, saved_lse = result.clone(), lse.clone()
    other = list(args)
    other[0] = torch.randn_like(q)
    other[3] = torch.empty_like(q)
    op(*other, budget, workspace)
    assert torch.equal(result.view(torch.int16), saved_result.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), saved_lse.view(torch.int32))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op(*args, budget, workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, lse = op(*args, budget, workspace)
    reference = list(args)
    reference[3] = torch.empty_like(q)
    reference_workspace = torch.empty_like(workspace)
    for step in range(12):
        args[0].normal_()
        args[5].copy_(torch.tensor([257 - step, 513 - step], device="cuda"))
        args[6].copy_(args[6].roll(1, dims=1))
        expected, expected_lse = op(*reference, budget, reference_workspace)
        graph.replay()
        assert torch.equal(result.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(lse.view(torch.int32), expected_lse.view(torch.int32))
        assert (storage[:256] == 173).all() and (storage[256 + budget :] == 173).all()


@pytest.mark.parametrize(
    "mode,eager,group32,max_query,max_kv,expected_bound,expected_op",
    [
        ("FULL", False, True, 1, 8193, 16384, "group32"),
        ("FULL", False, True, 1, 16384, 16384, "group32"),
        ("FULL", False, True, 1, 32768, 32768, "group32"),
        ("FULL_DECODE_ONLY", False, True, 1, 8193, 16384, "group32"),
        ("FULL_AND_PIECEWISE", False, True, 1, 8193, 16384, "group32"),
        ("NONE", False, True, 1, 8193, 8193, "group32"),
        ("PIECEWISE", False, True, 1, 8193, 8193, "group32"),
        ("FULL", True, True, 1, 8193, 8193, "group32"),
        ("FULL", False, False, 1, 8193, 8193, "original"),
        ("FULL", False, True, 2, 8193, 8193, "original"),
    ],
)
def test_group32_decode_keeps_capture_split_bound_in_eager_fallback(
    monkeypatch, mode, eager, group32, max_query, max_kv, expected_bound, expected_op
):
    """FULL Graph fallback must keep KV reduction partitions; preserve other paths."""
    import vllm.v1.attention.backends.flash_attn_qkv_fp8_sm80_fused as backend
    from vllm.config.compilation import CUDAGraphMode

    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=16384, enforce_eager=eager),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode[mode]),
    )
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    impl = object.__new__(backend.FlashAttentionQkvFp8Sm80FusedImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 16, 8, 256
    impl.scale = 0.0625
    impl._group32_prefill_budget = 0
    impl._group32_decode_bound = 0
    if group32:
        impl.prepare_group32_decode()
    calls = []
    ops = torch.ops._vllm_fa2_sm80_fp8_C
    for name, label in [("varlen_fwd", "original"), ("varlen_fwd_group32", "group32")]:
        monkeypatch.setattr(ops, name, lambda *a, label=label: calls.append((label, a)))
    query = torch.empty((max_query, 16, 256), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    cache = torch.empty((1, 8, 16, 512), dtype=torch.uint8)
    scale = torch.ones(1, dtype=torch.float32)
    layer = SimpleNamespace(
        _q_scale=scale, _k_scale=scale, _v_scale=scale, _sm80_group32_decode=group32
    )
    metadata = SimpleNamespace(
        num_actual_tokens=max_query,
        use_cascade=False,
        max_query_len=max_query,
        max_seq_len=max_kv,
        causal=True,
        max_num_splits=0,
        query_start_loc=torch.tensor([0, max_query], dtype=torch.int32),
        seq_lens=torch.tensor([max_kv], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    assert impl.forward(layer, query, None, None, cache, metadata, output) is output
    assert len(calls) == 1
    label, args = calls[0]
    assert label == expected_op and args[11] == expected_bound
    assert args[5] is metadata.seq_lens and metadata.max_seq_len == max_kv
    assert args[7] is scale and args[8] is scale and args[9] is scale
    if group32 and max_query == 1:
        assert args[-1] == metadata.max_num_splits
