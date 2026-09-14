# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 native FP8 attention with FA2 tiling and FA3's FP32 softmax recipe."""

import torch

from vllm.model_executor.layers.quantization.utils.fp8_sm89 import (
    _fp32_to_e4m3_rn,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _add_fp32(a, b):
    # Keep the online state out of the reduced-precision FP8 MMA accumulator.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _paged_fp8_fwd(
    Q,
    KV,
    OUT,
    LSE,
    Q_START,
    SEQ_LEN,
    BLOCK_TABLE,
    QS,
    KS,
    VS,
    q_stride_token: tl.constexpr,
    q_stride_head: tl.constexpr,
    kv_stride_block: tl.constexpr,
    kv_stride_head: tl.constexpr,
    kv_stride_token: tl.constexpr,
    o_stride_token: tl.constexpr,
    o_stride_head: tl.constexpr,
    bt_stride: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    STORE_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr = 1,
    GROUPED_DECODE: tl.constexpr = False,
    FULL_QK_UNROLL: tl.constexpr = False,
):
    row_block = tl.program_id(0) // SPLITS
    split = tl.program_id(0) % SPLITS
    head = tl.program_id(1)
    seq = tl.program_id(2)
    q_start = tl.load(Q_START + seq)
    q_len = tl.load(Q_START + seq + 1) - q_start
    if row_block * BLOCK_M >= q_len:
        return
    kv_len = tl.load(SEQ_LEN + seq)
    kv_head = head // (NUM_HEADS // NUM_KV_HEADS)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    if GROUPED_DECODE:
        kv_head = head
        query_head = head * (NUM_HEADS // NUM_KV_HEADS) + rows
        valid_rows = rows < NUM_HEADS // NUM_KV_HEADS
        q_base = q_start * q_stride_token + query_head[:, None] * q_stride_head
        o_base = q_start * o_stride_token + query_head[:, None] * o_stride_head
        lse_row = q_start * NUM_HEADS + query_head
    else:
        valid_rows = rows < q_len
        q_base = (q_start + rows[:, None]) * q_stride_token + head * q_stride_head
        o_base = (q_start + rows[:, None]) * o_stride_token + head * o_stride_head
        lse_row = (q_start + rows) * NUM_HEADS + head
    ns = tl.arange(0, BLOCK_N)
    ds = tl.arange(0, HEAD_SIZE)
    inner_ds = tl.arange(0, 32)
    score_scale = tl.load(QS) * tl.load(KS) * SOFTMAX_SCALE * 1.4426950408889634
    value_scale = tl.load(VS)
    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denom = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_SIZE), tl.float32)
    stop = tl.minimum(kv_len, (row_block + 1) * BLOCK_M + kv_len - q_len)
    if GROUPED_DECODE:
        stop = kv_len
    begin = 0
    if SPLITS > 1:
        segment = tl.cdiv(kv_len, BLOCK_N * SPLITS) * BLOCK_N
        begin = split * segment
        stop = tl.minimum(stop, begin + segment)
    for start in range(begin, stop, BLOCK_N):
        tokens = start + ns
        pages = tl.load(
            BLOCK_TABLE + seq * bt_stride + tokens // PAGE_SIZE,
            tokens < kv_len,
            other=0,
        )
        kv_base = (
            pages.to(tl.int64) * kv_stride_block
            + kv_head * kv_stride_head
            + (tokens % PAGE_SIZE) * kv_stride_token
        )
        scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        if FULL_QK_UNROLL:
            for inner_start in tl.static_range(0, HEAD_SIZE, 32):
                q = tl.load(
                    Q + q_base + inner_start + inner_ds[None, :],
                    valid_rows[:, None],
                    other=0.0,
                )
                k = tl.load(
                    KV + kv_base[None, :] + inner_start + inner_ds[:, None],
                    tokens[None, :] < kv_len,
                    other=0.0,
                )
                scores = _add_fp32(scores, tl.dot(q, k, out_dtype=tl.float32))
        else:
            for inner_start in tl.range(
                0,
                HEAD_SIZE,
                32,
                loop_unroll_factor=4 if HEAD_SIZE == 256 else 1,
            ):
                q = tl.load(
                    Q + q_base + inner_start + inner_ds[None, :],
                    valid_rows[:, None],
                    other=0.0,
                )
                k = tl.load(
                    KV + kv_base[None, :] + inner_start + inner_ds[:, None],
                    tokens[None, :] < kv_len,
                    other=0.0,
                )
                scores = _add_fp32(scores, tl.dot(q, k, out_dtype=tl.float32))
        scores *= score_scale
        valid = valid_rows[:, None] & (tokens[None, :] < kv_len)
        if not GROUPED_DECODE:
            valid &= tokens[None, :] <= rows[:, None] + kv_len - q_len
        scores = tl.where(valid, scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(scores - safe_m[:, None] + 8.0)
        # The denominator uses FP32 exponentials before the P-to-FP8 cast.
        denom = denom * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        v = tl.load(
            KV + kv_base[:, None] + HEAD_SIZE + ds[None, :],
            tokens[:, None] < kv_len,
            other=0.0,
        )
        acc = _add_fp32(acc, tl.dot(_fp32_to_e4m3_rn(p), v, out_dtype=tl.float32))
        m = new_m
    safe_denom = tl.where(denom > 0, denom, 1.0)
    result = acc * (value_scale / safe_denom[:, None])
    if SPLITS > 1:
        partial_row = lse_row * SPLITS + split
        tl.store(
            OUT + partial_row[:, None] * HEAD_SIZE + ds[None, :],
            result,
            valid_rows[:, None],
        )
        lse = tl.where(denom > 0, m + tl.log2(safe_denom) - 8.0, -float("inf"))
        tl.store(LSE + partial_row, lse, valid_rows)
    else:
        tl.store(
            OUT + o_base + ds[None, :],
            result,
            valid_rows[:, None],
        )
    if STORE_LSE and SPLITS == 1:
        lse = tl.where(
            denom > 0,
            (m + tl.log2(safe_denom) - 8.0) * 0.6931471805599453,
            -float("inf"),
        )
        tl.store(LSE + lse_row, lse, valid_rows)


@triton.jit
def _merge_fp8_splits(
    PARTIAL,
    PARTIAL_LSE,
    OUT,
    LSE,
    Q_START,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    o_stride_token: tl.constexpr,
    o_stride_head: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    head = tl.program_id(0)
    seq = tl.program_id(1)
    row = tl.load(Q_START + seq)
    if row == tl.load(Q_START + seq + 1):
        return
    ss = tl.arange(0, SPLITS)
    ds = tl.arange(0, HEAD_SIZE)
    partial_row = (row * NUM_HEADS + head) * SPLITS + ss
    lse = tl.load(PARTIAL_LSE + partial_row)
    maximum = tl.max(lse, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(lse - maximum)
    partials = tl.load(PARTIAL + partial_row[:, None] * HEAD_SIZE + ds[None, :])
    denom = tl.sum(weights, 0)
    safe_denom = tl.where(denom > 0, denom, 1.0)
    value = tl.sum(partials * weights[:, None], 0) / safe_denom
    tl.store(OUT + row * o_stride_token + head * o_stride_head + ds, value)
    if STORE_LSE:
        result_lse = tl.where(
            denom > 0,
            (maximum + tl.log2(safe_denom)) * 0.6931471805599453,
            -float("inf"),
        )
        tl.store(LSE + row * NUM_HEADS + head, result_lse)


def _decode_num_splits(num_seqs: int, num_heads: int, max_seq_len: int) -> int:
    # Keep at least one 64-token tile per split and enough independent CTAs
    # for a small decode batch; large batches need fewer splits.
    tiles = max_seq_len // 64
    if tiles <= 1:
        return 1
    parallel_splits = triton.next_power_of_2(triton.cdiv(1024, num_seqs * num_heads))
    return min(32, 1 << (tiles.bit_length() - 1), parallel_splits)


def sm89_fp8_paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    lse: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Causal ragged attention over packed (block, head, token, K|V) FP8 KV."""
    assert query.dtype == kv_cache.dtype == torch.float8_e4m3fn
    assert output.dtype in (torch.bfloat16, torch.float16)
    assert query.ndim == output.ndim == 3 and kv_cache.ndim == 4
    assert query.shape == output.shape
    assert query.stride(-1) == kv_cache.stride(-1) == output.stride(-1) == 1
    assert block_table.ndim == 2 and block_table.stride(-1) == 1
    assert query_start_loc.is_contiguous() and seq_lens.is_contiguous()
    num_heads, head_size = query.shape[1:]
    num_kv_heads, page_size = kv_cache.shape[1:3]
    assert head_size in (64, 128, 256)
    assert kv_cache.shape[-1] == 2 * head_size
    assert num_heads % num_kv_heads == 0 and page_size % 16 == 0
    assert q_scale.numel() == k_scale.numel() == v_scale.numel() == 1
    num_seqs = query_start_loc.numel() - 1
    assert seq_lens.numel() >= num_seqs and block_table.shape[0] >= num_seqs
    if lse is not None:
        assert lse.shape == query.shape[:2] and lse.is_contiguous()
        assert lse.dtype == torch.float32
    if num_seqs == 0 or max_query_len == 0:
        return output
    splits = 1
    if max_query_len == 1 and max_seq_len is not None:
        splits = _decode_num_splits(num_seqs, num_heads, max_seq_len)
    block_m = 32 if max_query_len >= 512 and head_size == 256 else 16
    # Decode rows share K/V across query heads without changing softmax tiles
    # or split boundaries. Larger groups retain the generic path.
    grouped_decode = max_query_len == 1 and 1 < num_heads // num_kv_heads <= block_m
    grid_heads = num_kv_heads if grouped_decode else num_heads
    kernel_output, kernel_lse = output, lse
    if splits > 1:
        kernel_output = torch.empty(
            (*query.shape[:2], splits, head_size),
            device=query.device,
            dtype=torch.float32,
        )
        kernel_lse = torch.empty(
            (*query.shape[:2], splits), device=query.device, dtype=torch.float32
        )
    _paged_fp8_fwd[
        (triton.cdiv(max_query_len, block_m) * splits, grid_heads, num_seqs)
    ](
        query,
        kv_cache,
        kernel_output,
        kernel_lse,
        query_start_loc,
        seq_lens,
        block_table,
        q_scale,
        k_scale,
        v_scale,
        *query.stride()[:2],
        *kv_cache.stride()[:3],
        *output.stride()[:2],
        block_table.stride(0),
        num_heads,
        num_kv_heads,
        head_size,
        page_size,
        softmax_scale,
        lse is not None,
        block_m,
        64,
        splits,
        grouped_decode,
        head_size == 256 and max_query_len > 1 and block_m == 16,
        num_warps=4,
        num_stages=2,
    )
    if splits > 1:
        _merge_fp8_splits[(num_heads, num_seqs)](
            kernel_output,
            kernel_lse,
            output,
            lse,
            query_start_loc,
            num_heads,
            head_size,
            splits,
            *output.stride()[:2],
            lse is not None,
            num_warps=4,
        )
    return output
