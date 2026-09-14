# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
        should be padded to a multiple of block_size,
    - ignore_invalid_experts: A flag indicating whether to ignore invalid
        experts. When False, all expert_ids in topk_ids will participate in
        counting and ranking, but invalid experts in expert_ids will be marked
        as -1. When True, all invalid expert_ids in topk_ids will be ignored
        and will not participate in counting or ranking, and there will be no
        -1 in expert_ids.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


@triton.jit
def _small_stable_align_kernel(
    IDS,
    SORTED,
    EXPERTS,
    TOTAL,
    PAIRS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    SORT_SIZE: tl.constexpr,
    FILL_SIZE: tl.constexpr,
    EXPERT_FILL_SIZE: tl.constexpr,
):
    i = tl.arange(0, SORT_SIZE)
    expert = tl.load(IDS + i, i < PAIRS, other=-1)
    valid = (i < PAIRS) & (expert >= 0) & (expert < NUM_EXPERTS)
    safe_expert = tl.where(valid, expert, 0).to(tl.int32)
    counts = tl.histogram(safe_expert, NUM_EXPERTS, mask=valid)
    raw_begin = tl.cumsum(counts) - counts
    padded = tl.cdiv(counts, BLOCK) * BLOCK
    padded_begin = tl.cumsum(padded) - padded
    tl.store(TOTAL, tl.sum(padded, 0))

    # Unique integer keys preserve flattened token order within each expert.
    keys = tl.where(valid, safe_expert * SORT_SIZE + i, (NUM_EXPERTS + 1) * SORT_SIZE)
    keys = tl.sort(keys, descending=False)
    sorted_expert = keys // SORT_SIZE
    sorted_token = keys % SORT_SIZE
    valid_sorted = sorted_expert < NUM_EXPERTS
    gather_expert = tl.minimum(sorted_expert, NUM_EXPERTS - 1)
    within_expert = i - tl.gather(raw_begin, gather_expert, 0)
    destination = tl.gather(padded_begin, gather_expert, 0) + within_expert

    fill = tl.arange(0, FILL_SIZE)
    tl.store(SORTED + fill, PAIRS, fill < CAPACITY)
    blocks = tl.arange(0, EXPERT_FILL_SIZE)
    tl.store(EXPERTS + blocks, -1, blocks < MAX_BLOCKS)
    # Fill and scatter can address the same positions from different warps.
    tl.debug_barrier()
    tl.store(SORTED + destination, sorted_token, valid_sorted)
    tl.store(
        EXPERTS + destination // BLOCK,
        sorted_expert,
        valid_sorted & (within_expert % BLOCK == 0),
    )


@triton.jit
def _tile_count_sort_fill(
    IDS,
    KEYS,
    COUNTS,
    SORTED,
    EXPERT_IDS,
    PAIRS: tl.constexpr,
    TILES: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    TILE: tl.constexpr,
    E: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, TILE)
    flat = tile * TILE + lane
    # Initialization finishes before the scatter kernel starts.
    tl.store(SORTED + flat, PAIRS, flat < CAPACITY)
    tl.store(EXPERT_IDS + flat, -1, flat < MAX_BLOCKS)
    if tile < TILES:
        expert = tl.load(IDS + flat, flat < PAIRS, other=-1)
        valid = (flat < PAIRS) & (expert >= 0) & (expert < E)
        safe = tl.where(valid, expert, 0).to(tl.int32)
        counts = tl.histogram(safe, E, mask=valid)
        e = tl.arange(0, E)
        tl.store(COUNTS + tile * E + e, counts)
        key = tl.where(valid, safe * TILE + lane, E * TILE)
        key = tl.sort(key, descending=False)
        tl.store(KEYS + flat, key)


@triton.jit
def _tile_prefix_scatter(
    KEYS,
    COUNTS,
    SORTED,
    EXPERT_IDS,
    TOTAL,
    TILES: tl.constexpr,
    TILES_PAD: tl.constexpr,
    E: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    all_tiles = tl.arange(0, TILES_PAD)
    expert = tl.arange(0, E)
    counts = tl.load(
        COUNTS + all_tiles[:, None] * E + expert[None, :],
        all_tiles[:, None] < TILES,
        other=0,
    )
    totals = tl.sum(counts, axis=0)
    preceding = tl.sum(tl.where(all_tiles[:, None] < tile, counts, 0), axis=0)
    local_counts = tl.load(COUNTS + tile * E + expert)
    raw_begin = tl.cumsum(local_counts, axis=0) - local_counts
    padded = tl.cdiv(totals, BLOCK) * BLOCK
    global_begin = tl.cumsum(padded, axis=0) - padded + preceding
    if tile == 0:
        tl.store(TOTAL, tl.sum(padded, axis=0))

    lane = tl.arange(0, TILE)
    key = tl.load(KEYS + tile * TILE + lane)
    sorted_expert = key // TILE
    original = key % TILE
    valid = sorted_expert < E
    safe = tl.minimum(sorted_expert, E - 1)
    within_tile = lane - tl.gather(raw_begin, safe, axis=0)
    destination = tl.gather(global_begin, safe, axis=0) + within_tile
    tl.store(SORTED + destination, tile * TILE + original, valid)
    tl.store(
        EXPERT_IDS + destination // BLOCK,
        sorted_expert,
        valid & (destination % BLOCK == 0),
    )


def _large_stable_align(
    topk_ids: torch.Tensor, block_size: int, num_experts: int, pad_sorted_ids: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = topk_ids.numel()
    capacity = pairs + num_experts * (block_size - 1)
    if pad_sorted_ids:
        capacity = triton.cdiv(capacity, block_size) * block_size
    blocks = triton.cdiv(capacity, block_size)
    tile = 256
    tiles = triton.cdiv(pairs, tile)
    options = dict(dtype=torch.int32, device=topk_ids.device)
    sorted_ids = torch.empty((capacity,), **options)
    expert_ids = torch.empty((blocks,), **options)
    total = torch.empty((1,), **options)
    keys = torch.empty((tiles, tile), **options)
    counts = torch.empty((tiles, num_experts), **options)
    _tile_count_sort_fill[(triton.cdiv(capacity, tile),)](
        topk_ids,
        keys,
        counts,
        sorted_ids,
        expert_ids,
        pairs,
        tiles,
        capacity,
        blocks,
        tile,
        num_experts,
        num_warps=4,
    )
    _tile_prefix_scatter[(tiles,)](
        keys,
        counts,
        sorted_ids,
        expert_ids,
        total,
        tiles,
        triton.next_power_of_2(tiles),
        num_experts,
        tile,
        block_size,
        num_warps=8 if tiles >= 16 else 4,
    )
    return sorted_ids, expert_ids, total


def moe_align_block_size_sm89(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve flattened token order within each expert on SM89.

    Small shapes use one CTA; larger shapes use tiled counting and a fused
    prefix/scatter pass. Unmeasured shapes retain the general fallback.
    Out-of-range IDs are ignored by the fast paths.
    """
    pairs = topk_ids.numel()
    if not (
        num_experts == 256
        and expert_map is None
        and 0 < pairs <= 16384
        and (
            (pairs <= 512 and block_size in (8, 16))
            or (pairs > 512 and block_size in (8, 16, 32, 48, 64))
        )
        and topk_ids.is_cuda
        and topk_ids.is_contiguous()
        and topk_ids.dtype in (torch.int32, torch.int64)
        and current_platform.is_device_capability(89)
    ):
        return moe_align_block_size(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )
    if pairs > 512:
        return _large_stable_align(topk_ids, block_size, num_experts, pad_sorted_ids)
    capacity = pairs + num_experts * (block_size - 1)
    if pad_sorted_ids:
        capacity = round_up(capacity, block_size)
    if pairs < num_experts:
        capacity = min(pairs * block_size, capacity)
    max_blocks = triton.cdiv(capacity, block_size)
    sorted_ids = torch.empty((capacity,), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=topk_ids.device)
    total = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    _small_stable_align_kernel[(1,)](
        topk_ids,
        sorted_ids,
        expert_ids,
        total,
        pairs,
        num_experts,
        block_size,
        capacity,
        max_blocks,
        triton.next_power_of_2(pairs),
        triton.next_power_of_2(capacity),
        triton.next_power_of_2(max_blocks),
        num_warps=4,
    )
    return sorted_ids, expert_ids, total


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Parameters:
    - max_tokens_per_batch (int): Number of tokens in each batch (both
        valid and invalid).
    - block_size (int): block_size to align the data to.
    - expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
        the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.
    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.
    """

    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
