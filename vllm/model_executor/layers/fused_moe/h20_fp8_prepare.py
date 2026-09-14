# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""H20 FP8 routing and metadata for native grouped GEMM."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _store_group(
    PTR1,
    PTR2,
    PROB1,
    PROB2,
    Y,
    Q2,
    O1,
    O2,
    S1,
    S2,
    W1,
    W2,
    C1,
    C2,
    G1,
    G2,
    slot,
    start,
    expert,
    count,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    expert = tl.minimum(expert, 255)
    first = (
        Y.to(tl.int64) + start * K,
        W1.to(tl.int64) + expert * K * N,
        O1.to(tl.int64) + start * (N * 2) * 2,
        S1.to(tl.int64) + start * 4,
        C1.to(tl.int64) + expert * (N * 2) * 4,
        G1.to(tl.int64) + expert * (K // 128) * (N * 2) * 8,
    )
    second = (
        Q2.to(tl.int64) + start * N,
        W2.to(tl.int64) + expert * N * K // 2,
        O2.to(tl.int64) + start * K * 2,
        S2.to(tl.int64) + start * 4,
        C2.to(tl.int64) + expert * K * 4,
        G2.to(tl.int64) + expert * (N // 128) * K * 8,
    )
    for index in tl.static_range(6):
        tl.store(PTR1 + index * GROUPS + slot, first[index])
        tl.store(PTR2 + index * GROUPS + slot, second[index])
    tl.store(PROB1 + slot * 3, N * 2)
    tl.store(PROB1 + slot * 3 + 1, count)
    tl.store(PROB1 + slot * 3 + 2, K)
    tl.store(PROB2 + slot * 3, K)
    tl.store(PROB2 + slot * 3 + 1, count)
    tl.store(PROB2 + slot * 3 + 2, N)


@triton.jit
def _fused_fp8_permute_prepare_kernel(
    X,
    S,
    IDS,
    Y,
    SY,
    OFFSETS,
    INV,
    PERM,
    Q2,
    S2,
    O1,
    O2,
    W1,
    W2,
    C1,
    C2,
    G1,
    G2,
    PTR1,
    PTR2,
    PROB1,
    PROB2,
    ROWS,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    COMPACT: tl.constexpr,
):
    p = tl.program_id(0)
    idx = tl.arange(0, BLOCK_ROWS)
    ids = tl.load(IDS + idx, idx < ROWS, other=2147483647)
    start = 0
    expert = 0
    destination = 0
    if p <= 256:
        start = tl.sum((ids < p).to(tl.int32), 0)
        tl.store(OFFSETS + p, start.to(tl.int64))
    if p < ROWS:
        # Router IDs are 0..255, or 256 for disabled experts.
        expert = tl.load(IDS + p).to(tl.int32)
        earlier = (ids < expert) | ((ids == expert) & (idx < p))
        destination = tl.sum(earlier.to(tl.int32), 0)
        tl.store(INV + p, destination)
        tl.store(PERM + destination, p)
        columns = tl.arange(0, K)
        tl.store(Y + destination * K + columns, tl.load(X + (p // 8) * K + columns))
        tl.store(SY + destination, tl.load(S + p // 8))
    if COMPACT:
        if p < ROWS:
            _store_group(
                PTR1,
                PTR2,
                PROB1,
                PROB2,
                Y,
                Q2,
                O1,
                O2,
                SY,
                S2,
                W1,
                W2,
                C1,
                C2,
                G1,
                G2,
                destination,
                destination,
                expert,
                (expert < 256).to(tl.int32),
                GROUPS,
                K,
                N,
            )
    else:
        if p < 256:
            count = tl.sum((ids == p).to(tl.int32), 0)
            _store_group(
                PTR1,
                PTR2,
                PROB1,
                PROB2,
                Y,
                Q2,
                O1,
                O2,
                SY,
                S2,
                W1,
                W2,
                C1,
                C2,
                G1,
                G2,
                p,
                start,
                p,
                count,
                GROUPS,
                K,
                N,
            )


def fused_fp8_prepare(
    hidden,
    scales,
    ids,
    output,
    scratch,
    quant2,
    scale2,
    out1,
    out2,
    w1,
    w2,
    channel1,
    channel2,
    group1,
    group2,
    ptr1,
    ptr2,
    problem1,
    problem2,
):
    tokens, width = hidden.shape
    assert width == 2048 and ids.shape == (tokens, 8)
    assert hidden.dtype == output.dtype == torch.float8_e4m3fn
    assert 0 < tokens <= 256
    assert scales.dtype == torch.float32 and scales.shape == (tokens, 1)
    assert ids.dtype in (torch.int32, torch.int64)
    assert hidden.is_contiguous() and scales.is_contiguous() and ids.is_contiguous()
    assert scratch.num_local_experts == scratch.num_experts == 256
    scratch.validate(hidden, ids)
    rows = tokens * 8
    groups = 8 if tokens == 1 else 256
    assert ptr1.shape == ptr2.shape == (6, groups)
    assert problem1.shape == problem2.shape == (groups, 3)
    offsets = scratch.expert_first_token_offset
    inv = scratch.inv_permuted_idx[:rows]
    perm = scratch.permuted_idx[:rows]
    scale1 = torch.empty((rows, 1), device=hidden.device, dtype=torch.float32)
    _fused_fp8_permute_prepare_kernel[(max(rows, 257),)](
        hidden.view(torch.uint8),
        scales.view(torch.int32),
        ids,
        output.view(torch.uint8),
        scale1.view(torch.int32),
        offsets,
        inv,
        perm,
        quant2,
        scale2,
        out1,
        out2,
        w1,
        w2,
        channel1,
        channel2,
        group1,
        group2,
        ptr1,
        ptr2,
        problem1,
        problem2,
        rows,
        K=2048,
        N=256,
        BLOCK_ROWS=triton.next_power_of_2(rows),
        GROUPS=groups,
        COMPACT=tokens == 1,
        num_warps=4,
    )
    return output, scale1, offsets, inv, perm
