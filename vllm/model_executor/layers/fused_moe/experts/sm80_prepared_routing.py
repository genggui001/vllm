# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded, prewarmed SM80 integer routing launch handles."""

import threading

import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_fp8_qdq_fused_moe import (
    _fp8_qdq_partial_scatter,
)
from vllm.triton_utils import tl, triton

_kernels = {}
_ready = set()
_lock = threading.Lock()


@triton.jit(do_not_specialize=["N", "MAX_SORT", "MAX_BLOCKS"])
def _sm80_prepared_histogram(
    IDs,
    Hist,
    Counters,
    Sorted,
    Experts,
    N,
    MAX_SORT,
    MAX_BLOCKS,
    E: tl.constexpr,
    B: tl.constexpr,
    BG: tl.constexpr,
    BH: tl.constexpr,
    FULL: tl.constexpr,
):
    if FULL:
        N = BG * BH
        MAX_SORT = N + E * (B - 1)
        MAX_BLOCKS = tl.cdiv(MAX_SORT, B)
    pid = tl.program_id(0)
    if pid < BG:
        i = pid * BH + tl.arange(0, BH)
        if FULL:
            ids = tl.load(IDs + i)
            valid = (ids >= 0) & (ids < E)
        else:
            ids = tl.load(IDs + i, i < N, -1)
            valid = (i < N) & (ids >= 0) & (ids < E)
        ids = tl.where(valid, ids, -1).to(tl.int32)
        counts = tl.histogram(ids, E, mask=valid)
        tl.store(Hist + pid * E + tl.arange(0, E), counts)
    elif pid == BG:
        e = tl.arange(0, E)
        tl.store(Counters + e, 0)
        i = tl.arange(0, 1024)
        for offset in range(0, MAX_SORT, 1024):
            tl.store(Sorted + offset + i, N, offset + i < MAX_SORT)
        for offset in range(0, MAX_BLOCKS, E):
            tl.store(Experts + offset + e, -1, offset + e < MAX_BLOCKS)


def _prewarm(device, dtype):
    pair = (device, dtype)
    if pair in _ready:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    with _lock:
        if pair in _ready:
            return True
        for groups in (8, 16):
            for full_hist, full_scatter in (
                (False, False),
                (False, True),
                (True, True),
            ):
                n = (
                    groups * 1024
                    if full_hist
                    else groups * 512 + (256 if full_scatter else 8)
                )
                for block in (8, 16, 32, 48, 64):
                    max_sorted = n + 256 * (block - 1)
                    max_blocks = (max_sorted + block - 1) // block
                    hist = _sm80_prepared_histogram.warmup(
                        dtype,
                        torch.int32,
                        torch.int32,
                        torch.int32,
                        torch.int32,
                        n,
                        max_sorted,
                        max_blocks,
                        E=256,
                        B=block,
                        BG=groups,
                        BH=1024,
                        FULL=full_hist,
                        num_warps=4,
                        num_stages=1,
                        grid=(groups + 1, 1, 1),
                    )
                    scatter = _fp8_qdq_partial_scatter.warmup(
                        dtype,
                        torch.int32,
                        torch.int32,
                        torch.int32,
                        torch.int32,
                        torch.int32,
                        n,
                        E=256,
                        B=block,
                        BG=groups,
                        BS=256,
                        FULL=full_scatter,
                        num_warps=4,
                        num_stages=1,
                        grid=((n + 255) // 256, 1, 1),
                    )
                    hist_launch = hist[(groups + 1, 1, 1)]
                    scatter[(1, 1, 1)]  # Load handles before any graph capture.
                    _kernels[
                        (device, dtype, block, groups, full_hist, full_scatter)
                    ] = (hist_launch, scatter)
        _ready.add(pair)
    return True


def compiled_prepare(ids, block):
    # MockTensor warmup specializes pointers to 16-byte alignment.
    # Unaligned contiguous views must keep the ordinary JIT path.
    if ids.data_ptr() % 16:
        return None
    if torch.accelerator.current_device_index() != ids.device.index:
        with torch.accelerator.device_index(ids.device.index):
            return compiled_prepare(ids, block)
    if not _prewarm(ids.device, ids.dtype):
        return None
    n = ids.numel()
    groups = triton.next_power_of_2(triton.cdiv(n, 1024))
    full_hist, full_scatter = n == groups * 1024, n % 256 == 0
    key = (ids.device, ids.dtype, block, groups, full_hist, full_scatter)
    pair = _kernels.get(key)
    if pair is None:
        return None
    max_sorted = n + 256 * (block - 1)
    max_blocks = (max_sorted + block - 1) // block
    hist = torch.empty((groups, 256), device=ids.device, dtype=torch.int32)
    counters = torch.empty(256, device=ids.device, dtype=torch.int32)
    sorted_ids = torch.empty(max_sorted, device=ids.device, dtype=torch.int32)
    experts = torch.empty(max_blocks, device=ids.device, dtype=torch.int32)
    total = torch.empty(1, device=ids.device, dtype=torch.int32)
    stream = torch.cuda.current_stream(ids.device).cuda_stream
    pair[0](
        ids,
        hist,
        counters,
        sorted_ids,
        experts,
        n,
        max_sorted,
        max_blocks,
        256,
        block,
        groups,
        1024,
        full_hist,
        stream=stream,
    )
    pair[1][((n + 255) // 256, 1, 1)](
        ids,
        hist,
        counters,
        sorted_ids,
        experts,
        total,
        n,
        256,
        block,
        groups,
        256,
        full_scatter,
        stream=stream,
    )
    return sorted_ids, experts, total
