# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Group32 GDN state writes with deterministic native indexing semantics."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _store_state(
    State,
    Ids,
    Value,
    rows: tl.constexpr,
    states: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    I0: tl.constexpr,
    BI: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(Ids + row * I0)
    idx = tl.where(idx < 0, idx + states, idx)
    # Match deterministic index_put: when destinations repeat, the last row wins.
    others = tl.arange(0, BI)
    all_idx = tl.load(Ids + others * I0, others < rows, other=states)
    all_idx = tl.where(all_idx < 0, all_idx + states, all_idx)
    last = tl.max(tl.where((others < rows) & (all_idx == idx), others, -1), 0)
    if row == last:
        offset = tl.program_id(1) * B + tl.arange(0, B)
        h = offset // (K * V)
        k = (offset // V) % K
        v = offset % V
        mask = (offset < H * K * V) & (idx >= 0) & (idx < states)
        # Match native deterministic index_put's BF16 NaN conversion explicitly;
        # preserve all other codes without a floating-point conversion.
        code = tl.uint32 if Value.dtype.element_ty == tl.float32 else tl.uint16
        raw_value = Value.to(tl.pointer_type(code))
        raw_state = State.to(tl.pointer_type(code))
        value = tl.load(raw_value + row * T0 + h * T1 + k * T2 + v * T3, mask, other=0)
        if Value.dtype.element_ty == tl.bfloat16:
            nan = ((value & 0x7F80) == 0x7F80) & ((value & 0x007F) != 0)
            value = tl.where(nan, 0x7FFF, value)
        tl.store(raw_state + idx * S0 + h * S1 + k * S2 + v * S3, value, mask)


def store_group32_gdn_state(
    state: torch.Tensor, indices: torch.Tensor, value: torch.Tensor
) -> None:
    """Update valid cache rows without cloning the padded state allocation.

    Callers provide non-overlapping state storage and valid native indices.
    Repeated indices keep the last row; BF16 NaNs follow deterministic index_put.
    """
    if (
        state.ndim != 4
        or value.ndim != 4
        or indices.ndim != 1
        or state.shape[1:] != value.shape[1:]
        or value.shape[0] != indices.numel()
        or not state.is_cuda
        or not state.device == indices.device == value.device
        or state.dtype not in (torch.float32, torch.bfloat16)
        or value.dtype != state.dtype
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Expected matching CUDA FP32/BF16 state rows and integer indices"
        )
    if not indices.numel():
        return
    _store_state[
        (
            indices.numel(),
            triton.cdiv(state.shape[1] * state.shape[2] * state.shape[3], 2048),
        )
    ](
        state,
        indices,
        value,
        indices.numel(),
        state.shape[0],
        *state.shape[1:],
        *state.stride(),
        *value.stride(),
        indices.stride(0),
        triton.next_power_of_2(indices.numel()),
        2048,
        num_warps=4,
    )
