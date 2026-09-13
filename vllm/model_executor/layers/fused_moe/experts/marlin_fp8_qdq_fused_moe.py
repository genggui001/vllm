# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80-friendly fused FP8 QDQ helpers for Marlin W4A16 MoE.

The Marlin FP8 activation GEMM uses FP8 tensor-core instructions and therefore
cannot run on SM80.  This opt-in backend keeps the proven W4A16 Marlin GEMMs,
but folds the QAT-compatible dynamic per-token FP8 round trip into one Triton
kernel.  The FC2 path also folds SwiGLU into that same kernel.
"""

import threading

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.triton_utils import tl, triton

_silu_tables: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_silu_tables_lock = threading.Lock()


@triton.jit
def _initialize_silu_table(table, block_size: tl.constexpr):
    code = tl.program_id(0) * block_size + tl.arange(0, block_size)
    gate = code.to(tl.uint16).to(table.dtype.element_ty, bitcast=True).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + tl.extra.cuda.libdevice.exp(-gate))
    tl.store(table + code, silu)


def _get_silu_table(input: torch.Tensor) -> torch.Tensor | None:
    key = (input.device, input.dtype)
    table = _silu_tables.get(key)
    if table is not None:
        return table
    with torch.accelerator.device_index(input.device.index):
        # A first call during graph capture keeps the arithmetic fallback.
        if torch.cuda.is_current_stream_capturing():
            return None
        with _silu_tables_lock:
            table = _silu_tables.get(key)
            if table is None:
                table = torch.empty(65536, device=input.device, dtype=input.dtype)
                _initialize_silu_table[(256,)](
                    table, block_size=256, num_warps=4, num_stages=1
                )
                # Publish only after initialization, including for other streams.
                torch.cuda.current_stream(input.device).synchronize()
                _silu_tables[key] = table
    return table


@triton.jit
def _e4m3fn_qdq_software(values):
    """Round FP32 values to finite E4M3 and decode, without FP8 hardware."""
    magnitude = tl.abs(values)
    bits = magnitude.to(tl.int32, bitcast=True)
    # Keep three fraction bits. The retained LSB supplies the ties-to-even bit.
    rounded_bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -0x100000
    normal = rounded_bits.to(tl.float32, bitcast=True)
    subnormal = tl.extra.cuda.libdevice.rint(magnitude * 512.0) * 0.001953125
    quantized_magnitude = tl.where(magnitude < 0.015625, subnormal, normal)
    quantized_magnitude = tl.minimum(quantized_magnitude, 448.0)
    # Preserve the E4M3 sign bit, including an input negative zero.
    sign = values.to(tl.int32, bitcast=True) & -2147483648
    magnitude_bits = quantized_magnitude.to(tl.int32, bitcast=True)
    return (magnitude_bits | sign).to(tl.float32, bitcast=True)


@triton.jit
def _fp8_e4m3_qdq_row(
    X, Y, row, input_stride, output_stride, C: tl.constexpr, BC: tl.constexpr
):
    c = tl.arange(0, BC)
    v = tl.load(X + row.to(tl.int64) * input_stride + c, c < C, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(v), 0) * (1.0 / 448.0), 1.1754943508222875e-38)
    normalized = tl.clamp(tl.div_rn(v, scale), -448.0, 448.0)
    y = _e4m3fn_qdq_software(normalized) * scale
    tl.store(Y + row.to(tl.int64) * output_stride + c, y, c < C)


@triton.jit(do_not_specialize=["N", "MAX_SORT", "MAX_BLOCKS"])
def _fp8_qdq_prepare_metadata(
    X,
    Y,
    IDs,
    Counts,
    Sorted,
    Experts,
    Total,
    N,
    FULL: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
    MAX_SORT,
    MAX_BLOCKS,
):
    # Exact powers of two keep their unmasked layout; other lengths share code.
    if FULL:
        N = BN
        MAX_SORT = BN + E * (B - 1)
        if BN < E:
            MAX_SORT = BN * B
        MAX_BLOCKS = tl.cdiv(MAX_SORT, B)
    pid = tl.program_id(0)
    if pid == 0:
        i = tl.arange(0, BN)
        ids = tl.load(IDs + i, i < N, -1)
        ids = tl.where((ids >= 0) & (ids < E), ids, -1).to(tl.int32)
        counts = tl.histogram(ids, E, mask=(i < N) & (ids >= 0) & (ids < E))
        e = tl.arange(0, E)
        blocks = tl.cdiv(counts, B)
        ends = tl.cumsum(blocks, 0)
        starts = ends - blocks
        tl.store(Counts + e, starts * B)
        total_blocks = tl.sum(blocks, 0)
        tl.store(Total, total_blocks * B)
        for block in range(tl.max(blocks, 0)):
            tl.store(Experts + starts + block, e, block < blocks)
        for offset in range(total_blocks, MAX_BLOCKS, E):
            tail = offset + e
            tl.store(Experts + tail, -1, tail < MAX_BLOCKS)
    elif pid == 1:
        i = tl.arange(0, 1024)
        for offset in range(0, MAX_SORT, 1024):
            tl.store(Sorted + offset + i, N, offset + i < MAX_SORT)
    else:
        _fp8_e4m3_qdq_row(X, Y, pid - 2, C, C, C, BC)


@triton.jit(do_not_specialize=["N"])
def _fp8_qdq_sort_tokens(IDs, Counts, Sorted, N, E: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    ids = tl.load(IDs + i, i < N, -1)
    valid = (i < N) & (ids >= 0) & (ids < E)
    rank = tl.atomic_add(Counts + ids, 1, valid, sem="relaxed")
    tl.store(Sorted + rank, i, valid)


@triton.jit(do_not_specialize=["N", "MAX_SORT", "MAX_BLOCKS"])
def _fp8_qdq_align_small(
    X,
    Y,
    IDs,
    Sorted,
    Experts,
    Total,
    N,
    FULL: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
    MAX_SORT,
    MAX_BLOCKS,
    BP: tl.constexpr,
    BM: tl.constexpr,
):
    # Exact powers of two keep their unmasked layout; other lengths share code.
    if FULL:
        N = BN
        MAX_SORT = BN + E * (B - 1)
        if BN < E:
            MAX_SORT = BN * B
        MAX_BLOCKS = tl.cdiv(MAX_SORT, B)
    pid = tl.program_id(0)
    if pid < E:
        i = tl.arange(0, BN)
        ids = tl.load(IDs + i, i < N, -1)
        ids = tl.where((ids >= 0) & (ids < E), ids, -1).to(tl.int32)
        counts = tl.histogram(ids, E, mask=(i < N) & (ids >= 0) & (ids < E))
        padded = tl.cdiv(counts, B) * B
        e = tl.arange(0, E)
        start = tl.sum(tl.where(e < pid, padded, 0), 0)
        valid = (i < N) & (ids == pid)
        count = tl.sum(valid.to(tl.int32), 0)
        rank = tl.cumsum(valid.to(tl.int32), 0) - 1
        tl.store(Sorted + start + rank, i, valid)
        p = tl.arange(0, BP)
        padded_count = tl.cdiv(count, B) * B
        tl.store(Sorted + start + count + p, N, count + p < padded_count)
        m = tl.arange(0, BM)
        tl.store(Experts + start // B + m, pid, m < padded_count // B)
        if pid == E - 1:
            total = tl.sum(padded, 0)
            tl.store(Total, total)
            for offset in range(total, MAX_SORT, BN):
                tail = offset + i
                tl.store(Sorted + tail, N, tail < MAX_SORT)
            for offset in range(total // B, MAX_BLOCKS, BM):
                tail = offset + m
                tl.store(Experts + tail, -1, tail < MAX_BLOCKS)
    else:
        _fp8_e4m3_qdq_row(X, Y, pid - E, C, C, C, BC)


@triton.jit
def _fp8_e4m3_per_token_qdq_kernel(
    input_ptr,
    output_ptr,
    input_stride,
    output_stride,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    _fp8_e4m3_qdq_row(
        input_ptr, output_ptr, row, input_stride, output_stride, n_cols, block_size
    )


@triton.jit
def _silu_mul_fp8_e4m3_per_token_qdq_kernel(
    input_ptr,
    output_ptr,
    silu_table,
    input_stride,
    output_stride,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
    use_lookup: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    input_row = input_ptr + row * input_stride

    gate = tl.load(input_row + offsets, mask=mask, other=0.0)
    up = tl.load(input_row + n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
    if use_lookup:
        code = gate.to(tl.uint16, bitcast=True).to(tl.int32)
        silu = tl.load(silu_table + code).to(tl.float32)
    else:
        gate = gate.to(tl.float32)
        silu = tl.div_rn(gate, 1.0 + tl.extra.cuda.libdevice.exp(-gate))
        # Match the packed CUDA activation's rounding before multiplication.
        silu = silu.to(input_ptr.dtype.element_ty).to(tl.float32)
    activated = (silu * up).to(input_ptr.dtype.element_ty).to(tl.float32)
    absmax = tl.max(tl.abs(activated), axis=0)
    scale = tl.maximum(absmax * (1.0 / 448.0), 1.1754943508222875e-38)
    normalized = tl.clamp(tl.div_rn(activated, scale), -448.0, 448.0)
    dequantized = _e4m3fn_qdq_software(normalized) * scale

    tl.store(
        output_ptr + row * output_stride + offsets,
        dequantized,
        mask=mask,
    )


def _launch_config(n_cols: int, rows: int, *, swiglu: bool = False) -> tuple[int, int]:
    if n_cols <= 0:
        raise ValueError("FP8 per-token QDQ requires a positive row width.")
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError(f"FP8 per-token QDQ row is too wide: {n_cols}.")
    num_warps = 4 if block_size <= 2048 else 8
    # Small FC1 batches benefit from more lanes per row; large batches need
    # more resident CTAs. Keep other widths on the established layout.
    if not swiglu and n_cols == 2048:
        if rows <= 128:
            num_warps = 16
        elif rows <= 512:
            num_warps = 8
    elif swiglu and n_cols == 256:
        if rows <= 64:
            num_warps = 8
        elif rows >= 8192:
            num_warps = 2
    return block_size, num_warps


def fp8_e4m3_per_token_qdq_fused(
    x: torch.Tensor, output: torch.Tensor | None = None
) -> torch.Tensor:
    """Run dynamic per-token E4M3 QDQ in one Triton kernel."""
    if x.ndim != 2:
        raise ValueError(f"FP8 per-token QDQ expects a 2D tensor, got {x.shape}.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"FP8 per-token QDQ expects float16 or bfloat16 input, got {x.dtype}."
        )
    if not x.is_contiguous():
        raise ValueError("FP8 per-token QDQ expects contiguous input.")
    if not x.is_cuda:
        raise ValueError("FP8 per-token QDQ expects a CUDA input.")
    if output is None:
        output = torch.empty_like(x)
    if output.shape != x.shape or output.dtype != x.dtype or not output.is_contiguous():
        raise ValueError("FP8 per-token QDQ output must match the contiguous input.")
    if output.device != x.device:
        raise ValueError("FP8 per-token QDQ tensors must be on the same CUDA device.")

    rows, n_cols = x.shape
    block_size, num_warps = _launch_config(n_cols, rows)
    if rows == 0:
        return output
    _fp8_e4m3_per_token_qdq_kernel[(rows,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_cols=n_cols,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def silu_mul_fp8_e4m3_per_token_qdq_fused(
    input: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Run SwiGLU and dynamic per-token E4M3 QDQ in one Triton kernel."""
    if input.ndim != 2 or output.ndim != 2:
        raise ValueError("Fused SwiGLU FP8 QDQ expects 2D tensors.")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported input dtype {input.dtype}.")
    if not input.is_cuda or output.device != input.device:
        raise ValueError(
            "Fused SwiGLU FP8 QDQ tensors must be on the same CUDA device."
        )
    if input.shape[0] != output.shape[0] or input.shape[1] != output.shape[1] * 2:
        raise ValueError(
            f"Expected input [M, 2N] and output [M, N], got "
            f"{input.shape} and {output.shape}."
        )
    if (
        input.dtype != output.dtype
        or not input.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("Fused SwiGLU FP8 QDQ expects matching contiguous tensors.")

    rows, n_cols = output.shape
    block_size, num_warps = _launch_config(n_cols, rows, swiglu=True)
    if rows == 0:
        return output
    table = _get_silu_table(input) if n_cols == 256 and rows >= 512 else None
    if table is not None:
        num_warps = 2 if rows >= 8192 else (8 if rows <= 512 else 4)
    _silu_mul_fp8_e4m3_per_token_qdq_kernel[(rows,)](
        input,
        output,
        table,
        input.stride(0),
        output.stride(0),
        n_cols=n_cols,
        block_size=block_size,
        use_lookup=table is not None,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _fp8_e4m3_qdq_align_fallback(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare QDQ activations and Marlin's padded expert indices together."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    rows = hidden_states.shape[0]
    if rows == 0:
        quantized = fp8_e4m3_per_token_qdq_fused(hidden_states)
        empty_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
        total = torch.zeros(1, dtype=torch.int32, device=hidden_states.device)
        return quantized, empty_ids, empty_ids, total
    supported = (
        hidden_states.ndim == 2
        and hidden_states.shape[1] == 2048
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and hidden_states.is_cuda
        and hidden_states.is_contiguous()
        and 0 < rows <= 2048
        and topk_ids.shape == (rows, 8)
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_ids.device == hidden_states.device
        and topk_ids.is_contiguous()
        and num_experts == 256
        and block_size in (8, 16, 32, 48, 64)
        and expert_map is None
    )
    if not supported:
        quantized = fp8_e4m3_per_token_qdq_fused(hidden_states)
        return quantized, *moe_align_block_size(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

    quantized = torch.empty_like(hidden_states)
    numel = topk_ids.numel()
    max_sorted = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_sorted = min(max_sorted, numel * block_size)
    max_blocks = triton.cdiv(max_sorted, block_size)
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=topk_ids.device)
    total = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    if rows <= 128:
        _fp8_qdq_align_small[(num_experts + rows,)](
            hidden_states,
            quantized,
            topk_ids,
            sorted_ids,
            expert_ids,
            total,
            N=numel,
            FULL=numel == triton.next_power_of_2(numel),
            E=num_experts,
            B=block_size,
            C=2048,
            BN=triton.next_power_of_2(numel),
            BC=2048,
            MAX_SORT=max_sorted,
            MAX_BLOCKS=max_blocks,
            BP=triton.next_power_of_2(block_size),
            BM=triton.next_power_of_2(triton.cdiv(numel, block_size)),
            num_warps=4,
            num_stages=1,
        )
    else:
        counts = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
        _fp8_qdq_prepare_metadata[(rows + 2,)](
            hidden_states,
            quantized,
            topk_ids,
            counts,
            sorted_ids,
            expert_ids,
            total,
            N=numel,
            FULL=numel == triton.next_power_of_2(numel),
            E=num_experts,
            B=block_size,
            C=2048,
            BN=triton.next_power_of_2(numel),
            BC=2048,
            MAX_SORT=max_sorted,
            MAX_BLOCKS=max_blocks,
            num_warps=8,
            num_stages=1,
        )
        _fp8_qdq_sort_tokens[(triton.cdiv(numel, 256),)](
            topk_ids,
            counts,
            sorted_ids,
            N=numel,
            E=num_experts,
            num_warps=8,
            num_stages=1,
        )
    return quantized, sorted_ids, expert_ids, total


class MarlinFp8QdqFusedExperts(MarlinExperts):
    """Opt-in Marlin W4A16 experts with fused QAT-compatible FP8 QDQ."""

    input_prequantized: bool = False

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if getattr(self, "input_prequantized", False):
            assert self._lora_context is None
            return fp8_e4m3_qdq_align_prepared(
                hidden_states, topk_ids, block_size, num_experts, expert_map
            )
        return fp8_e4m3_qdq_align(
            hidden_states, topk_ids, block_size, num_experts, expert_map
        )

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
        valid_rows: torch.Tensor | None = None,
    ) -> None:
        if activation != MoEActivation.SILU:
            raise ValueError(
                "marlin_fp8_qdq_fused currently supports only the SILU/SwiGLU "
                f"activation, got {activation.value}."
            )
        if self.activation_config.clamp_limit is not None:
            raise ValueError("marlin_fp8_qdq_fused does not support clamped SwiGLU.")
        # Plain SwiGLU is expert-independent. Routing/remapping is handled by
        # the surrounding Marlin GEMMs exactly as in the original backend.
        silu_mul_fp8_e4m3_per_token_qdq_fused(input, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self.input_dtype is not None:
            raise ValueError(
                "marlin_fp8_qdq_fused keeps Marlin GEMMs W4A16. Unset "
                "VLLM_MARLIN_INPUT_DTYPE."
            )
        if apply_router_weight_on_input:
            raise ValueError(
                "marlin_fp8_qdq_fused requires router probabilities after FC2."
            )

        # LoRA has its own alignment path and keeps the separate QDQ helper.
        qdq_hidden_states = hidden_states
        if self._lora_context is not None:
            qdq_hidden_states = fp8_e4m3_per_token_qdq_fused(hidden_states)
        super().apply(
            output=output,
            hidden_states=qdq_hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=False,
        )


@triton.jit(do_not_specialize=["N", "MAX_SORT", "MAX_BLOCKS"])
def _fp8_qdq_partial_histogram(
    X,
    Y,
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
    else:
        _fp8_e4m3_qdq_row(X, Y, pid - BG - 1, 2048, 2048, 2048, 2048)


@triton.jit(do_not_specialize=["N"])
def _fp8_qdq_partial_scatter(
    IDs,
    Hist,
    Counters,
    Sorted,
    Experts,
    Total,
    N,
    E: tl.constexpr,
    B: tl.constexpr,
    BG: tl.constexpr,
    BS: tl.constexpr,
    FULL: tl.constexpr,
):
    pid = tl.program_id(0)
    e = tl.arange(0, E)
    g = tl.arange(0, BG)
    counts = tl.sum(tl.load(Hist + g[:, None] * E + e[None, :]), axis=0)
    blocks = tl.cdiv(counts, B)
    ends = tl.cumsum(blocks, axis=0)
    starts = ends - blocks
    if pid == 0:
        tl.store(Total, tl.sum(blocks, axis=0) * B)
        for block in range(tl.max(blocks, axis=0)):
            tl.store(Experts + starts + block, e, block < blocks)
    i = pid * BS + tl.arange(0, BS)
    if FULL:
        ids = tl.load(IDs + i)
        valid = (ids >= 0) & (ids < E)
    else:
        ids = tl.load(IDs + i, i < N, -1)
        valid = (i < N) & (ids >= 0) & (ids < E)
    safe_ids = tl.where(valid, ids, 0).to(tl.int32)
    base = tl.gather(starts, safe_ids, axis=0) * B
    rank = tl.atomic_add(Counters + safe_ids, 1, valid, sem="relaxed")
    tl.store(Sorted + base + rank, i, valid)


_prewarmed_preparation_inputs = set()
_preparation_warmup_lock = threading.Lock()


def _prewarm_fp8_preparation(hidden, ids):
    key = (hidden.device, hidden.dtype, ids.dtype)
    if key in _prewarmed_preparation_inputs:
        return
    with torch.accelerator.device_index(hidden.device.index):
        if torch.cuda.is_current_stream_capturing():
            return
        with _preparation_warmup_lock:
            if key in _prewarmed_preparation_inputs:
                return
            # Representatives cover the default Marlin B/BG/FULL combinations.
            for rows in (513, 544, 922, 960, 1024, 1025, 1056, 1383, 1408, 2048):
                for block in (8, 16, 32, 48, 64):
                    if rows * 8 / 256 / block < 0.9:
                        break
                n = rows * 8
                groups = triton.next_power_of_2(triton.cdiv(n, 1024))
                max_sorted = n + 256 * (block - 1)
                max_blocks = triton.cdiv(max_sorted, block)
                _fp8_qdq_partial_histogram.warmup(
                    hidden.dtype,
                    hidden.dtype,
                    ids.dtype,
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
                    FULL=n == groups * 1024,
                    grid=(groups + 1 + rows,),
                    num_warps=4,
                    num_stages=1,
                )
                _fp8_qdq_partial_scatter.warmup(
                    ids.dtype,
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
                    FULL=n % 256 == 0,
                    grid=(triton.cdiv(n, 256),),
                    num_warps=4,
                    num_stages=1,
                )
            _prewarmed_preparation_inputs.add(key)


def fp8_e4m3_qdq_align(
    hidden,
    ids,
    block_size,
    num_experts,
    expert_map=None,
):
    histogram_block, scatter_block = 1024, 256
    first_warps = second_warps = 4
    rows = hidden.shape[0]
    if not (
        hidden.ndim == 2
        and hidden.shape[1] == 2048
        and hidden.dtype in (torch.float16, torch.bfloat16)
        and hidden.is_cuda
        and hidden.is_contiguous()
        and 512 < rows <= 2048
        and ids.shape == (rows, 8)
        and ids.dtype in (torch.int32, torch.int64)
        and ids.device == hidden.device
        and ids.is_contiguous()
        and num_experts == 256
        and block_size in (8, 16, 32, 48, 64)
        and expert_map is None
    ):
        return _fp8_e4m3_qdq_align_fallback(
            hidden, ids, block_size, num_experts, expert_map
        )
    _prewarm_fp8_preparation(hidden, ids)
    n = ids.numel()
    groups = triton.next_power_of_2(triton.cdiv(n, histogram_block))
    max_sorted = n + num_experts * (block_size - 1)
    max_blocks = triton.cdiv(max_sorted, block_size)
    output = torch.empty_like(hidden)
    hist = torch.empty((groups, num_experts), device=ids.device, dtype=torch.int32)
    counters = torch.empty(num_experts, device=ids.device, dtype=torch.int32)
    sorted_ids = torch.empty(max_sorted, device=ids.device, dtype=torch.int32)
    expert_ids = torch.empty(max_blocks, device=ids.device, dtype=torch.int32)
    total = torch.empty(1, device=ids.device, dtype=torch.int32)
    _fp8_qdq_partial_histogram[(groups + 1 + rows,)](
        hidden,
        output,
        ids,
        hist,
        counters,
        sorted_ids,
        expert_ids,
        n,
        max_sorted,
        max_blocks,
        E=num_experts,
        B=block_size,
        BG=groups,
        BH=histogram_block,
        FULL=n == groups * histogram_block,
        num_warps=first_warps,
        num_stages=1,
    )
    _fp8_qdq_partial_scatter[(triton.cdiv(n, scatter_block),)](
        ids,
        hist,
        counters,
        sorted_ids,
        expert_ids,
        total,
        n,
        E=num_experts,
        B=block_size,
        BG=groups,
        BS=scatter_block,
        FULL=n % scatter_block == 0,
        num_warps=second_warps,
        num_stages=1,
    )
    return output, sorted_ids, expert_ids, total


def _fp8_e4m3_qdq_align_prepared_fallback(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare QDQ activations and Marlin's padded expert indices together."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    rows = hidden_states.shape[0]
    if rows == 0:
        quantized = hidden_states
        empty_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
        total = torch.zeros(1, dtype=torch.int32, device=hidden_states.device)
        return quantized, empty_ids, empty_ids, total
    supported = (
        hidden_states.ndim == 2
        and hidden_states.shape[1] == 2048
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and hidden_states.is_cuda
        and hidden_states.is_contiguous()
        and 0 < rows <= 2048
        and topk_ids.shape == (rows, 8)
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_ids.device == hidden_states.device
        and topk_ids.is_contiguous()
        and num_experts == 256
        and block_size in (8, 16, 32, 48, 64)
        and expert_map is None
    )
    if not supported:
        quantized = hidden_states
        return quantized, *moe_align_block_size(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

    quantized = hidden_states
    numel = topk_ids.numel()
    max_sorted = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_sorted = min(max_sorted, numel * block_size)
    max_blocks = triton.cdiv(max_sorted, block_size)
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=topk_ids.device)
    total = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    if rows <= 128:
        _fp8_qdq_align_small[(num_experts,)](
            hidden_states,
            quantized,
            topk_ids,
            sorted_ids,
            expert_ids,
            total,
            N=numel,
            FULL=numel == triton.next_power_of_2(numel),
            E=num_experts,
            B=block_size,
            C=2048,
            BN=triton.next_power_of_2(numel),
            BC=2048,
            MAX_SORT=max_sorted,
            MAX_BLOCKS=max_blocks,
            BP=triton.next_power_of_2(block_size),
            BM=triton.next_power_of_2(triton.cdiv(numel, block_size)),
            num_warps=4,
            num_stages=1,
        )
    else:
        counts = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
        _fp8_qdq_prepare_metadata[(2,)](
            hidden_states,
            quantized,
            topk_ids,
            counts,
            sorted_ids,
            expert_ids,
            total,
            N=numel,
            FULL=numel == triton.next_power_of_2(numel),
            E=num_experts,
            B=block_size,
            C=2048,
            BN=triton.next_power_of_2(numel),
            BC=2048,
            MAX_SORT=max_sorted,
            MAX_BLOCKS=max_blocks,
            num_warps=8,
            num_stages=1,
        )
        _fp8_qdq_sort_tokens[(triton.cdiv(numel, 256),)](
            topk_ids,
            counts,
            sorted_ids,
            N=numel,
            E=num_experts,
            num_warps=8,
            num_stages=1,
        )
    return quantized, sorted_ids, expert_ids, total


def fp8_e4m3_qdq_align_prepared(
    hidden,
    ids,
    block_size,
    num_experts,
    expert_map=None,
):
    histogram_block, scatter_block = 1024, 256
    first_warps = second_warps = 4
    rows = hidden.shape[0]
    if not (
        hidden.ndim == 2
        and hidden.shape[1] == 2048
        and hidden.dtype in (torch.float16, torch.bfloat16)
        and hidden.is_cuda
        and hidden.is_contiguous()
        and 512 < rows <= 2048
        and ids.shape == (rows, 8)
        and ids.dtype in (torch.int32, torch.int64)
        and ids.device == hidden.device
        and ids.is_contiguous()
        and num_experts == 256
        and block_size in (8, 16, 32, 48, 64)
        and expert_map is None
    ):
        return _fp8_e4m3_qdq_align_prepared_fallback(
            hidden, ids, block_size, num_experts, expert_map
        )
    if rows < 2048 and ids.dtype == torch.int32:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        # The norm epilogue already applied QDQ. Reuse native CUDA routing
        # without another activation copy or quantization round trip.
        return hidden, *moe_align_block_size(
            ids, block_size, num_experts, ignore_invalid_experts=True
        )
    from vllm.model_executor.layers.fused_moe.experts.sm80_prepared_routing import (
        compiled_prepare,
    )

    prepared = compiled_prepare(ids, block_size)
    if prepared is not None:
        return hidden, *prepared
    _prewarm_fp8_preparation(hidden, ids)
    n = ids.numel()
    groups = triton.next_power_of_2(triton.cdiv(n, histogram_block))
    max_sorted = n + num_experts * (block_size - 1)
    max_blocks = triton.cdiv(max_sorted, block_size)
    output = hidden
    hist = torch.empty((groups, num_experts), device=ids.device, dtype=torch.int32)
    counters = torch.empty(num_experts, device=ids.device, dtype=torch.int32)
    sorted_ids = torch.empty(max_sorted, device=ids.device, dtype=torch.int32)
    expert_ids = torch.empty(max_blocks, device=ids.device, dtype=torch.int32)
    total = torch.empty(1, device=ids.device, dtype=torch.int32)
    _fp8_qdq_partial_histogram[(groups + 1,)](
        hidden,
        output,
        ids,
        hist,
        counters,
        sorted_ids,
        expert_ids,
        n,
        max_sorted,
        max_blocks,
        E=num_experts,
        B=block_size,
        BG=groups,
        BH=histogram_block,
        FULL=n == groups * histogram_block,
        num_warps=first_warps,
        num_stages=1,
    )
    _fp8_qdq_partial_scatter[(triton.cdiv(n, scatter_block),)](
        ids,
        hist,
        counters,
        sorted_ids,
        expert_ids,
        total,
        n,
        E=num_experts,
        B=block_size,
        BG=groups,
        BS=scatter_block,
        FULL=n % scatter_block == 0,
        num_warps=second_warps,
        num_stages=1,
    )
    return output, sorted_ids, expert_ids, total
