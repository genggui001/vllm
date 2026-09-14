# SPDX-License-Identifier: Apache-2.0
# Host layouts and launch contracts adapted from FlashInfer's
# gdn_kernels/delta_rule_dsl/delta_rule_cp_sm90.py (Apache-2.0).
"""Call-scoped argument reuse for the existing FlashInfer H20 CP kernels.

The four GPU kernels and their launch arguments remain unchanged.
Tensor descriptors and scalar wrappers are shared only within one invocation.
"""

from functools import lru_cache

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm90 as cp


@lru_cache(maxsize=2)
def _kernels(has_initial_state):
    return (
        cp.CPDeltaRuleTPrecomputeSm90(cutlass.BFloat16),
        cp.CPDeltaRuleMNPrecomputeSm90(cutlass.BFloat16),
        cp.CPDeltaRuleFixupSimtSm90(has_initial_state, 8),
        cp.CPDeltaRulePrefillSm90(
            cutlass.BFloat16, needs_initial_state=has_initial_state
        ),
    )


def _descriptor(tensor, alignment, leading_dim=None):
    result = cute.runtime.from_dlpack(
        tensor, assumed_align=alignment, enable_tvm_ffi=True
    )
    if leading_dim is None:
        return result.mark_layout_dynamic()
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def eligible(o, state, q, k, v, alpha, beta, cu_seqlens, initial_state):
    """Check the supported TP2 GDN layout on H20."""
    if q.ndim != 3 or q.shape[0] <= 0 or q.shape[1:] != (8, 128):
        return False
    if not q.is_cuda or "H20" not in torch.cuda.get_device_name(q.device).split():
        return False
    if q.dtype != torch.bfloat16 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        return False
    tokens = q.shape[0]
    sequences = cu_seqlens.numel() - 1
    for tensor, shape, dtype in (
        (q, (tokens, 8, 128), torch.bfloat16),
        (k, (tokens, 8, 128), torch.bfloat16),
        (v, (tokens, 16, 128), torch.bfloat16),
        (o, (tokens, 16, 128), torch.bfloat16),
        (alpha, (tokens, 16), torch.float32),
        (beta, (tokens, 16), torch.float32),
        (state, (sequences, 16, 128, 128), torch.float32),
        (cu_seqlens, (sequences + 1,), torch.int64),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return False
    return not (
        initial_state is not None
        and (
            initial_state.shape != state.shape
            or initial_state.dtype != torch.float32
            or initial_state.device != q.device
            or not initial_state.is_contiguous()
        )
    )


def cp_prefill(
    o,
    state,
    q,
    k,
    v,
    alpha,
    beta,
    cu_seqlens,
    scale,
    *,
    initial_state=None,
    max_seqlen=None,
    cp_chunk_len=None,
    cp_chunk_len_granularity=cp.CP_CHUNK_LEN_GRANULARITY,
):
    """Run the original four native GDN kernels with shared host arguments."""
    if not eligible(o, state, q, k, v, alpha, beta, cu_seqlens, initial_state):
        return cp.cp_delta_rule_dsl_sm90(
            o,
            state,
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            scale,
            initial_state=initial_state,
            max_seqlen=max_seqlen,
            cp_chunk_len=cp_chunk_len,
            cp_chunk_len_granularity=cp_chunk_len_granularity,
        )
    total_seqlen = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1
    if max_seqlen is None and num_seqs == 1:
        max_seqlen = total_seqlen
    if max_seqlen is None or max_seqlen <= 0:
        return cp.cp_delta_rule_dsl_sm90(
            o,
            state,
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            scale,
            initial_state=initial_state,
            max_seqlen=max_seqlen,
            cp_chunk_len=cp_chunk_len,
            cp_chunk_len_granularity=cp_chunk_len_granularity,
        )
    if cp_chunk_len is None:
        cp_chunk_len = cp.choose_cp_chunk_len_host(
            max_seqlen,
            16,
            cp.get_device_sm_count(q.device),
            chunk_len_granularity=cp_chunk_len_granularity,
            device_capability=torch.cuda.get_device_capability(q.device),
            total_seqlen=total_seqlen,
            device_name=torch.cuda.get_device_properties(q.device).name,
        )
    if cp_chunk_len <= 0 or cp_chunk_len % 64 != 0:
        return cp.cp_delta_rule_dsl_sm90(
            o,
            state,
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            scale,
            initial_state=initial_state,
            max_seqlen=max_seqlen,
            cp_chunk_len=cp_chunk_len,
            cp_chunk_len_granularity=cp_chunk_len_granularity,
        )

    total_t_blocks = cp.workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    total_cp_chunks = cp.workspace_num_chunks_host(
        cu_seqlens, cp_chunk_len, total_seqlen
    )
    max_t_blocks = cp.max_num_chunks_host(max_seqlen, 64)
    max_cp_chunks = cp.max_num_chunks_host(max_seqlen, cp_chunk_len)
    t = torch.empty((total_t_blocks, 16, 64, 64), dtype=q.dtype, device=q.device)
    transfer = torch.empty(
        (total_cp_chunks, 16, 128, 128), dtype=torch.float32, device=q.device
    )
    local_state = torch.empty_like(transfer)
    fixed_state = torch.empty_like(local_state)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    kernels = _kernels(initial_state is not None)
    options = (cute.GPUArch("sm_90a"),)

    # Identical dynamic layouts and alignment assumptions to the four wrappers.
    k_view = k.as_strided((128, total_seqlen, 8), (1, 8 * 128, 128))
    v_view = v.as_strided((128, total_seqlen, 16), (1, 16 * 128, 128))
    t_view = t.as_strided((64, 64, 16, total_t_blocks), (64, 1, 4096, 16 * 4096))
    dk = _descriptor(k_view, 16, 0)
    d_beta = _descriptor(beta.reshape(-1), 16)
    dt_flat = _descriptor(t.view(-1), 128)
    dcu = _descriptor(cu_seqlens, 8)
    nk, nv, nh = cutlass.Int32(8), cutlass.Int32(16), cutlass.Int32(16)
    nt, mt = cutlass.Int32(total_t_blocks), cutlass.Int32(max_t_blocks)
    nc, mc = cutlass.Int32(total_cp_chunks), cutlass.Int32(max_cp_chunks)
    ns, chunk = cutlass.Int32(num_seqs), cutlass.Int32(cp_chunk_len)
    args = (dk, d_beta, dt_flat, dcu, nk, nh, nt, mt, ns, stream)
    cp.cached_compile(kernels[0], *args, compile_options=options)(*args)

    dv = _descriptor(v_view, 16, 0)
    dt = _descriptor(t_view, 16, 1)
    da = _descriptor(alpha.view(-1), 16)
    dtransfer_flat = _descriptor(transfer.view(-1), 16)
    dlocal_flat = _descriptor(local_state.view(-1), 16)
    args = (
        dk,
        dv,
        dt,
        da,
        dtransfer_flat,
        dlocal_flat,
        dcu,
        chunk,
        nk,
        nv,
        nh,
        nc,
        mc,
        ns,
        stream,
    )
    cp.cached_compile(kernels[1], *args, compile_options=options)(*args)

    transfer_view = transfer.as_strided(
        (128, 128, 16, total_cp_chunks), (128, 1, 16384, 16 * 16384)
    )
    local_view = local_state.as_strided(
        (128, 128, 16, total_cp_chunks), (128, 1, 16384, 16 * 16384)
    )
    dtransfer = _descriptor(transfer_view, 128, 1)
    dlocal = _descriptor(local_view, 128, 1)
    dinitial = (
        _descriptor(initial_state.reshape(-1), 16)
        if initial_state is not None
        else None
    )
    dfixed128 = _descriptor(fixed_state.reshape(-1), 128)
    args = (dtransfer, dlocal, dinitial, dfixed128, dcu, chunk, nc, ns, nh, stream)
    cp.cached_compile(kernels[2], *args, compile_options=options)(*args)

    q_view = q.as_strided((total_seqlen, 128, 8), (8 * 128, 1, 128))
    o_view = o.as_strided((128, total_seqlen, 16), (1, 16 * 128, 128))
    # A separate cache key per stream avoids aliasing a different stream's
    # temporary TMA descriptor storage. The buffer is only ~16 KiB on H20.
    stream_id = torch.cuda.current_stream(q.device).cuda_stream
    maps = cp._get_cache_buf(
        f"h20_cp_prefill_tensormaps_{stream_id}",
        cp.get_device_sm_count(q.device) * 128,
        q.device,
    )
    dq = _descriptor(q_view, 16, 1)
    do = _descriptor(o_view, 16, 0)
    dstate = _descriptor(state.reshape(-1), 16)
    dfixed16 = _descriptor(fixed_state.reshape(-1), 16)
    dmaps = _descriptor(maps, 128)
    args = (
        dq,
        dk,
        dv,
        dt,
        do,
        da,
        dstate,
        dfixed16,
        dinitial,
        dmaps,
        dcu,
        cutlass.Float32(scale),
        nk,
        nk,
        nv,
        nh,
        chunk,
        nc,
        mc,
        ns,
        stream,
    )
    cp.cached_compile(kernels[3], *args, compile_options=options)(*args)
