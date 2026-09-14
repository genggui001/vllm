# SPDX-License-Identifier: Apache-2.0
# CP dispatch and allocation contracts follow FlashInfer's gdn_prefill.py.
"""vLLM adapter for FlashInfer GDN with H20 CP argument reuse."""

import math

import torch


def chunk_gated_delta_rule(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
):
    """Apply the vLLM GDN contract, preserving FlashInfer's CP heuristic.

    Args:
        q: Contiguous normalized queries, shaped [tokens, query_heads, dim].
        k: Contiguous normalized keys, shaped [tokens, key_heads, dim].
        v: Contiguous values, shaped [tokens, value_heads, dim].
        g: FP32 forget gate after exponentiation.
        beta: FP32 update gate.
        initial_state: Optional packed FP32 recurrent state.
        output_final_state: Whether to return the final state with the output.
        cu_seqlens: Device cumulative sequence lengths.

    Returns:
        Output, or (output, final_state) when output_final_state is true.
    """
    import flashinfer.gdn_prefill as flashinfer_gdn

    original = flashinfer_gdn.chunk_gated_delta_rule

    arguments = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    if (
        not q.is_cuda
        or q.ndim != 3
        or q.shape[0] <= 0
        or q.shape[1:] != (8, 128)
        or q.dtype != torch.bfloat16
        or "H20" not in torch.cuda.get_device_name(q.device).split()
        or torch.cuda.get_device_capability(q.device) != (9, 0)
        or getattr(flashinfer_gdn, "cp_delta_rule_dsl_sm90", None) is None
        or not callable(getattr(flashinfer_gdn, "should_use_cp_host", None))
        or cu_seqlens is None
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
    ):
        return original(**arguments)

    tokens = q.shape[0]
    sequences = cu_seqlens.numel() - 1
    for tensor, shape, dtype in (
        (q, (tokens, 8, 128), torch.bfloat16),
        (k, (tokens, 8, 128), torch.bfloat16),
        (v, (tokens, 16, 128), torch.bfloat16),
        (g, (tokens, 16), torch.float32),
        (beta, (tokens, 16), torch.float32),
        (cu_seqlens, (sequences + 1,), cu_seqlens.dtype),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return original(**arguments)
    state_shape = (sequences, 16, 128, 128)
    if initial_state is not None and (
        tuple(initial_state.shape) != state_shape
        or initial_state.dtype != torch.float32
        or initial_state.device != q.device
        or not initial_state.is_contiguous()
    ):
        return original(**arguments)

    if not flashinfer_gdn.should_use_cp_host(
        sequences * 16,
        flashinfer_gdn.get_device_sm_count(q.device),
        torch.cuda.get_device_properties(q.device).name,
    ):
        return original(**arguments)

    from vllm.model_executor.layers.mamba.ops.h20_cp_prepare import cp_prefill

    output = torch.empty((tokens, 16, 128), dtype=q.dtype, device=q.device)
    state = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    cp_prefill(
        output,
        state,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens.to(torch.int64),
        1.0 / math.sqrt(128),
        initial_state=initial_state,
        max_seqlen=tokens,
    )
    if output_final_state:
        return output, state
    return output
