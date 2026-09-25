# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 FP8 G32 execution with original floating-point scales."""

import torch

from vllm import _custom_ops as ops  # noqa: F401
from vllm.utils.torch_utils import direct_register_custom_op


def is_g32(weight, activation, *, weight_bits: int) -> bool:
    if weight is None or activation is None:
        return False
    return (
        weight.num_bits == weight_bits
        and weight.type == ("float" if weight_bits == 8 else "int")
        and weight.strategy == "group"
        and weight.group_size == 32
        and weight.symmetric
        and not weight.dynamic
        and weight.actorder is None
        and weight.block_structure is None
        and activation.num_bits == 8
        and activation.type == "float"
        and activation.strategy == "group"
        and activation.group_size == 32
        and activation.symmetric
        and activation.dynamic
        and activation.block_structure is None
    )


def validate_partition(global_k: int, local_k: int, rank: int, name: str) -> None:
    offset = 0 if global_k == local_k else rank * local_k
    if (
        min(global_k, local_k) <= 0
        or rank < 0
        or global_k % local_k
        or offset + local_k > global_k
        or any(v % 32 for v in (global_k, local_k, offset))
    ):
        raise ValueError(
            f"{name}: G32 requires whole original K groups: rank={rank}, "
            f"global_K={global_k}, K_offset={offset}, local_K={local_k}"
        )


def require_sm90() -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or not current_platform.is_device_capability(90):
        raise ValueError("The cutlass_g32 backend requires an SM90 CUDA device")
    required = (
        ("h20_linear_g32", "gemm"),
        ("h20_linear_nt32_g32", "gemm"),
        ("h20_quant_g32", "quant"),
        ("_moe_C_exact", "h20_w4a8_cutlass_groupwise_mm"),
        ("_moe_C_gemmfollowup", "h20_w4a8_cutlass_groupwise_mm"),
        ("_moe_C", "h20_fp8_group_quant"),
    )
    for namespace, name in required:
        if not hasattr(getattr(torch.ops, namespace), name):
            raise RuntimeError(f"Missing SM90 G32 CUDA kernel: {namespace}.{name}")


_tail_streams: dict[int, torch.cuda.Stream] = {}


def prepare_linear_stream(device: torch.device) -> None:
    """Initialize reusable auxiliary stream before graph capture."""
    index = device.index
    assert index is not None
    if index not in _tail_streams:
        with torch.cuda.device(device):
            _tail_streams[index] = torch.cuda.Stream(device=device)


def _linear(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    shape = x.shape[:-1]
    k, n = weight.shape[1], weight.shape[0]
    if x.dtype != torch.bfloat16 or x.shape[-1] != k:
        raise ValueError("SM90 G32 Linear requires BF16 input with matching K")
    x = x.reshape(-1, k).contiguous()
    m = x.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if m:
        q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
        scale = torch.empty_strided(
            (m, k // 32), (1, m), device=x.device, dtype=torch.float32
        )
        # Split-K is restricted to small M; reserve its disjoint FP32 partials.
        scratch = torch.empty(
            65536 + 16 * min(m, 16) * n * 4, device=x.device, dtype=torch.uint8
        )
        torch.ops.h20_quant_g32.quant(x, q, scale, 1e-12)
        if n == 6144 and k == 2048 and m == 33:
            # Measured main-first overlap: 32-row head plus a resident tail.
            # Both scales keep the full matrix leading stride and original bits.
            prepare_linear_stream(x.device)
            main = torch.cuda.current_stream(x.device)
            tail = _tail_streams[x.device.index]
            tail_scratch = torch.empty(65536, device=x.device, dtype=torch.uint8)
            tail.wait_stream(main)
            torch.ops.h20_linear_nt32_g32.gemm(
                q[:32], weight, scale[:32], scales, out[:32], scratch, False
            )
            with torch.cuda.stream(tail):
                torch.ops.h20_linear_g32.gemm(
                    q[32:], weight, scale[32:], scales, out[32:], tail_scratch, 101
                )
            main.wait_stream(tail)
        elif n == 6144 and k == 2048 and 33 < m <= 128:
            torch.ops.h20_linear_nt32_g32.gemm(
                q, weight, scale, scales, out, scratch, True
            )
        else:
            torch.ops.h20_linear_g32.gemm(q, weight, scale, scales, out, scratch, 0)
    return out.reshape(*shape, n)


def _linear_fake(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="sm90_g32_linear",
    op_func=_linear,
    mutates_args=[],
    fake_impl=_linear_fake,
)
