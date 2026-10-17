# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 Dense FP8 G32 contract and lossless, load-time weight preparation.

The CUDA operator owns shape dispatch. There is no BF16 GEMM fallback here.
Activation producers pass codes and scales explicitly, never through mutable
layer state (shared experts may execute on a different CUDA stream).
"""

from typing import NamedTuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.sm80_group32_qdq import (
    group32_fp8_quant,
)


class Group32FP8Input(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor


def quantize_input(x: torch.Tensor, *, swiglu: bool = False) -> Group32FP8Input:
    codes, scales, _ = group32_fp8_quant(x, swiglu=swiglu)
    return Group32FP8Input(codes, scales)


def validate_dense_shape(n: int, k: int, max_tokens: int) -> None:
    padded_n = 64 if n <= 32 else ((n + 127) // 128) * 128
    if (
        n <= 0
        or k < 128
        or k > 65536
        or k % 128
        or max_tokens < 0
        or max_tokens > 524280
        or max(padded_n * k, max_tokens * k, max_tokens * padded_n) > 2147483647
    ):
        raise ValueError(
            f"Unsupported SM80 Dense FP8 shape N={n}, K={k}, "
            f"max_tokens={max_tokens}: K must be a multiple of 128 in "
            "[128, 65536], with supported 32-bit indexing. BF16 rollback "
            "is disabled; change the model/TP/scheduler configuration."
        )


def prepare_weight(w: torch.Tensor, scales: torch.Tensor):
    """Pack original E4M3 bytes and FP32 scales without requantizing either."""
    n, k = w.shape
    validate_dense_shape(n, k, 0)
    if not hasattr(torch.ops._C, "sm80_fp8_g32_gemm"):
        raise RuntimeError(
            "This vLLM extension lacks the SM80 Dense FP8 kernel; rebuild "
            "the candidate extension before starting. BF16 rollback is disabled."
        )
    if (
        not w.is_cuda
        or torch.cuda.get_device_capability(w.device) != (8, 0)
        or w.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float32
        or scales.device != w.device
        or scales.shape != (n, k // 32)
    ):
        raise ValueError("SM80 Dense requires CUDA E4M3 W and FP32 G32 scales")
    if not torch.isfinite(w.float()).all().item():
        raise ValueError("Dense FP8 weights contain nonfinite E4M3 codes")
    if not (torch.isfinite(scales) & (scales > 0)).all().item():
        raise ValueError("Dense FP8 scales must be positive finite FP32 values")
    pn = 64 if n <= 32 else ((n + 127) // 128) * 128
    wp = torch.zeros((pn, k), device=w.device, dtype=w.dtype)
    sp = torch.ones((pn, k // 32), device=w.device, dtype=torch.float32)
    wp[:n].copy_(w)
    sp[:n].copy_(scales)
    packed = ops.gptq_marlin_repack(
        wp.view(torch.int32).T.contiguous(),
        torch.empty(0, device=w.device, dtype=torch.int32),
        k,
        pn,
        8,
        False,
    )
    order = [i + 8 * j for i in range(8) for j in range(8)]
    packed_scales = sp.T.contiguous().reshape(-1, 64)[:, order]
    packed_scales = packed_scales.reshape(k // 32, pn).contiguous()
    auxiliary = torch.empty(0, device=w.device, dtype=torch.int32)
    if n <= 32 and k == 2048:
        words = packed.reshape(-1, 8)
        if torch.count_nonzero(words[:, 4:]).item():
            raise ValueError("Nonzero padding in the compact BA weight layout")
        auxiliary = words[:, :4].contiguous().reshape(k // 16, pn * 2)
    elif (n, k) in ((512, 2048), (1024, 4096)):
        tile_n = 128 if n == 512 else 256
        records = packed.reshape(k // 32, 2, pn // tile_n, tile_n * 4)
        records = records.permute(0, 2, 1, 3).reshape(k // 32, pn // tile_n, tile_n * 8)
        scale_records = packed_scales.view(torch.int32).reshape(
            k // 32, pn // tile_n, tile_n
        )
        auxiliary = torch.cat((records, scale_records), dim=2).contiguous()
    return packed, packed_scales, auxiliary


def dense_fp8(x: Group32FP8Input, w, sw, auxiliary, n: int):
    # The kernel returns padded physical N. Slicing retains valid row strides,
    # including N=32 BA, and does not add an output-copy kernel.
    return torch.ops._C.sm80_fp8_g32_gemm(x.codes, x.scales, w, sw, auxiliary, n)[:, :n]
