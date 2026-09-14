# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 FP8 conversion and activation quantization with CUDA rounding."""

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice


@triton.jit
def _fp32_to_e4m3_rn(x):
    # Avoid the generic SM89 cast's intermediate FP16 rounding.
    return tl.inline_asm_elementwise(
        """{
        .reg .b16 lo, hi;
        cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
        cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
        mov.b32 $0, {lo, hi};
        }""",
        constraints="=r,f,f,f,f",
        args=[x],
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _silu_token_fp8(
    X, Q, S, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    gate = tl.load(X + row * STRIDE + col, col < N, other=0).to(tl.float32)
    up = tl.load(X + row * STRIDE + N + col, col < N, other=0).to(tl.float32)
    # SiluAndMul applies its runtime beta=+0, including to negative zero.
    up = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[up, 0.0],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # Preserve both dtype-rounding steps of the unfused CUDA activation.
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    silu = silu.to(X.dtype.element_ty).to(tl.float32)
    value = (silu * up).to(X.dtype.element_ty).to(tl.float32)
    scale = tl.maximum(
        tl.div_rn(tl.max(tl.abs(value), 0), 448.0), 1.0 / (448.0 * 512.0)
    )
    normalized = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
    tl.store(Q + row * N + col, _fp32_to_e4m3_rn(normalized), col < N)
    tl.store(S + row, scale)


def silu_and_mul_token_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse CUDA-equivalent SiLU gating and dynamic per-token E4M3 quantization."""
    assert x.ndim == 2 and x.stride(-1) == 1 and x.shape[-1] % 2 == 0
    assert x.dtype in (torch.bfloat16, torch.float16)
    n = x.shape[-1] // 2
    assert n > 0
    q = torch.empty((x.shape[0], n), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    if x.shape[0]:
        _silu_token_fp8[(x.shape[0],)](
            x,
            q,
            scales,
            n,
            x.stride(0),
            triton.next_power_of_2(n),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, scales
