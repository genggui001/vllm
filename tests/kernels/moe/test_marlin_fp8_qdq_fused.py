# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_fp8_qdq_fused_moe import (
    fp8_e4m3_per_token_qdq_fused,
    silu_mul_fp8_e4m3_per_token_qdq_fused,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_fp8_qdq_moe import (
    fp8_e4m3_per_token_qdq,
)


def _require_sm80() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("This backend targets SM80")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 512), (37, 2560), (128, 896)])
def test_fused_per_token_qdq_matches_reference(dtype: torch.dtype, shape):
    _require_sm80()
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    row_scales = torch.logspace(-4, 3, shape[0], device="cuda").unsqueeze(1)
    x = (x.float() * row_scales).to(dtype)
    x[0].zero_()

    expected = fp8_e4m3_per_token_qdq(x)
    actual = fp8_e4m3_per_token_qdq_fused(x)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 512), (37, 2560), (128, 896)])
def test_fused_swiglu_qdq_matches_reference(dtype: torch.dtype, shape):
    _require_sm80()
    torch.manual_seed(1)
    rows, n_cols = shape
    x = torch.randn((rows, n_cols * 2), device="cuda", dtype=dtype)
    expected_activation = torch.empty(shape, device="cuda", dtype=dtype)
    torch.ops._C.silu_and_mul(expected_activation, x)
    expected = fp8_e4m3_per_token_qdq(expected_activation)

    actual = torch.empty_like(expected)
    silu_mul_fp8_e4m3_per_token_qdq_fused(x, actual)

    assert torch.equal(actual, expected)
