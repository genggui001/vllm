# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_fp8_qdq_moe import (
    fp8_e4m3_per_token_qdq,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_e4m3_per_token_qdq_matches_qat_formula(dtype: torch.dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FP8 conversion")

    torch.manual_seed(0)
    x = torch.randn((37, 512), device="cuda", dtype=dtype)
    x = (x.float() * torch.logspace(-4, 3, 37, device="cuda").unsqueeze(1)).to(dtype)
    x[0].zero_()

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    expected_scales = x.abs().amax(dim=1, keepdim=True).float() / fp8_max
    expected_scales = expected_scales.clamp_min(torch.finfo(torch.float32).tiny)
    expected_q = (x / expected_scales).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    expected = (expected_q.float() * expected_scales).to(dtype)

    actual = fp8_e4m3_per_token_qdq(x)

    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
