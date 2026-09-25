# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original K-group boundaries and graph replay regressions for native G32."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.sm90_g32 import (
    _linear,
    validate_partition,
)


@pytest.mark.parametrize(
    "global_k,local_k,rank", [(2048, 2048, 1), (4096, 2048, 1), (512, 256, 1)]
)
def test_tp_keeps_complete_groups(global_k, local_k, rank):
    validate_partition(global_k, local_k, rank, "layer")


@pytest.mark.parametrize(
    "global_k,local_k,rank",
    [
        (0, 0, 0),
        (64, 16, 1),
        (96, 48, 1),
        (96, 64, 1),
        (2049, 2049, 0),
        (64, 32, 2),
        (64, 32, -1),
    ],
)
def test_tp_rejects_incomplete_groups(global_k, local_k, rank):
    with pytest.raises(ValueError, match="whole original K groups"):
        validate_partition(global_k, local_k, rank, "layer")


@pytest.mark.parametrize("m,n", [(1, 32), (33, 6144), (64, 6144)])
def test_dynamic_graph_preserves_bits(m, n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires compiled SM90 G32 extension and SM90 GPU")
    torch.manual_seed(971)
    x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, 2048, device="cuda") * 10).to(torch.float8_e4m3fn)
    scales = (torch.rand(n, 64, device="cuda") + 0.1).t().contiguous().t()
    for _ in range(3):
        _linear(x, w, scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _linear(x, w, scales)
    for _ in range(5):
        x.copy_(torch.randn_like(x))
        graph.replay()
        reference = _linear(x, w, scales)
        assert torch.isfinite(output).all()
        assert torch.equal(output, reference)
