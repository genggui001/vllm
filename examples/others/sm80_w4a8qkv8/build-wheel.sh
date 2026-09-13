#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Run with a prepared CUDA/PyTorch build environment activated.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-12}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.29.0+sm80w4a8qkv8}"
export VLLM_TARGET_DEVICE=cuda

# Conda compiler activation can inject an absolute RPATH into linker flags.
# Replace those flags and keep wheel runtime lookups relative to site-packages.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--as-needed -DCMAKE_MODULE_LINKER_FLAGS=-Wl,--as-needed -DCMAKE_EXE_LINKER_FLAGS=-Wl,--as-needed -DCMAKE_INSTALL_RPATH=\$ORIGIN/../torch/lib;\$ORIGIN/../../torch/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF"
uv build --wheel --no-build-isolation --out-dir "${WHEEL_OUTPUT_DIR:-dist}"
# Some compiler wrappers append their own RPATH after CMake's linker flags.
uv run --no-project --with patchelf==0.17.2.4 --with wheel==0.45.1 \
    python examples/others/sm80_w4a8qkv8/repair-wheel.py "${WHEEL_OUTPUT_DIR:-dist}"
