# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the release's complete SM90 MoE extension with existing dependencies.

The output is isolated from the checkout and Python installation. Unchanged
upstream extensions are supplied separately when assembling the release wheel.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    repo, root = args.source.resolve(), args.output.resolve()
    assert args.jobs > 0 and not root.exists()
    assert root != repo and repo not in root.parents, (
        "Use a build directory outside the checkout"
    )
    root.mkdir(parents=True)
    toolkit = Path(sys.prefix)
    site = Path(sysconfig.get_paths()["purelib"])
    cutlass = site / "flashinfer/data/cutlass"
    assert (cutlass / "include/cutlass/cutlass.h").exists()
    assert importlib.metadata.version("torch").split("+")[0] == "2.13.0"
    assert importlib.metadata.version("flashinfer-python") == "0.6.16.post3"
    for tool in ["cmake", "ninja", "nvcc"]:
        assert (toolkit / "bin" / tool).is_file(), tool

    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo,
    ).split(b"\0")
    snapshot = root / "source"
    snapshot.mkdir()
    manifest = {}
    for item in sorted(set(filter(None, files))):
        rel = Path(os.fsdecode(item))
        assert not rel.is_absolute() and ".." not in rel.parts
        original = repo / rel
        if not original.is_file():
            continue
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        manifest[str(rel)] = sha(original)
        assert sha(target) == manifest[str(rel)]
    write(root / "source-sha256.json", manifest)
    text = (snapshot / "CMakeLists.txt").read_text()
    marker = "# For CUDA and HIP builds also build the triton_kernels external package."
    assert text.count(marker) == 1
    prefix, suffix = text.split(marker, 1)
    assert "cmake/h20_w4a8.cmake" in prefix
    assert "define_extension_target(" not in suffix
    view = root / "cmake-moe-only"
    view.mkdir()
    for child in snapshot.iterdir():
        if child.name != "CMakeLists.txt":
            (view / child.name).symlink_to(child, target_is_directory=child.is_dir())
    (view / "CMakeLists.txt").write_text(prefix + "\n# MoE component build.\n")

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("H20_") and k != "PYTHONPATH"
    }
    env.update(
        CUDA_HOME=str(toolkit),
        TORCH_CUDA_ARCH_LIST="9.0a",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_CUTLASS_SRC_DIR=str(cutlass),
        MAX_JOBS=str(args.jobs),
        CUDA_INC_PATH=str(toolkit / "targets/x86_64-linux"),
    )
    env["PATH"] = f"/usr/bin:/bin:{toolkit}/bin:" + env.get("PATH", "")
    for name, value in [
        ("CPATH", toolkit / "targets/x86_64-linux/include"),
        ("LIBRARY_PATH", toolkit / "lib"),
    ]:
        env[name] = str(value) + (":" + env[name] if env.get(name) else "")
    build = root / "build"
    cmake = str(toolkit / "bin/cmake")
    configure = [
        cmake,
        "-S",
        str(view),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={toolkit}/bin/ninja",
        "-DCMAKE_CXX_COMPILER=/usr/bin/c++",
        "-DCMAKE_CXX_FLAGS=-B/usr/bin",
        "-DCMAKE_CUDA_FLAGS=-Xcompiler=-B/usr/bin",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INCLUDE_PATH={toolkit}/targets/x86_64-linux/include",
        "-DVLLM_TARGET_DEVICE=cuda",
        f"-DVLLM_PYTHON_EXECUTABLE={sys.executable}",
        "-DVLLM_PYTHON_PATH=" + ":".join(sys.path),
        f"-DCMAKE_CUDA_COMPILER={toolkit}/bin/nvcc",
        f"-DVLLM_CUTLASS_SRC_DIR={cutlass}",
        f"-DCUTLASS_DIR={cutlass}",
        f"-DCUTLASS_INCLUDE_DIR={cutlass}/include",
        f"-DCUTLASS_TOOLS_UTIL_INCLUDE_DIR={cutlass}/tools/util/include",
        f"-DCUDA_TOOLKIT_ROOT_DIR={toolkit}",
        f"-DCUDA_TOOLKIT_INCLUDE={toolkit}/targets/x86_64-linux/include",
        f"-DCUDAToolkit_INCLUDE_DIR={toolkit}/targets/x86_64-linux/include",
        f"-DCUDA_CUDART_LIBRARY={toolkit}/lib/libcudart.so",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        f"-DFETCHCONTENT_BASE_DIR={root}/deps",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DNVCC_THREADS=1",
        "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
        f"-DCMAKE_JOB_POOLS:STRING=compile={args.jobs}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DCMAKE_INSTALL_RPATH=$ORIGIN/../torch/lib;$ORIGIN/../nvidia/cuda_runtime/lib",
        "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF",
    ]
    command = [
        cmake,
        "--build",
        str(build),
        "--target",
        "_moe_C_stable_libtorch",
        "--parallel",
        str(args.jobs),
    ]
    state = dict(
        running=True,
        phase="configure",
        started=time.time(),
        pid=os.getpid(),
        configure=configure,
        build_command=command,
        omp_num_threads=env.get("OMP_NUM_THREADS"),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        source_branch=subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip(),
        source_status=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ),
        source_manifest_sha256=sha(root / "source-sha256.json"),
        installed_environment_modified=False,
        all_extensions_recompiled=False,
    )
    write(root / "status.json", state)
    try:
        with (root / "configure.log").open("x") as log:
            subprocess.run(
                configure,
                env=env,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        entries = json.loads((build / "compile_commands.json").read_text())
        selected = [r for r in entries if "/moe/h20_w4a8/" in r["file"]]
        assert len(selected) == 14
        for row in selected:
            for flag in [
                "sm_90a",
                "ENABLE_FP8",
                "TORCH_TARGET_VERSION",
                "USE_CUDA",
                "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL",
            ]:
                assert flag in row["command"], (row["file"], flag)
            assert "--use_fast_math" not in row["command"]
        write(root / "selected-compile-commands.json", selected)
        state["phase"] = "compile"
        write(root / "status.json", state)
        with (root / "compile.log").open("x") as log:
            subprocess.run(
                command,
                env=env,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=18000,
            )
        libraries = list(build.glob("_moe_C_stable_libtorch*.so"))
        assert len(libraries) == 1
        library = libraries[0]
        dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True)
        (root / "elf-dynamic.txt").write_text(dynamic)
        for line in dynamic.splitlines():
            if "RPATH" in line or "RUNPATH" in line:
                assert (
                    "$ORIGIN" in line
                    and str(toolkit) not in line
                    and str(root) not in line
                )
        for name, expected in manifest.items():
            assert sha(snapshot / name) == sha(repo / name) == expected, name
        state.update(
            running=False,
            phase="complete",
            exit_code=0,
            library=str(library),
            library_sha256=sha(library),
            selected_cuda_sources=14,
        )
    except BaseException as error:
        state.update(running=False, exit_code=1, error=repr(error))
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["started"]
        write(root / "status.json", state)


if __name__ == "__main__":
    main()
