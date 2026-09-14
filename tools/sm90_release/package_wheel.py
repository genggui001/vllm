# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble a complete SM90 wheel from audited upstream files and rebuilt MoE.

Use the unmodified vLLM 0.28.0 installation as the source of unchanged native
libraries and vendored assets. The six release Python files and complete MoE
extension come from the release checkout and build. No installation is changed.
"""

import argparse
import base64
import csv
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from build_moe import sha, write
from packaging.utils import parse_wheel_filename
from wheel.wheelfile import WheelFile

PYTHON_FILES = (
    "vllm/_custom_ops.py",
    "vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py",
    "vllm/model_executor/layers/fused_moe/h20_fp8_prepare.py",
    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "vllm/model_executor/layers/mamba/ops/h20_cp_prepare.py",
    "vllm/model_executor/layers/mamba/ops/h20_gdn_prefill.py",
)
VERSION = "0.28.0+sm90w4a8qkv8.cu132"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, build, root = (
        args.source.resolve(),
        args.build.resolve(),
        args.output.resolve(),
    )
    assert sys.version_info[:2] == (3, 12), "This release is validated for CPython 3.12"
    assert not root.exists() and repo not in root.parents
    status = json.loads((build / "status.json").read_text())
    assert status["exit_code"] == 0 and not status["running"]
    library = Path(status["library"])
    assert sha(library) == status["library_sha256"]
    source_hashes = json.loads((build / "source-sha256.json").read_text())
    for rel, expected in source_hashes.items():
        assert sha(repo / rel) == expected, rel
    assert (
        subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip()
        == "v0.28.0-sm90w4a8qkv8"
    )
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo), (
        "Commit the release sources before packaging"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    distribution = importlib.metadata.distribution("vllm")
    assert distribution.version == "0.28.0", (
        "Use the unmodified upstream base distribution"
    )
    assert importlib.metadata.version("torch") == "2.13.0+cu132"
    base = Path(distribution.locate_file("")).resolve()
    old_info = "vllm-0.28.0.dist-info"
    new_info = f"vllm-{VERSION}.dist-info"
    record = distribution.read_text("RECORD")
    assert record is not None
    root.mkdir(parents=True)
    stage = root / "staging"
    stage.mkdir()
    base_hashes = {}
    skipped = []
    for rel, digest, size in csv.reader(record.splitlines()):
        path = Path(rel)
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            skipped.append(rel)
            continue
        if rel.startswith("../"):
            assert path.name == "vllm", rel
            skipped.append(rel)
            continue
        assert not path.is_absolute() and ".." not in path.parts
        assert rel.startswith("vllm/") or rel.startswith(old_info + "/"), rel
        original = base / rel
        assert original.is_file(), rel
        actual_sha = sha(original)
        if digest:
            algorithm, encoded = digest.split("=", 1)
            assert algorithm == "sha256"
            actual = (
                base64.urlsafe_b64encode(bytes.fromhex(actual_sha))
                .rstrip(b"=")
                .decode()
            )
            assert actual == encoded and original.stat().st_size == int(size), rel
        if rel.startswith(old_info + "/") and path.name in {
            "RECORD",
            "INSTALLER",
            "REQUESTED",
            "direct_url.json",
        }:
            skipped.append(rel)
            continue
        base_hashes[rel] = actual_sha
        # Upstream Python files outside the selected six must match this tag's
        # checkout, preventing accidental mixing of two vLLM versions.
        source = repo / rel
        if path.suffix == ".py" and source.is_file() and rel not in PYTHON_FILES:
            assert sha(source) == actual_sha, rel
        destination = stage / rel.replace(old_info + "/", new_info + "/", 1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        assert sha(destination) == actual_sha
    assert len(base_hashes) > 1000
    write(root / "base-distribution-sha256.json", base_hashes)
    for rel in PYTHON_FILES:
        assert sha(repo / rel) == source_hashes[rel], rel
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, target)
    native_rel = "vllm/_moe_C_stable_libtorch.abi3.so"
    assert (stage / native_rel).is_file()
    shutil.copy2(library, stage / native_rel)
    version_file = stage / "vllm/_version.py"
    version_file.write_text(
        "# Generated by the SM90 release builder.\n"
        f"__version__ = version = {VERSION!r}\n"
        "__version_tuple__ = version_tuple = (0, 28, 0, 'sm90w4a8qkv8.cu132')\n"
        f"__commit_id__ = commit_id = {commit!r}\n"
    )
    metadata = stage / new_info / "METADATA"
    text, replacements = re.subn(
        r"(?m)^Version: 0\.28\.0$", f"Version: {VERSION}", metadata.read_text()
    )
    assert replacements == 1
    metadata.write_text(text)
    (stage / new_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: vllm-sm90-release\n"
        "Root-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n"
    )
    native_hashes = {
        rel: digest
        for rel, digest in base_hashes.items()
        if rel.endswith(".so") and rel != native_rel
    }
    provenance = dict(
        version=VERSION,
        source_commit=commit,
        source_branch="v0.28.0-sm90w4a8qkv8",
        base_version=distribution.version,
        base_record_sha256=sha(base / old_info / "RECORD"),
        compiled_component="_moe_C_stable_libtorch",
        all_extensions_recompiled=False,
        rebuilt_library_sha256=sha(library),
        reused_native_libraries=native_hashes,
        python_source_sha256={rel: sha(repo / rel) for rel in PYTHON_FILES},
        torch="2.13.0+cu132",
        flashinfer="0.6.16.post3",
        cuda="13.2",
        specialization="H20 SM90a, native FP8 MoE; original FP8 FlashAttention 3",
        installed_environment_modified=False,
    )
    write(stage / "vllm/sm90_release.json", provenance)
    # These were development entry points, never runtime dependencies.
    assert not (
        stage / "vllm/model_executor/layers/fused_moe/fused_fp8_permute.py"
    ).exists()
    assert not (stage / "sitecustomize.py").exists()
    dist = root / "dist"
    dist.mkdir()
    wheel = dist / f"vllm-{VERSION}-cp312-cp312-linux_x86_64.whl"
    with WheelFile(str(wheel), "w") as archive:
        archive.write_files(stage)
    name, version, _, tags = parse_wheel_filename(wheel.name)
    assert name == "vllm" and str(version) == VERSION
    assert {str(t) for t in tags} == {"cp312-cp312-linux_x86_64"}
    unpacked = root / "wheel-test/site"
    unpacked.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.testzip() is None
        members = archive.namelist()
        assert len(members) == len(set(members))
        for member in members:
            path = Path(member)
            assert not path.is_absolute() and ".." not in path.parts
        archive.extractall(unpacked)
        for info in archive.infolist():
            if not info.is_dir():
                os.chmod(unpacked / info.filename, (info.external_attr >> 16) & 0o777)
    rows = list(csv.reader((unpacked / new_info / "RECORD").read_text().splitlines()))
    assert {r[0] for r in rows} == set(members)
    for rel, digest, size in rows:
        path = unpacked / rel
        if rel.endswith(".dist-info/RECORD"):
            assert not digest and not size
            continue
        algorithm, expected = digest.split("=", 1)
        assert algorithm == "sha256"
        actual = (
            base64.urlsafe_b64encode(bytes.fromhex(sha(path))).rstrip(b"=").decode()
        )
        assert actual == expected and path.stat().st_size == int(size), rel
    for rel, expected in native_hashes.items():
        assert sha(unpacked / rel) == expected, rel
    assert sha(unpacked / native_rel) == sha(library)
    write(
        root / "package-status.json",
        dict(
            complete=True,
            wheel=str(wheel),
            sha256=sha(wheel),
            bytes=wheel.stat().st_size,
            source_commit=commit,
            record_entries=len(rows),
            record_verified=True,
            upstream_files_checked=len(base_hashes),
            reused_native_libraries=len(native_hashes),
            rebuilt_moe_sha256=sha(library),
            unpacked_site=str(unpacked),
            installed_environment_modified=False,
            runtime_validation_pending=True,
            skipped_installer_files=skipped,
        ),
    )
    print(
        json.dumps(
            dict(wheel=str(wheel), sha256=sha(wheel), unpacked_site=str(unpacked))
        )
    )


if __name__ == "__main__":
    main()
