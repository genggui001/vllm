# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remove build-host RPATHs from a source-built SM80 release wheel."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    wheels = list(args.directory.resolve().glob("vllm-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Use an output directory containing exactly one vLLM wheel.")
    wheel = wheels[0]
    with tempfile.TemporaryDirectory(
        prefix="vllm-wheel-rpath-", dir=wheel.parent
    ) as temporary:
        root = Path(temporary)
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", str(root)],
            check=True,
        )
        unpacked = next(root.iterdir())
        for extension in sorted((unpacked / "vllm").rglob("*.so")):
            rpath = "$ORIGIN/../torch/lib:$ORIGIN/../../torch/lib"
            subprocess.run(
                ["patchelf", "--set-rpath", rpath, str(extension)], check=True
            )
            actual = subprocess.check_output(
                ["patchelf", "--print-rpath", str(extension)], text=True
            ).strip()
            if actual != rpath:
                raise RuntimeError(f"Unexpected RPATH in {extension.name}: {actual}")
        output = root / "repacked"
        output.mkdir()
        # wheel pack regenerates RECORD hashes for every modified native library.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(unpacked),
                "-d",
                str(output),
            ],
            check=True,
        )
        repacked = output / wheel.name
        if not repacked.is_file():
            raise RuntimeError("Repacking unexpectedly changed the wheel tags.")
        repacked.replace(wheel)


if __name__ == "__main__":
    main()
