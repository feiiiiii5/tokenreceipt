"""Verify the pinned checkpoint, then convert it with the released CPU runtime."""
import argparse
import fcntl
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import ctranslate2
import torch


WHEEL_NAME = "ctranslate2-4.9.0-cp311-cp311-macosx_11_0_arm64.whl"
WHEEL_SHA256 = "78a0a59721031193e52732b1e565bfaee7c22ec0d1c48d40e9ed4bc7341a843e"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/41/6b/"
    "ca9fe0dbc7c4f76d57c1d759a23cf059c3e323effc29b2c21208908b5bb7/"
    + WHEEL_NAME
)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    if digest(args.wheel) != WHEEL_SHA256 or ctranslate2.__version__ != "4.9.0":
        raise ValueError("Wheel or installed engine version does not match the frozen release.")
    study = Path(__file__).resolve().parent
    freeze = json.loads((study / "freeze.json").read_text())
    manifest_sha256 = digest(args.manifest)
    if manifest_sha256 != freeze["model_manifest_sha256"]:
        raise ValueError("Model manifest differs from the frozen protocol.")
    original = json.loads(args.manifest.read_text())
    for filename, expected in original["sha256"].items():
        if digest(args.checkpoint / filename) != expected:
            raise ValueError(f"Original checkpoint hash mismatch for {filename}.")

    with Path("/tmp/native-engine-review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.set_num_threads(1)
        started = time.monotonic()
        converter = ctranslate2.converters.TransformersConverter(
            str(args.checkpoint), low_cpu_mem_usage=True
        )
        converter.convert(str(args.converted), quantization="float32")
        provenance = {
            "wheel": {"name": args.wheel.name, "sha256": WHEEL_SHA256, "source": WHEEL_URL,
                      "published_sha256": WHEEL_SHA256},
            "original_model": original,
            "model_manifest_sha256": manifest_sha256,
            "converted_float32_model": {
                path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                for path in sorted(args.converted.iterdir())
            },
            "packages": {
                package: version(package)
                for package in ("ctranslate2", "numpy", "torch", "transformers", "tokenizers",
                                "safetensors", "huggingface-hub")
            },
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            "conversion_seconds": time.monotonic() - started,
        }
        args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
