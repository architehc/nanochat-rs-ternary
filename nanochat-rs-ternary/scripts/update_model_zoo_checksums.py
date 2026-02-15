#!/usr/bin/env python3
"""Update nanochat/model_zoo_checksums.json from local GGUF + mHC artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update model-zoo checksums for a published model artifact pair."
    )
    parser.add_argument("--model", required=True, help="Model registry key, e.g. nanochat-3b")
    parser.add_argument("--gguf", required=True, type=Path, help="Path to model.gguf")
    parser.add_argument("--mhc", required=True, type=Path, help="Path to model.mhc")
    parser.add_argument(
        "--checksums-file",
        type=Path,
        default=Path("nanochat/model_zoo_checksums.json"),
        help="Path to checksums JSON file (default: nanochat/model_zoo_checksums.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.gguf.is_file():
        raise SystemExit(f"GGUF file not found: {args.gguf}")
    if not args.mhc.is_file():
        raise SystemExit(f"mHC file not found: {args.mhc}")

    checksums_path = args.checksums_file
    checksums_path.parent.mkdir(parents=True, exist_ok=True)

    if checksums_path.exists():
        with checksums_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data[args.model] = {
        "sha256_gguf": sha256_file(args.gguf),
        "sha256_mhc": sha256_file(args.mhc),
    }

    with checksums_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Updated {checksums_path} for model '{args.model}'")
    print(f"sha256_gguf={data[args.model]['sha256_gguf']}")
    print(f"sha256_mhc={data[args.model]['sha256_mhc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
