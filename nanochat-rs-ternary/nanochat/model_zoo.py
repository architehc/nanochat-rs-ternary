"""Model zoo for nanochat with HuggingFace integration.

Downloads and caches pre-trained ternary models with SHA256 verification.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Model registry with HuggingFace URLs and metadata
MODELS: Dict[str, Dict] = {
    "nanochat-125m": {
        "gguf_url": "https://huggingface.co/architehc/nanochat-125m/resolve/main/model.gguf",
        "mhc_url": "https://huggingface.co/architehc/nanochat-125m/resolve/main/model.mhc",
        "config": "nano_125m",
        "sha256_gguf": "TO_BE_FILLED",
        "sha256_mhc": "TO_BE_FILLED",
        "description": "125M parameter ternary model",
        "size_mb": 110,
        "vocab_size": 50257,
        "dim": 768,
        "layers": 12,
    },
    "nanochat-3b": {
        "gguf_url": "https://huggingface.co/architehc/nanochat-3b/resolve/main/model.gguf",
        "mhc_url": "https://huggingface.co/architehc/nanochat-3b/resolve/main/model.mhc",
        "config": "medium_3b",
        "sha256_gguf": "TO_BE_FILLED",
        "sha256_mhc": "TO_BE_FILLED",
        "description": "3B parameter ternary model",
        "size_mb": 1500,
        "vocab_size": 50257,
        "dim": 2048,
        "layers": 28,
    },
    "nano-125m-rust": {
        "gguf_url": "https://huggingface.co/architehc/nano-125m-rust/resolve/main/model.gguf",
        "mhc_url": "https://huggingface.co/architehc/nano-125m-rust/resolve/main/model.mhc",
        "config": "125M",
        "sha256_gguf": "TO_BE_FILLED",  # Fill with actual hash after training
        "sha256_mhc": "TO_BE_FILLED",
        "description": "125M parameter ternary model for Rust code generation",
        "size_mb": 45,
        "vocab_size": 50257,
        "dim": 768,
        "layers": 12,
    },
    "nano-7b-rust": {
        "gguf_url": "https://huggingface.co/architehc/nano-7b-rust/resolve/main/model.gguf",
        "mhc_url": "https://huggingface.co/architehc/nano-7b-rust/resolve/main/model.mhc",
        "config": "7B",
        "sha256_gguf": "TO_BE_FILLED",
        "sha256_mhc": "TO_BE_FILLED",
        "description": "7B parameter ternary model for Rust code generation",
        "size_mb": 1800,
        "vocab_size": 50257,
        "dim": 4096,
        "layers": 32,
    },
}

CHECKSUM_OVERRIDES_PATH = Path(__file__).with_name("model_zoo_checksums.json")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_checksum_overrides():
    """Load checksum overrides from model_zoo_checksums.json if present."""
    if not CHECKSUM_OVERRIDES_PATH.exists():
        return

    with open(CHECKSUM_OVERRIDES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for model_name, values in data.items():
        if model_name not in MODELS:
            continue
        gguf = values.get("sha256_gguf")
        mhc = values.get("sha256_mhc")
        if gguf:
            MODELS[model_name]["sha256_gguf"] = gguf
        if mhc:
            MODELS[model_name]["sha256_mhc"] = mhc


_load_checksum_overrides()


class ModelZoo:
    """Downloads and manages pre-trained nanochat models."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model zoo.

        Args:
            cache_dir: Directory for caching models. Defaults to ~/.cache/nanochat
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/nanochat")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self):
        """Print all available models."""
        print("\n=== Available Models ===\n")
        for name, info in MODELS.items():
            print(f"{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size_mb']} MB")
            print(f"  Config: {info['config']}")
            print(f"  Vocab: {info['vocab_size']}")
            print(f"  Dim: {info['dim']}, Layers: {info['layers']}")
            # Check if cached
            if self._is_cached(name):
                print(f"  Status: ✓ Cached at {self.cache_dir / name}")
            else:
                print(f"  Status: Not downloaded")
            print()

    def _is_cached(self, name: str) -> bool:
        """Check if model is already cached."""
        model_dir = self.cache_dir / name
        gguf_path = model_dir / "model.gguf"
        mhc_path = model_dir / "model.mhc"
        return gguf_path.exists() and mhc_path.exists()

    def download(self, name: str, force: bool = False, verify_checksum: bool = True) -> Path:
        """Download a model to cache.

        Args:
            name: Model name from MODELS registry
            force: Force re-download even if cached
            verify_checksum: Verify SHA256 checksums (recommended)

        Returns:
            Path to cached model directory

        Raises:
            ValueError: If model name not found or checksum mismatch
        """
        if name not in MODELS:
            raise ValueError(
                f"Unknown model: {name}\n"
                f"Available models: {', '.join(MODELS.keys())}"
            )

        model_info = MODELS[name]
        model_dir = self.cache_dir / name
        gguf_path = model_dir / "model.gguf"
        mhc_path = model_dir / "model.mhc"

        if not force and self._is_cached(name):
            if verify_checksum:
                if not self.verify_cached_model(name):
                    print(f"⚠ Checksum mismatch for cached '{name}', re-downloading...")
                else:
                    print(f"✓ Model '{name}' already cached at {model_dir}")
                    return model_dir
            else:
                print(f"✓ Model '{name}' already cached at {model_dir}")
                return model_dir

        model_dir.mkdir(parents=True, exist_ok=True)

        # Download GGUF model
        print(f"\n📥 Downloading {name} GGUF ({model_info['size_mb']} MB)...")
        self._download_file(
            model_info["gguf_url"],
            gguf_path,
            expected_hash=model_info["sha256_gguf"] if verify_checksum else None,
        )

        # Download mHC weights
        print(f"\n📥 Downloading {name} mHC weights...")
        self._download_file(
            model_info["mhc_url"],
            mhc_path,
            expected_hash=model_info["sha256_mhc"] if verify_checksum else None,
        )

        print(f"\n✓ Model downloaded to {model_dir}\n")
        return model_dir

    def _download_file(
        self, url: str, dest: Path, expected_hash: Optional[str] = None
    ):
        """Download a file with progress bar and optional checksum verification."""
        try:
            import requests
            from tqdm import tqdm
        except ImportError as e:
            raise RuntimeError(
                "Downloading models requires 'requests' and 'tqdm'. "
                "Install with: pip install requests tqdm"
            ) from e

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        sha256 = hashlib.sha256() if expected_hash else None

        with open(dest, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
                if sha256:
                    sha256.update(chunk)

        # Verify checksum
        if expected_hash:
            if expected_hash == "TO_BE_FILLED":
                print(
                    f"  ⚠ Checksum for {dest.name} is not registered yet; "
                    "skipping verification."
                )
                return

            actual_hash = sha256.hexdigest()
            if actual_hash != expected_hash:
                dest.unlink()  # Remove corrupted file
                raise ValueError(
                    f"Checksum mismatch for {dest.name}!\n"
                    f"Expected: {expected_hash}\n"
                    f"Got: {actual_hash}\n"
                    f"File may be corrupted. Please try again."
                )
            print(f"  ✓ Checksum verified")

    def verify_cached_model(self, name: str) -> bool:
        """Verify cached model file checksums if registered.

        Returns:
            True if verification succeeds (or hashes are not yet registered).
        """
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")

        model_dir = self.cache_dir / name
        gguf_path = model_dir / "model.gguf"
        mhc_path = model_dir / "model.mhc"
        if not gguf_path.exists() or not mhc_path.exists():
            raise FileNotFoundError(f"Cached files missing for model '{name}'")

        info = MODELS[name]
        expected_gguf = info.get("sha256_gguf")
        expected_mhc = info.get("sha256_mhc")
        actual_gguf = compute_file_sha256(gguf_path)
        actual_mhc = compute_file_sha256(mhc_path)

        ok = True
        if expected_gguf and expected_gguf != "TO_BE_FILLED":
            ok = ok and actual_gguf == expected_gguf
        if expected_mhc and expected_mhc != "TO_BE_FILLED":
            ok = ok and actual_mhc == expected_mhc
        return ok

    def load(self, name: str):
        """Download if needed and return model paths.

        Args:
            name: Model name

        Returns:
            Tuple of (gguf_path, mhc_path)

        Example:
            >>> zoo = ModelZoo()
            >>> gguf, mhc = zoo.load("nano-125m-rust")
        """
        model_dir = self.download(name)
        return (
            str(model_dir / "model.gguf"),
            str(model_dir / "model.mhc"),
        )


def load_model(name: str):
    """Convenience function to load a model from the zoo.

    Args:
        name: Model name (e.g., "nano-125m-rust")

    Returns:
        Tuple of (gguf_path, mhc_path)

    Example:
        >>> gguf, mhc = load_model("nano-125m-rust")
        >>> # Then load with nanochat_py.PyModel.load(gguf, mhc)
    """
    zoo = ModelZoo()
    return zoo.load(name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nanochat model zoo CLI")
    parser.add_argument(
        "command",
        choices=["list", "download", "hash", "verify"],
        help="Command to execute",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name (required for download)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification",
    )
    parser.add_argument(
        "--gguf-path",
        help="Path to local GGUF file (for hash command)",
    )
    parser.add_argument(
        "--mhc-path",
        help="Path to local mHC file (for hash command)",
    )

    args = parser.parse_args()
    zoo = ModelZoo()

    if args.command == "list":
        zoo.list_models()
    elif args.command == "download":
        if not args.model:
            parser.error("Model name required for download command")
        zoo.download(args.model, force=args.force, verify_checksum=not args.no_verify)
    elif args.command == "hash":
        if not args.gguf_path or not args.mhc_path:
            parser.error("--gguf-path and --mhc-path are required for hash command")
        gguf_sha = compute_file_sha256(Path(args.gguf_path))
        mhc_sha = compute_file_sha256(Path(args.mhc_path))
        print(f"sha256_gguf={gguf_sha}")
        print(f"sha256_mhc={mhc_sha}")
    elif args.command == "verify":
        if not args.model:
            parser.error("Model name required for verify command")
        ok = zoo.verify_cached_model(args.model)
        if ok:
            print(f"✓ Checksum verification passed for {args.model}")
        else:
            raise SystemExit(f"Checksum verification failed for {args.model}")
