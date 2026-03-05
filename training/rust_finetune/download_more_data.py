#!/usr/bin/env python3
"""
Download significantly more Rust code for 7-day training on RTX 5090.

Expands from ~87M tokens to 500M+ tokens by:
1. Cloning 100+ high-quality Rust repos (vs original 47)
2. Downloading Rust stdlib source
3. Processing into longer sequences (8192 tokens)
4. 30% FIM rate (reduced from 50% for better pretraining signal)
"""

import os
import sys
import random
import json
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B"
MAX_SEQ_LEN = 8192  # Doubled from 4096 for deeper context
FIM_RATE = 0.30  # Reduced: more pure code = better pretraining
SEED = 42
MIN_FILE_SIZE = 200
MAX_FILE_SIZE = 100_000  # Increased max to include larger files
CLONE_DIR = SCRIPT_DIR / "data" / "rust_repos_v2"

# Qwen2.5-Coder FIM tokens
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

# Expanded repo list: 100+ repos covering all domains of Rust
RUST_REPOS = [
    # === Core ecosystem (highest quality) ===
    "rust-lang/rust-clippy",
    "rust-lang/rustfmt",
    "rust-lang/cargo",
    "rust-lang/rust-analyzer",
    "rust-lang/hashbrown",
    "rust-lang/mdBook",
    "rust-lang/miri",
    "rust-lang/rustlings",
    "dtolnay/syn",
    "dtolnay/serde",
    "dtolnay/anyhow",
    "dtolnay/thiserror",
    "dtolnay/proc-macro2",
    "dtolnay/quote",
    "dtolnay/semver",
    "dtolnay/paste",

    # === Async runtime & networking ===
    "tokio-rs/tokio",
    "tokio-rs/axum",
    "tokio-rs/tracing",
    "tokio-rs/mio",
    "tokio-rs/bytes",
    "tokio-rs/console",
    "hyperium/hyper",
    "hyperium/tonic",  # gRPC
    "seanmonstar/reqwest",
    "actix/actix-web",
    "rustls/rustls",
    "cloudflare/quiche",
    "quinn-rs/quinn",

    # === Serialization & data formats ===
    "serde-rs/json",
    "toml-rs/toml",
    "dtolnay/serde-yaml",  # Actually BurntSushi/toml moved
    "csv-rs/csv",  # Actually BurntSushi/csv

    # === CLI tools (excellent systems code) ===
    "BurntSushi/ripgrep",
    "sharkdp/fd",
    "sharkdp/bat",
    "sharkdp/hyperfine",
    "dandavison/delta",
    "eza-community/eza",
    "ajeetdsouza/zoxide",
    "starship/starship",
    "ogham/exa",
    "bootandy/dust",
    "dalance/procs",
    "ClementTsang/bottom",
    "imsnif/bandwhich",
    "XAMPPRocky/tokei",

    # === Data & concurrency ===
    "rayon-rs/rayon",
    "crossbeam-rs/crossbeam",
    "rust-lang-nursery/lazy-static.rs",

    # === Shells & terminals ===
    "nushell/nushell",
    "alacritty/alacritty",
    "zellij-org/zellij",
    "wez/wezterm",

    # === Compilers & language tools ===
    "nickel-lang/nickel",
    "gluon-lang/gluon",
    "pest-parser/pest",
    "tree-sitter/tree-sitter",
    "lalrpop/lalrpop",
    "grmtools/grmtools",
    "softdevteam/grmtools",

    # === Databases & storage ===
    "tikv/tikv",
    "surrealdb/surrealdb",
    "apache/datafusion",
    "launchbadge/sqlx",
    "SeaQL/sea-orm",
    "diesel-rs/diesel",

    # === Web frameworks ===
    "poem-web/poem",
    "salvo-rs/salvo",
    "iron/iron",

    # === Crypto & security ===
    "briansmith/ring",
    "RustCrypto/hashes",
    "RustCrypto/block-ciphers",

    # === Graphics & games ===
    "bevyengine/bevy",
    "gfx-rs/wgpu",
    "nannou-org/nannou",

    # === Embedded & OS ===
    "embassy-rs/embassy",
    "rcore-os/rCore-Tutorial-v3",
    "phil-opp/blog_os",

    # === Error handling & logging ===
    "yaahc/eyre",
    "dtolnay/trybuild",
    "rust-lang/log",
    "env-logger-rs/env_logger",

    # === Testing ===
    "assert-rs/predicates-rs",
    "proptest-rs/proptest",

    # === Image & media ===
    "image-rs/image",
    "BurntSushi/ripgrep",

    # === Math & scientific ===
    "rust-ndarray/ndarray",
    "dimforge/nalgebra",
    "huggingface/tokenizers",
    "huggingface/candle",

    # === Build tools ===
    "aspect-build/bazel-lib",
    "nickel-lang/nickel",

    # === Popular crates (misc) ===
    "clap-rs/clap",
    "chronotope/chrono",
    "rust-itertools/itertools",
    "regex-rs/regex",
    "BurntSushi/memchr",
    "bitflags/bitflags",
    "uuid-rs/uuid",
    "rust-random/rand",
    "contain-rs/linked-hash-map",
    "indexmap-rs/indexmap",
    "bluss/petgraph",

    # === ML & AI in Rust ===
    "tracel-ai/burn",
    "LaurentMazare/tch-rs",
    "sonos/tract",

    # === Additional high-value repos ===
    "PyO3/pyo3",
    "nickel-org/nickel.rs",
    "mozilla/sccache",
    "meilisearch/meilisearch",
    "astral-sh/ruff",
    "astral-sh/uv",
    "prefix-dev/pixi",
    "typst/typst",
    "denoland/deno_core",
    "nickel-lang/nickel",
    "libp2p/rust-libp2p",
    "paritytech/substrate",
    "solana-labs/solana",
    "foundry-rs/foundry",
    "paradigmxyz/reth",
    "aptos-labs/aptos-core",
]

# Deduplicate
RUST_REPOS = list(dict.fromkeys(RUST_REPOS))


def clone_repo(repo_url, dest_dir, timeout=180):
    """Shallow clone a GitHub repo."""
    repo_name = repo_url.replace("/", "_")
    clone_path = dest_dir / repo_name

    if clone_path.exists():
        return clone_path, True, "cached"

    full_url = f"https://github.com/{repo_url}.git"
    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch", full_url, str(clone_path)],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return clone_path, True, "cloned"
        return clone_path, False, result.stderr[:200]
    except subprocess.TimeoutExpired:
        return clone_path, False, "timeout"
    except Exception as e:
        return clone_path, False, str(e)[:200]


def extract_rust_files(repo_path):
    """Extract all .rs files from a cloned repo with quality filtering."""
    texts = []
    try:
        for rs_file in repo_path.rglob("*.rs"):
            path_str = str(rs_file)
            skip_dirs = ["/target/", "/vendor/", "/.git/", "/test_data/",
                        "/fixtures/", "/testdata/", "/fuzz/", "/benches/"]
            if any(d in path_str for d in skip_dirs):
                continue
            try:
                content = rs_file.read_text(errors="replace")
                if quality_filter(content):
                    texts.append(content)
            except Exception:
                continue
    except Exception:
        pass
    return texts


def quality_filter(content):
    """Filter for high-quality Rust code."""
    if not content:
        return False
    size = len(content)
    if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
        return False

    # Must have Rust indicators
    rust_indicators = ["fn ", "let ", "use ", "struct ", "impl ", "pub ", "mod ", "trait ", "enum "]
    count = sum(1 for ind in rust_indicators if ind in content)
    if count < 2:
        return False

    # Skip auto-generated
    first_500 = content[:500]
    skip = ["// Generated by", "// Auto-generated", "// DO NOT EDIT",
            "// @generated", "GENERATED BY", "This file was auto"]
    if any(pat in first_500 for pat in skip):
        return False

    # Skip files that are mostly comments or macros
    lines = content.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('//')]
    if len(code_lines) < 5:
        return False

    return True


def apply_fim(text, rng):
    """Apply Fill-in-the-Middle transformation."""
    if rng.random() > FIM_RATE:
        return text

    lines = text.split('\n')
    if len(lines) < 5:
        return text

    n = len(lines)
    span = max(1, rng.randint(1, max(1, n // 4)))
    start = rng.randint(0, n - span)

    prefix = '\n'.join(lines[:start])
    middle = '\n'.join(lines[start:start + span])
    suffix = '\n'.join(lines[start + span:])

    if prefix:
        prefix += '\n'
    if suffix:
        suffix = '\n' + suffix

    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("=" * 70)
    print("EXPANDED DATA DOWNLOAD FOR 7-DAY RTX 5090 TRAINING")
    print(f"Target: 500M+ tokens from {len(RUST_REPOS)} repos")
    print("=" * 70)

    # Step 1: Clone repos
    print(f"\nCloning {len(RUST_REPOS)} repos...")
    ok_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(clone_repo, repo, CLONE_DIR): repo for repo in RUST_REPOS}
        for future in as_completed(futures):
            repo = futures[future]
            path, success, msg = future.result()
            name = repo.split("/")[-1]
            if success:
                ok_count += 1
                status = "OK" if msg == "cloned" else "CACHED"
                print(f"  [{status}] {name}")
            else:
                fail_count += 1
                print(f"  [FAIL] {name}: {msg}")

    print(f"\nCloned: {ok_count}/{len(RUST_REPOS)} repos ({fail_count} failed)")

    # Step 2: Extract files
    print(f"\nExtracting Rust files...")
    all_texts = []
    repo_stats = {}

    for repo_dir in sorted(CLONE_DIR.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
            continue
        texts = extract_rust_files(repo_dir)
        if texts:
            mb = sum(len(t) for t in texts) / 1e6
            print(f"  {repo_dir.name}: {len(texts)} files ({mb:.1f} MB)")
            repo_stats[repo_dir.name] = {"files": len(texts), "mb": round(mb, 1)}
            all_texts.extend(texts)

    # Add existing data
    print("\nAdding existing local data...")
    project_root = SCRIPT_DIR.parent.parent
    local_count = 0
    for rs_file in project_root.rglob("*.rs"):
        skip = [".venv", "target", "rust_repos", "data/rust_repos"]
        if any(s in str(rs_file) for s in skip):
            continue
        try:
            content = rs_file.read_text(errors="replace")
            if quality_filter(content):
                all_texts.append(content)
                local_count += 1
        except:
            continue
    print(f"  Added {local_count} local .rs files")

    # Add combined_rust.txt
    for data_dir in [project_root / "nanochat-rs-ternary" / "data", project_root / "data"]:
        combined = data_dir / "combined_rust.txt"
        if combined.exists():
            content = combined.read_text(errors="replace")
            for i in range(0, len(content), 8000):
                chunk = content[i:i + 8000]
                if len(chunk) > 200:
                    all_texts.append(chunk)
            print(f"  Added {len(content)//8000} chunks from {combined}")

    rng.shuffle(all_texts)
    total_mb = sum(len(t) for t in all_texts) / 1e6
    print(f"\nTotal: {len(all_texts)} files, {total_mb:.1f} MB raw text")

    # Step 3: Tokenize
    print("\nTokenizing...")
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def process_batch(examples):
        all_ids = []
        all_mask = []
        all_labels = []

        for content in examples["content"]:
            text = apply_fim(content, rng)
            tokens = tokenizer(text, truncation=False, add_special_tokens=True,
                             return_attention_mask=False)
            ids = tokens["input_ids"]

            for i in range(0, max(1, len(ids) - 128), MAX_SEQ_LEN):
                chunk = ids[i:i + MAX_SEQ_LEN]
                if len(chunk) < 256:
                    continue

                pad = MAX_SEQ_LEN - len(chunk)
                mask = [1] * len(chunk) + [0] * pad
                chunk = chunk + [tokenizer.pad_token_id or 0] * pad
                labels = [c if m == 1 else -100 for c, m in zip(chunk, mask)]

                all_ids.append(chunk)
                all_mask.append(mask)
                all_labels.append(labels)

        return {"input_ids": all_ids, "attention_mask": all_mask, "labels": all_labels}

    raw_ds = Dataset.from_dict({"content": all_texts})
    processed = raw_ds.map(process_batch, batched=True, batch_size=200,
                          remove_columns=["content"],
                          num_proc=min(16, os.cpu_count() or 1),
                          desc="Tokenizing")

    n_seq = len(processed)
    n_tokens = n_seq * MAX_SEQ_LEN
    print(f"\nSequences: {n_seq}")
    print(f"Tokens: ~{n_tokens / 1e6:.0f}M")

    # Split
    n_val = max(200, int(n_seq * 0.01))
    split = processed.train_test_split(test_size=n_val, seed=SEED)
    final = DatasetDict({"train": split["train"], "validation": split["test"]})

    save_path = OUTPUT_DIR / "rust_code_v2"
    final.save_to_disk(str(save_path))

    stats = {
        "total_repos": ok_count,
        "total_raw_files": len(all_texts),
        "total_raw_mb": round(total_mb, 1),
        "total_sequences": n_seq,
        "total_tokens_approx": n_tokens,
        "train_sequences": len(final["train"]),
        "val_sequences": len(final["validation"]),
        "max_seq_len": MAX_SEQ_LEN,
        "fim_rate": FIM_RATE,
        "model": MODEL_NAME,
        "repo_stats": repo_stats,
    }

    with open(OUTPUT_DIR / "data_stats_v2.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print(f"  Sequences: {n_seq:,} ({n_tokens/1e6:.0f}M tokens)")
    print(f"  Train: {len(final['train']):,}")
    print(f"  Val: {len(final['validation']):,}")
    print(f"  Saved to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
