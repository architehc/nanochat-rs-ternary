#!/usr/bin/env python3
"""
Combine all .rs files from rust-repos/ into a large training corpus.
Produces:
  - data/rust_big/combined.txt (raw text)
  - data/rust_big/tokens.bin (u32 LE tokenized)
  - data/rust_big/train_tokens.bin (95% train split)
  - data/rust_big/test_tokens.bin (5% test split)
  - data/rust_big/eval_prompts.jsonl (evaluation prompts)
"""

import os
import json
import struct
import random
import re
from pathlib import Path
from tokenizers import Tokenizer

REPO_DIR = "data/rust-repos"
OUTPUT_DIR = "data/rust_big"
TOKENIZER_PATH = "data/rust_v2_prepared/tokenizer.json"
TEST_SPLIT = 0.05  # 5% for test
SEED = 42

# Files/dirs to skip
SKIP_PATTERNS = [
    "/target/", "/.git/", "/test_data/", "/fixtures/",
    "/benches/", "_test.rs", "/tests/ui/", "/tests/run-make/",
    "/tests/rustdoc/", "/tests/pretty/",
]

def should_skip(path: str) -> bool:
    for pat in SKIP_PATTERNS:
        if pat in path:
            return True
    return False

def extract_rs_files(repo_dir: str) -> list[tuple[str, str]]:
    """Extract all .rs files with their content."""
    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        # Skip .git and target dirs
        dirs[:] = [d for d in dirs if d not in ('.git', 'target', 'node_modules')]
        for fname in filenames:
            if not fname.endswith('.rs'):
                continue
            path = os.path.join(root, fname)
            if should_skip(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if len(content.strip()) < 50:  # Skip tiny files
                    continue
                if len(content) > 500_000:  # Skip huge generated files
                    continue
                files.append((path, content))
            except Exception:
                continue
    return files

def extract_eval_prompts(files: list[tuple[str, str]]) -> list[dict]:
    """Extract function signatures and struct definitions for eval prompts."""
    prompts = []

    # Pattern: fn name(args) -> RetType {
    fn_pattern = re.compile(
        r'((?:pub\s+)?(?:async\s+)?fn\s+\w+[^{]*\{)',
        re.MULTILINE
    )

    # Pattern: struct Name { or impl Name {
    struct_pattern = re.compile(
        r'((?:pub\s+)?struct\s+\w+[^{]*\{)',
        re.MULTILINE
    )

    impl_pattern = re.compile(
        r'(impl(?:<[^>]+>)?\s+\w+[^{]*\{)',
        re.MULTILINE
    )

    seen = set()
    random.seed(SEED)
    sampled_files = random.sample(files, min(1000, len(files)))

    for path, content in sampled_files:
        # Extract function signatures
        for m in fn_pattern.finditer(content):
            sig = m.group(1).strip()
            if len(sig) < 10 or len(sig) > 200:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            # Get the full function body for reference
            start = m.start()
            depth = 0
            end = start
            for i, c in enumerate(content[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            body = content[start:end]
            if 20 < len(body) < 2000:
                prompts.append({
                    "type": "fn_completion",
                    "prompt": sig + "\n",
                    "reference": body,
                    "source": os.path.relpath(path, REPO_DIR),
                })

    # Also add some standard prompts
    standard_prompts = [
        {"type": "standard", "prompt": "fn main() {\n", "reference": ""},
        {"type": "standard", "prompt": "use std::collections::HashMap;\n\nfn ", "reference": ""},
        {"type": "standard", "prompt": "pub struct Config {\n", "reference": ""},
        {"type": "standard", "prompt": "impl Iterator for ", "reference": ""},
        {"type": "standard", "prompt": "pub fn parse(input: &str) -> Result<", "reference": ""},
        {"type": "standard", "prompt": "#[derive(Debug, Clone)]\npub struct ", "reference": ""},
        {"type": "standard", "prompt": "async fn handle_request(req: Request) -> Response {\n", "reference": ""},
        {"type": "standard", "prompt": "impl Display for Error {\n    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {\n", "reference": ""},
        {"type": "standard", "prompt": "pub trait ", "reference": ""},
        {"type": "standard", "prompt": "match self {\n", "reference": ""},
        {"type": "standard", "prompt": "/// Returns the number of elements in the collection.\npub fn len(&self) -> usize {\n", "reference": ""},
        {"type": "standard", "prompt": "fn from_str(s: &str) -> Result<Self, Self::Err> {\n", "reference": ""},
        {"type": "standard", "prompt": "let mut result = Vec::new();\n    for ", "reference": ""},
        {"type": "standard", "prompt": "#[test]\nfn test_", "reference": ""},
        {"type": "standard", "prompt": "use serde::{Deserialize, Serialize};\n\n#[derive(Debug, Serialize, Deserialize)]\npub struct ", "reference": ""},
    ]
    prompts.extend(standard_prompts)

    # Sample down to 200 fn_completion prompts
    fn_prompts = [p for p in prompts if p["type"] == "fn_completion"]
    other_prompts = [p for p in prompts if p["type"] != "fn_completion"]

    if len(fn_prompts) > 200:
        random.shuffle(fn_prompts)
        fn_prompts = fn_prompts[:200]

    return other_prompts + fn_prompts

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    print(f"Scanning {REPO_DIR} for .rs files...")
    all_files = extract_rs_files(REPO_DIR)
    print(f"Found {len(all_files)} .rs files")

    # Sort by path for reproducibility
    all_files.sort(key=lambda x: x[0])

    # Shuffle with seed for train/test split
    random.seed(SEED)
    random.shuffle(all_files)

    # Split into train/test
    n_test = max(1, int(len(all_files) * TEST_SPLIT))
    test_files = all_files[:n_test]
    train_files = all_files[n_test:]

    print(f"Train files: {len(train_files)}, Test files: {n_test}")

    # Extract eval prompts from test set
    print("Extracting evaluation prompts...")
    eval_prompts = extract_eval_prompts(test_files)
    eval_path = os.path.join(OUTPUT_DIR, "eval_prompts.jsonl")
    with open(eval_path, 'w') as f:
        for p in eval_prompts:
            f.write(json.dumps(p) + '\n')
    print(f"Saved {len(eval_prompts)} eval prompts to {eval_path}")

    # Combine and tokenize train set
    print("Combining and tokenizing train set...")
    combined_path = os.path.join(OUTPUT_DIR, "combined.txt")
    train_tokens = []
    test_tokens = []

    total_chars = 0
    with open(combined_path, 'w') as out:
        for i, (path, content) in enumerate(train_files):
            # Add file separator
            rel = os.path.relpath(path, REPO_DIR)
            header = f"\n// === {rel} ===\n"
            out.write(header + content + "\n")

            # Tokenize
            encoded = tokenizer.encode(header + content + "\n")
            train_tokens.extend(encoded.ids)
            total_chars += len(content)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(train_files)} files, "
                      f"{total_chars/1e6:.1f}MB text, {len(train_tokens)/1e6:.1f}M tokens")

    print(f"\nTrain: {total_chars/1e6:.1f}MB text, {len(train_tokens)/1e6:.1f}M tokens")

    # Tokenize test set
    print("Tokenizing test set...")
    for path, content in test_files:
        rel = os.path.relpath(path, REPO_DIR)
        header = f"\n// === {rel} ===\n"
        encoded = tokenizer.encode(header + content + "\n")
        test_tokens.extend(encoded.ids)

    print(f"Test: {len(test_tokens)/1e6:.1f}M tokens")

    # Save tokens as u32 LE binary
    print("Writing train tokens...")
    train_path = os.path.join(OUTPUT_DIR, "train_tokens.bin")
    with open(train_path, 'wb') as f:
        for tok in train_tokens:
            f.write(struct.pack('<I', tok))

    # Also save as combined tokens.bin (for backward compat with train CLI)
    all_tokens_path = os.path.join(OUTPUT_DIR, "tokens.bin")
    with open(all_tokens_path, 'wb') as f:
        for tok in train_tokens:
            f.write(struct.pack('<I', tok))

    print("Writing test tokens...")
    test_path = os.path.join(OUTPUT_DIR, "test_tokens.bin")
    with open(test_path, 'wb') as f:
        for tok in test_tokens:
            f.write(struct.pack('<I', tok))

    # Copy tokenizer for convenience
    import shutil
    shutil.copy(TOKENIZER_PATH, os.path.join(OUTPUT_DIR, "tokenizer.json"))

    print(f"\nDone!")
    print(f"  Train: {train_path} ({os.path.getsize(train_path)/1e6:.1f}MB, {len(train_tokens):,} tokens)")
    print(f"  Test:  {test_path} ({os.path.getsize(test_path)/1e6:.1f}MB, {len(test_tokens):,} tokens)")
    print(f"  Eval:  {eval_path} ({len(eval_prompts)} prompts)")
    print(f"  Tokenizer: {OUTPUT_DIR}/tokenizer.json")

if __name__ == "__main__":
    main()
