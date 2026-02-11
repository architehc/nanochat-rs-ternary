#!/usr/bin/env python3
"""Extract Rust code from repositories and tokenize for training."""

import os
import sys
from pathlib import Path
from tokenizers import Tokenizer
import struct
from tqdm import tqdm

def find_rust_files(root_dir):
    """Find all .rs files recursively."""
    rust_files = []
    for path in Path(root_dir).rglob("*.rs"):
        # Skip test files and examples to focus on library code
        path_str = str(path)
        if "/tests/" in path_str or "/benches/" in path_str:
            continue
        rust_files.append(path)
    return rust_files

def extract_rust_code(rust_files, output_file, max_size_mb=500):
    """Extract Rust code into a single text file."""
    max_bytes = max_size_mb * 1024 * 1024
    total_bytes = 0
    files_included = 0

    print(f"Extracting Rust code (max {max_size_mb}MB)...")

    with open(output_file, 'w', encoding='utf-8') as out:
        for rust_file in tqdm(rust_files):
            try:
                with open(rust_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Skip very large files (probably generated code)
                if len(content) > 500_000:
                    continue

                # Skip files with too many non-ASCII chars (probably not real Rust)
                non_ascii = sum(1 for c in content if ord(c) > 127)
                if non_ascii > len(content) * 0.1:
                    continue

                out.write(content)
                out.write("\n\n")  # Separate files

                total_bytes += len(content.encode('utf-8'))
                files_included += 1

                # Stop if we hit size limit
                if total_bytes >= max_bytes:
                    print(f"Reached {max_size_mb}MB limit, stopping")
                    break

            except (UnicodeDecodeError, PermissionError):
                continue

    print(f"✓ Extracted {files_included} files ({total_bytes / 1024 / 1024:.1f}MB)")
    return files_included, total_bytes

def tokenize_and_save(text_file, output_file, tokenizer_path):
    """Tokenize text and save as binary u32 array."""
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Reading text from {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Tokenizing {len(text)} characters...")
    encoding = tokenizer.encode(text)
    tokens = encoding.ids

    print(f"Saving {len(tokens)} tokens to {output_file}...")
    with open(output_file, 'wb') as f:
        for token in tqdm(tokens):
            f.write(struct.pack('<I', token))  # Little-endian u32

    print(f"✓ Tokenized: {len(tokens):,} tokens")
    return len(tokens)

def main():
    # Paths
    repos_dir = "data/rust-repos"
    text_output = "data/rust_corpus_large.txt"
    tokens_output = "data/rust_tokens_large.bin"
    tokenizer_path = "models/gpt2-tokenizer.json"

    print("═══════════════════════════════════════════════════════════")
    print("  Rust Code Extraction & Tokenization")
    print("═══════════════════════════════════════════════════════════")
    print()

    # Step 1: Find all Rust files
    print("Step 1: Finding Rust files...")
    rust_files = find_rust_files(repos_dir)
    print(f"✓ Found {len(rust_files):,} Rust files")
    print()

    # Step 2: Extract code (limit to 500MB to keep reasonable)
    print("Step 2: Extracting Rust code...")
    files_included, total_bytes = extract_rust_code(
        rust_files,
        text_output,
        max_size_mb=500  # 500MB text corpus
    )
    print()

    # Step 3: Tokenize
    print("Step 3: Tokenizing...")
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("Please download GPT-2 tokenizer first")
        sys.exit(1)

    num_tokens = tokenize_and_save(text_output, tokens_output, tokenizer_path)
    print()

    # Summary
    print("═══════════════════════════════════════════════════════════")
    print("  Summary")
    print("═══════════════════════════════════════════════════════════")
    print()
    print(f"Input:  {len(rust_files):,} Rust files scanned")
    print(f"        {files_included:,} files included")
    print(f"        {total_bytes / 1024 / 1024:.1f}MB text extracted")
    print()
    print(f"Output: {num_tokens:,} tokens")
    print(f"        {os.path.getsize(tokens_output) / 1024 / 1024:.1f}MB binary file")
    print()
    print(f"Original dataset: 4.2M tokens (17MB)")
    print(f"New dataset:      {num_tokens / 1_000_000:.1f}M tokens ({os.path.getsize(tokens_output) / 1024 / 1024:.0f}MB)")
    print(f"Growth: {num_tokens / 4_200_000:.1f}x larger")
    print()
    print("✓ Complete!")
    print()
    print("Next steps:")
    print("1. Backup old data: mv data/rust_tokens.bin data/rust_tokens_old.bin")
    print("2. Use new data:    mv data/rust_tokens_large.bin data/rust_tokens.bin")
    print("3. Retrain model with larger dataset")

if __name__ == "__main__":
    main()
