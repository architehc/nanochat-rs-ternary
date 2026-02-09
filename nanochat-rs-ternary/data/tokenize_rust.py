#!/usr/bin/env python3
"""Tokenize Rust code with GPT-2 tokenizer and save as binary."""

import sys
import struct
from pathlib import Path

# Check for tokenizers library
try:
    from tokenizers import Tokenizer
except ImportError:
    print("❌ tokenizers library not found")
    print("Install with: pip install tokenizers")
    sys.exit(1)

print("✓ tokenizers library available\n")

# Load tokenizer
tokenizer_path = Path(__file__).parent.parent / "models" / "gpt2-tokenizer.json"
print(f"Loading tokenizer from: {tokenizer_path}")
tokenizer = Tokenizer.from_file(str(tokenizer_path))
print(f"✓ Tokenizer loaded (vocab size: {tokenizer.get_vocab_size()})\n")

# Load Rust code
rust_file = Path(__file__).parent / "combined_rust.txt"
print(f"Loading Rust code from: {rust_file}")
with open(rust_file, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

print(f"✓ Loaded {len(text):,} characters\n")

# Tokenize
print("Tokenizing with GPT-2 BPE...")
encoding = tokenizer.encode(text)
tokens = encoding.ids

print(f"✓ Tokenized into {len(tokens):,} tokens\n")

# Save as binary (little-endian u32)
output_file = Path(__file__).parent / "rust_tokens.bin"
print(f"Writing binary token file: {output_file}")

with open(output_file, 'wb') as f:
    for token_id in tokens:
        f.write(struct.pack('<I', token_id))  # little-endian u32

output_size_mb = output_file.stat().st_size / (1024 * 1024)
print(f"✓ Wrote {len(tokens):,} tokens ({output_size_mb:.2f} MB)\n")

# Calculate training info
seq_len = 512
num_samples = len(tokens) // seq_len
print("=" * 60)
print("Dataset Statistics")
print("=" * 60)
print(f"Total tokens:    {len(tokens):,}")
print(f"Sequence length: {seq_len}")
print(f"Training samples: {num_samples:,} (at seq_len={seq_len})")
print(f"File size:       {output_size_mb:.2f} MB")
print(f"Vocab size:      {tokenizer.get_vocab_size():,}")
print()

# Estimate training steps
for batch_size in [4, 8, 16, 32]:
    steps_per_epoch = (num_samples + batch_size - 1) // batch_size
    print(f"Batch size {batch_size:2d}: {steps_per_epoch:,} steps per epoch")

print()
print("✓ Ready for training!")
print()
print("Next command:")
print(f"  cargo run --release -p nanochat-train --example train_stack_maxgpu -- \\")
print(f"    --data data/rust_tokens.bin \\")
print(f"    --checkpoint-dir checkpoints/rust-nano-125m")
