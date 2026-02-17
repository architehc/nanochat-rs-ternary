# Getting Started

## Prerequisites

- Rust 1.75+ (edition 2021)
- CUDA toolkit (optional, for GPU training)
- ~24GB GPU VRAM for nano-125m, or CPU-only for d20/tiny-cpu

## Build

```bash
# CPU only
cargo build --release -p nanochat-train

# With CUDA support (for GPU training)
cargo build --release -p nanochat-train --features cuda
```

## Validate

```bash
# Run all tests (535 tests)
cargo test --workspace

# Check for warnings
cargo clippy --workspace
```

## Train

### Option A: Synthetic data (smoke test, no data prep needed)

```bash
cargo run --release --features cuda -p nanochat-train -- train \
    --config d20 \
    --dataset synthetic \
    --n-samples 5000 \
    --batch-size 4 \
    --seq-len 128 \
    --log-interval 50 \
    --device cuda
```

### Option B: Real data on RTX 4090

```bash
# Tokenize your text corpus
cargo run --release -p nanochat-train -- prepare-data \
    --text data/your_corpus.txt \
    --vocab-size 50257 \
    --output data/

# Train nano-125m (127M params, ~15h for 50k steps)
cargo run --release --features cuda -p nanochat-train -- train \
    --config nano-125m \
    --dataset tokens \
    --data-path data/tokens.bin \
    --batch-size 2 \
    --seq-len 256 \
    --checkpoint-dir checkpoints/nano125m \
    --log-interval 50 \
    --checkpoint-interval 5000 \
    --keep-last-checkpoints 5 \
    --device cuda
```

See [training/README.md](../training/README.md) for full CLI reference, VRAM limits, and config details.

## Export

```bash
cargo run --release -p nanochat-train -- export \
    --checkpoint checkpoints/nano125m/final \
    --gguf model.gguf \
    --mhc model.mhc
```

## Serve

```bash
cargo run --release -p nanochat-serve -- \
    --model model.gguf \
    --mhc model.mhc \
    --tokenizer models/gpt2-tokenizer.json \
    --port 8000
```
