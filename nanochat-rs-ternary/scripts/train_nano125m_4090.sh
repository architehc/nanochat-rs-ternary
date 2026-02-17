#!/bin/bash
# Production training: nano_125m on RTX 4090 (24GB VRAM)
# Expected: ~15 hours, 50k steps, 102M tokens
#
# Hardware: RTX 4090, 224-core EPYC, 944GB RAM
# Model: nano_125m (127M params, dim=768, 12 layers)
# Data: rust_tokens_large.bin (68M tokens, ~1.5 epochs)
# Throughput: ~1,900 tok/s → ~0.93 optimizer steps/s
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${WORKSPACE}/data/rust_tokens_large.bin"
CKPT="${WORKSPACE}/checkpoints/nano125m_rust"

# Verify data exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: Training data not found at $DATA"
    exit 1
fi

echo "=== nano_125m Production Training ==="
echo "Data: $DATA ($(du -h "$DATA" | cut -f1))"
echo "Checkpoints: $CKPT"
echo "Model: 127M params (dim=768, 12 layers, vocab=50257)"
echo "Batch: 2 × 256 tokens × 4 grad_accum = 2048 tokens/step"
echo "Target: 50,000 steps (~15 hours)"
echo ""

cd "$WORKSPACE"
RUST_LOG=info cargo run --release --features cuda -p nanochat-train -- train \
    --config nano-125m \
    --dataset tokens \
    --data-path "$DATA" \
    --epochs 5 \
    --batch-size 2 \
    --seq-len 256 \
    --checkpoint-dir "$CKPT" \
    --log-interval 50 \
    --checkpoint-interval 5000 \
    --keep-last-checkpoints 5 \
    --device cuda \
    2>&1 | tee "$CKPT/training.log"

echo ""
echo "=== Training Complete ==="
echo "Checkpoints: $CKPT"
echo "Log: $CKPT/training.log"
