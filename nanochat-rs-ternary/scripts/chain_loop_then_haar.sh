#!/bin/bash
# Chain: run LoopLM to completion, then auto-start Haar v3 on same GPU
set -euo pipefail

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

echo "$(date): Waiting for LoopLM to complete..."

# Wait for LoopLM process to finish
while pgrep -f "nano-275m-loop-only" > /dev/null 2>&1; do
    sleep 60
done

echo "$(date): LoopLM completed. Starting Haar v3 on GPU 1..."

# Start Haar v3 with adjusted total_steps for remaining time
# Estimate ~4 hours available = ~10K steps at ~370 tok/s
CONFIG="nano-275m-haar-v3"
DEVICE="cuda:1"
DATA_PATH="data/rust_big/tokens.bin"
CHECKPOINT_DIR="checkpoints/nano-275m-haar-v3"
# Use 12K steps so WSD decay happens at 80% = step 9600
TOTAL_STEPS=12000

echo "=== nano-275m-haar-v3 (GPU 1, chained) ==="
echo "Architecture: 20 layers, 50% Haar wavefield"
echo "Steps: $TOTAL_STEPS"
echo ""

cargo run --release -p nanochat-train --features cuda -- train \
    --config "$CONFIG" \
    --dataset tokens \
    --data-path "$DATA_PATH" \
    --batch-size 2 \
    --seq-len 256 \
    --epochs 999 \
    --total-steps "$TOTAL_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval 2000 \
    --keep-last-checkpoints 5 \
    --log-interval 50 \
    --device "$DEVICE"

echo "$(date): Haar v3 training complete."
