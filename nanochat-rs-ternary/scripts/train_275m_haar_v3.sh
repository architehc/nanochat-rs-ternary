#!/bin/bash
# Train nano-275m-haar-v3 — fresh Haar run with matched hyperparams
# Architecture: 20 layers, 50% Haar wavefield (~277M params)
# Same hyperparams as loop-only and engram-only for fair comparison
set -euo pipefail

CONFIG="nano-275m-haar-v3"
DEVICE="cuda:1"
DATASET="tokens"
DATA_PATH="data/rust_big/tokens.bin"
CHECKPOINT_DIR="checkpoints/nano-275m-haar-v3"
SEQ_LEN=256
BATCH_SIZE=2
LOG_INTERVAL=50
CHECKPOINT_INTERVAL=2000
KEEP_CHECKPOINTS=5
TOTAL_STEPS=20000

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

echo "=== nano-275m-haar-v3 (GPU 0) ==="
echo "Architecture: 20 layers, 50% Haar wavefield"
echo "Dataset: $DATA_PATH"
echo "Steps: $TOTAL_STEPS"
echo ""

# Find latest checkpoint for resume
RESUME_ARG=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST=$(ls -1d ${CHECKPOINT_DIR}/step_* 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$LATEST" ] && [ -f "${LATEST}/model.safetensors" ]; then
        echo "Resuming from: $LATEST"
        RESUME_ARG="--resume $LATEST"
    fi
fi

cargo run --release -p nanochat-train --features cuda -- train \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --data-path "$DATA_PATH" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --epochs 999 \
    --total-steps "$TOTAL_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --keep-last-checkpoints "$KEEP_CHECKPOINTS" \
    --log-interval "$LOG_INTERVAL" \
    --device "$DEVICE" \
    $RESUME_ARG

echo "Training complete."
