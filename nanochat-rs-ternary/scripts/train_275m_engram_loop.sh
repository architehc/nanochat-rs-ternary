#!/bin/bash
# Train nano-275m-wave-engram-loop on GPU 1 (cuda:1)
# Architecture: LoopLM + Wavefield + Engram (~279M params)
# Chunked training: 5000 steps per chunk, resume from last checkpoint
set -euo pipefail

CONFIG="nano-275m-wave-engram-loop"
DEVICE="cuda:1"
DATASET="tokens"
DATA_PATH="data/rust_v2_prepared/tokens.bin"
CHECKPOINT_DIR="checkpoints/nano-275m-engram-loop"
SEQ_LEN=256
BATCH_SIZE=2
LOG_INTERVAL=50
CHECKPOINT_INTERVAL=1000
KEEP_CHECKPOINTS=3
CHUNK_SIZE=5000

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

# Build first
echo "Building nanochat-train with CUDA..."
cargo build --release -p nanochat-train --features cuda 2>&1

# Find latest checkpoint for resume
RESUME_ARG=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST=$(ls -1d ${CHECKPOINT_DIR}/step_* 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$LATEST" ] && [ -f "${LATEST}/model.safetensors" ]; then
        echo "Resuming from: $LATEST"
        RESUME_ARG="--resume $LATEST"
    fi
fi

# Determine step range
CURRENT_STEP=0
if [ -n "$RESUME_ARG" ]; then
    CURRENT_STEP=$(basename "$LATEST" | sed 's/step_//')
fi

# Total target from config: 134764 steps
TOTAL_TARGET=134764
STOP_AT=$((CURRENT_STEP + CHUNK_SIZE))
if [ "$STOP_AT" -gt "$TOTAL_TARGET" ]; then
    STOP_AT=$TOTAL_TARGET
fi

echo "=== nano-275m-wave-engram-loop chunk ==="
echo "Steps: ${CURRENT_STEP} -> ${STOP_AT} (of ${TOTAL_TARGET})"
echo "Device: ${DEVICE}"
echo ""

cargo run --release -p nanochat-train --features cuda -- train \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --data-path "$DATA_PATH" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --epochs 999 \
    --total-steps "$STOP_AT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --keep-last-checkpoints "$KEEP_CHECKPOINTS" \
    --log-interval "$LOG_INTERVAL" \
    --device "$DEVICE" \
    $RESUME_ARG

echo "Chunk complete. Run again to continue training."
