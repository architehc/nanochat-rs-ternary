#!/bin/bash
# Chunked training for nano-500m-baseline on GPU 1
# Works around CUDA context sharing with GPU 0 process

set -eo pipefail

TARGET_STEPS="${1:-30000}"
DEVICE="cuda"
DATA_PATH="data/rust_v2_prepared/tokens.bin"
CHUNK_SIZE=100
CHECKPOINT_DIR="checkpoints/nano-500m-baseline"
CONFIG="nano-500m-baseline"

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "═══════════════════════════════════════════════"
echo "  nano-500m-baseline Chunked Training (GPU 1)"
echo "═══════════════════════════════════════════════"

mkdir -p "$CHECKPOINT_DIR"

find_last_step() {
    local last=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1)
    echo "${last:-0}"
}

CURRENT_STEP=$(find_last_step)
echo "Starting from step: $CURRENT_STEP"

if [ "$CURRENT_STEP" -ge "$TARGET_STEPS" ]; then
    echo "Already at target. Done!"
    exit 0
fi

while [ "$CURRENT_STEP" -lt "$TARGET_STEPS" ]; do
    CHUNK_TARGET=$((CURRENT_STEP + CHUNK_SIZE))
    if [ "$CHUNK_TARGET" -gt "$TARGET_STEPS" ]; then
        CHUNK_TARGET=$TARGET_STEPS
    fi

    echo "--- steps $CURRENT_STEP → $CHUNK_TARGET ($(date +%H:%M:%S)) ---"

    RESUME_ARG=""
    if [ "$CURRENT_STEP" -gt 0 ]; then
        RESUME_DIR="$CHECKPOINT_DIR/step_$CURRENT_STEP"
        if [ -d "$RESUME_DIR" ]; then
            RESUME_ARG="--resume $RESUME_DIR"
        else
            echo "ERROR: checkpoint $RESUME_DIR not found"
            exit 1
        fi
    fi

    cargo run --release -p nanochat-train --features cuda -- train \
        --config "$CONFIG" \
        --dataset tokens --data-path "$DATA_PATH" \
        --device "$DEVICE" \
        --batch-size 2 --seq-len 512 --epochs 1 \
        --log-interval 50 \
        --checkpoint-interval $CHUNK_SIZE \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --keep-last-checkpoints 3 \
        --total-steps $CHUNK_TARGET \
        $RESUME_ARG \
        2>&1 || true

    NEW_STEP=$(find_last_step)
    if [ "$NEW_STEP" -le "$CURRENT_STEP" ]; then
        echo "No progress (still at step $CURRENT_STEP). Retrying in 10s..."
        sleep 10
        continue
    fi
    CURRENT_STEP=$NEW_STEP
    echo "  → at step $CURRENT_STEP"
    sleep 3
done

echo "Training complete! Final step: $CURRENT_STEP"
