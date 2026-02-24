#!/bin/bash
# Chunked training for nano-500m-wave-haar config
# Works around Candle's CUDA memory leak by restarting every CHUNK_SIZE steps
#
# Usage: ./scripts/train_haar_chunked.sh [target_steps] [device]

set -eo pipefail

# Configuration
TARGET_STEPS="${1:-10000}"
DEVICE="${2:-cuda}"
CHUNK_SIZE=100          # restart before OOM (~150 step limit)
CHECKPOINT_DIR="checkpoints/nano-500m-haar"
DATA_PATH="data/rust_prepared/tokens.bin"
BATCH_SIZE=2
SEQ_LEN=512
LOG_INTERVAL=50
CONFIG="nano-500m-wave-haar"

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "═══════════════════════════════════════════════════════════"
echo "  nano-500m-wave-haar Chunked Training"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Config:           $CONFIG"
echo "  Target steps:     $TARGET_STEPS"
echo "  Chunk size:       $CHUNK_SIZE steps"
echo "  Checkpoint dir:   $CHECKPOINT_DIR"
echo "  Batch size:       $BATCH_SIZE"
echo "  Seq len:          $SEQ_LEN"
echo "  Device:           $DEVICE"
echo "  Data:             $DATA_PATH"
echo ""

mkdir -p "$CHECKPOINT_DIR"

# Find last checkpoint step
find_last_step() {
    local last=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1)
    echo "${last:-0}"
}

CURRENT_STEP=$(find_last_step)
echo "Starting from step: $CURRENT_STEP"

if [ "$CURRENT_STEP" -ge "$TARGET_STEPS" ]; then
    echo "Already at target steps ($CURRENT_STEP >= $TARGET_STEPS). Done!"
    exit 0
fi

STEPS_REMAINING=$((TARGET_STEPS - CURRENT_STEP))
CHUNKS_NEEDED=$(( (STEPS_REMAINING + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "Chunks needed: $CHUNKS_NEEDED (~$(( CHUNKS_NEEDED * 7 / 10 )) minutes)"
echo ""

CHUNK_NUM=0
while [ "$CURRENT_STEP" -lt "$TARGET_STEPS" ]; do
    CHUNK_NUM=$((CHUNK_NUM + 1))
    CHUNK_TARGET=$((CURRENT_STEP + CHUNK_SIZE))
    if [ "$CHUNK_TARGET" -gt "$TARGET_STEPS" ]; then
        CHUNK_TARGET=$TARGET_STEPS
    fi

    echo "───────────────────────────────────────────────────────────"
    echo "Chunk $CHUNK_NUM: steps $CURRENT_STEP → $CHUNK_TARGET ($(date +%H:%M:%S))"
    echo "───────────────────────────────────────────────────────────"

    # Build resume arg
    RESUME_ARG=""
    if [ "$CURRENT_STEP" -gt 0 ]; then
        RESUME_DIR="$CHECKPOINT_DIR/step_$CURRENT_STEP"
        if [ -d "$RESUME_DIR" ]; then
            RESUME_ARG="--resume $RESUME_DIR"
        else
            echo "WARNING: Expected checkpoint at $RESUME_DIR not found!"
            echo "Available checkpoints:"
            ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null || echo "  (none)"
            exit 1
        fi
    fi

    # Run training chunk
    # Use --total-steps in config to limit this chunk, but the config has total_steps=10000
    # The checkpoint interval ensures we save at the right points
    if cargo run --release -p nanochat-train --features cuda -- train \
        --config "$CONFIG" \
        --dataset tokens \
        --data-path "$DATA_PATH" \
        --device "$DEVICE" \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --epochs 1 \
        --log-interval $LOG_INTERVAL \
        --checkpoint-interval $CHUNK_SIZE \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --keep-last-checkpoints 5 \
        --total-steps $CHUNK_TARGET \
        $RESUME_ARG \
        2>&1 | tee -a "$CHECKPOINT_DIR/training.log"; then

        echo "✓ Chunk $CHUNK_NUM complete"
    else
        EXIT_CODE=$?
        echo "Chunk $CHUNK_NUM exited with code $EXIT_CODE"

        if grep -q "CUDA_ERROR_OUT_OF_MEMORY" "$CHECKPOINT_DIR/training.log" 2>/dev/null; then
            echo "  (CUDA OOM — expected with Candle memory leak, continuing...)"
        else
            echo "  (Unexpected error — check training.log)"
        fi
    fi

    # Update step from last checkpoint
    NEW_STEP=$(find_last_step)
    if [ "$NEW_STEP" -le "$CURRENT_STEP" ]; then
        echo "ERROR: No progress (still at step $CURRENT_STEP). Aborting."
        exit 1
    fi
    CURRENT_STEP=$NEW_STEP
    echo "  → Resumed at step $CURRENT_STEP"

    # Brief pause for GPU cleanup
    sleep 3
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "Training Complete!"
echo "═══════════════════════════════════════════════════════════"
echo "Final step: $CURRENT_STEP"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""
