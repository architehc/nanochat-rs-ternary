#!/bin/bash
# Training script with automatic restarts to work around CUDA memory leak
# Trains in 500-step chunks, restarting after each chunk to free GPU memory

set -e

# Configuration
CHECKPOINT_DIR="${1:-checkpoints/rust-nano-d20}"
TARGET_STEPS="${2:-10000}"
CHUNK_SIZE=500
BATCH_SIZE=2
DATA_PATH="data/rust_tokens.bin"
DEVICE="cuda:0"

echo "═══════════════════════════════════════════════════════════"
echo "  Rust Model Training with Automatic Restarts"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Target steps:     $TARGET_STEPS"
echo "  Chunk size:       $CHUNK_SIZE steps"
echo "  Checkpoint dir:   $CHECKPOINT_DIR"
echo "  Batch size:       $BATCH_SIZE"
echo "  Device:           $DEVICE"
echo ""
echo "Strategy: Train in $CHUNK_SIZE-step chunks, restart after each chunk"
echo "         to work around Candle CUDA memory leak"
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Find last checkpoint
CURRENT_STEP=0
if [ -d "$CHECKPOINT_DIR" ]; then
    LAST_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CHECKPOINT" ]; then
        CURRENT_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
        echo "✓ Found checkpoint at step $CURRENT_STEP"
        echo "  Resuming from: $LAST_CHECKPOINT"
        echo ""
    fi
fi

# Calculate chunks needed
STEPS_REMAINING=$((TARGET_STEPS - CURRENT_STEP))
CHUNKS_NEEDED=$(( (STEPS_REMAINING + CHUNK_SIZE - 1) / CHUNK_SIZE ))

if [ $STEPS_REMAINING -le 0 ]; then
    echo "✓ Training already complete ($CURRENT_STEP >= $TARGET_STEPS steps)"
    exit 0
fi

echo "Training plan:"
echo "  Current step:     $CURRENT_STEP"
echo "  Steps remaining:  $STEPS_REMAINING"
echo "  Chunks needed:    $CHUNKS_NEEDED"
echo "  Est. time:        ~$(( CHUNKS_NEEDED * 2 )) minutes (with restarts)"
echo ""

# Training loop
for ((chunk=1; chunk<=CHUNKS_NEEDED; chunk++)); do
    CHUNK_START_STEP=$CURRENT_STEP
    CHUNK_END_STEP=$((CHUNK_START_STEP + CHUNK_SIZE))
    if [ $CHUNK_END_STEP -gt $TARGET_STEPS ]; then
        CHUNK_END_STEP=$TARGET_STEPS
    fi

    STEPS_THIS_CHUNK=$((CHUNK_END_STEP - CHUNK_START_STEP))

    echo "───────────────────────────────────────────────────────────"
    echo "Chunk $chunk/$CHUNKS_NEEDED: Training steps $CHUNK_START_STEP → $CHUNK_END_STEP"
    echo "───────────────────────────────────────────────────────────"

    # Train this chunk
    cd /home/habitat/ternary-clawd/nanochat-rs-ternary

    if cargo run --release --example train_rust_maxgpu --features nanochat-train/cuda -- \
        --data "$DATA_PATH" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --device "$DEVICE" \
        --batch-size $BATCH_SIZE \
        --total-steps $CHUNK_END_STEP \
        --checkpoint-interval $CHUNK_SIZE \
        --log-interval 50 \
        2>&1 | tee -a "$CHECKPOINT_DIR/training.log"; then

        echo ""
        echo "✓ Chunk $chunk complete"
        CURRENT_STEP=$CHUNK_END_STEP

        # Clean up GPU memory between chunks
        sleep 2

    else
        EXIT_CODE=$?
        echo ""
        echo "✗ Chunk $chunk failed with exit code $EXIT_CODE"

        # Check if it's the expected OOM error
        if grep -q "CUDA_ERROR_OUT_OF_MEMORY" "$CHECKPOINT_DIR/training.log" | tail -50; then
            echo "  (CUDA OOM - expected with memory leak)"
            echo "  Checkpoint should be saved, continuing..."

            # Update current step from last checkpoint
            LAST_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sort -V | tail -1)
            if [ -n "$LAST_CHECKPOINT" ]; then
                CURRENT_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
                echo "  Recovered to step $CURRENT_STEP from checkpoint"
            fi

            # Continue to next chunk after brief pause for GPU cleanup
            sleep 5
        else
            echo "  (Unexpected error - aborting)"
            exit 1
        fi
    fi

    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "Training Complete! 🎉"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Final checkpoint: $CHECKPOINT_DIR/step_$TARGET_STEPS"
echo "Training log: $CHECKPOINT_DIR/training.log"
echo ""
echo "Next steps:"
echo "1. Export to ternary GGUF:"
echo "   cargo run --release -p nanochat-train --example export_checkpoint -- \\"
echo "     --checkpoint $CHECKPOINT_DIR/step_$TARGET_STEPS \\"
echo "     --output models/rust-d20.gguf"
echo ""
echo "2. Test Rust code generation:"
echo "   cargo run --release -p nanochat-serve -- \\"
echo "     --model models/rust-d20.gguf \\"
echo "     --mhc models/rust-d20.mhc \\"
echo "     --tokenizer models/gpt2-tokenizer.json \\"
echo "     --port 8085"
echo ""
