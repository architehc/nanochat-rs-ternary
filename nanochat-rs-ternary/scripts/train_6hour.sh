#!/bin/bash
# 6-hour continuous training with automatic OOM recovery
# Trains continuously for 6 hours, restarting after each OOM event

set -eo pipefail

CHECKPOINT_DIR="${1:-checkpoints/rust-6hour}"
DURATION_HOURS="${2:-6}"
BATCH_SIZE=2
DATA_PATH="data/rust_tokens.bin"
DEVICE="cuda:0"

DURATION_SECONDS=$((DURATION_HOURS * 3600))
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_SECONDS))

echo "═══════════════════════════════════════════════════════════"
echo "  6-Hour Continuous Rust Code Training"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Duration:         $DURATION_HOURS hours (until $(date -d @$END_TIME '+%H:%M:%S'))"
echo "  Checkpoint dir:   $CHECKPOINT_DIR"
echo "  Batch size:       $BATCH_SIZE"
echo "  Device:           $DEVICE"
echo "  Data:             $DATA_PATH"
echo ""
echo "Strategy: Train continuously, restart after OOM, repeat until time limit"
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Find starting step
CURRENT_STEP=0
if [ -d "$CHECKPOINT_DIR" ]; then
    LAST_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CHECKPOINT" ]; then
        CURRENT_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
        echo "✓ Resuming from checkpoint: step_$CURRENT_STEP"
        echo ""
    fi
fi

RUN_COUNT=0
TOTAL_STEPS_TRAINED=0

# Training loop - run until time limit
while true; do
    CURRENT_TIME=$(date +%s)
    TIME_REMAINING=$((END_TIME - CURRENT_TIME))

    if [ $TIME_REMAINING -le 0 ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  Time limit reached - Training complete!"
        echo "═══════════════════════════════════════════════════════════"
        break
    fi

    RUN_COUNT=$((RUN_COUNT + 1))
    HOURS_ELAPSED=$(( (CURRENT_TIME - START_TIME) / 3600 ))
    MINUTES_ELAPSED=$(( ((CURRENT_TIME - START_TIME) % 3600) / 60 ))
    HOURS_REMAINING=$(( TIME_REMAINING / 3600 ))
    MINUTES_REMAINING=$(( (TIME_REMAINING % 3600) / 60 ))

    echo "───────────────────────────────────────────────────────────"
    echo "Run #$RUN_COUNT | Elapsed: ${HOURS_ELAPSED}h${MINUTES_ELAPSED}m | Remaining: ${HOURS_REMAINING}h${MINUTES_REMAINING}m | Steps: $CURRENT_STEP"
    echo "───────────────────────────────────────────────────────────"

    # Train for a large number of steps - it will hit OOM and we'll restart
    # We use 50000 as a "forever" target - OOM will stop it first
    cd /home/habitat/ternary-clawd/nanochat-rs-ternary

    cargo run --release --example train_rust_maxgpu --features nanochat-train/cuda -- \
        --data "$DATA_PATH" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --device "$DEVICE" \
        --batch-size $BATCH_SIZE \
        --total-steps 50000 \
        --checkpoint-interval 500 \
        --log-interval 100 \
        2>&1 | tee -a "$CHECKPOINT_DIR/training_6hour.log" || {

        EXIT_CODE=$?
        echo ""
        echo "Training run ended (exit code: $EXIT_CODE)"

        # Check if it was OOM (expected) or another error
        if tail -20 "$CHECKPOINT_DIR/training_6hour.log" | grep -q "CUDA_ERROR_OUT_OF_MEMORY"; then
            echo "  → CUDA OOM (expected with memory leak)"
        elif tail -20 "$CHECKPOINT_DIR/training_6hour.log" | grep -q "Error"; then
            echo "  → Unexpected error detected"
            tail -10 "$CHECKPOINT_DIR/training_6hour.log"
        fi

        # Update current step from last checkpoint
        LAST_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sort -V | tail -1)
        if [ -n "$LAST_CHECKPOINT" ]; then
            NEW_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
            STEPS_THIS_RUN=$((NEW_STEP - CURRENT_STEP))
            TOTAL_STEPS_TRAINED=$((TOTAL_STEPS_TRAINED + STEPS_THIS_RUN))
            CURRENT_STEP=$NEW_STEP
            echo "  → Checkpoint saved: step_$CURRENT_STEP (+$STEPS_THIS_RUN steps)"
        fi

        echo "  → Waiting 3s for GPU cleanup..."
        sleep 3
    }

    echo ""
done

# Final statistics
TOTAL_ELAPSED=$((DURATION_SECONDS / 60))
AVG_STEPS_PER_MINUTE=$(( CURRENT_STEP / (TOTAL_ELAPSED + 1) ))

echo ""
echo "Training Statistics:"
echo "  Total duration:    $DURATION_HOURS hours"
echo "  Training runs:     $RUN_COUNT"
echo "  Final step:        $CURRENT_STEP"
echo "  Steps trained:     $TOTAL_STEPS_TRAINED"
echo "  Avg steps/min:     $AVG_STEPS_PER_MINUTE"
echo ""
echo "Final checkpoint:    $CHECKPOINT_DIR/step_$CURRENT_STEP"
echo "Training log:        $CHECKPOINT_DIR/training_6hour.log"
echo ""
echo "Next: Export to GGUF and test Rust code generation!"
echo ""
