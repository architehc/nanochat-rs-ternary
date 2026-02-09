#!/bin/bash
# Monitor 6-hour training progress

CHECKPOINT_DIR="${1:-checkpoints/rust-6hour}"

echo "═══════════════════════════════════════════════════════════"
echo "  6-Hour Training Status"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if wrapper is running
if pgrep -f "train_6hour.sh" > /dev/null; then
    echo "✓ Training wrapper is RUNNING"
    WRAPPER_PID=$(pgrep -f "train_6hour.sh")
    echo "  Wrapper PID: $WRAPPER_PID"
else
    echo "⚠ Training wrapper is NOT running"
fi

# Check if actual training is running
if pgrep -f "train_rust_maxgpu" > /dev/null; then
    echo "✓ Training process is ACTIVE"
    TRAIN_PID=$(pgrep -f "train_rust_maxgpu")
    echo "  Training PID: $TRAIN_PID"
else
    echo "  (Training between restarts or completing)"
fi

echo ""

# GPU status
echo "GPU Status:"
nvidia-smi | grep -A2 "GeForce RTX" || echo "  Could not check GPU"
echo ""

# Check checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Progress:"
    LAST_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CHECKPOINT" ]; then
        CURRENT_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
        CHECKPOINT_COUNT=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | wc -l)
        echo "  Latest checkpoint: step_$CURRENT_STEP"
        echo "  Total checkpoints: $CHECKPOINT_COUNT"
        echo "  Checkpoint dir:    $(du -sh $CHECKPOINT_DIR | cut -f1)"
    else
        echo "  No checkpoints yet"
    fi
    echo ""

    # Recent training output
    if [ -f "$CHECKPOINT_DIR/training_6hour.log" ]; then
        echo "Recent training output:"
        echo "───────────────────────────────────────────────────────────"
        tail -15 "$CHECKPOINT_DIR/training_6hour.log" | grep -E "loss=|Run #|Elapsed|Error|✓" || tail -5 "$CHECKPOINT_DIR/training_6hour.log"
        echo "───────────────────────────────────────────────────────────"
    fi
else
    echo "  Checkpoint directory not found"
fi

echo ""
echo "Monitor live: tail -f $CHECKPOINT_DIR/training_6hour.log"
echo ""
