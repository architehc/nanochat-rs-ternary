#!/bin/bash
# Check training progress

CHECKPOINT_DIR="${1:-checkpoints/rust-nano-d20}"

echo "═══════════════════════════════════════════════════════════"
echo "  Training Status Check"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if training is running
if pgrep -f train_rust_maxgpu > /dev/null; then
    echo "✓ Training process is RUNNING"
    echo ""
else
    echo "⚠ Training process is NOT running"
    echo ""
fi

# Check GPU usage
echo "GPU Status:"
nvidia-smi | grep -A2 "GeForce RTX" || echo "  (Could not check GPU)"
echo ""

# Check checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoints found:"
    ls -lh "$CHECKPOINT_DIR"/step_* 2>/dev/null | tail -5 || echo "  No checkpoints yet"
    echo ""

    # Get last checkpoint
    LAST_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CHECKPOINT" ]; then
        CURRENT_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
        echo "Latest checkpoint: step_$CURRENT_STEP"
        echo ""
    fi

    # Check training log
    if [ -f "$CHECKPOINT_DIR/training.log" ]; then
        echo "Recent training output:"
        echo "───────────────────────────────────────────────────────────"
        tail -20 "$CHECKPOINT_DIR/training.log" | grep -E "loss=|Chunk|✓|Error"
        echo "───────────────────────────────────────────────────────────"
    fi
else
    echo "Checkpoint directory not found: $CHECKPOINT_DIR"
fi

echo ""
