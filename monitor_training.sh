#!/bin/bash
# monitor_training.sh - Monitor training progress every 10 minutes

LOG_FILE="/tmp/training_125m.log"
CHECKPOINT_DIR=$(grep "Checkpoint:" "$LOG_FILE" | tail -1 | awk -F': ' '{print $2}')

echo "=== Training Progress Monitor ==="
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""
echo "Checking progress every 10 minutes..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    echo "==================== $(date) ===================="

    # Show latest training logs
    echo ""
    echo "Latest training output:"
    tail -30 "$LOG_FILE" | grep -E "(step|epoch|loss|Step|Epoch|Loss|Error)" || echo "Waiting for training logs..."

    # Show GPU usage if available
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU not in use (CPU training)"

    # Show CPU usage
    echo ""
    echo "CPU Usage:"
    top -bn1 | grep "nanochat-train" | head -3

    # Check for checkpoints
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo ""
        echo "Checkpoints:"
        ls -lh "$CHECKPOINT_DIR"/checkpoint_* 2>/dev/null | tail -5 || echo "No checkpoints yet"
    fi

    # Check if training is still running
    if ! pgrep -f "nanochat-train" > /dev/null; then
        echo ""
        echo "Training process not found! Training may have completed or crashed."
        echo "Check full log: $LOG_FILE"
        exit 1
    fi

    echo ""
    echo "Next check in 10 minutes..."
    echo "========================================================"
    echo ""

    sleep 600  # 10 minutes
done
