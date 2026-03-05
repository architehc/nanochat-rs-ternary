#!/bin/bash
# GPU and training monitor for 7-day RTX 5090 training
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output_5090"
LOG_DIR="$OUTPUT_DIR/logs"

echo "RTX 5090 Training Monitor - $(date)"
while true; do
    echo ""
    echo "--- $(date) ---"
    nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    LATEST_LOG=$(ls -t "$LOG_DIR"/train_7day_*.log 2>/dev/null | head -1)
    [ -n "$LATEST_LOG" ] && echo "Latest:" && tail -3 "$LATEST_LOG" 2>/dev/null
    pgrep -f "train_7day.py" > /dev/null && echo "Status: RUNNING" || echo "Status: NOT RUNNING"
    sleep 3600
done
