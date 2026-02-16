#!/bin/bash
# Monitor training progress in real-time

set -eo pipefail

LOG_FILE=${1:-"training_fresh.log"}

echo "========================================================"
echo "  Training Monitor"
echo "  Log: $LOG_FILE"
echo "========================================================"
echo ""

# Check that LOG_FILE exists before processing
if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    echo "Usage: $0 <path-to-log-file>"
    exit 1
fi

# Check if training is running
if ! ps aux | grep -q "[t]rain_rust_maxgpu\|[n]anochat-train"; then
    echo "Warning: No training process found!"
    echo ""
fi

# Show latest progress
echo "Latest Progress:"
PROGRESS=$(grep -E "\[.*/" "$LOG_FILE" 2>/dev/null | tail -5 || true)
if [ -n "$PROGRESS" ]; then
    echo "$PROGRESS"
else
    echo "  (no progress entries found)"
fi
echo ""

# Show loss trend
echo "Loss Trend (last 10 checkpoints):"
LOSS_TREND=$(grep "loss=" "$LOG_FILE" 2>/dev/null | tail -10 | awk '{print $2, $3}' | column -t 2>/dev/null || true)
if [ -n "$LOSS_TREND" ]; then
    echo "$LOSS_TREND"
else
    echo "  (no loss entries found)"
fi
echo ""

# Calculate progress
CURRENT=$(grep -E "\[.*/" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*\[\s*\([0-9]*\).*/\1/' || true)
TOTAL=$(grep -E "\[.*/" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*\/\s*\([0-9]*\).*/\1/' || true)

if [ -n "$CURRENT" ] && [ -n "$TOTAL" ] && [ "$TOTAL" -gt 0 ] 2>/dev/null; then
    PERCENT=$((CURRENT * 100 / TOTAL))
    REMAINING=$((TOTAL - CURRENT))
    echo "Progress: $CURRENT/$TOTAL ($PERCENT%) - $REMAINING steps remaining"
    echo ""
else
    echo "Progress: (unable to determine from log)"
    echo ""
fi

# Check for issues
ISSUES=$(grep -i "error\|oom\|killed" "$LOG_FILE" 2>/dev/null | tail -3 || true)
if [ -n "$ISSUES" ]; then
    echo "Warning: Issues detected:"
    echo "$ISSUES"
    echo ""
fi

# Show checkpoints
if [ -d "checkpoints/stable-v2" ]; then
    echo "Saved Checkpoints:"
    ls -lth checkpoints/stable-v2/ | grep "^d" | head -5 || echo "  (no checkpoint directories found)"
    echo ""
fi

echo "--------------------------------------------------------"
echo "Commands:"
echo "  Watch live:  tail -f $LOG_FILE"
echo "  Full log:    less $LOG_FILE"
echo "  Kill:        pkill -f 'train_rust_maxgpu|nanochat-train'"
echo "========================================================"
