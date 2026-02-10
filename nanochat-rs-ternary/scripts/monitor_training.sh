#!/bin/bash
# Monitor training progress in real-time

LOG_FILE=${1:-"training_fresh.log"}

echo "═══════════════════════════════════════════════════════════"
echo "  Training Monitor"
echo "  Log: $LOG_FILE"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if training is running
if ! ps aux | grep -q "[t]rain_rust_maxgpu"; then
    echo "⚠️  Warning: No training process found!"
    echo ""
fi

# Show latest progress
echo "Latest Progress:"
grep -E "\[.*/" "$LOG_FILE" | tail -5
echo ""

# Show loss trend
echo "Loss Trend (last 10 checkpoints):"
grep "loss=" "$LOG_FILE" | tail -10 | awk '{print $2, $3}' | column -t
echo ""

# Calculate progress
CURRENT=$(grep -E "\[.*/" "$LOG_FILE" | tail -1 | sed 's/.*\[\s*\([0-9]*\).*/\1/')
TOTAL=$(grep -E "\[.*/" "$LOG_FILE" | tail -1 | sed 's/.*\/\s*\([0-9]*\).*/\1/')

if [ -n "$CURRENT" ] && [ -n "$TOTAL" ]; then
    PERCENT=$((CURRENT * 100 / TOTAL))
    REMAINING=$((TOTAL - CURRENT))
    echo "Progress: $CURRENT/$TOTAL ($PERCENT%) - $REMAINING steps remaining"
    echo ""
fi

# Check for issues
if grep -qi "error\|oom\|killed" "$LOG_FILE"; then
    echo "⚠️  Issues detected:"
    grep -i "error\|oom\|killed" "$LOG_FILE" | tail -3
    echo ""
fi

# Show checkpoints
if [ -d "checkpoints/stable-v2" ]; then
    echo "Saved Checkpoints:"
    ls -lth checkpoints/stable-v2/ | grep "^d" | head -5
    echo ""
fi

echo "─────────────────────────────────────────────────────────"
echo "Commands:"
echo "  Watch live:  tail -f $LOG_FILE"
echo "  Full log:    less $LOG_FILE"
echo "  Kill:        pkill -f train_rust_maxgpu"
echo "═══════════════════════════════════════════════════════════"
