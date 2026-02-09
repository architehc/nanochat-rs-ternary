#!/bin/bash
# Monitor production training progress

echo "═══════════════════════════════════════════════════════════"
echo "  Production Training Monitor"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Find most recent log file
LOG_FILE=$(ls -t production_training_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No training log found. Training may not have started yet."
    exit 1
fi

echo "Monitoring log: $LOG_FILE"
echo ""

# Check if training is running
if pgrep -f "train_rust_maxgpu\|train_rl" > /dev/null; then
    echo "✓ Training process is ACTIVE"
    TRAIN_PID=$(pgrep -f "train_rust_maxgpu\|train_rl")
    echo "  PID: $TRAIN_PID"
else
    echo "⚠ Training process is NOT running (may be between restarts)"
fi

echo ""

# GPU Status
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader | \
    awk -F, '{printf "  GPU Util: %s | VRAM Util: %s | VRAM Used: %s /%s\n", $1, $2, $3, $4}'

echo ""

# Current phase detection
if grep -q "PHASE 1" "$LOG_FILE" && ! grep -q "Phase 1 Complete" "$LOG_FILE"; then
    CURRENT_PHASE="Phase 1: Supervised Pre-training"
elif grep -q "PHASE 2" "$LOG_FILE" && ! grep -q "Phase 2 Complete" "$LOG_FILE"; then
    CURRENT_PHASE="Phase 2: RL Fine-tuning"
elif grep -q "PHASE 3" "$LOG_FILE"; then
    CURRENT_PHASE="Phase 3: Evaluation"
else
    CURRENT_PHASE="Starting..."
fi

echo "Current Phase: $CURRENT_PHASE"
echo ""

# Check supervised training progress
if [ -d "checkpoints/production-supervised" ]; then
    LAST_SUPERVISED=$(ls -d checkpoints/production-supervised/step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_SUPERVISED" ]; then
        CURRENT_STEP=$(basename "$LAST_SUPERVISED" | sed 's/step_//')
        PROGRESS=$((($CURRENT_STEP - 2000) * 100 / 13000))
        echo "Supervised Training Progress:"
        echo "  Latest checkpoint: step_$CURRENT_STEP"
        echo "  Progress: ${PROGRESS}% (2000 → 15000)"
        echo "  Remaining: $((15000 - CURRENT_STEP)) steps"

        # Show progress bar
        FILLED=$((PROGRESS / 2))
        printf "  ["
        for i in $(seq 1 50); do
            if [ $i -le $FILLED ]; then
                printf "="
            else
                printf " "
            fi
        done
        printf "] ${PROGRESS}%%\n"

        # Storage used
        SIZE=$(du -sh checkpoints/production-supervised 2>/dev/null | cut -f1)
        COUNT=$(ls -d checkpoints/production-supervised/step_* 2>/dev/null | wc -l)
        echo "  Checkpoints: $COUNT (${SIZE:-0B})"
    fi
fi

echo ""

# Check RL training progress
if [ -f "rl_training.log" ]; then
    RL_ITERATIONS=$(wc -l < rl_training.log)
    if [ $RL_ITERATIONS -gt 0 ]; then
        echo "RL Training Progress:"
        echo "  Iterations completed: $RL_ITERATIONS / 1000"

        # Recent stats
        RECENT=$(tail -1 rl_training.log)
        AVG_REWARD=$(echo "$RECENT" | cut -d, -f2)
        COMPILE_RATE=$(echo "$RECENT" | cut -d, -f4)
        echo "  Recent avg reward: $AVG_REWARD"
        echo "  Compilation rate: $(echo "$COMPILE_RATE * 100" | bc)%"
    fi
    echo ""
fi

# Recent training output
echo "Recent Training Output:"
echo "───────────────────────────────────────────────────────────"
tail -25 "$LOG_FILE" | grep -E "(loss=|Progress:|Step|Iteration|✓|→|Error|OOM)" || tail -10 "$LOG_FILE"
echo "───────────────────────────────────────────────────────────"

echo ""
echo "Commands:"
echo "  Watch live:        tail -f $LOG_FILE"
echo "  GPU monitor:       watch -n 1 nvidia-smi"
echo "  Re-run monitor:    ./scripts/monitor_production.sh"
echo ""
