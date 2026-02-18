#!/bin/bash
# Monitor training run every 2 hours for 14 hours (7 checks)
LOG="/tmp/claude-1000/-home-habitat-ternary-clawd/tasks/b682522.output"
REPORT="/home/habitat/ternary-clawd/nanochat-rs-ternary/checkpoints/nano125m_rust/monitor_report.txt"

echo "=== Training Monitor Started $(date) ===" | tee "$REPORT"
echo "Checking every 2 hours for 14 hours (7 checks)" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

for i in $(seq 1 7); do
    if [ $i -gt 1 ]; then
        sleep 7200  # 2 hours
    fi
    
    echo "=== Check $i/7 — $(date) ===" | tee -a "$REPORT"
    
    # Get latest training line
    LAST_LINE=$(tail -1 "$LOG" 2>/dev/null || echo "N/A")
    echo "Latest: $LAST_LINE" | tee -a "$REPORT"
    
    # Extract step number
    STEP=$(echo "$LAST_LINE" | grep -oP '\[\s*\K\d+' | head -1)
    if [ -n "$STEP" ]; then
        PCT=$((STEP * 100 / 50000))
        echo "Progress: step $STEP/50000 ($PCT%)" | tee -a "$REPORT"
    fi
    
    # GPU stats
    GPU_STATS=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null)
    echo "GPU: $GPU_STATS" | tee -a "$REPORT"
    
    # Checkpoints
    CKPTS=$(ls -d /home/habitat/ternary-clawd/nanochat-rs-ternary/checkpoints/nano125m_rust/step_* 2>/dev/null | wc -l)
    echo "Checkpoints saved: $CKPTS" | tee -a "$REPORT"
    
    # Check if process is still running
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
        echo "Status: RUNNING" | tee -a "$REPORT"
    else
        echo "Status: STOPPED (GPU process not found)" | tee -a "$REPORT"
    fi
    echo "" | tee -a "$REPORT"
done

echo "=== Monitor Complete $(date) ===" | tee -a "$REPORT"
