#!/bin/bash
# Monitor training runs, check for failures, disk space, etc.
MAX_HOURS=10
CHECK_INTERVAL=900
MIN_DISK_GB=10
START_TIME=$(date +%s)
LOG_FILE="monitor.log"

echo "$(date): Training monitor started" | tee -a "$LOG_FILE"
echo "  Monitoring: Engram (GPU 0), LoopLM (GPU 1)" | tee -a "$LOG_FILE"

check_iteration=0
while true; do
    check_iteration=$((check_iteration + 1))
    elapsed=$(( ($(date +%s) - START_TIME) / 60 ))
    echo "=== Check #${check_iteration} at $(date) (${elapsed}m elapsed) ===" | tee -a "$LOG_FILE"

    disk_free_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    echo "  Disk free: ${disk_free_gb}GB" | tee -a "$LOG_FILE"
    [ "$disk_free_gb" -lt "$MIN_DISK_GB" ] && echo "  WARNING: Low disk!" | tee -a "$LOG_FILE"

    nvidia-smi --query-gpu=index,memory.used,temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
        echo "  GPU: $line" | tee -a "$LOG_FILE"
    done

    for model in engram loop; do
        if [ "$model" = "engram" ]; then
            logf="training_engram_only.log"
            grep_pat="nano-275m-engram-only"
        else
            logf="training_loop_only.log"
            grep_pat="nano-275m-loop-only"
        fi
        pid=$(pgrep -f "$grep_pat" 2>/dev/null | head -1)
        if [ -n "$pid" ]; then
            last=$(tail -1 "$logf" 2>/dev/null)
            step=$(echo "$last" | grep -oP '\[\s*\K\d+' | head -1)
            loss=$(echo "$last" | grep -oP 'loss=\K[0-9.]+')
            toks=$(echo "$last" | grep -oP 'tok/s=\K[0-9]+')
            echo "  ${model}: PID=$pid step=$step loss=$loss tok/s=$toks" | tee -a "$LOG_FILE"
            if tail -5 "$logf" 2>/dev/null | grep -qi "nan.*loss\|Error:\|panic\|ABORT"; then
                echo "  ALERT: ${model} FAILED!" | tee -a "$LOG_FILE"
                tail -3 "$logf" | tee -a "$LOG_FILE"
            fi
        else
            echo "  ${model}: NOT RUNNING" | tee -a "$LOG_FILE"
            tail -2 "$logf" 2>/dev/null | tee -a "$LOG_FILE"
        fi
    done

    [ -z "$(pgrep -f 'nano-275m-engram-only')" ] && [ -z "$(pgrep -f 'nano-275m-loop-only')" ] && {
        echo "  Both done." | tee -a "$LOG_FILE"; break
    }

    elapsed_hours=$(( ($(date +%s) - START_TIME) / 3600 ))
    [ "$elapsed_hours" -ge "$MAX_HOURS" ] && { echo "  Max time reached." | tee -a "$LOG_FILE"; break; }

    for dir in checkpoints/nano-275m-engram-only checkpoints/nano-275m-loop-only; do
        latest=$(ls -1d ${dir}/step_* 2>/dev/null | sort -V | tail -1 || true)
        [ -n "$latest" ] && echo "  ckpt: $(basename $dir)/$(basename $latest)" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
    sleep "$CHECK_INTERVAL"
done

echo "$(date): Monitor finished." | tee -a "$LOG_FILE"
