#!/usr/bin/env bash
# monitor.sh — GPU and training health monitor for nanochat 7B training.
#
# Usage:
#   bash scripts/monitor.sh                    # Single check
#   while true; do bash scripts/monitor.sh; sleep 900; done  # Every 15 min
#
# Writes to logs/monitor.log with timestamps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/monitor.log"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints_7b}"
TRAINING_LOG="${TRAINING_LOG:-${PROJECT_DIR}/training.log}"

# Alert thresholds
GPU_MEM_WARN_GB=90
GPU_TEMP_WARN=85
GPU_UTIL_WARN=50
LOSS_DIVERGE_THRESHOLD=20.0
LOSS_SPIKE_FACTOR=5
STALL_TIMEOUT_MIN=30
DISK_WARN_PCT=85

mkdir -p "$LOG_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*" | tee -a "$LOG_FILE"
}

alert() {
    echo "[$(timestamp)] ALERT: $*" | tee -a "$LOG_FILE" >&2
}

# --- GPU Check ---
check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        log "GPU: nvidia-smi not found, skipping GPU check"
        return
    fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || echo "")

    if [[ -z "$gpu_info" ]]; then
        alert "GPU: nvidia-smi failed"
        return
    fi

    IFS=',' read -r mem_used mem_total temp util <<< "$gpu_info"
    mem_used=$(echo "$mem_used" | tr -d ' ')
    mem_total=$(echo "$mem_total" | tr -d ' ')
    temp=$(echo "$temp" | tr -d ' ')
    util=$(echo "$util" | tr -d ' ')

    local mem_gb=$((mem_used / 1024))
    local mem_total_gb=$((mem_total / 1024))

    log "GPU: ${mem_gb}/${mem_total_gb}GB VRAM, ${temp}C, ${util}% util"

    if (( mem_gb > GPU_MEM_WARN_GB )); then
        alert "GPU VRAM high: ${mem_gb}GB / ${mem_total_gb}GB (threshold: ${GPU_MEM_WARN_GB}GB)"
    fi
    if (( temp > GPU_TEMP_WARN )); then
        alert "GPU temp high: ${temp}C (threshold: ${GPU_TEMP_WARN}C) — throttling risk"
    fi
    if (( util < GPU_UTIL_WARN )); then
        alert "GPU utilization low: ${util}% (threshold: ${GPU_UTIL_WARN}%) — data loading bottleneck?"
    fi
}

# --- Process Check ---
check_process() {
    if pgrep -f "nanochat-train" > /dev/null 2>&1; then
        local pid
        pid=$(pgrep -f "nanochat-train" | head -1)
        log "Process: nanochat-train alive (PID $pid)"
    else
        alert "Process: nanochat-train NOT RUNNING"
    fi
}

# --- Training Progress Check ---
check_training_progress() {
    if [[ ! -f "$TRAINING_LOG" ]]; then
        log "Training log not found: $TRAINING_LOG"
        return
    fi

    # Parse latest training log line with step info
    local latest_line
    latest_line=$(grep -E '^\[' "$TRAINING_LOG" 2>/dev/null | tail -1 || echo "")

    if [[ -z "$latest_line" ]]; then
        log "Training: no step data found in log"
        return
    fi

    # Extract step, loss, tok/s from format: [  step/total ] loss=X.XXXX ...
    local step loss toks_per_sec
    step=$(echo "$latest_line" | grep -oP '\[\s*\K\d+' || echo "0")
    loss=$(echo "$latest_line" | grep -oP 'loss=\K[0-9.]+' || echo "0")
    toks_per_sec=$(echo "$latest_line" | grep -oP 'tok/s=\K[0-9.]+' || echo "0")

    log "Training: step=$step, loss=$loss, tok/s=$toks_per_sec"

    # Check for divergence
    if command -v bc &>/dev/null; then
        if (( $(echo "$loss > $LOSS_DIVERGE_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
            alert "Loss divergence: $loss > threshold $LOSS_DIVERGE_THRESHOLD"
        fi

        if (( $(echo "$toks_per_sec < 100" | bc -l 2>/dev/null || echo 0) )); then
            alert "Low throughput: ${toks_per_sec} tok/s — data loading bottleneck?"
        fi
    fi

    # Check for stall (compare log file modification time)
    local log_mtime
    log_mtime=$(stat -c %Y "$TRAINING_LOG" 2>/dev/null || echo 0)
    local now
    now=$(date +%s)
    local age_min=$(( (now - log_mtime) / 60 ))

    if (( age_min > STALL_TIMEOUT_MIN )); then
        alert "Training stall: log not updated for ${age_min} minutes (threshold: ${STALL_TIMEOUT_MIN})"
    fi
}

# --- Disk Check ---
check_disk() {
    local usage_pct
    usage_pct=$(df "$PROJECT_DIR" --output=pcent 2>/dev/null | tail -1 | tr -d ' %' || echo "0")

    local avail_gb
    avail_gb=$(df "$PROJECT_DIR" --output=avail --block-size=1G 2>/dev/null | tail -1 | tr -d ' ' || echo "0")

    log "Disk: ${usage_pct}% used, ${avail_gb}GB available"

    if (( usage_pct > DISK_WARN_PCT )); then
        alert "Disk usage high: ${usage_pct}% (threshold: ${DISK_WARN_PCT}%)"
    fi

    # Check checkpoint dir size
    if [[ -d "$CHECKPOINT_DIR" ]]; then
        local ckpt_size
        ckpt_size=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1 || echo "unknown")
        log "Checkpoints: $ckpt_size in $CHECKPOINT_DIR"
    fi
}

# --- Main ---
log "=== Monitor Check ==="
check_gpu
check_process
check_training_progress
check_disk
log "=== Done ==="
