#!/bin/bash
# =============================================================================
# RTX 5090 Training Monitor — runs every hour, logs metrics, pushes to git
# Usage: nohup bash training/monitor_5090.sh &
# =============================================================================

set -euo pipefail

REPO_DIR="/home/galic/nanochat-rs-ternary"
TRAIN_LOG="$REPO_DIR/training/5090_v2_train.log"
MONITOR_LOG="$REPO_DIR/training/monitor_5090.log"
PID_FILE="$REPO_DIR/training/5090_v2_train.pid"
REPORT_FILE="$REPO_DIR/training/TRAINING_PROGRESS.md"
BRANCH="5090"
INTERVAL_SECS=3600  # 1 hour

cd "$REPO_DIR/nanochat-rs-ternary"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

check_training() {
    local pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0  # running
    fi
    return 1  # not running
}

get_latest_metrics() {
    tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP '\[\s*\K\d+(?=/\d+\])' || echo "0"
}

get_latest_loss() {
    tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'loss=\K[0-9.]+' || echo "N/A"
}

get_gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo "N/A"
}

get_gpu_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %' || echo "0"
}

get_gpu_mem() {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0, 0"
}

get_gpu_power() {
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null || echo "0"
}

update_report() {
    local step=$(get_latest_metrics)
    local loss=$(get_latest_loss)
    local temp=$(get_gpu_temp)
    local util=$(get_gpu_util)
    local mem=$(get_gpu_mem)
    local power=$(get_gpu_power)
    local tok_s=$(tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'tok/s=\K[0-9]+' || echo "N/A")
    local lr=$(tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'lr=\K[0-9.]+' || echo "N/A")
    local gnorm=$(tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'gnorm=\K[0-9.]+' || echo "N/A")
    local elapsed=$(tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'elapsed=\K[0-9]+' || echo "0")
    local hours=$(echo "$elapsed / 3600" | bc 2>/dev/null || echo "0")

    cat > "$REPORT_FILE" << EOF
# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps

## Latest Metrics ($(date '+%Y-%m-%d %H:%M'))
| Metric | Value |
|--------|-------|
| Step | $step / 150,000 |
| Loss | $loss |
| Learning Rate | $lr |
| Grad Norm | $gnorm |
| Tokens/sec | $tok_s |
| Elapsed | ${hours}h (${elapsed}s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | ${temp}°C |
| Utilization | ${util}% |
| VRAM | ${mem} MiB |
| Power | ${power}W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **$step/150K** | **$loss** | **Current run** |

## Loss Trajectory (last 20 readings)
\`\`\`
$(tail -20 "$TRAIN_LOG" 2>/dev/null | grep -oP '\[\s*\d+/\d+\] loss=[0-9.]+' || echo "No data yet")
\`\`\`

---
*Auto-updated by monitor_5090.sh at $(date '+%Y-%m-%d %H:%M:%S')*
EOF
}

git_push() {
    cd "$REPO_DIR"
    git add -A training/TRAINING_PROGRESS.md training/monitor_5090.log 2>/dev/null || true
    git add -A nanochat-rs-ternary/crates/nanochat-train/src/config.rs 2>/dev/null || true
    git add -A nanochat-rs-ternary/crates/nanochat-train/src/main.rs 2>/dev/null || true
    git add -A training/generate_rust_dataset.py 2>/dev/null || true

    local step=$(get_latest_metrics)
    local loss=$(get_latest_loss)

    if git diff --cached --quiet 2>/dev/null; then
        log "No changes to commit"
        return
    fi

    git commit -m "5090-v2: step $step, loss=$loss (hourly update)" 2>/dev/null || true
    git push origin "$BRANCH" 2>/dev/null && log "Pushed to $BRANCH" || log "Push failed"
}

# ============================================================================
# MAIN LOOP
# ============================================================================

log "=== Monitor started ==="
log "Training log: $TRAIN_LOG"
log "Check interval: ${INTERVAL_SECS}s"

while true; do
    log "--- Hourly Check ---"

    if check_training; then
        step=$(get_latest_metrics)
        loss=$(get_latest_loss)
        temp=$(get_gpu_temp)
        util=$(get_gpu_util)

        log "RUNNING: step=$step loss=$loss temp=${temp}C util=${util}%"

        # Thermal protection: if GPU > 85C, alert
        if [ "$temp" -gt 85 ] 2>/dev/null; then
            log "WARNING: GPU temperature ${temp}C exceeds 85C threshold!"
        fi

        # Update report and push
        update_report
        git_push
    else
        log "TRAINING NOT RUNNING! Checking if completed or crashed..."
        if tail -3 "$TRAIN_LOG" 2>/dev/null | grep -q "Training complete"; then
            log "Training completed successfully!"
            update_report
            git_push
            log "Starting GRPO RL phase..."
            # TODO: launch GRPO here
            break
        else
            log "Training crashed! Last lines:"
            tail -5 "$TRAIN_LOG" 2>/dev/null | while read line; do log "  $line"; done
            log "Attempting restart..."
            # Auto-restart from latest checkpoint
            latest_ckpt=$(ls -td checkpoints/5090_v2/step_* 2>/dev/null | head -1)
            if [ -n "$latest_ckpt" ]; then
                log "Resuming from $latest_ckpt"
                nohup bash -c "CUDA_ARCH=sm_120 cargo run --release -p nanochat-train --features cuda -- train \
                  --config nano-275m-engram-5090-v2 \
                  --dataset tokens \
                  --data-path data/rust_v3_4k/tokens.bin \
                  --epochs 200 \
                  --batch-size 2 \
                  --seq-len 512 \
                  --checkpoint-dir checkpoints/5090_v2 \
                  --checkpoint-interval 2000 \
                  --keep-last-checkpoints 10 \
                  --log-interval 50 \
                  --device cuda:0 \
                  --threads 16 \
                  --total-steps 150000 \
                  --resume $latest_ckpt" >> "$TRAIN_LOG" 2>&1 &
                echo $! > "$PID_FILE"
                log "Restarted with PID $(cat $PID_FILE)"
            else
                log "No checkpoint found, cannot restart"
                break
            fi
        fi
    fi

    sleep $INTERVAL_SECS
done

log "=== Monitor stopped ==="
