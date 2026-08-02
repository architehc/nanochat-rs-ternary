#!/bin/bash
# =============================================================================
# RTX 5090 Training Monitor — runs every 30 minutes, logs metrics, pushes to git
# Usage: nohup bash training/monitor_5090.sh &
# =============================================================================

set -euo pipefail

REPO_DIR="/home/galic/nanochat-rs-ternary"
TRAIN_LOG="$REPO_DIR/training/5090_v4_train.log"
MONITOR_LOG="$REPO_DIR/training/monitor_5090.log"
PID_FILE="$REPO_DIR/training/5090_v4_train.pid"
REPORT_FILE="$REPO_DIR/training/TRAINING_PROGRESS.md"
BRANCH="5090"
INTERVAL_SECS=1800  # 30 minutes

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

get_latest_gnorm() {
    tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'gnorm=\K[0-9.]+' || echo "N/A"
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
    local gnorm=$(get_latest_gnorm)
    local temp=$(get_gpu_temp)
    local util=$(get_gpu_util)
    local mem=$(get_gpu_mem)
    local power=$(get_gpu_power)
    local tok_s=$(tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'tok/s=\K[0-9]+' || echo "N/A")
    local lr=$(tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'lr=\K[0-9.]+' || echo "N/A")
    local elapsed=$(tail -1 "$TRAIN_LOG" 2>/dev/null | grep -oP 'elapsed=\K[0-9]+' || echo "0")
    local hours=$(echo "$elapsed / 3600" | bc 2>/dev/null || echo "0")

    cat > "$REPORT_FILE" << EOF
# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v4
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v4 (143M tokens, 65K files from 187 repos, vocab=4096)
- **Includes**: tokio, serde, bevy, rust-lang/rust stdlib, solana, reth, CS algorithms
- **Schedule**: WSD lr=0.006, decay at 40% (step 20K), 50K total steps
- **Resumed from**: v3/final (loss 2.72, 30K steps)

## Latest Metrics ($(date '+%Y-%m-%d %H:%M'))
| Metric | Value |
|--------|-------|
| Step | $step / 50,000 |
| Loss | $loss |
| Grad Norm | $gnorm |
| Learning Rate | $lr |
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
| 5090-v2 | 106M tok | 21K/150K | ~4.1 | gnorm blowup (lr=0.012), killed |
| 5090-v3 | 106M tok | 30K | 2.72 | lr=0.008, decay@20%, stable |
| **5090-v4** | **143M tok** | **$step/50K** | **$loss** | **Current (lr=0.006, 65K files, 187 repos)** |

## Loss Trajectory (last 20 readings)
\`\`\`
$(tail -20 "$TRAIN_LOG" 2>/dev/null | grep -oP '\[\s*\d+/\d+\] loss=[0-9.]+ .* gnorm=[0-9.]+' || echo "No data yet")
\`\`\`

---
*Auto-updated by monitor_5090.sh every 30 min at $(date '+%Y-%m-%d %H:%M:%S')*
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

    git commit -m "5090-v4: step $step, loss=$loss (30-min update)" 2>/dev/null || true
    git push origin "$BRANCH" 2>/dev/null && log "Pushed to $BRANCH" || log "Push failed"
}

# ============================================================================
# MAIN LOOP
# ============================================================================

log "=== Monitor started (v3) ==="
log "Training log: $TRAIN_LOG"
log "Check interval: ${INTERVAL_SECS}s (30 min)"

while true; do
    log "--- 30-min Check ---"

    if check_training; then
        step=$(get_latest_metrics)
        loss=$(get_latest_loss)
        gnorm=$(get_latest_gnorm)
        temp=$(get_gpu_temp)
        util=$(get_gpu_util)

        log "RUNNING: step=$step loss=$loss gnorm=$gnorm temp=${temp}C util=${util}%"

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
            log "Pre-training done! Ready for GRPO RL phase."
            log "Launch with: bash training/launch_grpo.sh"
            break
        else
            log "Training crashed! Last lines:"
            tail -5 "$TRAIN_LOG" 2>/dev/null | while read line; do log "  $line"; done
            log "Attempting restart..."
            # Auto-restart from latest v3 checkpoint
            latest_ckpt=$(ls -td checkpoints/5090_v4/step_* 2>/dev/null | head -1)
            if [ -z "$latest_ckpt" ]; then
                # Fall back to v3 final if no v4 checkpoints yet
                latest_ckpt="checkpoints/5090_v3/final"
            fi
            if [ -d "$latest_ckpt" ]; then
                log "Resuming from $latest_ckpt"
                nohup bash -c "CUDA_ARCH=sm_120 cargo run --release -p nanochat-train --features cuda -- train \
                  --config nano-275m-engram-5090-v4 \
                  --dataset tokens \
                  --data-path data/rust_v4_4k_v3tok/tokens.bin \
                  --epochs 200 \
                  --batch-size 2 \
                  --seq-len 512 \
                  --checkpoint-dir checkpoints/5090_v4 \
                  --checkpoint-interval 2000 \
                  --keep-last-checkpoints 10 \
                  --log-interval 50 \
                  --device cuda:0 \
                  --threads 16 \
                  --total-steps 50000 \
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
