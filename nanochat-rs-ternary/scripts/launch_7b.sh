#!/usr/bin/env bash
# launch_7b.sh — Launch 7B ternary model training with monitoring.
#
# Usage:
#   bash scripts/launch_7b.sh                     # Fresh start
#   bash scripts/launch_7b.sh --resume step_500   # Resume from checkpoint
#
# Prerequisites:
#   - CUDA toolkit available
#   - Data tokenized at data/owt_train/tokens.bin
#   - ~1.5TB free disk space (checkpoints + model)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_7b"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/owt_train/tokens.bin}"
TRAINING_LOG="${PROJECT_DIR}/training.log"

# Training config
CONFIG="large-7b-6day"
DEVICE="cuda"
EPOCHS=100  # total_steps=1500 is the real limit
LOG_INTERVAL=10
CHECKPOINT_INTERVAL=100
KEEP_LAST_CHECKPOINTS=5   # ~140GB for 5 checkpoints (conservative for disk)
MONITOR_INTERVAL=900  # 15 minutes

# Parse arguments
RESUME_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_DIR="${CHECKPOINT_DIR}/${2:?Missing checkpoint name}"
            if [[ ! -d "$RESUME_DIR" ]]; then
                echo "ERROR: Checkpoint directory not found: $RESUME_DIR"
                exit 1
            fi
            RESUME_ARG="--resume $RESUME_DIR"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: launch_7b.sh [--resume step_N]"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "============================================"
echo "  nanochat 7B Ternary Training Launch"
echo "============================================"
echo ""
echo "Config:       $CONFIG"
echo "Device:       $DEVICE"
echo "Data:         $DATA_PATH"
echo "Checkpoints:  $CHECKPOINT_DIR"
echo "Logs:         $LOG_DIR"
echo "Resume:       ${RESUME_ARG:-fresh start}"
echo ""

# Verify prerequisites
echo "--- Pre-flight checks ---"

# Check CUDA
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found — CUDA required"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check data
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: Training data not found: $DATA_PATH"
    echo "Run: nanochat-train prepare-data --text <text_file> --vocab-size 128000 --output data/owt_train/"
    exit 1
fi
DATA_SIZE=$(du -h "$DATA_PATH" | cut -f1)
echo "Training data: $DATA_PATH ($DATA_SIZE)"

# Check disk space
DISK_AVAIL_GB=$(df "$PROJECT_DIR" --output=avail --block-size=1G | tail -1 | tr -d ' ')
echo "Disk available: ${DISK_AVAIL_GB}GB"
if (( DISK_AVAIL_GB < 300 )); then
    echo "WARNING: Less than 300GB free — checkpoints may fill disk"
    echo "Consider reducing --keep-last-checkpoints or cleaning up"
fi
echo ""

# Build
echo "--- Building with CUDA support ---"
cargo build --release -p nanochat-train --features cuda 2>&1 | tail -5
echo ""

# Start monitoring daemon in background
echo "--- Starting monitor daemon (every ${MONITOR_INTERVAL}s) ---"
(
    export CHECKPOINT_DIR TRAINING_LOG
    while true; do
        bash "${SCRIPT_DIR}/monitor.sh" 2>/dev/null || true
        sleep "$MONITOR_INTERVAL"
    done
) &
MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping monitor (PID $MONITOR_PID)..."
    kill "$MONITOR_PID" 2>/dev/null || true
    echo "Training ended at $(date)"
}
trap cleanup EXIT

# Launch training
echo ""
echo "--- Launching training ---"
echo "Training output: $TRAINING_LOG"
echo "Press Ctrl+C to stop (checkpoint will be saved)"
echo ""

RUST_LOG=nanochat_train=info \
cargo run --release -p nanochat-train --features cuda -- train \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --dataset tokens \
    --data-path "$DATA_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS" \
    --log-interval "$LOG_INTERVAL" \
    --epochs "$EPOCHS" \
    $RESUME_ARG \
    2>&1 | tee "$TRAINING_LOG"

echo ""
echo "=== Training Complete ==="
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"
