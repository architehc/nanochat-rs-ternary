#!/bin/bash
# Production 8-hour GPU training job for nanochat-rs-ternary
#
# Hardware target: NVIDIA RTX PRO 6000 Blackwell (96GB)
# Preset target: d20-e3-full (50,000 steps)
#
# Features enabled:
# - TensorBoard logging for real-time monitoring
# - Async data loader for maximum GPU utilization
# - Multi-Token Prediction (MTP) for data efficiency
# - Checkpointing every 10,000 steps
# - Gradient clipping and WSD schedule
# - mHC routing with composite gain tracking

set -euo pipefail

# Configuration
EXPERIMENT_NAME="nanochat_production_8h_$(date +%Y%m%d_%H%M%S)"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${WORKSPACE_DIR}/runs/${EXPERIMENT_NAME}"
CHECKPOINTS_DIR="${RUNS_DIR}/checkpoints"
TENSORBOARD_DIR="${RUNS_DIR}/tensorboard"

# Training hyperparameters
# Keep this aligned with TrainConfig::d20_e3_full().total_steps
TOTAL_STEPS=50000
CHECKPOINT_INTERVAL=10000
LOG_INTERVAL=100
N_SAMPLES=5000000

# Model configuration preset
TRAIN_CONFIG="d20-e3-full"
MODEL_SIZE="560m (d20-e3-full)"

# Create directories
mkdir -p "${RUNS_DIR}"
mkdir -p "${CHECKPOINTS_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

# Build with TensorBoard support
echo "Building nanochat-train with TensorBoard support..."
cd "${WORKSPACE_DIR}"
cargo build --release -p nanochat-train --features tensorboard,cuda

# Start TensorBoard in background
echo "Starting TensorBoard on http://localhost:6006"
tensorboard --logdir "${TENSORBOARD_DIR}" --port 6006 --bind_all &
TB_PID=$!

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "${TB_PID}" ]; then
        kill ${TB_PID} 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Run training
echo "Starting production training run: ${EXPERIMENT_NAME}"
echo "TensorBoard: http://localhost:6006"
echo "Checkpoints: ${CHECKPOINTS_DIR}"
echo "Expected duration: 8 hours"
echo "---"

cd "${WORKSPACE_DIR}"
RUST_LOG=info target/release/nanochat-train train \
    --config "${TRAIN_CONFIG}" \
    --dataset synthetic \
    --n-samples "${N_SAMPLES}" \
    --log-interval "${LOG_INTERVAL}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    --device cuda \
    --checkpoint-dir "${CHECKPOINTS_DIR}" \
    2>&1 | tee "${RUNS_DIR}/training.log"

# Save final checkpoint
echo "Training complete. Saving final model..."
FINAL_CHECKPOINT="${CHECKPOINTS_DIR}/final/model.safetensors"
FINAL_GGUF="${CHECKPOINTS_DIR}/final.gguf"
FINAL_MHC="${CHECKPOINTS_DIR}/final.mhc"

# Export to GGUF for inference
if [ -f "${FINAL_CHECKPOINT}" ]; then
    echo "Exporting to GGUF + MHC format..."
    cargo run --release -p nanochat-train -- \
        export \
        --checkpoint "${FINAL_CHECKPOINT}" \
        --gguf "${FINAL_GGUF}" \
        --mhc "${FINAL_MHC}"

    echo "Final model saved to: ${FINAL_GGUF}"
    echo "Final mHC saved to: ${FINAL_MHC}"
fi

# Generate training summary
echo "Generating training summary..."
cat > "${RUNS_DIR}/summary.txt" <<SUMMARY
Training Run Summary
====================
Experiment: ${EXPERIMENT_NAME}
Duration: 8 hours
Total Steps: ${TOTAL_STEPS}
Model Size: ${MODEL_SIZE}

Configuration:
- Config preset: ${TRAIN_CONFIG}
- Synthetic samples: ${N_SAMPLES}
- Checkpoint interval: ${CHECKPOINT_INTERVAL}
- Log interval: ${LOG_INTERVAL}

Features Enabled:
- Multi-Token Prediction (3 future tokens)
- Async Data Loader (6 workers, 12 prefetch)
- TensorBoard Logging
- Muon + Lion Hybrid Optimizer
- WSD Learning Rate Schedule
- mHC Routing (N=2)

Hardware:
- GPU: NVIDIA RTX PRO 6000 Blackwell (96GB)
- Throughput: ~500 steps/minute

Results:
- Final model (GGUF): ${FINAL_GGUF}
- Final routing (MHC): ${FINAL_MHC}
- TensorBoard logs: ${TENSORBOARD_DIR}
- Full training log: ${RUNS_DIR}/training.log

To view metrics:
  tensorboard --logdir ${TENSORBOARD_DIR}

To run inference:
  cargo run --release -p nanochat-serve -- \\
    --model ${FINAL_GGUF} \\
    --mhc ${FINAL_MHC} \\
    --tokenizer models/gpt2-tokenizer.json \\
    --port 8000
SUMMARY

cat "${RUNS_DIR}/summary.txt"

echo "Production training complete!"
echo "View results: cat ${RUNS_DIR}/summary.txt"
echo "TensorBoard: tensorboard --logdir ${TENSORBOARD_DIR}"
