#!/bin/bash
# Train Nano 125M model - Quick iteration and testing
# Duration: ~2 hours | Memory: 4GB GPU | Steps: 50K

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_NAME="nano_125m_$(date +%Y%m%d_%H%M%S)"
RUNS_DIR="${WORKSPACE_DIR}/runs/${EXPERIMENT_NAME}"

mkdir -p "${RUNS_DIR}"
cd "${WORKSPACE_DIR}"

echo "Starting Nano 125M training: ${EXPERIMENT_NAME}"
echo "Target: 50K steps (~2 hours)"
echo ""

# Build with TensorBoard
cargo build --release -p nanochat-train --features tensorboard,cuda

# Start TensorBoard
tensorboard --logdir "${RUNS_DIR}/tensorboard" --port 6006 &
TB_PID=$!
trap "kill $TB_PID 2>/dev/null || true" EXIT

# Train
RUST_LOG=info target/release/nanochat-train train \
    --config nano-125m \
    --dataset synthetic \
    --device cuda \
    --checkpoint-dir "${RUNS_DIR}/checkpoints" \
    2>&1 | tee "${RUNS_DIR}/training.log"

echo ""
echo "Training complete! Model saved to: ${RUNS_DIR}"
echo "View metrics: tensorboard --logdir ${RUNS_DIR}/tensorboard"
