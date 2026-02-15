#!/bin/bash
# Train Small 560M model (d20 architecture) - Production code generation
# Duration: ~8 hours | Memory: 16GB GPU | Steps: 100K

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_NAME="small_560m_$(date +%Y%m%d_%H%M%S)"
RUNS_DIR="${WORKSPACE_DIR}/runs/${EXPERIMENT_NAME}"

mkdir -p "${RUNS_DIR}"
cd "${WORKSPACE_DIR}"

echo "Starting Small 560M (d20) training: ${EXPERIMENT_NAME}"
echo "Target: 100K steps (~8 hours)"
echo "Features: Hybrid attention (80% MHA, 20% DeltaNet)"
echo ""

# Build with TensorBoard
cargo build --release -p nanochat-train --features tensorboard,cuda

# Start TensorBoard
tensorboard --logdir "${RUNS_DIR}/tensorboard" --port 6006 &
TB_PID=$!
trap "kill $TB_PID 2>/dev/null || true" EXIT

# Train
RUST_LOG=info target/release/nanochat-train train \
    --config d20 \
    --dataset synthetic \
    --device cuda \
    --checkpoint-dir "${RUNS_DIR}/checkpoints" \
    2>&1 | tee "${RUNS_DIR}/training.log"

echo ""
echo "Training complete! Model saved to: ${RUNS_DIR}"
echo ""
echo "To export to GGUF:"
echo "  cargo run --release -p nanochat-train -- export \\"
echo "    --checkpoint ${RUNS_DIR}/final.safetensors \\"
echo "    --output models/nanochat-560m.gguf"
