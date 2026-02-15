#!/bin/bash
# Train Medium 3B MoE model - High-quality code generation
# Duration: ~24 hours | Memory: 48GB GPU | Steps: 200K

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_NAME="medium_3b_moe_$(date +%Y%m%d_%H%M%S)"
RUNS_DIR="${WORKSPACE_DIR}/runs/${EXPERIMENT_NAME}"

mkdir -p "${RUNS_DIR}"
cd "${WORKSPACE_DIR}"

echo "Starting Medium 3B MoE training: ${EXPERIMENT_NAME}"
echo "Target: 200K steps (~24 hours)"
echo "Features:"
echo "  - 8 experts, 2 active (MoE)"
echo "  - Hybrid attention (70% MHA, 30% DeltaNet)"
echo "  - 8-bit Muon optimizer"
echo "  - mHC N=4 routing"
echo ""

# Build with TensorBoard
cargo build --release -p nanochat-train --features tensorboard,cuda

# Start TensorBoard
tensorboard --logdir "${RUNS_DIR}/tensorboard" --port 6006 &
TB_PID=$!
trap "kill $TB_PID 2>/dev/null || true" EXIT

echo "TensorBoard: http://localhost:6006"
echo ""

# Train
RUST_LOG=info target/release/nanochat-train train \
    --config medium-3b \
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
echo "    --gguf models/nanochat-3b-moe.gguf \\"
echo "    --mhc models/nanochat-3b-moe.mhc"
echo ""
echo "Expected quality: Competitive with GPT-3.5 on Rust code generation"
