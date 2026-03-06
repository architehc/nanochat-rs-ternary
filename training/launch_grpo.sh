#!/bin/bash
# =============================================================================
# GRPO RL Pipeline — runs after pre-training produces a good checkpoint
#
# Usage: bash training/launch_grpo.sh [checkpoint_dir]
#
# Default: uses latest checkpoint from 5090_v2 training
# The RL pipeline:
#   1. Loads pre-trained checkpoint
#   2. Generates Rust code from coding prompts
#   3. Compiles each sample with rustc for ground-truth reward
#   4. AST analysis for code quality metrics
#   5. GRPO policy gradient updates
# =============================================================================

set -euo pipefail

REPO_DIR="/home/galic/nanochat-rs-ternary/nanochat-rs-ternary"
LOG_FILE="/home/galic/nanochat-rs-ternary/training/grpo_train.log"

cd "$REPO_DIR"

# Find latest checkpoint if not specified
if [ -n "${1:-}" ]; then
    CHECKPOINT_DIR="$1"
else
    CHECKPOINT_DIR=$(ls -td checkpoints/5090_v2/step_* 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "ERROR: No checkpoints found in checkpoints/5090_v2/"
        exit 1
    fi
fi

# Set tokenizer path
export NANOCHAT_TOKENIZER="data/rust_v3_4k/tokenizer.json"

echo "======================================================================="
echo "GRPO RL Training Pipeline"
echo "======================================================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Tokenizer:  $NANOCHAT_TOKENIZER"
echo "Log:        $LOG_FILE"
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_DIR/model.safetensors" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_DIR"
    echo "Available checkpoints:"
    ls -td checkpoints/5090_v2/step_* 2>/dev/null | head -5
    exit 1
fi

# Build with CUDA
echo "Building nanochat-rl..."
CUDA_ARCH=sm_120 cargo build --release --example train_rl -p nanochat-rl --features cuda 2>&1 | tail -3

echo ""
echo "Launching GRPO training..."

# Launch GRPO with compiler feedback
nohup cargo run --release --example train_rl -p nanochat-rl --features cuda -- \
    --checkpoint "$CHECKPOINT_DIR" \
    --device cuda:0 \
    --iterations 500 \
    --n-samples 4 \
    --batch-size 2 \
    --lr 1e-5 \
    --kl-coef 0.1 \
    > "$LOG_FILE" 2>&1 &

GRPO_PID=$!
echo "$GRPO_PID" > /home/galic/nanochat-rs-ternary/training/grpo_train.pid
echo "GRPO PID: $GRPO_PID"
echo ""
echo "Monitor with: tail -f $LOG_FILE"
