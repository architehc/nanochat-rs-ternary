#!/bin/bash
# train_config_a.sh - Threadripper 3995WX Pro + 96GB Blackwell Training Script
# Optimized for maximum memory bandwidth and large-batch training

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== nanochat-rs-ternary Training: Config A (Threadripper + Blackwell) ===${NC}"
echo ""

# Environment setup
echo -e "${YELLOW}Setting up environment...${NC}"
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1"
export CUDA_VISIBLE_DEVICES=0
export NUM_THREADS=128

# NUMA optimization (critical for Threadripper)
export NUMA_AWARE=1
export NUMA_NODES=8
export OMP_NUM_THREADS=128
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Memory optimization
export MALLOC_CONF="background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000"

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nanochat-rs-ternary"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/config_a"
CONFIG_FILE="${SCRIPT_DIR}/config_a_threadripper_blackwell.toml"

mkdir -p "${CHECKPOINT_DIR}"

echo -e "${GREEN}Environment configured:${NC}"
echo "  Project dir: ${PROJECT_DIR}"
echo "  Data dir: ${DATA_DIR}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "  Config: ${CONFIG_FILE}"
echo ""

# Stage 1: Data Preprocessing
echo -e "${YELLOW}=== Stage 1: Data Preprocessing ===${NC}"
if [ ! -f "${DATA_DIR}/processed/train_verified.bin" ]; then
    echo "Preprocessing training data with semantic verification..."
    cargo run -p nanochat-train --release -- preprocess \
        --input "${DATA_DIR}/raw" \
        --output "${DATA_DIR}/processed" \
        --config "${CONFIG_FILE}" \
        --workers 64 \
        --verify-compilation \
        --min-compile-rate 0.85 \
        --numa-aware \
        --use-semantic-verification
    echo -e "${GREEN}Data preprocessing complete!${NC}"
else
    echo -e "${GREEN}Preprocessed data found, skipping...${NC}"
fi
echo ""

# Stage 2: LoopLM Pre-training
echo -e "${YELLOW}=== Stage 2: LoopLM Pre-training (100K steps) ===${NC}"
echo "Starting LoopLM training with entropy-regularized depth allocation..."
echo "This stage will take approximately 4-5 days."
echo ""

# Run with numactl for NUMA optimization
numactl --interleave=all cargo run -p nanochat-train --release -- train \
    --config "${CONFIG_FILE}" \
    --data-path "${DATA_DIR}/processed" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage1" \
    --n-loops 4 \
    --entropy-weight 0.05 \
    --batch-size 32 \
    --seq-len 8192 \
    --total-steps 100000 \
    --eval-every 5000 \
    --save-every 1000 \
    --numa-aware \
    --preferred-kernel AVX512 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage1/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 1 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 1 complete!${NC}"
echo ""

# Stage 3: Compiler-Verified MaxRL Fine-tuning
echo -e "${YELLOW}=== Stage 3: Compiler-Verified MaxRL Fine-tuning (50K steps) ===${NC}"
echo "Fine-tuning with compiler feedback and semantic verification..."
echo ""

numactl --interleave=all cargo run -p nanochat-train --release -- train \
    --config "${CONFIG_FILE}" \
    --base-checkpoint "${CHECKPOINT_DIR}/stage1/final" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage2" \
    --compiler-verification \
    --semantic-analysis \
    --reward-threshold 0.9 \
    --total-steps 50000 \
    --eval-every 2500 \
    --save-every 1000 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage2/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 2 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 2 complete!${NC}"
echo ""

# Stage 4: Long-Context Training
echo -e "${YELLOW}=== Stage 4: Long-Context Training (20K steps, 64K seq len) ===${NC}"
echo "Training with extended context length..."
echo ""

numactl --interleave=all cargo run -p nanochat-train --release -- train \
    --config "${CONFIG_FILE}" \
    --base-checkpoint "${CHECKPOINT_DIR}/stage2/final" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage3" \
    --seq-len 65536 \
    --total-steps 20000 \
    --batch-size 4 \
    --gradient-accumulation 8 \
    --eval-every 2000 \
    --save-every 1000 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage3/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 3 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 3 complete!${NC}"
echo ""

# Export to GGUF
echo -e "${YELLOW}=== Exporting to GGUF Format ===${NC}"
cargo run -p nanochat-train --release -- export \
    --checkpoint "${CHECKPOINT_DIR}/stage3/final" \
    --output "${PROJECT_DIR}/models/nanochat-7b-config-a.gguf" \
    --quantize ternary \
    --group-size 128

echo -e "${GREEN}Export complete!${NC}"
echo ""

# Final Report
echo -e "${GREEN}=== Training Complete! ===${NC}"
echo ""
echo "Model saved to: ${PROJECT_DIR}/models/nanochat-7b-config-a.gguf"
echo ""
echo "Training Summary:"
echo "  - Total steps: 170K"
echo "  - Final model: 7B parameters"
echo "  - Context length: 64K tokens"
echo "  - Quantization: Ternary (1.58-bit)"
echo "  - Expected compilation success rate: >90%"
echo ""
echo "Next steps:"
echo "  1. Evaluate model on HumanEval-Rust"
echo "  2. Run semantic correctness benchmarks"
echo "  3. Publish to Hugging Face"
echo ""
