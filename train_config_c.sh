#!/bin/bash
# train_config_c.sh - Dual EPYC 56-core + RTX 4090 Training Script
# Optimized for massive CPU parallelism with NUMA awareness

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== nanochat-rs-ternary Training: Config C (Dual EPYC + RTX 4090) ===${NC}"
echo ""

# Environment setup
echo -e "${YELLOW}Setting up environment...${NC}"
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export CUDA_VISIBLE_DEVICES=0

# NUMA optimization (critical for dual-socket)
export NUMA_AWARE=1
export NUMA_NODES=16
export OMP_NUM_THREADS=56
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_DYNAMIC=false

# CPU affinity - use all cores
export CPU_AFFINITY="0-111"

# Memory policy
export NUCTL_MEMORY_POLICY=interleave

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nanochat-rs-ternary"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/config_c"
CONFIG_FILE="${SCRIPT_DIR}/config_c_dual_epyc_4090.toml"

mkdir -p "${CHECKPOINT_DIR}"

echo -e "${GREEN}Environment configured:${NC}"
echo "  CPU threads: ${OMP_NUM_THREADS}"
echo "  NUMA nodes: ${NUMA_NODES}"
echo "  Memory policy: interleave"
echo ""

# Stage 1: Data Preprocessing (massive parallelism)
echo -e "${YELLOW}=== Stage 1: Data Preprocessing ===${NC}"
if [ ! -f "${DATA_DIR}/processed/train_verified.bin" ]; then
    echo "Preprocessing with 112 workers..."
    numactl --interleave=all cargo run -p nanochat-train --release -- preprocess \
        --input "${DATA_DIR}/raw" \
        --output "${DATA_DIR}/processed" \
        --config "${CONFIG_FILE}" \
        --workers 112 \
        --verify-compilation \
        --min-compile-rate 0.87 \
        --numa-aware \
        --use-semantic-verification
    echo -e "${GREEN}Data preprocessing complete!${NC}"
else
    echo -e "${GREEN}Preprocessed data found, skipping...${NC}"
fi
echo ""

# Stage 2: LoopLM Pre-training
echo -e "${YELLOW}=== Stage 2: LoopLM Pre-training (120K steps) ===${NC}"
echo "Starting training with massive CPU parallelism..."
echo "This stage will take approximately 4-5 days."
echo ""

numactl --interleave=all cargo run -p nanochat-train --release -- train \
    --config "${CONFIG_FILE}" \
    --data-path "${DATA_DIR}/processed" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage1" \
    --n-loops 4 \
    --entropy-weight 0.06 \
    --batch-size 24 \
    --seq-len 6144 \
    --total-steps 120000 \
    --eval-every 5000 \
    --save-every 1000 \
    --numa-aware \
    --preferred-kernel AVX2 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage1/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 1 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 1 complete!${NC}"
echo ""

# Stage 3: Compiler-Verified MaxRL
echo -e "${YELLOW}=== Stage 3: Compiler-Verified MaxRL (60K steps) ===${NC}"
echo "Fine-tuning with compiler feedback..."
echo ""

numactl --interleave=all cargo run -p nanochat-train --release -- train \
    --config "${CONFIG_FILE}" \
    --base-checkpoint "${CHECKPOINT_DIR}/stage1/final" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage2" \
    --compiler-verification \
    --semantic-analysis \
    --reward-threshold 0.90 \
    --total-steps 60000 \
    --eval-every 3000 \
    --save-every 1000 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage2/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 2 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 2 complete!${NC}"
echo ""

# Export
echo -e "${YELLOW}=== Exporting to GGUF Format ===${NC}"
cargo run -p nanochat-train --release -- export \
    --checkpoint "${CHECKPOINT_DIR}/stage2/final" \
    --output "${PROJECT_DIR}/models/nanochat-5b-config-c.gguf" \
    --quantize ternary \
    --group-size 128

echo -e "${GREEN}Export complete!${NC}"
echo ""

# Final Report
echo -e "${GREEN}=== Training Complete! ===${NC}"
echo ""
echo "Model saved to: ${PROJECT_DIR}/models/nanochat-5b-config-c.gguf"
echo ""
echo "Training Summary:"
echo "  - Total steps: 180K"
echo "  - Final model: 5B parameters"
echo "  - Context length: 6K tokens"
echo "  - Quantization: Ternary (1.58-bit)"
echo "  - Expected compilation success rate: >87%"
echo ""
