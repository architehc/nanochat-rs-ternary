#!/bin/bash
# train_config_b.sh - Ryzen 9800X3D + 2× RTX 4090 Training Script
# Optimized for fast GPU training with tensor parallelism

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== nanochat-rs-ternary Training: Config B (9800X3D + 2× RTX 4090) ===${NC}"
echo ""

# Environment setup
echo -e "${YELLOW}Setting up environment...${NC}"
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export CUDA_VISIBLE_DEVICES=0,1

# Multi-GPU setup
export WORLD_SIZE=2
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

# X3D optimization
export AMD_CPU_OPTIMIZATIONS=1
export OMP_NUM_THREADS=16

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/config_b"
CONFIG_FILE="${PROJECT_DIR}/configs/config_b_9800x3d_dual4090.toml"

mkdir -p "${CHECKPOINT_DIR}"

echo -e "${GREEN}Environment configured:${NC}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  World size: ${WORLD_SIZE}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# Stage 1: Data Preprocessing
echo -e "${YELLOW}=== Stage 1: Data Preprocessing ===${NC}"
if [ ! -f "${DATA_DIR}/processed/train_verified.bin" ]; then
    echo "Preprocessing training data..."
    cargo run --release --example preprocess_data --         --input "${DATA_DIR}/raw"         --output "${DATA_DIR}/processed"         --config "${CONFIG_FILE}"         --workers 8         --verify-compilation         --min-compile-rate 0.88
    echo -e "${GREEN}Data preprocessing complete!${NC}"
else
    echo -e "${GREEN}Preprocessed data found, skipping...${NC}"
fi
echo ""

# Stage 2: LoopLM Pre-training with Tensor Parallelism
echo -e "${YELLOW}=== Stage 2: LoopLM Pre-training with Tensor Parallelism (150K steps) ===${NC}"
echo "Starting distributed training across 2× RTX 4090..."
echo "This stage will take approximately 3-4 days."
echo ""

# Use torchrun for distributed training
torchrun     --nproc_per_node=2     --nnodes=1     --master_addr=${MASTER_ADDR}     --master_port=${MASTER_PORT}     cargo run --release --example train_looplm_distributed --     --config "${CONFIG_FILE}"     --data "${DATA_DIR}/processed"     --checkpoint-dir "${CHECKPOINT_DIR}/stage1"     --n-loops 4     --tensor-parallel-size 2     --batch-size 4     --seq-len 4096     --total-steps 150000     --eval-every 10000     --save-every 2000     2>&1 | tee "${CHECKPOINT_DIR}/stage1/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 1 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 1 complete!${NC}"
echo ""

# Stage 3: Compiler-Verified MaxRL
echo -e "${YELLOW}=== Stage 3: Compiler-Verified MaxRL (75K steps) ===${NC}"
echo "Fine-tuning with compiler feedback..."
echo ""

torchrun     --nproc_per_node=2     --nnodes=1     cargo run --release --example train_maxrl_verified --     --config "${CONFIG_FILE}"     --base-checkpoint "${CHECKPOINT_DIR}/stage1/final"     --checkpoint-dir "${CHECKPOINT_DIR}/stage2"     --compiler-verification     --reward-threshold 0.92     --total-steps 75000     --eval-every 5000     --save-every 2000     2>&1 | tee "${CHECKPOINT_DIR}/stage2/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 2 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 2 complete!${NC}"
echo ""

# Export
echo -e "${YELLOW}=== Exporting to GGUF Format ===${NC}"
cargo run --release --example export_gguf --     --checkpoint "${CHECKPOINT_DIR}/stage2/final"     --output "${PROJECT_DIR}/models/nanochat-3b-config-b.gguf"     --quantize ternary     --group-size 128

echo -e "${GREEN}Export complete!${NC}"
echo ""

# Final Report
echo -e "${GREEN}=== Training Complete! ===${NC}"
echo ""
echo "Model saved to: ${PROJECT_DIR}/models/nanochat-3b-config-b.gguf"
echo ""
echo "Training Summary:"
echo "  - Total steps: 225K"
echo "  - Final model: 3B parameters"
echo "  - Context length: 4K tokens"
echo "  - Quantization: Ternary (1.58-bit)"
echo "  - Expected compilation success rate: >88%"
echo ""
