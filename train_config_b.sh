#!/bin/bash
# train_config_b.sh - Ryzen 9800X3D + 2x RTX 4090 Training Script
# Optimized for fast GPU training

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== nanochat-rs-ternary Training: Config B (9800X3D + 2x RTX 4090) ===${NC}"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/nanochat-rs-ternary"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/config_b"
CONFIG_FILE="${SCRIPT_DIR}/config_b_9800x3d_dual4090.toml"

mkdir -p "${CHECKPOINT_DIR}"

echo -e "${GREEN}Environment configured:${NC}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  World size: ${WORLD_SIZE}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# Stage 1: Data Preprocessing
echo -e "${YELLOW}=== Stage 1: Data Preprocessing ===${NC}"
if [ ! -f "${DATA_DIR}/processed/tokens.bin" ]; then
    if [ -f "${DATA_DIR}/raw/train.txt" ]; then
        echo "Preprocessing training data..."
        cargo run -p nanochat-train --release --manifest-path "${PROJECT_DIR}/Cargo.toml" -- \
            prepare-data \
            --text "${DATA_DIR}/raw/train.txt" \
            --output "${DATA_DIR}/processed"
        echo -e "${GREEN}Data preprocessing complete!${NC}"
    else
        echo -e "${YELLOW}No raw text found, expecting pre-tokenized data.${NC}"
    fi
else
    echo -e "${GREEN}Preprocessed data found, skipping...${NC}"
fi
echo ""

# Stage 2: Pre-training (150K steps)
echo -e "${YELLOW}=== Stage 2: Pre-training (150K steps) ===${NC}"
echo "Starting training across 2x RTX 4090..."
echo "This stage will take approximately 3-4 days."
echo ""

cargo run -p nanochat-train --release --manifest-path "${PROJECT_DIR}/Cargo.toml" -- train \
    --config medium-3b \
    --dataset tokens \
    --data-path "${DATA_DIR}/processed/tokens.bin" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage1" \
    --batch-size 4 \
    --seq-len 4096 \
    --epochs 1 \
    --log-interval 100 \
    --checkpoint-interval 2000 \
    --keep-last-checkpoints 3 \
    --threads 16 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage1/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 1 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 1 complete!${NC}"
echo ""

# Stage 3: Resume for fine-tuning (75K steps)
echo -e "${YELLOW}=== Stage 3: Fine-tuning (75K steps) ===${NC}"
echo "Fine-tuning from stage 1 checkpoint..."
echo ""

LATEST_CKPT=$(ls -d "${CHECKPOINT_DIR}/stage1/step_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "${LATEST_CKPT}" ]; then
    echo -e "${RED}No checkpoint found from stage 1!${NC}"
    exit 1
fi

cargo run -p nanochat-train --release --manifest-path "${PROJECT_DIR}/Cargo.toml" -- train \
    --config medium-3b \
    --dataset tokens \
    --data-path "${DATA_DIR}/processed/tokens.bin" \
    --resume "${LATEST_CKPT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage2" \
    --batch-size 4 \
    --seq-len 4096 \
    --epochs 1 \
    --log-interval 100 \
    --checkpoint-interval 2000 \
    --keep-last-checkpoints 3 \
    --threads 16 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage2/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 2 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 2 complete!${NC}"
echo ""

# Export
echo -e "${YELLOW}=== Exporting to GGUF Format ===${NC}"
FINAL_CKPT=$(ls -d "${CHECKPOINT_DIR}/stage2/step_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "${FINAL_CKPT}" ]; then
    FINAL_CKPT="${LATEST_CKPT}"
fi

mkdir -p "${PROJECT_DIR}/models"
cargo run -p nanochat-train --release --manifest-path "${PROJECT_DIR}/Cargo.toml" -- export \
    --checkpoint "${FINAL_CKPT}" \
    --gguf "${PROJECT_DIR}/models/nanochat-3b-config-b.gguf" \
    --mhc "${PROJECT_DIR}/models/nanochat-3b-config-b.mhc"

echo -e "${GREEN}Export complete!${NC}"
echo ""

# Final Report
echo -e "${GREEN}=== Training Complete! ===${NC}"
echo ""
echo "Model saved to: ${PROJECT_DIR}/models/nanochat-3b-config-b.gguf"
echo ""
echo "Training Summary:"
echo "  - Final model: 3B parameters"
echo "  - Context length: 4K tokens"
echo "  - Quantization: Ternary (1.58-bit)"
echo ""
