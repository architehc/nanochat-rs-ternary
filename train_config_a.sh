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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/nanochat-rs-ternary"
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
if [ ! -f "${DATA_DIR}/processed/tokens.bin" ]; then
    if [ -f "${DATA_DIR}/raw/train.txt" ]; then
        echo "Preprocessing training data..."
        cargo run -p nanochat-train --release --manifest-path "${PROJECT_DIR}/Cargo.toml" -- \
            prepare-data \
            --text "${DATA_DIR}/raw/train.txt" \
            --output "${DATA_DIR}/processed"
        echo -e "${GREEN}Data preprocessing complete!${NC}"
    else
        echo -e "${YELLOW}No raw text found at ${DATA_DIR}/raw/train.txt, skipping preprocessing.${NC}"
        echo "Expecting pre-tokenized data at ${DATA_DIR}/processed/tokens.bin"
    fi
else
    echo -e "${GREEN}Preprocessed data found, skipping...${NC}"
fi
echo ""

# Stage 2: Pre-training (100K steps)
echo -e "${YELLOW}=== Stage 2: Pre-training (100K steps) ===${NC}"
echo "Starting training with entropy-regularized depth allocation..."
echo "This stage will take approximately 4-5 days."
echo ""

# Run with numactl for NUMA optimization
numactl --interleave=all cargo run -p nanochat-train --release \
    --manifest-path "${PROJECT_DIR}/Cargo.toml" -- train \
    --config large-7b \
    --dataset tokens \
    --data-path "${DATA_DIR}/processed/tokens.bin" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage1" \
    --batch-size 32 \
    --seq-len 8192 \
    --epochs 1 \
    --log-interval 100 \
    --checkpoint-interval 1000 \
    --keep-last-checkpoints 3 \
    --threads 128 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage1/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 1 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 1 complete!${NC}"
echo ""

# Stage 3: Resume with fine-tuning
echo -e "${YELLOW}=== Stage 3: Fine-tuning (50K steps) ===${NC}"
echo "Fine-tuning from stage 1 checkpoint..."
echo ""

LATEST_CKPT=$(ls -d "${CHECKPOINT_DIR}/stage1/step_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "${LATEST_CKPT}" ]; then
    echo -e "${RED}No checkpoint found from stage 1!${NC}"
    exit 1
fi

numactl --interleave=all cargo run -p nanochat-train --release \
    --manifest-path "${PROJECT_DIR}/Cargo.toml" -- train \
    --config large-7b \
    --dataset tokens \
    --data-path "${DATA_DIR}/processed/tokens.bin" \
    --resume "${LATEST_CKPT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage2" \
    --batch-size 32 \
    --seq-len 8192 \
    --epochs 1 \
    --log-interval 100 \
    --checkpoint-interval 1000 \
    --keep-last-checkpoints 3 \
    --threads 128 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage2/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 2 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 2 complete!${NC}"
echo ""

# Export to GGUF
echo -e "${YELLOW}=== Exporting to GGUF Format ===${NC}"
FINAL_CKPT=$(ls -d "${CHECKPOINT_DIR}/stage2/step_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "${FINAL_CKPT}" ]; then
    FINAL_CKPT="${LATEST_CKPT}"
fi

mkdir -p "${PROJECT_DIR}/models"
cargo run -p nanochat-train --release --manifest-path "${PROJECT_DIR}/Cargo.toml" -- export \
    --checkpoint "${FINAL_CKPT}" \
    --gguf "${PROJECT_DIR}/models/nanochat-7b-config-a.gguf" \
    --mhc "${PROJECT_DIR}/models/nanochat-7b-config-a.mhc"

echo -e "${GREEN}Export complete!${NC}"
echo ""

# Final Report
echo -e "${GREEN}=== Training Complete! ===${NC}"
echo ""
echo "Model saved to: ${PROJECT_DIR}/models/nanochat-7b-config-a.gguf"
echo ""
echo "Training Summary:"
echo "  - Final model: 7B parameters"
echo "  - Context length: 8K tokens"
echo "  - Quantization: Ternary (1.58-bit)"
echo ""
