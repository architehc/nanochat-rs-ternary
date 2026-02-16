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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/nanochat-rs-ternary"
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
if [ ! -f "${DATA_DIR}/processed/tokens.bin" ]; then
    if [ -f "${DATA_DIR}/raw/train.txt" ]; then
        echo "Preprocessing with NUMA-aware workers..."
        numactl --interleave=all cargo run -p nanochat-train --release \
            --manifest-path "${PROJECT_DIR}/Cargo.toml" -- \
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

# Stage 2: Pre-training (120K steps)
echo -e "${YELLOW}=== Stage 2: Pre-training (120K steps) ===${NC}"
echo "Starting training with massive CPU parallelism..."
echo "This stage will take approximately 4-5 days."
echo ""

numactl --interleave=all cargo run -p nanochat-train --release \
    --manifest-path "${PROJECT_DIR}/Cargo.toml" -- train \
    --config medium-3b \
    --dataset tokens \
    --data-path "${DATA_DIR}/processed/tokens.bin" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage1" \
    --batch-size 24 \
    --seq-len 6144 \
    --epochs 1 \
    --log-interval 100 \
    --checkpoint-interval 1000 \
    --keep-last-checkpoints 3 \
    --threads 112 \
    2>&1 | tee "${CHECKPOINT_DIR}/stage1/training.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Stage 1 training failed! Check logs.${NC}"
    exit 1
fi

echo -e "${GREEN}Stage 1 complete!${NC}"
echo ""

# Stage 3: Resume for fine-tuning (60K steps)
echo -e "${YELLOW}=== Stage 3: Fine-tuning (60K steps) ===${NC}"
echo "Fine-tuning from stage 1 checkpoint..."
echo ""

LATEST_CKPT=$(ls -d "${CHECKPOINT_DIR}/stage1/step_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "${LATEST_CKPT}" ]; then
    echo -e "${RED}No checkpoint found from stage 1!${NC}"
    exit 1
fi

numactl --interleave=all cargo run -p nanochat-train --release \
    --manifest-path "${PROJECT_DIR}/Cargo.toml" -- train \
    --config medium-3b \
    --dataset tokens \
    --data-path "${DATA_DIR}/processed/tokens.bin" \
    --resume "${LATEST_CKPT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}/stage2" \
    --batch-size 24 \
    --seq-len 6144 \
    --epochs 1 \
    --log-interval 100 \
    --checkpoint-interval 1000 \
    --keep-last-checkpoints 3 \
    --threads 112 \
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
numactl --interleave=all cargo run -p nanochat-train --release \
    --manifest-path "${PROJECT_DIR}/Cargo.toml" -- export \
    --checkpoint "${FINAL_CKPT}" \
    --gguf "${PROJECT_DIR}/models/nanochat-5b-config-c.gguf" \
    --mhc "${PROJECT_DIR}/models/nanochat-5b-config-c.mhc"

echo -e "${GREEN}Export complete!${NC}"
echo ""

# Final Report
echo -e "${GREEN}=== Training Complete! ===${NC}"
echo ""
echo "Model saved to: ${PROJECT_DIR}/models/nanochat-5b-config-c.gguf"
echo ""
echo "Training Summary:"
echo "  - Final model: 5B parameters"
echo "  - Context length: 6K tokens"
echo "  - Quantization: Ternary (1.58-bit)"
echo ""
