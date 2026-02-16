#!/bin/bash
# train_1b_epyc.sh - Train 1B Model on Dual EPYC + RTX 4090
# Using largest available config (nano-1b) with NUMA optimization

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Starting 1B Model Training on Dual EPYC + RTX 4090 ===${NC}"
echo ""

# Environment setup for dual-EPYC NUMA
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export CUDA_VISIBLE_DEVICES=0
export NUMA_AWARE=1
export NUMA_NODES=16
export OMP_NUM_THREADS=112
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export RAYON_NUM_THREADS=112

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nanochat-rs-ternary"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/1b_epyc_$(date +%Y%m%d_%H%M%S)"
DATA_PATH="${PROJECT_DIR}/data/rust_tokens_large.bin"

mkdir -p "${CHECKPOINT_DIR}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Model: nano-1b (1B params, 2048 dim, 20 layers)"
echo "  Data: ${DATA_PATH}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Batch size: 8"
echo "  Sequence length: 512"
echo "  Device: cuda (RTX 4090 24GB)"
echo "  NUMA nodes: 16"
echo "  CPU threads: 112"
echo "  Epochs: 10"
echo ""

# Check GPU
echo -e "${YELLOW}GPU Status:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo -e "${YELLOW}Starting training at $(date)...${NC}"
echo ""

cd "${PROJECT_DIR}"

# Start training with numactl for NUMA optimization
numactl --interleave=all \
./target/release/nanochat-train train \
    --config nano-1b \
    --data-path "${DATA_PATH}" \
    --dataset tokens \
    --epochs 10 \
    --batch-size 2 \
    --seq-len 256 \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-interval 50 \
    --checkpoint-interval 1000 \
    --keep-last-checkpoints 5 \
    --device cuda \
    --threads 112 \
    2>&1 | tee "${CHECKPOINT_DIR}/training.log"

echo ""
echo -e "${GREEN}Training completed!${NC}"
echo "Logs: ${CHECKPOINT_DIR}/training.log"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
