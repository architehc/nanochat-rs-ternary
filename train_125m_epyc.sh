#!/bin/bash
# train_125m_epyc.sh - Train 125M Model on Dual EPYC + RTX 4090
# Using nano-125m config that fits in 24GB VRAM

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Starting 125M Model Training on Dual EPYC + RTX 4090 ===${NC}"
echo ""

# Environment setup for dual-EPYC NUMA
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
# Let Candle select GPU automatically
#export CUDA_VISIBLE_DEVICES=0
export NUMA_AWARE=1
export NUMA_NODES=16
export OMP_NUM_THREADS=112
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export RAYON_NUM_THREADS=112

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nanochat-rs-ternary"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/125m_epyc_$(date +%Y%m%d_%H%M%S)"
DATA_PATH="${PROJECT_DIR}/data/rust_tokens_large.bin"

mkdir -p "${CHECKPOINT_DIR}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Model: nano-125m (125M params, 768 dim, 12 layers)"
echo "  Data: ${DATA_PATH}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Batch size: 4"
echo "  Sequence length: 512"
echo "  Device: cpu (Dual EPYC 9654, 224 threads)"
echo "  NUMA nodes: 16"
echo "  CPU threads: 112"
echo "  Epochs: 20"
echo ""

# Check GPU
echo -e "${YELLOW}GPU Status:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo -e "${YELLOW}Starting training at $(date)...${NC}"
echo "This will take several days. Progress will be monitored every 10 minutes."
echo ""

cd "${PROJECT_DIR}"

# Start training with numactl for NUMA optimization
numactl --interleave=all \
./target/release/nanochat-train train \
    --config nano-125m \
    --data-path "${DATA_PATH}" \
    --dataset tokens \
    --epochs 20 \
    --batch-size 4 \
    --seq-len 512 \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-interval 50 \
    --checkpoint-interval 1000 \
    --keep-last-checkpoints 5 \
    --device cpu \
    --threads 112 \
    2>&1 | tee "${CHECKPOINT_DIR}/training.log"

echo ""
echo -e "${GREEN}Training completed!${NC}"
echo "Logs: ${CHECKPOINT_DIR}/training.log"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
