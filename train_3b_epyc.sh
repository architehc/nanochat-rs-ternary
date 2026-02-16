#!/bin/bash
# train_3b_epyc.sh - Train 3B LoopLM Model on Dual EPYC 9654 + RTX 4090
# Optimized for massive CPU parallelism with NUMA awareness

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== nanochat-rs-ternary Training: 3B LoopLM on Dual EPYC + RTX 4090 ===${NC}"
echo ""

# Environment setup
echo -e "${YELLOW}Setting up environment...${NC}"
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export CUDA_VISIBLE_DEVICES=0

# NUMA optimization (critical for dual-socket EPYC)
export NUMA_AWARE=1
export NUMA_NODES=16
export OMP_NUM_THREADS=112
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_DYNAMIC=false

# CPU affinity - use all 224 threads
export CPU_AFFINITY="0-223"

# Memory policy
export NUMACTL_MEMORY_POLICY=interleave

# CUDA optimizations for RTX 4090
export CUDA_LAUNCH_BLOCKING=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/nanochat-rs-ternary"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/3b_epyc"
LOG_FILE="${CHECKPOINT_DIR}/training.log"

mkdir -p "${CHECKPOINT_DIR}"

echo -e "${GREEN}Environment configured:${NC}"
echo "  CPU threads: ${OMP_NUM_THREADS}"
echo "  NUMA nodes: ${NUMA_NODES}"
echo "  GPU: RTX 4090 (24GB)"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo ""

# Check GPU
echo -e "${YELLOW}Checking GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Stage 1: Data check
echo -e "${YELLOW}=== Stage 1: Checking Training Data ===${NC}"
if [ ! -f "${DATA_DIR}/rust_maxgpu_tokenized.bin" ]; then
    echo -e "${RED}Training data not found at ${DATA_DIR}/rust_maxgpu_tokenized.bin${NC}"
    echo "Please ensure training data is prepared first."
    exit 1
else
    echo -e "${GREEN}Training data found!${NC}"
    ls -lh "${DATA_DIR}"/rust_maxgpu*.bin
fi
echo ""

# Stage 2: Start Training
echo -e "${YELLOW}=== Stage 2: Starting Training (150K steps) ===${NC}"
echo "This will take approximately 5 days on dual EPYC + RTX 4090."
echo "Training with:"
echo "  - Batch size: 16"
echo "  - Sequence length: 4096"
echo "  - NUMA-aware allocation across 16 nodes"
echo ""
echo -e "${BLUE}Starting training at $(date)...${NC}"
echo ""

# Run with numactl for NUMA optimization
# train_rust_maxgpu accepts: --data, --checkpoint-dir, --total-steps,
#   --warmup-steps, --log-interval, --checkpoint-interval,
#   --keep-last-checkpoints, --device, --batch-size, --seq-len, --lr, --grad-clip, --resume
cd "${PROJECT_DIR}"
numactl --interleave=all cargo run --release --example train_rust_maxgpu -- \
    --data "${DATA_DIR}/rust_maxgpu_tokenized.bin" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --batch-size 16 \
    --seq-len 4096 \
    --total-steps 150000 \
    --log-interval 100 \
    --checkpoint-interval 1000 \
    --keep-last-checkpoints 5 \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -ne 0 ]; then
    echo -e "${RED}Training failed with exit code ${EXIT_CODE}! Check logs at ${LOG_FILE}${NC}"
    exit 1
fi

echo -e "${GREEN}Training completed successfully!${NC}"
echo ""

# Export to GGUF
echo -e "${YELLOW}=== Exporting to GGUF Format ===${NC}"
FINAL_CKPT=$(ls -d "${CHECKPOINT_DIR}/step_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "${FINAL_CKPT}" ]; then
    echo -e "${RED}No checkpoint found to export!${NC}"
    exit 1
fi

mkdir -p "${PROJECT_DIR}/models"
cargo run -p nanochat-train --release -- export \
    --checkpoint "${FINAL_CKPT}" \
    --gguf "${PROJECT_DIR}/models/nanochat-3b-epyc.gguf" \
    --mhc "${PROJECT_DIR}/models/nanochat-3b-epyc.mhc"

echo -e "${GREEN}Export complete!${NC}"
echo ""

# Final Report
echo -e "${GREEN}=== Training Complete! ===${NC}"
echo ""
echo "Model saved to: ${PROJECT_DIR}/models/nanochat-3b-epyc.gguf"
echo ""
echo "Training Summary:"
echo "  - Total steps: 150K"
echo "  - Model size: 3B parameters"
echo "  - Context length: 4K tokens"
echo "  - Quantization: Ternary (1.58-bit)"
echo ""
echo "Logs saved to: ${LOG_FILE}"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
echo ""
