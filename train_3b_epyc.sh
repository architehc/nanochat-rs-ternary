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

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nanochat-rs-ternary"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/3b_epyc"
CONFIG_FILE="${SCRIPT_DIR}/training_config_3b_epyc.toml"
LOG_FILE="${CHECKPOINT_DIR}/training.log"

mkdir -p "${CHECKPOINT_DIR}"

echo -e "${GREEN}Environment configured:${NC}"
echo "  CPU threads: ${OMP_NUM_THREADS}"
echo "  NUMA nodes: ${NUMA_NODES}"
echo "  GPU: RTX 4090 (24GB)"
echo "  Config: ${CONFIG_FILE}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo ""

# Check GPU
echo -e "${YELLOW}Checking GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Stage 1: Data Preprocessing (if needed)
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

# Stage 2: Start LoopLM Training
echo -e "${YELLOW}=== Stage 2: Starting LoopLM Training (150K steps) ===${NC}"
echo "This will take approximately 5 days on dual EPYC + RTX 4090."
echo "Training with:"
echo "  - LoopLM architecture (4 recurrent steps)"
echo "  - Entropy regularization (weight=0.05)"
echo "  - Compiler verification enabled"
echo "  - NUMA-aware allocation across 16 nodes"
echo "  - 112 data workers for massive parallelism"
echo ""
echo -e "${BLUE}Starting training at $(date)...${NC}"
echo ""

# Run with numactl for NUMA optimization
cd "${PROJECT_DIR}"
numactl --interleave=all cargo run --release --example train_rust_maxgpu -- \
    --data "${DATA_DIR}/rust_maxgpu_tokenized.bin" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --config "${CONFIG_FILE}" \
    --batch-size 16 \
    --seq-len 4096 \
    --total-steps 150000 \
    --eval-every 5000 \
    --save-every 1000 \
    --log-every 100 \
    --entropy-weight 0.05 \
    --min-compile-rate 0.88 \
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
cargo run -p nanochat-train --release -- export \
    --checkpoint "${CHECKPOINT_DIR}/checkpoint_150000" \
    --output "${PROJECT_DIR}/models/nanochat-3b-epyc.gguf" \
    --quantize ternary \
    --group-size 128

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
echo "  - Expected compilation success rate: >88%"
echo "  - Expected HumanEval-Rust pass@1: >65%"
echo ""
echo "Logs saved to: ${LOG_FILE}"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Evaluate model: cargo run --release --example evaluate_model"
echo "  2. Run benchmarks: cargo bench"
echo "  3. Test inference: cargo run --release --bin nanochat-serve"
echo ""
