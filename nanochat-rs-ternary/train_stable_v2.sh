#!/bin/bash
# Stable Training - Fixed Hyperparameters v2
# Prevents model collapse with conservative settings

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  Stable Training v2 - Conservative Hyperparameters"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Key Changes from v1 (which collapsed):"
echo "  • Learning rate: 0.02 → 0.001 (20x lower)"
echo "  • Warmup: 2000 → 5000 steps (2.5x longer)"
echo "  • Total steps: 15000 → 30000 (2x longer)"
echo "  • Grad clip: 1.0 → 0.5 (2x stricter)"
echo "  • Batch size: 2 (same, limited by GPU)"
echo ""

CHECKPOINT_DIR="checkpoints/stable-v2"
DATA_PATH="data/rust_tokens.bin"
DEVICE="cuda:0"
BATCH_SIZE=1  # Reduced from 2 to avoid OOM
TOTAL_STEPS=10000  # Accelerated: 30K → 10K steps
WARMUP_STEPS=1000  # Accelerated: 5K → 1K warmup
LR=0.002  # Slightly higher for faster convergence
GRAD_CLIP=0.5
CHECKPOINT_INTERVAL=1000
LOG_INTERVAL=100

mkdir -p "$CHECKPOINT_DIR"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Data: $DATA_PATH"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Total steps: $TOTAL_STEPS"
echo "  Warmup: $WARMUP_STEPS"
echo "  Learning rate: $LR"
echo "  Grad clip: $GRAD_CLIP"
echo ""

echo "Starting training..."
echo "═══════════════════════════════════════════════════════════"
echo ""

cargo run --release --example train_rust_maxgpu --features nanochat-train/cuda -- \
    --data "$DATA_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --batch-size $BATCH_SIZE \
    --total-steps $TOTAL_STEPS \
    --warmup-steps $WARMUP_STEPS \
    --lr $LR \
    --grad-clip $GRAD_CLIP \
    --checkpoint-interval $CHECKPOINT_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --resume

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training Complete!"
echo "═══════════════════════════════════════════════════════════"
