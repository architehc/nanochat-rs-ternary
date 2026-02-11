#!/bin/bash
# Stable Training - Fixed Hyperparameters v2
# Prevents model collapse with conservative settings

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  Stable Training v3 - Large Dataset (68M tokens)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Dataset Upgrade:"
echo "  • Training data: 4.2M → 68M tokens (16.2x larger!)"
echo "  • Repositories: 3 → 13 (rust-lang, servo, cargo, tikv, etc.)"
echo "  • Total steps: 10K → 20K (2x longer for larger dataset)"
echo "  • Label smoothing: eps=0.1 (prevents overconfidence)"
echo ""
echo "Why this matters:"
echo "  Previous run (small dataset) showed severe overfitting:"
echo "    - Loss: 3.0 (good on training)"
echo "    - Generation: Repeats tokens (bad)"
echo "  Larger dataset should provide better generalization"
echo ""

CHECKPOINT_DIR="checkpoints/stable-v2"
DATA_PATH="data/rust_tokens.bin"  # Now points to large dataset (68M tokens)
DEVICE="cuda:0"
BATCH_SIZE=1
SEQ_LEN=256
TOTAL_STEPS=20000  # 2x longer for 16x more data
WARMUP_STEPS=2000  # Proportionally longer warmup
LR=0.002
GRAD_CLIP=0.5
CHECKPOINT_INTERVAL=500
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
    --seq-len $SEQ_LEN \
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
