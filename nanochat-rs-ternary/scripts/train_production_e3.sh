#!/bin/bash
# Production Training: d20 (560M) with Full E3 Stack
#
# Hardware: NVIDIA RTX PRO 6000 Ada (96GB VRAM)
# Dataset: 68M tokens of Rust code
# Expected throughput: ~65K tokens/step with all E3 optimizations
# Expected runtime: ~24-36 hours for 50K steps

set -e

# Configuration
CONFIG_PRESET="d20-e3-full"
DATA_PATH="data/rust_tokens_large.bin"
CHECKPOINT_DIR="checkpoints/production-e3-d20"
DEVICE="cuda"
EPOCHS=100  # Will train for 50K steps (overrides epoch count)

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Verify data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    echo "Please run data preparation first"
    exit 1
fi

# Print configuration summary
echo "================================================"
echo "Production Training: d20 (560M) + Full E3 Stack"
echo "================================================"
echo ""
echo "Model:"
echo "  - Architecture: d20 (768 dim, 24 layers)"
echo "  - Parameters: ~560M"
echo "  - Context: 2048 tokens"
echo ""
echo "Dataset:"
echo "  - Path: $DATA_PATH"
echo "  - Size: 68M tokens"
echo "  - Epochs: ~48 (3.25B total tokens)"
echo ""
echo "E3 Optimizations ENABLED:"
echo "  ✓ Multi-Token Prediction (n=3, weight=0.2)"
echo "  ✓ Collider Token Filtering (sparsity=0.35)"
echo "  ✓ Async Data Loader (workers=6, prefetch=12)"
echo "  ✓ mHC Analysis (health monitoring)"
echo "  ✓ FIRE capability (not auto-triggered)"
echo ""
echo "Training:"
echo "  - Batch size: 8 × 2048 = 16K tokens/batch"
echo "  - Gradient accumulation: 4 steps"
echo "  - Effective batch: 65K tokens"
echo "  - Total steps: 50K"
echo "  - Warmup: 2K steps"
echo "  - Device: $DEVICE"
echo ""
echo "Expected:"
echo "  - Throughput: ~65K tokens/step (with E3)"
echo "  - Runtime: 24-36 hours"
echo "  - Checkpoints: Every 1K steps"
echo ""
echo "================================================"
echo ""

# Start training
cargo run --release --bin nanochat-train -- train \
    --config "$CONFIG_PRESET" \
    --dataset tokens \
    --data-path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --log-interval 100 \
    --checkpoint-interval 1000 \
    --keep-last-checkpoints 5

echo ""
echo "Training complete! Checkpoints saved to: $CHECKPOINT_DIR"
