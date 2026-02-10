#!/bin/bash
# MaxRL Training Script - Only learns from compilable code
# 20x more efficient than GRPO according to paper

set -e

# Configuration
CHECKPOINT=${1:-"checkpoints/stable-v2/step_10000"}
OUTPUT_DIR="checkpoints/maxrl-rust"
ITERATIONS=200  # Accelerated: 200 instead of 1000
N_SAMPLES=8     # More samples for MaxRL
BATCH_SIZE=3
CORRECTNESS_THRESHOLD=20.0  # Reward > 20 = compilable
TEMPERATURE=1.0
LR=1e-5
DEVICE="cuda:0"

mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  MaxRL Training - Learn from Correct Code Only"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Base checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT_DIR"
echo "  Iterations: $ITERATIONS (accelerated)"
echo "  Samples per prompt: $N_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Correctness threshold: $CORRECTNESS_THRESHOLD"
echo "  Temperature: $TEMPERATURE"
echo "  Learning rate: $LR"
echo "  Device: $DEVICE"
echo ""
echo "Key Difference from GRPO:"
echo "  • GRPO: Uses ALL samples (correct + incorrect)"
echo "  • MaxRL: Uses ONLY correct samples (reward > threshold)"
echo "  • Expected: 20x better test-time scaling"
echo ""

if [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo "Please train base model first or specify valid checkpoint"
    exit 1
fi

echo "Starting MaxRL training..."
echo "═══════════════════════════════════════════════════════════"
echo ""

cargo run --release -p nanochat-rl --example train_maxrl --features nanochat-train/cuda -- \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR" \
    --iterations $ITERATIONS \
    --n-samples $N_SAMPLES \
    --batch-size $BATCH_SIZE \
    --correctness-threshold $CORRECTNESS_THRESHOLD \
    --temperature $TEMPERATURE \
    --lr $LR \
    --device "$DEVICE"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  MaxRL Training Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Final checkpoint: $OUTPUT_DIR/final"
echo ""
echo "Next steps:"
echo "  1. Test generation:"
echo "     cargo run --example test_generation_simple -- \\"
echo "       --checkpoint $OUTPUT_DIR/final"
echo ""
echo "  2. Evaluate compilation rate:"
echo "     cargo run -p nanochat-rl --example evaluate_model -- \\"
echo "       --checkpoint $OUTPUT_DIR/final"
echo ""
