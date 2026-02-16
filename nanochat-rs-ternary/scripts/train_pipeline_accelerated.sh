#!/bin/bash
# Accelerated Training Pipeline: Supervised → MaxRL → Evaluate
# Complete end-to-end training automation

set -eo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/pipeline_$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  Accelerated Training Pipeline"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Pipeline stages:"
echo "  1. Base supervised training (10K steps, ~3 hours)"
echo "  2. MaxRL refinement (200 iterations, ~1 hour)"
echo "  3. Evaluation and testing"
echo ""
echo "Logs: $LOG_DIR"
echo ""

# ═══════════════════════════════════════════════════════════
# STAGE 1: Base Supervised Training
# ═══════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════"
echo "  STAGE 1: Supervised Pre-training"
echo "═══════════════════════════════════════════════════════════"
echo ""

SUPERVISED_LOG="$LOG_DIR/01_supervised.log"

echo "Starting supervised training..."
echo "  Steps: 10,000 (accelerated)"
echo "  LR: 0.002"
echo "  Batch size: 1"
echo "  Log: $SUPERVISED_LOG"
echo ""

./train_stable_v2.sh > >(tee "$SUPERVISED_LOG") 2>&1 &
SUPERVISED_PID=$!

echo "Training PID: $SUPERVISED_PID"
echo "Monitor: tail -f $SUPERVISED_LOG"
echo ""

# Wait for training to complete
echo "Waiting for supervised training to complete..."
wait $SUPERVISED_PID
SUPERVISED_EXIT=$?

if [ $SUPERVISED_EXIT -ne 0 ]; then
    echo "ERROR: Supervised training failed with exit code $SUPERVISED_EXIT"
    echo "Check log: $SUPERVISED_LOG"
    exit 1
fi

echo "✓ Supervised training complete"
echo ""

# Find final checkpoint
SUPERVISED_CHECKPOINT=$(ls -d checkpoints/stable-v2/step_* | sort -V | tail -1)
echo "Final checkpoint: $SUPERVISED_CHECKPOINT"
echo ""

# Test generation quality
echo "Testing supervised model quality..."
cargo run --release --example test_generation_simple --features nanochat-train/cuda \
    2>&1 | tee "$LOG_DIR/01_supervised_test.log"
echo ""

# ═══════════════════════════════════════════════════════════
# STAGE 2: MaxRL Refinement
# ═══════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════"
echo "  STAGE 2: MaxRL Refinement"
echo "═══════════════════════════════════════════════════════════"
echo ""

MAXRL_LOG="$LOG_DIR/02_maxrl.log"

echo "Starting MaxRL training..."
echo "  Base: $SUPERVISED_CHECKPOINT"
echo "  Iterations: 200 (accelerated)"
echo "  Method: Learn only from compilable code"
echo "  Log: $MAXRL_LOG"
echo ""

./scripts/train_maxrl.sh "$SUPERVISED_CHECKPOINT" > >(tee "$MAXRL_LOG") 2>&1 &
MAXRL_PID=$!

echo "MaxRL PID: $MAXRL_PID"
echo "Monitor: tail -f $MAXRL_LOG"
echo ""

# Wait for MaxRL to complete
echo "Waiting for MaxRL training to complete..."
wait $MAXRL_PID
MAXRL_EXIT=$?

if [ $MAXRL_EXIT -ne 0 ]; then
    echo "WARNING: MaxRL training failed with exit code $MAXRL_EXIT"
    echo "Check log: $MAXRL_LOG"
    echo "Continuing with supervised checkpoint..."
    FINAL_CHECKPOINT="$SUPERVISED_CHECKPOINT"
else
    echo "✓ MaxRL training complete"
    FINAL_CHECKPOINT="checkpoints/maxrl-rust/final"
fi
echo ""

# ═══════════════════════════════════════════════════════════
# STAGE 3: Evaluation
# ═══════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════"
echo "  STAGE 3: Final Evaluation"
echo "═══════════════════════════════════════════════════════════"
echo ""

EVAL_LOG="$LOG_DIR/03_evaluation.log"

echo "Evaluating final model..."
echo "  Checkpoint: $FINAL_CHECKPOINT"
echo "  Log: $EVAL_LOG"
echo ""

# Test generation quality
echo "--- Generation Test ---" | tee "$EVAL_LOG"
cargo run --release --example test_generation_simple --features nanochat-train/cuda \
    2>&1 | tee -a "$EVAL_LOG"
echo ""

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════

PIPELINE_END=$(date +%s)
PIPELINE_START=$(stat -c %Y "$LOG_DIR" 2>/dev/null || date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

echo "═══════════════════════════════════════════════════════════"
echo "  Pipeline Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Duration: $((PIPELINE_DURATION / 3600))h $((PIPELINE_DURATION % 3600 / 60))m"
echo "Logs: $LOG_DIR"
echo ""
echo "Checkpoints:"
echo "  Supervised: $SUPERVISED_CHECKPOINT"
echo "  MaxRL: $FINAL_CHECKPOINT"
echo ""
echo "Next steps:"
echo "  1. Test the model:"
echo "     cargo run --example test_generation_simple"
echo ""
echo "  2. Export to GGUF:"
echo "     cargo run --example export_checkpoint -- \\"
echo "       --checkpoint $FINAL_CHECKPOINT \\"
echo "       --output models/rust-nano-maxrl.gguf"
echo ""
echo "  3. Deploy inference server:"
echo "     cargo run -p nanochat-serve -- \\"
echo "       --model models/rust-nano-maxrl.gguf \\"
echo "       --port 8080"
echo ""
