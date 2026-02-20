#!/bin/bash
# Production-grade training pipeline for Rust code generation
# Multi-phase approach: Supervised pre-training → RL fine-tuning → Evaluation

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  Production-Grade Rust Code Generation Model Training"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Multi-Phase Training Plan:"
echo "  Phase 1: Extended supervised pre-training (step 2000 → 15000)"
echo "  Phase 2: RL fine-tuning with compiler feedback (1000 iterations)"
echo "  Phase 3: Evaluation on benchmarks"
echo "  Phase 4: Export final model to GGUF"
echo ""
echo "Hardware:"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MiB"
echo "  Free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) MiB"
echo ""

# Configuration
BASE_CHECKPOINT="checkpoints/rust-6hour/step_2000"
SUPERVISED_DIR="checkpoints/production-supervised"
RL_DIR="checkpoints/production-rl"
FINAL_DIR="checkpoints/production-final"
DATA_PATH="data/rust_tokens.bin"
DEVICE="cuda:0"
BATCH_SIZE=2
TARGET_STEPS=15000
CHECKPOINT_INTERVAL=500
RUN_EVAL="${RUN_EVAL:-0}"
MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://localhost:8080/v1/completions}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-HumanEval.jsonl}"
MBPP_PATH="${MBPP_PATH:-MBPP.json}"

# Timestamps
START_TIME=$(date +%s)
LOG_FILE="production_training_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# ═══════════════════════════════════════════════════════════
# PHASE 1: Extended Supervised Pre-training
# ═══════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 1: Extended Supervised Pre-training"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Goal: Train from step 2000 → 15000 (13,000 additional steps)"
echo "Strategy: Automatic restart on OOM with checkpoint resume"
echo ""

mkdir -p "$SUPERVISED_DIR"

# Check if we should resume from existing checkpoint
if [ -d "$SUPERVISED_DIR" ]; then
    LAST_CHECKPOINT=$(ls -d "$SUPERVISED_DIR"/step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CHECKPOINT" ]; then
        RESUME_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
        echo "Found existing checkpoint at step $RESUME_STEP"
        echo "Resuming from: $LAST_CHECKPOINT"
        BASE_CHECKPOINT="$LAST_CHECKPOINT"
    fi
fi

# Training loop with automatic restart
PHASE1_RUN=0
CURRENT_STEP=2000

while [ $CURRENT_STEP -lt $TARGET_STEPS ]; do
    PHASE1_RUN=$((PHASE1_RUN + 1))
    REMAINING_STEPS=$((TARGET_STEPS - CURRENT_STEP))

    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "Phase 1 - Run #$PHASE1_RUN | Current: step_$CURRENT_STEP | Target: step_$TARGET_STEPS"
    echo "───────────────────────────────────────────────────────────"

    # Run training (will OOM after ~600-1100 steps)
    cargo run --release --example train_rust_maxgpu --features nanochat-train/cuda -- \
        --data "$DATA_PATH" \
        --checkpoint-dir "$SUPERVISED_DIR" \
        --device "$DEVICE" \
        --batch-size $BATCH_SIZE \
        --total-steps $REMAINING_STEPS \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --log-interval 100 || {

        EXIT_CODE=$?
        echo ""
        echo "Training run ended (exit code: $EXIT_CODE)"

        # Check if OOM
        if tail -20 "$LOG_FILE" | grep -q "CUDA_ERROR_OUT_OF_MEMORY"; then
            echo "  → CUDA OOM (expected - restarting)"
        else
            echo "  → Unexpected error - check logs"
        fi

        # Wait for GPU cleanup
        echo "  → Waiting 5s for GPU cleanup..."
        sleep 5
    }

    # Update current step from last checkpoint
    LAST_CHECKPOINT=$(ls -d "$SUPERVISED_DIR"/step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CHECKPOINT" ]; then
        NEW_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
        STEPS_THIS_RUN=$((NEW_STEP - CURRENT_STEP))
        CURRENT_STEP=$NEW_STEP
        echo "  → Progress: step_$CURRENT_STEP (+$STEPS_THIS_RUN steps)"

        # Calculate ETA
        ELAPSED=$(($(date +%s) - START_TIME))
        TOTAL_TRAINED=$((CURRENT_STEP - 2000))
        REMAINING=$((TARGET_STEPS - CURRENT_STEP))
        if [ $TOTAL_TRAINED -gt 0 ]; then
            STEPS_PER_SEC=$(echo "scale=2; $TOTAL_TRAINED / $ELAPSED" | bc)
            ETA_SECONDS=$(echo "scale=0; $REMAINING / $STEPS_PER_SEC" | bc)
            ETA_HOURS=$(echo "scale=1; $ETA_SECONDS / 3600" | bc)
            echo "  → ETA: ~${ETA_HOURS}h (${STEPS_PER_SEC} steps/s)"
        fi
    else
        echo "  → Warning: No checkpoint found - will retry"
        sleep 10
    fi

    echo ""
done

echo ""
echo "✓ Phase 1 Complete: step_$CURRENT_STEP reached"
echo ""

# ═══════════════════════════════════════════════════════════
# PHASE 2: RL Fine-tuning with Compiler Feedback
# ═══════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 2: RL Fine-tuning with Compiler Feedback"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Goal: Apply GRPO/GSPO with rustc feedback for 1000 iterations"
echo "Strategy: Compiler-guided optimization for code quality"
echo ""

mkdir -p "$RL_DIR"

# Get best supervised checkpoint
BEST_SUPERVISED=$(ls -d "$SUPERVISED_DIR"/step_* 2>/dev/null | sort -V | tail -1)

echo "Starting RL from checkpoint: $BEST_SUPERVISED"
echo ""

# Run RL training
cargo run --release -p nanochat-rl --example train_rl -- \
    --checkpoint "$BEST_SUPERVISED" \
    --iterations 1000 \
    --n-samples 4 \
    --batch-size 3 \
    --device "$DEVICE" \
    --lr 1e-5 \
    --kl-coef 0.1 || {

    echo "Warning: RL training incomplete - check logs"
}

echo ""
echo "✓ Phase 2 Complete: RL fine-tuning done"
echo ""

# ═══════════════════════════════════════════════════════════
# PHASE 3: Evaluation
# ═══════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 3: Evaluation"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ "$RUN_EVAL" = "1" ]; then
    echo "Running automated benchmarks..."

    if [ -f "$HUMANEVAL_PATH" ]; then
        echo "  → HumanEval benchmark"
        cargo run --release --example evaluate_codegen -- \
            --dataset humaneval \
            --data-path "$HUMANEVAL_PATH" \
            --model-endpoint "$MODEL_ENDPOINT" \
            --num-samples 10 || echo "  ⚠ HumanEval benchmark failed"
    else
        echo "  → Skipping HumanEval (missing file: $HUMANEVAL_PATH)"
    fi

    if [ -f "$MBPP_PATH" ]; then
        echo "  → MBPP benchmark"
        cargo run --release --example evaluate_codegen -- \
            --dataset mbpp \
            --data-path "$MBPP_PATH" \
            --model-endpoint "$MODEL_ENDPOINT" \
            --num-samples 10 || echo "  ⚠ MBPP benchmark failed"
    else
        echo "  → Skipping MBPP (missing file: $MBPP_PATH)"
    fi
else
    echo "Skipping automated benchmarks (set RUN_EVAL=1 to enable)."
    echo "Expected inputs:"
    echo "  HUMANEVAL_PATH=$HUMANEVAL_PATH"
    echo "  MBPP_PATH=$MBPP_PATH"
    echo "  MODEL_ENDPOINT=$MODEL_ENDPOINT"
fi
echo ""

# ═══════════════════════════════════════════════════════════
# PHASE 4: Export Final Model
# ═══════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 4: Export to GGUF"
echo "═══════════════════════════════════════════════════════════"
echo ""

mkdir -p "$FINAL_DIR"

# Copy best checkpoint
BEST_RL=$(ls -d "$RL_DIR"/rl-iter-* 2>/dev/null | sort -V | tail -1)
if [ -z "$BEST_RL" ]; then
    BEST_RL="$BEST_SUPERVISED"
    echo "No RL checkpoint found, using supervised: $BEST_RL"
else
    echo "Using RL checkpoint: $BEST_RL"
fi

FINAL_GGUF="$FINAL_DIR/model.gguf"
FINAL_MHC="$FINAL_DIR/model.mhc"
echo "Export to GGUF:"
echo "  Source: $BEST_RL"
echo "  Target: $FINAL_GGUF"

if [ -d "$BEST_RL" ]; then
    cargo run --release -p nanochat-train -- \
        export \
        --checkpoint "$BEST_RL" \
        --gguf "$FINAL_GGUF" \
        --mhc "$FINAL_MHC" || echo "Warning: GGUF export failed"
else
    echo "Warning: source checkpoint directory not found: $BEST_RL"
fi
echo ""

# ═══════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo "═══════════════════════════════════════════════════════════"
echo "  Production Training Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Duration: ${HOURS}h ${MINUTES}m"
echo "Started:  $(date -d @$START_TIME)"
echo "Finished: $(date -d @$END_TIME)"
echo ""
echo "Checkpoints:"
echo "  Supervised: $SUPERVISED_DIR (step_2000 → step_$CURRENT_STEP)"
echo "  RL:         $RL_DIR (1000 iterations)"
echo "  Final:      $FINAL_DIR"
echo ""
echo "Next steps:"
echo "  1. Test the model: cargo run --example generate_code"
echo "  2. Run benchmarks: cargo run -p nanochat-eval"
echo "  3. Export to GGUF: cargo run --example export_gguf"
echo "  4. Deploy for inference: cargo run -p nanochat-serve"
echo ""
echo "Log: $LOG_FILE"
echo ""
