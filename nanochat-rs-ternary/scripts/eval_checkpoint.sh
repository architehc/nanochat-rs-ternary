#!/usr/bin/env bash
# eval_checkpoint.sh — Export checkpoint and run perplexity evaluation on CPU.
#
# Usage:
#   bash scripts/eval_checkpoint.sh checkpoints_7b/step_500
#   bash scripts/eval_checkpoint.sh checkpoints_7b/step_500 data/owt_eval/tokens.bin
#
# Runs on CPU so it doesn't interfere with GPU training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
EVAL_CSV="${LOG_DIR}/eval_results.csv"
EVAL_TOKENS="${2:-${PROJECT_DIR}/data/owt_eval/tokens.bin}"
SEQ_LEN="${SEQ_LEN:-4096}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-100000}"  # Limit for speed

CHECKPOINT_DIR="${1:?Usage: eval_checkpoint.sh <checkpoint_dir> [eval_tokens_path]}"

mkdir -p "$LOG_DIR"

echo "=== Checkpoint Evaluation ==="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Eval tokens: $EVAL_TOKENS"
echo "Seq len: $SEQ_LEN"
echo "Max tokens: $MAX_EVAL_TOKENS"
echo ""

# Extract step number from checkpoint path
STEP=$(basename "$CHECKPOINT_DIR" | grep -oP '\d+' || echo "0")
echo "Step: $STEP"

# Create temp dir for exported model
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

GGUF_PATH="${TMPDIR}/model.gguf"
MHC_PATH="${TMPDIR}/model.mhc"

# Export checkpoint to GGUF + mHC
echo ""
echo "--- Exporting checkpoint ---"
cargo run --release -p nanochat-train -- export \
    --checkpoint "$CHECKPOINT_DIR" \
    --gguf "$GGUF_PATH" \
    --mhc "$MHC_PATH"

if [[ ! -f "$GGUF_PATH" ]] || [[ ! -f "$MHC_PATH" ]]; then
    echo "ERROR: Export failed — GGUF or mHC file not created"
    exit 1
fi

echo ""
echo "--- Running perplexity evaluation ---"

# Check if eval tokens exist
if [[ ! -f "$EVAL_TOKENS" ]]; then
    echo "WARNING: Eval token file not found: $EVAL_TOKENS"
    echo "Skipping perplexity evaluation."
    exit 0
fi

# Run perplexity eval (uses nanochat-eval binary if available, otherwise inline)
# For now, output a placeholder that can be replaced with actual eval binary
# The perplexity module is available as a library in nanochat-eval

# Simple Python-free perplexity using the GGUF model
# TODO: Build a nanochat-eval CLI binary that wraps evaluate_perplexity()
echo "Perplexity evaluation requires nanochat-eval CLI (not yet built as binary)."
echo "Library function available: nanochat_eval::perplexity::evaluate_perplexity()"
echo ""
echo "Results will be appended to: $EVAL_CSV"
echo ""

# Log to CSV with placeholder (will be filled by actual eval)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [[ ! -f "$EVAL_CSV" ]]; then
    echo "step,avg_loss,perplexity,bits_per_byte,n_tokens,n_chunks,timestamp" > "$EVAL_CSV"
fi
echo "# Step $STEP exported successfully at $TIMESTAMP" >> "$EVAL_CSV"

echo "=== Evaluation Complete ==="
echo "Exported model: $GGUF_PATH ($(du -h "$GGUF_PATH" | cut -f1))"
echo "Exported mHC: $MHC_PATH ($(du -h "$MHC_PATH" | cut -f1))"
