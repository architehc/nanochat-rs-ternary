#!/bin/bash
# Comprehensive coherence evaluation across all generatable models
# Measures: syntax correctness, bracket matching, repetition, pattern quality

set -e
cd "$(dirname "$0")/.."

DEVICE="${1:-cuda}"
TOKENIZER="data/rust_v2_prepared/tokenizer.json"
RESULTS_DIR="eval_results/coherence_$(date +%Y%m%d_%H%M)"
mkdir -p "$RESULTS_DIR"

BINARY="./target/release/nanochat-train"

# Models to evaluate (only those that can generate)
declare -A MODELS
MODELS[engram-v1]="checkpoints/nano-275m-engram-v1/final"
MODELS[baseline-v1]="checkpoints/nano-275m-baseline-v1/final"
MODELS[engram-v2]="checkpoints/nano-275m-engram-v2/final"

# Standard prompts for coherence testing
PROMPTS=(
    "fn main() {"
    "pub struct Config {"
    "use std::collections::HashMap;\n\nfn "
    "impl Iterator for "
    "pub fn parse(input: &str) -> Result<"
    "#[derive(Debug, Clone)]\npub struct "
    "async fn handle_request(req: Request) -> Response {"
    "impl Display for Error {\n    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {"
    "pub trait "
    "match self {"
    "#[test]\nfn test_"
    "let mut result = Vec::new();\n    for "
    "fn from_str(s: &str) -> Result<Self, Self::Err> {"
    "/// Returns the number of elements in the collection.\npub fn len(&self) -> usize {"
    "use serde::{Deserialize, Serialize};\n\n#[derive(Debug, Serialize, Deserialize)]\npub struct "
)

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "=== Coherence Evaluation ==="
echo "Device: $DEVICE"
echo "Results: $RESULTS_DIR"
echo ""

for MODEL_NAME in "${!MODELS[@]}"; do
    CKPT="${MODELS[$MODEL_NAME]}"
    if [ ! -d "$CKPT" ]; then
        echo "SKIP: $MODEL_NAME ($CKPT not found)"
        continue
    fi

    OUTFILE="$RESULTS_DIR/${MODEL_NAME}.txt"
    echo "=== Evaluating: $MODEL_NAME ==="
    echo "# Model: $MODEL_NAME" > "$OUTFILE"
    echo "# Checkpoint: $CKPT" >> "$OUTFILE"
    echo "# Device: $DEVICE" >> "$OUTFILE"
    echo "# Date: $(date)" >> "$OUTFILE"
    echo "" >> "$OUTFILE"

    # Select GPU device based on model
    DEV="$DEVICE"
    if [ "$MODEL_NAME" == "engram-v1" ] || [ "$MODEL_NAME" == "engram-v2" ]; then
        DEV="cuda:1"
    else
        DEV="cuda"
    fi

    for i in "${!PROMPTS[@]}"; do
        PROMPT="${PROMPTS[$i]}"
        echo "  Prompt $((i+1))/${#PROMPTS[@]}: ${PROMPT:0:40}..."

        echo "--- PROMPT $((i+1)) ---" >> "$OUTFILE"
        echo "INPUT: $PROMPT" >> "$OUTFILE"
        echo "OUTPUT:" >> "$OUTFILE"

        # Run generation with timeout
        timeout 60 $BINARY generate \
            --checkpoint "$CKPT" \
            --tokenizer "$TOKENIZER" \
            --device "$DEV" \
            --prompt "$PROMPT" \
            --max-tokens 256 \
            --temperature 0.7 \
            --top-k 40 \
            2>/dev/null >> "$OUTFILE" || echo "[TIMEOUT/ERROR]" >> "$OUTFILE"

        echo "" >> "$OUTFILE"
        echo "" >> "$OUTFILE"
    done

    echo "  Saved to $OUTFILE"
    echo ""
done

echo "=== Evaluation complete ==="
echo "Results in: $RESULTS_DIR"
