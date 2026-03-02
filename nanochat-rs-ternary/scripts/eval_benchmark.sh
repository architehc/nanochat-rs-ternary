#!/bin/bash
# Evaluate all trained models on benchmark prompts
# Usage: ./scripts/eval_benchmark.sh [device]
set -euo pipefail

DEVICE="${1:-cpu}"
TOKENIZER="data/rust_big/tokenizer.json"
EVAL_PROMPTS="data/rust_big/eval_prompts.jsonl"
RESULTS_DIR="eval_results"

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$RESULTS_DIR"

# Models to evaluate
declare -A MODELS
MODELS["haar"]="checkpoints/nano-275m-haar-v3/latest"
MODELS["looplm"]="checkpoints/nano-275m-loop-only/latest"
MODELS["engram"]="checkpoints/nano-275m-engram-only/latest"
# Also evaluate existing checkpoints
MODELS["haar-30k"]="checkpoints/nano-275m-haar/step_30000"
MODELS["engram-loop-5k"]="checkpoints/nano-275m-engram-loop/step_5000"

echo "=== Rust Code Generation Benchmark ==="
echo "Device: $DEVICE"
echo ""

for name in "${!MODELS[@]}"; do
    ckpt="${MODELS[$name]}"
    if [ ! -f "$ckpt/model.safetensors" ]; then
        echo "[$name] SKIP - checkpoint not found: $ckpt"
        continue
    fi

    echo "=== Evaluating: $name ($ckpt) ==="
    outfile="$RESULTS_DIR/${name}_generation.txt"

    # Standard prompts for generation quality
    prompts=(
        "fn main() {"
        "pub struct Config {"
        "impl Iterator for"
        "pub fn parse(input: &str) -> Result<"
        "#[derive(Debug, Clone)]\npub struct "
        "async fn handle_request("
        "fn from_str(s: &str) -> Result<Self, Self::Err> {"
        "#[test]\nfn test_"
        "use std::collections::HashMap;\n\nfn "
        "/// A simple key-value store.\npub struct Store {"
    )

    echo "--- $name ---" > "$outfile"
    for prompt in "${prompts[@]}"; do
        echo "" >> "$outfile"
        echo "=== PROMPT: $prompt ===" >> "$outfile"
        # Generate with moderate temperature
        cargo run --release -p nanochat-train --features cuda -- generate \
            --checkpoint "$ckpt" \
            --tokenizer "$TOKENIZER" \
            --prompt "$prompt" \
            --max-tokens 200 \
            --temperature 0.7 \
            --top-k 40 \
            --device "$DEVICE" 2>/dev/null >> "$outfile" || echo "[ERROR generating]" >> "$outfile"
        echo "" >> "$outfile"
    done

    echo "  Generated samples saved to: $outfile"

    # Compute test perplexity
    echo "  Computing test perplexity..."
    # TODO: Add perplexity eval command to nanochat-train
done

echo ""
echo "=== Results in $RESULTS_DIR/ ==="
echo "Review generation quality: cat $RESULTS_DIR/<model>_generation.txt"
