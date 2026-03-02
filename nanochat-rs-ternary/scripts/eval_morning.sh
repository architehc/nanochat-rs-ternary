#!/bin/bash
# Morning evaluation: generate samples from all trained checkpoints
# Run this after overnight training completes
set -euo pipefail

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

TOKENIZER="data/rust_big/tokenizer.json"
RESULTS_DIR="eval_results"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "  NANOCHAT ARCHITECTURE COMPARISON - $(date)"
echo "================================================================"

# Find best checkpoints for each model
declare -A CKPTS
declare -A LABELS

for dir in checkpoints/nano-275m-engram-only checkpoints/nano-275m-loop-only checkpoints/nano-275m-haar/step_30000; do
    name=$(basename "$dir")
    if [ "$name" = "step_30000" ]; then
        name="haar-30k"
        ckpt="$dir"
    else
        ckpt=$(ls -1d ${dir}/step_* 2>/dev/null | sort -V | tail -1 || true)
    fi
    if [ -n "$ckpt" ] && [ -f "$ckpt/model.safetensors" ]; then
        CKPTS["$name"]="$ckpt"
        echo "Found: $name -> $ckpt"
    else
        echo "SKIP: $name (no checkpoint)"
    fi
done

echo ""

# Standard prompts for Rust code generation
PROMPTS=(
    "fn main() {"
    "pub struct Config {"
    "pub fn parse(input: &str) -> Result<"
    "#[derive(Debug, Clone)]\npub struct "
    "impl Iterator for"
    "pub fn sort(arr: &mut [i32]) {"
    "fn fibonacci(n: u64) -> u64 {"
    "async fn fetch_url(url: &str) -> Result<String, Box<dyn std::error::Error>> {"
    "use std::collections::HashMap;\n\nfn count_words(text: &str) -> HashMap<&str, usize> {"
    "#[test]\nfn test_"
)

for name in "${!CKPTS[@]}"; do
    ckpt="${CKPTS[$name]}"
    outfile="$RESULTS_DIR/${name}_samples.txt"

    echo "============================================" | tee "$outfile"
    echo " MODEL: $name" | tee -a "$outfile"
    echo " CHECKPOINT: $ckpt" | tee -a "$outfile"
    
    # Read loss from meta.json
    if [ -f "$ckpt/meta.json" ]; then
        loss=$(python3 -c "import json; d=json.load(open('$ckpt/meta.json')); print(f'loss={d.get(\"loss\",\"?\")}, step={d.get(\"step\",\"?\")}')" 2>/dev/null)
        echo " STATUS: $loss" | tee -a "$outfile"
    fi
    echo "============================================" | tee -a "$outfile"
    echo "" | tee -a "$outfile"

    for prompt in "${PROMPTS[@]}"; do
        echo "--- PROMPT: $prompt ---" | tee -a "$outfile"
        timeout 120 cargo run --release -p nanochat-train --features cuda -- generate \
            --checkpoint "$ckpt" \
            --tokenizer "$TOKENIZER" \
            --prompt "$prompt" \
            --max-tokens 256 \
            --temperature 0.7 \
            --top-k 40 \
            --device cpu 2>/dev/null >> "$outfile" 2>&1 || echo "[TIMEOUT/ERROR]" >> "$outfile"
        echo "" | tee -a "$outfile"
        echo "" >> "$outfile"
    done

    echo "" | tee -a "$outfile"
done

echo ""
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"

# Print training loss summary
for name in "${!CKPTS[@]}"; do
    ckpt="${CKPTS[$name]}"
    if [ -f "$ckpt/meta.json" ]; then
        loss=$(python3 -c "import json; d=json.load(open('$ckpt/meta.json')); print(d.get('loss','?'))" 2>/dev/null)
        step=$(python3 -c "import json; d=json.load(open('$ckpt/meta.json')); print(d.get('step','?'))" 2>/dev/null)
        echo "  $name: loss=$loss at step=$step"
    fi
done

echo ""
echo "Generation samples saved in: $RESULTS_DIR/"
echo "  cat eval_results/<model>_samples.txt"
echo ""
echo "Compare: diff eval_results/nano-275m-engram-only_samples.txt eval_results/nano-275m-loop-only_samples.txt"
