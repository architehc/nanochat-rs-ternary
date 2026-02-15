#!/bin/bash
# Extended benchmark runner (compilation + quality metrics).
#
# Requires a checkpoint directory with:
# - model.safetensors
# - meta.json
#
# Example:
#   CHECKPOINT=checkpoints/stable-v2/step_20000 ./scripts/run_extended_benchmarks.sh

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_DIR}"

CHECKPOINT="${CHECKPOINT:-}"
OUTPUT="${OUTPUT:-benchmark_extended_$(date +%Y%m%d_%H%M%S).json}"
N_SAMPLES="${N_SAMPLES:-25}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-200}"
DEVICE="${DEVICE:-cpu}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "Set CHECKPOINT to a checkpoint directory (contains meta.json + model.safetensors)."
  exit 1
fi

echo "========================================"
echo "Running extended benchmarks"
echo "========================================"
echo "checkpoint=${CHECKPOINT}"
echo "output=${OUTPUT}"
echo "samples=${N_SAMPLES}"
echo

cargo run --release -p nanochat-eval --example benchmark_model -- \
  --checkpoint "${CHECKPOINT}" \
  --n-samples "${N_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}"

echo
echo "Benchmark results saved to ${OUTPUT}"
