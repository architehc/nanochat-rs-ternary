#!/bin/bash
# Benchmark all Muon optimizer variants and emit JSON report.
#
# Variants:
# - muon
# - muon_8bit
# - galore2
# - galore2_muon_8bit

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${STEPS:-100}"
OUTPUT_DIR="${WORKSPACE_DIR}/benchmark_results_optimizers_$(date +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${OUTPUT_DIR}/benchmark_optimizers.json"

mkdir -p "${OUTPUT_DIR}"
cd "${WORKSPACE_DIR}"

echo "========================================"
echo "Optimizer Benchmark Suite"
echo "========================================"
echo "Steps per variant: ${STEPS}"
echo "Output: ${OUTPUT_JSON}"
echo

cargo run --release -p nanochat-train --example benchmark_optimizers -- \
  --steps "${STEPS}" \
  --output "${OUTPUT_JSON}"

echo
echo "========================================"
echo "Benchmark Complete"
echo "========================================"
echo "Report: ${OUTPUT_JSON}"
echo
cat "${OUTPUT_JSON}"
