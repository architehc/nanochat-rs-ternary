#!/bin/bash
# Launch large (~7B) training profile.
#
# Notes:
# - Intended for multi-GPU or very high-memory hosts.
# - Uses 8-bit optimizer + GaLore + Collider + MTP defaults from `large-7b`.

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_DIR}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/large_7b_$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-synthetic}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
KEEP_LAST="${KEEP_LAST:-5}"

echo "========================================"
echo "Training large-7b profile"
echo "========================================"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "dataset=${DATASET}"
echo "device=${DEVICE}"
echo

cargo run --release -p nanochat-train -- train \
  --config large-7b \
  --dataset "${DATASET}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --log-interval "${LOG_INTERVAL}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --keep-last-checkpoints "${KEEP_LAST}"
