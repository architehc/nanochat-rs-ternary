#!/bin/bash
# Publish GGUF + mHC artifacts to Hugging Face.
#
# Usage:
#   HF_REPO=architehc/nanochat-3b \
#   GGUF_PATH=path/to/model.gguf \
#   MHC_PATH=path/to/model.mhc \
#   ./scripts/publish_hf.sh

set -euo pipefail

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli not found. Install with: pip install huggingface_hub"
  exit 1
fi

HF_REPO="${HF_REPO:-}"
GGUF_PATH="${GGUF_PATH:-}"
MHC_PATH="${MHC_PATH:-}"
MODEL_CARD="${MODEL_CARD:-docs/huggingface_model_card_template.md}"

if [[ -z "${HF_REPO}" || -z "${GGUF_PATH}" || -z "${MHC_PATH}" ]]; then
  echo "Missing required env vars:"
  echo "  HF_REPO=<org/model>"
  echo "  GGUF_PATH=<path/to/model.gguf>"
  echo "  MHC_PATH=<path/to/model.mhc>"
  exit 1
fi

if [[ ! -f "${GGUF_PATH}" ]]; then
  echo "GGUF not found: ${GGUF_PATH}"
  exit 1
fi

if [[ ! -f "${MHC_PATH}" ]]; then
  echo "mHC not found: ${MHC_PATH}"
  exit 1
fi

if [[ ! -f "${MODEL_CARD}" ]]; then
  echo "Model card not found: ${MODEL_CARD}"
  exit 1
fi

echo "========================================"
echo "Publishing to Hugging Face"
echo "========================================"
echo "repo=${HF_REPO}"
echo "gguf=${GGUF_PATH}"
echo "mhc=${MHC_PATH}"
echo

huggingface-cli repo create "${HF_REPO}" --type model --exist-ok
huggingface-cli upload "${HF_REPO}" "${GGUF_PATH}" model.gguf
huggingface-cli upload "${HF_REPO}" "${MHC_PATH}" model.mhc
huggingface-cli upload "${HF_REPO}" "${MODEL_CARD}" README.md

echo
echo "Publish complete: https://huggingface.co/${HF_REPO}"
