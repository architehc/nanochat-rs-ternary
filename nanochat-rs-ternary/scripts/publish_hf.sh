#!/bin/bash
# Publish GGUF + mHC artifacts to Hugging Face.
#
# Usage:
#   HF_REPO=architehc/nanochat-3b \
#   MODEL_NAME=nanochat-3b \
#   GGUF_PATH=path/to/model.gguf \
#   MHC_PATH=path/to/model.mhc \
#   ./scripts/publish_hf.sh

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli not found. Install with: pip install huggingface_hub"
  exit 1
fi

HF_REPO="${HF_REPO:-}"
MODEL_NAME="${MODEL_NAME:-}"
GGUF_PATH="${GGUF_PATH:-}"
MHC_PATH="${MHC_PATH:-}"
MODEL_CARD="${MODEL_CARD:-docs/huggingface_model_card_template.md}"
UPDATE_MODEL_ZOO="${UPDATE_MODEL_ZOO:-1}"

if [[ -z "${HF_REPO}" || -z "${GGUF_PATH}" || -z "${MHC_PATH}" ]]; then
  echo "Missing required env vars:"
  echo "  HF_REPO=<org/model>"
  echo "  MODEL_NAME=<model-zoo key>   # optional but recommended"
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
echo "model_name=${MODEL_NAME:-<unset>}"
echo "gguf=${GGUF_PATH}"
echo "mhc=${MHC_PATH}"
echo

GGUF_SHA256="$(sha256sum "${GGUF_PATH}" | awk '{print $1}')"
MHC_SHA256="$(sha256sum "${MHC_PATH}" | awk '{print $1}')"
echo "sha256_gguf=${GGUF_SHA256}"
echo "sha256_mhc=${MHC_SHA256}"
echo

huggingface-cli repo create "${HF_REPO}" --type model --exist-ok
huggingface-cli upload "${HF_REPO}" "${GGUF_PATH}" model.gguf
huggingface-cli upload "${HF_REPO}" "${MHC_PATH}" model.mhc
huggingface-cli upload "${HF_REPO}" "${MODEL_CARD}" README.md

if [[ "${UPDATE_MODEL_ZOO}" == "1" ]]; then
  if [[ -z "${MODEL_NAME}" ]]; then
    echo "⚠ MODEL_NAME not set; skipping model-zoo checksum update."
  else
    ./scripts/update_model_zoo_checksums.py \
      --model "${MODEL_NAME}" \
      --gguf "${GGUF_PATH}" \
      --mhc "${MHC_PATH}" \
      --checksums-file "nanochat/model_zoo_checksums.json"

    huggingface-cli upload "${HF_REPO}" "nanochat/model_zoo_checksums.json" model_zoo_checksums.json
    echo "Uploaded updated model_zoo_checksums.json"
  fi
fi

echo
echo "Publish complete: https://huggingface.co/${HF_REPO}"
