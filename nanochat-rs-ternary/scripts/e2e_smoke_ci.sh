#!/usr/bin/env bash
# Fast CI-oriented E2E smoke:
#   check -> key tests -> tiny CPU train/export -> serve/API smoke -> optional RL smoke
#
# Usage:
#   ./scripts/e2e_smoke_ci.sh
#
# Optional env vars:
#   SMOKE_ENABLE_RL=0|1       Run GRPO RL smoke (default: 0)
#   SMOKE_RL_ITERATIONS=2
#   SMOKE_RL_SAMPLES=2
#   SMOKE_RL_BATCH_SIZE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_DIR}"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

note() {
    echo "[$(timestamp)] $*"
}

section() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

latest_step_checkpoint() {
    local dir="$1"
    local ckpt
    ckpt="$(find "${dir}" -maxdepth 1 -type d -name 'step_*' | sort -V | tail -1 || true)"
    [[ -n "${ckpt}" ]] || return 1
    echo "${ckpt}"
}

wait_for_http() {
    local url="$1"
    local retries="${2:-45}"
    local delay_secs="${3:-2}"
    local i
    for ((i = 1; i <= retries; i++)); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep "${delay_secs}"
    done
    return 1
}

RUN_ID="${RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}"
ART="${ARTIFACT_ROOT:-${WORKSPACE_DIR}/artifacts/e2e/${RUN_ID}}"
LOG_DIR="${ART}/logs"
SMOKE_DIR="${ART}/smoke"
mkdir -p "${LOG_DIR}" "${SMOKE_DIR}/checkpoints"

SMOKE_SFT_CONFIG="${SMOKE_SFT_CONFIG:-test-8bit}"
SMOKE_SFT_SAMPLES="${SMOKE_SFT_SAMPLES:-4096}"
SMOKE_SFT_EPOCHS="${SMOKE_SFT_EPOCHS:-1}"
SMOKE_SFT_CHECKPOINT_INTERVAL="${SMOKE_SFT_CHECKPOINT_INTERVAL:-20}"
SMOKE_SFT_LOG_INTERVAL="${SMOKE_SFT_LOG_INTERVAL:-20}"
SMOKE_SERVE_PORT="${SMOKE_SERVE_PORT:-18082}"

SMOKE_ENABLE_RL="${SMOKE_ENABLE_RL:-0}"
SMOKE_RL_ITERATIONS="${SMOKE_RL_ITERATIONS:-2}"
SMOKE_RL_SAMPLES="${SMOKE_RL_SAMPLES:-2}"
SMOKE_RL_BATCH_SIZE="${SMOKE_RL_BATCH_SIZE:-1}"
SMOKE_RL_LR="${SMOKE_RL_LR:-1e-5}"
SMOKE_RL_KL_COEF="${SMOKE_RL_KL_COEF:-0.1}"

section "Preflight"
command -v cargo >/dev/null || fail "cargo not found"
command -v curl >/dev/null || fail "curl not found"
require_file "${WORKSPACE_DIR}/models/gpt2-tokenizer.json"

{
    echo "run_id=${RUN_ID}"
    echo "workspace=${WORKSPACE_DIR}"
    echo "artifact_root=${ART}"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
} | tee "${ART}/run_meta.txt"

section "CPU Check + Key Tests"
{
    note "cargo check --workspace --all-targets"
    cargo check --workspace --all-targets

    note "triangle_of_truth"
    cargo test --test triangle_of_truth -- --nocapture

    note "roundtrip_test"
    cargo test --test roundtrip_test -- --nocapture

    note "e2e_generate"
    cargo test --test e2e_generate -- --nocapture
} 2>&1 | tee "${LOG_DIR}/smoke_checks.log"

section "CPU Train + Export Smoke"
{
    note "training smoke checkpoint (${SMOKE_SFT_CONFIG})"
    cargo run --release -p nanochat-train -- train \
        --config "${SMOKE_SFT_CONFIG}" \
        --dataset synthetic \
        --n-samples "${SMOKE_SFT_SAMPLES}" \
        --epochs "${SMOKE_SFT_EPOCHS}" \
        --device cpu \
        --checkpoint-dir "${SMOKE_DIR}/checkpoints" \
        --log-interval "${SMOKE_SFT_LOG_INTERVAL}" \
        --checkpoint-interval "${SMOKE_SFT_CHECKPOINT_INTERVAL}"
} 2>&1 | tee "${LOG_DIR}/smoke_train.log"

SMOKE_CKPT="$(latest_step_checkpoint "${SMOKE_DIR}/checkpoints" || true)"
[[ -n "${SMOKE_CKPT}" ]] || fail "no step checkpoint found in ${SMOKE_DIR}/checkpoints"
require_file "${SMOKE_CKPT}/meta.json"
require_file "${SMOKE_CKPT}/model.safetensors"

{
    note "exporting smoke checkpoint"
    cargo run --release -p nanochat-train -- export \
        --checkpoint "${SMOKE_CKPT}" \
        --gguf "${SMOKE_DIR}/model.gguf" \
        --mhc "${SMOKE_DIR}/model.mhc"
} 2>&1 | tee "${LOG_DIR}/smoke_export.log"

require_file "${SMOKE_DIR}/model.gguf"
require_file "${SMOKE_DIR}/model.mhc"

section "Serve/API Smoke"
SERVE_PID=""
SERVE_PID_FILE="${ART}/serve.pid"
cleanup() {
    local pid="${SERVE_PID}"
    if [[ -z "${pid}" ]] && [[ -f "${SERVE_PID_FILE}" ]]; then
        pid="$(cat "${SERVE_PID_FILE}" 2>/dev/null || true)"
    fi
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
        wait "${pid}" >/dev/null 2>&1 || true
    fi
    rm -f "${SERVE_PID_FILE}"
}
trap cleanup EXIT

{
    cargo run --release -p nanochat-serve -- \
        --model "${SMOKE_DIR}/model.gguf" \
        --mhc "${SMOKE_DIR}/model.mhc" \
        --tokenizer "${WORKSPACE_DIR}/models/gpt2-tokenizer.json" \
        --port "${SMOKE_SERVE_PORT}" \
        > "${LOG_DIR}/smoke_serve_runtime.log" 2>&1 &
    SERVE_PID=$!
    echo "${SERVE_PID}" > "${SERVE_PID_FILE}"

    if ! wait_for_http "http://127.0.0.1:${SMOKE_SERVE_PORT}/health" 60 2; then
        fail "smoke server failed to become healthy"
    fi

    curl -fsS "http://127.0.0.1:${SMOKE_SERVE_PORT}/health" | tee "${ART}/health.txt"
    curl -fsS "http://127.0.0.1:${SMOKE_SERVE_PORT}/v1/models" | tee "${ART}/models.json"
    curl -fsS "http://127.0.0.1:${SMOKE_SERVE_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Write a Rust max function"}],"max_tokens":32}' \
        | tee "${ART}/completion.json"

    cleanup
} 2>&1 | tee "${LOG_DIR}/smoke_serve.log"

if [[ "${SMOKE_ENABLE_RL}" == "1" ]]; then
    section "Optional RL Smoke (GRPO)"
    RL_SMOKE_DIR="${ART}/rl_smoke"
    mkdir -p "${RL_SMOKE_DIR}"
    {
        pushd "${RL_SMOKE_DIR}" >/dev/null
        NANOCHAT_TOKENIZER="${WORKSPACE_DIR}/models/gpt2-tokenizer.json" \
            cargo run --release \
            --manifest-path "${WORKSPACE_DIR}/Cargo.toml" \
            -p nanochat-rl --example train_rl -- \
            --checkpoint "${SMOKE_CKPT}" \
            --iterations "${SMOKE_RL_ITERATIONS}" \
            --n-samples "${SMOKE_RL_SAMPLES}" \
            --batch-size "${SMOKE_RL_BATCH_SIZE}" \
            --device cpu \
            --lr "${SMOKE_RL_LR}" \
            --kl-coef "${SMOKE_RL_KL_COEF}"
        popd >/dev/null
    } 2>&1 | tee "${LOG_DIR}/smoke_rl.log"

    require_file "${RL_SMOKE_DIR}/checkpoints/rl-final/meta.json"
    require_file "${RL_SMOKE_DIR}/checkpoints/rl-final/model.safetensors"
fi

section "Smoke Complete"
cat <<EOF
Smoke pipeline finished successfully.

Artifacts:
  ${ART}

Checkpoint:
  ${SMOKE_CKPT}

Export:
  ${SMOKE_DIR}/model.gguf
  ${SMOKE_DIR}/model.mhc
EOF
