#!/usr/bin/env bash
# End-to-end pipeline orchestrator:
#   CPU checks -> GPU checks -> SFT (CPU/GPU) -> export -> serve/API -> GRPO RL -> benchmarks
#
# Usage:
#   ./scripts/e2e_full_pipeline.sh
#
# Important knobs (env vars):
#   ARTIFACT_ROOT                 Output root (default: artifacts/e2e/<timestamp>)
#   REQUIRE_GPU=1                 Fail if no GPU is visible
#   RUN_VALIDATE_E2E=1            Run scripts/validate_e2e.sh
#   RUN_BENCHMARKS=1              Run benchmark_model before/after RL
#   SKIP_RL=0                     Skip RL stage when set to 1
#   CPU_SFT_CONFIG=test-8bit      CPU supervised smoke config
#   GPU_SFT_CONFIG=test-8bit      GPU supervised smoke config
#   RL_ITERATIONS=40              GRPO iterations
#
# Notes:
# - This script uses GRPO RL (`examples/train_rl`), not `train_maxrl` (which is not wired to update weights yet).
# - Runs from repo root; creates timestamped artifact/log directories.

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
    local retries="${2:-60}"
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

assert_checkpoint_dir() {
    local dir="$1"
    require_file "${dir}/meta.json"
    require_file "${dir}/model.safetensors"
}

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${WORKSPACE_DIR}/artifacts/e2e/${RUN_ID}}"
LOG_DIR="${ARTIFACT_ROOT}/logs"
CPU_DIR="${ARTIFACT_ROOT}/cpu"
GPU_DIR="${ARTIFACT_ROOT}/gpu"
RL_DIR="${ARTIFACT_ROOT}/rl"
EVAL_DIR="${ARTIFACT_ROOT}/eval"

REQUIRE_GPU="${REQUIRE_GPU:-1}"
RUN_VALIDATE_E2E="${RUN_VALIDATE_E2E:-1}"
RUN_BENCHMARKS="${RUN_BENCHMARKS:-1}"
SKIP_RL="${SKIP_RL:-0}"

CPU_SFT_CONFIG="${CPU_SFT_CONFIG:-test-8bit}"
CPU_SFT_SAMPLES="${CPU_SFT_SAMPLES:-8192}"
CPU_SFT_EPOCHS="${CPU_SFT_EPOCHS:-1}"
CPU_SFT_CHECKPOINT_INTERVAL="${CPU_SFT_CHECKPOINT_INTERVAL:-20}"
CPU_SFT_LOG_INTERVAL="${CPU_SFT_LOG_INTERVAL:-20}"

GPU_SFT_CONFIG="${GPU_SFT_CONFIG:-test-8bit}"
GPU_SFT_SAMPLES="${GPU_SFT_SAMPLES:-20000}"
GPU_SFT_EPOCHS="${GPU_SFT_EPOCHS:-1}"
GPU_SFT_CHECKPOINT_INTERVAL="${GPU_SFT_CHECKPOINT_INTERVAL:-20}"
GPU_SFT_LOG_INTERVAL="${GPU_SFT_LOG_INTERVAL:-20}"

SERVE_PORT="${SERVE_PORT:-18081}"

RL_ITERATIONS="${RL_ITERATIONS:-40}"
RL_SAMPLES="${RL_SAMPLES:-4}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-2}"
RL_LR="${RL_LR:-1e-5}"
RL_KL_COEF="${RL_KL_COEF:-0.1}"

BENCHMARK_SAMPLES="${BENCHMARK_SAMPLES:-40}"
BENCHMARK_MAX_TOKENS="${BENCHMARK_MAX_TOKENS:-200}"
BENCHMARK_TEMPERATURE="${BENCHMARK_TEMPERATURE:-0.8}"

mkdir -p "${LOG_DIR}" "${CPU_DIR}" "${GPU_DIR}" "${RL_DIR}" "${EVAL_DIR}"

section "Preflight"
command -v rustc >/dev/null || fail "rustc not found"
command -v cargo >/dev/null || fail "cargo not found"
command -v curl >/dev/null || fail "curl not found"

require_file "${WORKSPACE_DIR}/models/gpt2-tokenizer.json"

HAVE_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAVE_GPU=1
fi

if [[ "${HAVE_GPU}" -eq 0 && "${REQUIRE_GPU}" == "1" ]]; then
    fail "no visible GPU; set REQUIRE_GPU=0 to allow CPU-only execution"
fi

if [[ "${HAVE_GPU}" -eq 1 ]]; then
    RL_DEVICE_DEFAULT="cuda:0"
else
    RL_DEVICE_DEFAULT="cpu"
fi
RL_DEVICE="${RL_DEVICE:-${RL_DEVICE_DEFAULT}}"

{
    echo "run_id=${RUN_ID}"
    echo "workspace=${WORKSPACE_DIR}"
    echo "artifact_root=${ARTIFACT_ROOT}"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    if [[ "${HAVE_GPU}" -eq 1 ]]; then
        echo "gpu=$(nvidia-smi -L | head -1)"
    else
        echo "gpu=none"
    fi
} | tee "${ARTIFACT_ROOT}/run_meta.txt"

section "Lane A: CPU Validation"
{
    note "cargo check --workspace --all-targets"
    cargo check --workspace --all-targets

    note "cargo test --workspace --lib"
    cargo test --workspace --lib

    note "cargo test --test triangle_of_truth"
    cargo test --test triangle_of_truth -- --nocapture

    note "cargo test --test roundtrip_test"
    cargo test --test roundtrip_test -- --nocapture

    note "cargo test --test export_roundtrip"
    cargo test --test export_roundtrip -- --nocapture

    note "cargo test --test e2e_generate"
    cargo test --test e2e_generate -- --nocapture

    note "cargo test -p nanochat-train --tests"
    cargo test -p nanochat-train --tests
} 2>&1 | tee "${LOG_DIR}/lane_a_cpu_validation.log"

section "Lane C: Supervised Fine-tune Smoke (CPU)"
CPU_SFT_OUT="${CPU_DIR}/sft"
mkdir -p "${CPU_SFT_OUT}/checkpoints"
{
    note "training CPU checkpoint (${CPU_SFT_CONFIG})"
    cargo run --release -p nanochat-train -- train \
        --config "${CPU_SFT_CONFIG}" \
        --dataset synthetic \
        --n-samples "${CPU_SFT_SAMPLES}" \
        --epochs "${CPU_SFT_EPOCHS}" \
        --device cpu \
        --checkpoint-dir "${CPU_SFT_OUT}/checkpoints" \
        --log-interval "${CPU_SFT_LOG_INTERVAL}" \
        --checkpoint-interval "${CPU_SFT_CHECKPOINT_INTERVAL}"
} 2>&1 | tee "${LOG_DIR}/lane_c_cpu_sft.log"

CPU_CKPT="$(latest_step_checkpoint "${CPU_SFT_OUT}/checkpoints" || true)"
[[ -n "${CPU_CKPT}" ]] || fail "no CPU step checkpoint found in ${CPU_SFT_OUT}/checkpoints"
assert_checkpoint_dir "${CPU_CKPT}"

{
    note "exporting CPU checkpoint"
    cargo run --release -p nanochat-train -- export \
        --checkpoint "${CPU_CKPT}" \
        --gguf "${CPU_SFT_OUT}/model.gguf" \
        --mhc "${CPU_SFT_OUT}/model.mhc"
} 2>&1 | tee "${LOG_DIR}/lane_c_cpu_export.log"

require_file "${CPU_SFT_OUT}/model.gguf"
require_file "${CPU_SFT_OUT}/model.mhc"

SERVE_GGUF="${CPU_SFT_OUT}/model.gguf"
SERVE_MHC="${CPU_SFT_OUT}/model.mhc"
BASELINE_CKPT="${CPU_CKPT}"

if [[ "${HAVE_GPU}" -eq 1 ]]; then
    section "Lane B: GPU Validation"
    {
        note "cargo build --release -p nanochat-train --features cuda,tensorboard"
        cargo build --release -p nanochat-train --features cuda,tensorboard

        note "GPU kernel tests"
        cargo test -p ternary-kernels --features cuda test_gpu_gemv_matches_cpu_scalar -- --nocapture
        cargo test -p ternary-kernels --features cuda test_gpu_shape_torture -- --nocapture

        if [[ "${RUN_VALIDATE_E2E}" == "1" ]]; then
            note "running scripts/validate_e2e.sh"
            ./scripts/validate_e2e.sh
        else
            note "RUN_VALIDATE_E2E=0, skipping scripts/validate_e2e.sh"
        fi
    } 2>&1 | tee "${LOG_DIR}/lane_b_gpu_validation.log"

    section "Lane C: Supervised Fine-tune Smoke (GPU)"
    GPU_SFT_OUT="${GPU_DIR}/sft"
    mkdir -p "${GPU_SFT_OUT}/checkpoints"
    {
        note "training GPU checkpoint (${GPU_SFT_CONFIG})"
        cargo run --release -p nanochat-train --features cuda -- train \
            --config "${GPU_SFT_CONFIG}" \
            --dataset synthetic \
            --n-samples "${GPU_SFT_SAMPLES}" \
            --epochs "${GPU_SFT_EPOCHS}" \
            --device cuda \
            --checkpoint-dir "${GPU_SFT_OUT}/checkpoints" \
            --log-interval "${GPU_SFT_LOG_INTERVAL}" \
            --checkpoint-interval "${GPU_SFT_CHECKPOINT_INTERVAL}"
    } 2>&1 | tee "${LOG_DIR}/lane_c_gpu_sft.log"

    GPU_CKPT="$(latest_step_checkpoint "${GPU_SFT_OUT}/checkpoints" || true)"
    [[ -n "${GPU_CKPT}" ]] || fail "no GPU step checkpoint found in ${GPU_SFT_OUT}/checkpoints"
    assert_checkpoint_dir "${GPU_CKPT}"

    {
        note "exporting GPU checkpoint"
        cargo run --release -p nanochat-train -- export \
            --checkpoint "${GPU_CKPT}" \
            --gguf "${GPU_SFT_OUT}/model.gguf" \
            --mhc "${GPU_SFT_OUT}/model.mhc"
    } 2>&1 | tee "${LOG_DIR}/lane_c_gpu_export.log"

    require_file "${GPU_SFT_OUT}/model.gguf"
    require_file "${GPU_SFT_OUT}/model.mhc"

    SERVE_GGUF="${GPU_SFT_OUT}/model.gguf"
    SERVE_MHC="${GPU_SFT_OUT}/model.mhc"
    BASELINE_CKPT="${GPU_CKPT}"
fi

section "Lane D: Serve/API E2E"
SERVE_PID=""
SERVE_PID_FILE="${ARTIFACT_ROOT}/serve.pid"
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
    note "starting server on port ${SERVE_PORT}"
    cargo run --release -p nanochat-serve -- \
        --model "${SERVE_GGUF}" \
        --mhc "${SERVE_MHC}" \
        --tokenizer "${WORKSPACE_DIR}/models/gpt2-tokenizer.json" \
        --port "${SERVE_PORT}" \
        > "${LOG_DIR}/serve_runtime.log" 2>&1 &
    SERVE_PID=$!
    echo "${SERVE_PID}" > "${SERVE_PID_FILE}"

    if ! wait_for_http "http://127.0.0.1:${SERVE_PORT}/health" 90 2; then
        fail "server failed to become healthy on port ${SERVE_PORT}"
    fi

    note "querying /health"
    curl -fsS "http://127.0.0.1:${SERVE_PORT}/health" | tee "${ARTIFACT_ROOT}/serve_health.txt"

    note "querying /v1/models"
    curl -fsS "http://127.0.0.1:${SERVE_PORT}/v1/models" \
        | tee "${ARTIFACT_ROOT}/serve_models.json"

    note "querying non-stream completion"
    curl -fsS "http://127.0.0.1:${SERVE_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Write a Rust add function"}],"max_tokens":64}' \
        | tee "${ARTIFACT_ROOT}/serve_completion.json"

    note "querying stream completion"
    curl -fsS -N --max-time 30 "http://127.0.0.1:${SERVE_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Write a Rust clamp function"}],"max_tokens":32,"stream":true}' \
        | tee "${ARTIFACT_ROOT}/serve_stream.txt" >/dev/null

    cleanup
} 2>&1 | tee "${LOG_DIR}/lane_d_serve_api.log"

if [[ "${SKIP_RL}" == "1" ]]; then
    note "SKIP_RL=1, skipping RL and benchmark stages"
else
    section "Lane D: GRPO RL Fine-tune"
    RL_RUN_DIR="${RL_DIR}/run"
    mkdir -p "${RL_RUN_DIR}"

    {
        note "starting RL training (GRPO)"
        pushd "${RL_RUN_DIR}" >/dev/null
        NANOCHAT_TOKENIZER="${WORKSPACE_DIR}/models/gpt2-tokenizer.json" \
            cargo run --release \
            --manifest-path "${WORKSPACE_DIR}/Cargo.toml" \
            -p nanochat-rl --example train_rl -- \
            --checkpoint "${BASELINE_CKPT}" \
            --iterations "${RL_ITERATIONS}" \
            --n-samples "${RL_SAMPLES}" \
            --batch-size "${RL_BATCH_SIZE}" \
            --device "${RL_DEVICE}" \
            --lr "${RL_LR}" \
            --kl-coef "${RL_KL_COEF}"
        popd >/dev/null
    } 2>&1 | tee "${LOG_DIR}/lane_d_rl_grpo.log"

    RL_FINAL_CKPT="${RL_RUN_DIR}/checkpoints/rl-final"
    assert_checkpoint_dir "${RL_FINAL_CKPT}"

    if [[ "${RUN_BENCHMARKS}" == "1" ]]; then
        section "Lane D: Benchmark Baseline vs RL"
        {
            note "benchmark baseline checkpoint"
            cargo run --release -p nanochat-eval --example benchmark_model -- \
                --checkpoint "${BASELINE_CKPT}" \
                --n-samples "${BENCHMARK_SAMPLES}" \
                --temperature "${BENCHMARK_TEMPERATURE}" \
                --max-tokens "${BENCHMARK_MAX_TOKENS}" \
                --output "${EVAL_DIR}/baseline.json" \
                --device "${RL_DEVICE}"

            note "benchmark RL checkpoint"
            cargo run --release -p nanochat-eval --example benchmark_model -- \
                --checkpoint "${RL_FINAL_CKPT}" \
                --n-samples "${BENCHMARK_SAMPLES}" \
                --temperature "${BENCHMARK_TEMPERATURE}" \
                --max-tokens "${BENCHMARK_MAX_TOKENS}" \
                --output "${EVAL_DIR}/rl.json" \
                --device "${RL_DEVICE}"
        } 2>&1 | tee "${LOG_DIR}/lane_d_benchmark.log"

        if command -v jq >/dev/null 2>&1; then
            BASE_COMPILE="$(jq -r '.compile_success_rate' "${EVAL_DIR}/baseline.json")"
            RL_COMPILE="$(jq -r '.compile_success_rate' "${EVAL_DIR}/rl.json")"
            note "compile_success_rate baseline=${BASE_COMPILE} rl=${RL_COMPILE}"
        fi
    else
        note "RUN_BENCHMARKS=0, skipping benchmark stage"
    fi
fi

section "Complete"
cat <<EOF
Pipeline finished successfully.

Artifacts:
  ${ARTIFACT_ROOT}

Key outputs:
  CPU checkpoint:  ${CPU_CKPT}
  Base checkpoint: ${BASELINE_CKPT}
  Serve model:     ${SERVE_GGUF}
  Serve mhc:       ${SERVE_MHC}
EOF

if [[ "${SKIP_RL}" != "1" ]]; then
    cat <<EOF
  RL final:        ${RL_DIR}/run/checkpoints/rl-final
EOF
    if [[ "${RUN_BENCHMARKS}" == "1" ]]; then
        cat <<EOF
  Benchmark base:  ${EVAL_DIR}/baseline.json
  Benchmark rl:    ${EVAL_DIR}/rl.json
EOF
    fi
fi
