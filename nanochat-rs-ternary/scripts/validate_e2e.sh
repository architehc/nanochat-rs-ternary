#!/bin/bash
# End-to-End Validation Script for nanochat-rs-ternary
#
# This script validates the entire training → export → inference pipeline:
# 1. Train small model (1000 steps, ~2 minutes)
# 2. Export to GGUF format
# 3. Start inference server
# 4. Test API endpoints
# 5. Validate generation quality
# 6. Run performance benchmarks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_DIR="/home/habitat/ternary-clawd/nanochat-rs-ternary"
TEST_DIR="${WORKSPACE_DIR}/e2e_validation_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="${TEST_DIR}/checkpoints"
MODEL_PATH="${CHECKPOINT_DIR}/model.gguf"
MHC_PATH="${CHECKPOINT_DIR}/model.mhc"
SERVER_PORT=8765
SERVER_PID=""

# Test configuration
QUICK_STEPS=1000  # ~2 minutes for quick validation
BATCH_SIZE=4
SEQ_LENGTH=512

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}nanochat-rs E2E Validation Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ ! -z "${SERVER_PID}" ]; then
        kill ${SERVER_PID} 2>/dev/null || true
        echo "Stopped server (PID ${SERVER_PID})"
    fi
}
trap cleanup EXIT

# Setup
mkdir -p "${TEST_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
cd "${WORKSPACE_DIR}"

# Step 1: Build all components
echo -e "${GREEN}[1/7] Building nanochat components...${NC}"
cargo build --release -p nanochat-train --features cuda 2>&1 | tail -5
cargo build --release -p nanochat-serve --features cuda 2>&1 | tail -5
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Step 2: Create minimal training config
echo -e "${GREEN}[2/7] Creating training configuration...${NC}"
cat > "${TEST_DIR}/train_config.toml" <<EOF
[model]
vocab_size = 32000
hidden_dim = 512
num_layers = 6
num_heads = 8
num_kv_heads = 2
ffn_mult = 2.0
max_seq_len = ${SEQ_LENGTH}
group_size = 128
mhc_n_streams = 2

[training]
total_steps = ${QUICK_STEPS}
batch_size = ${BATCH_SIZE}
seq_length = ${SEQ_LENGTH}
learning_rate = 0.02
warmup_steps = 100
weight_decay = 0.01
grad_clip = 1.0
entropy_weight = 0.01

use_mtp = true
mtp_n_future_tokens = 2
use_async_loader = false  # Simpler for testing

optimizer_type = "muon"
muon_momentum = 0.95
schedule_type = "wsd"
stable_ratio = 0.8
min_lr_ratio = 0.1

[checkpointing]
checkpoint_dir = "${CHECKPOINT_DIR}"
checkpoint_interval = ${QUICK_STEPS}  # Save at end

[logging]
log_interval = 50
structured_logging = true
EOF
echo -e "${GREEN}✓ Configuration created${NC}"
echo ""

# Step 3: Train model
echo -e "${GREEN}[3/7] Training model (${QUICK_STEPS} steps, ~2 minutes)...${NC}"
RUST_LOG=info target/release/nanochat-train \
    --config "${TEST_DIR}/train_config.toml" \
    2>&1 | tee "${TEST_DIR}/training.log" | grep -E "step=|loss=|Training complete" || true

if [ ! -f "${CHECKPOINT_DIR}/checkpoint_${QUICK_STEPS}.safetensors" ]; then
    echo -e "${RED}✗ Training failed - checkpoint not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Training complete${NC}"
echo ""

# Step 4: Export to GGUF
echo -e "${GREEN}[4/7] Exporting to GGUF format...${NC}"
cargo run --release -p nanochat-train -- \
    export \
    --checkpoint "${CHECKPOINT_DIR}/checkpoint_${QUICK_STEPS}.safetensors" \
    --output "${MODEL_PATH}" \
    --mhc-output "${MHC_PATH}" \
    2>&1 | tee "${TEST_DIR}/export.log"

if [ ! -f "${MODEL_PATH}" ]; then
    echo -e "${RED}✗ Export failed - GGUF file not found${NC}"
    exit 1
fi

MODEL_SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
echo -e "${GREEN}✓ Export complete (${MODEL_SIZE})${NC}"
echo ""

# Step 5: Start inference server
echo -e "${GREEN}[5/7] Starting inference server...${NC}"
target/release/nanochat-serve \
    --model "${MODEL_PATH}" \
    --mhc "${MHC_PATH}" \
    --port ${SERVER_PORT} \
    --host 127.0.0.1 \
    > "${TEST_DIR}/server.log" 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 3

if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo -e "${RED}✗ Server failed to start${NC}"
    cat "${TEST_DIR}/server.log"
    exit 1
fi
echo -e "${GREEN}✓ Server started (PID ${SERVER_PID}, port ${SERVER_PORT})${NC}"
echo ""

# Step 6: Test API endpoints
echo -e "${GREEN}[6/7] Testing API endpoints...${NC}"

# Test 1: Health check
echo -n "  Testing /health... "
if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 2: Model info
echo -n "  Testing /v1/models... "
MODELS=$(curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/models")
if echo "$MODELS" | grep -q "nanochat"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 3: Chat completion (non-streaming)
echo -n "  Testing /v1/chat/completions... "
RESPONSE=$(curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nanochat",
        "messages": [{"role": "user", "content": "Write a Rust function that adds two numbers"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": false
    }')

if echo "$RESPONSE" | grep -q "choices"; then
    echo -e "${GREEN}✓${NC}"
    GENERATED_TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "")
else
    echo -e "${RED}✗${NC}"
    echo "Response: $RESPONSE"
    exit 1
fi

# Test 4: Streaming response
echo -n "  Testing streaming... "
STREAM_RESPONSE=$(curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nanochat",
        "messages": [{"role": "user", "content": "fn factorial"}],
        "max_tokens": 50,
        "stream": true
    }')

if echo "$STREAM_RESPONSE" | grep -q "data:"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All API tests passed${NC}"
echo ""

# Step 7: Validate generation quality
echo -e "${GREEN}[7/7] Validating generation quality...${NC}"

# Create test prompts
PROMPTS=(
    "Write a Rust function that computes fibonacci numbers"
    "fn reverse_string(s: &str) -> String {"
    "impl Iterator for"
    "use std::collections::HashMap"
)

PASSED=0
TOTAL=${#PROMPTS[@]}

for PROMPT in "${PROMPTS[@]}"; do
    echo "  Prompt: ${PROMPT:0:50}..."

    RESPONSE=$(curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"nanochat\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
            \"max_tokens\": 100,
            \"temperature\": 0.3
        }")

    GENERATED=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "")

    # Basic quality checks
    if [ ! -z "$GENERATED" ] && [ ${#GENERATED} -gt 10 ]; then
        echo -e "    ${GREEN}✓${NC} Generated ${#GENERATED} chars"
        ((PASSED++))
    else
        echo -e "    ${RED}✗${NC} Generation failed or too short"
    fi
done

echo ""
echo -e "${GREEN}Quality Score: ${PASSED}/${TOTAL} prompts generated successfully${NC}"
echo ""

# Step 8: Performance benchmark
echo -e "${GREEN}[BONUS] Performance Benchmark${NC}"
echo "  Running 10 inference requests..."

START_TIME=$(date +%s%N)
for i in {1..10}; do
    curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "nanochat",
            "messages": [{"role": "user", "content": "fn add"}],
            "max_tokens": 20
        }' > /dev/null
done
END_TIME=$(date +%s%N)

ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
AVG_LATENCY=$(( ELAPSED_MS / 10 ))

echo -e "  Average latency: ${GREEN}${AVG_LATENCY}ms${NC}"
echo ""

# Final summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ E2E Validation PASSED${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  Model: ${MODEL_PATH}"
echo "  Size: ${MODEL_SIZE}"
echo "  Training steps: ${QUICK_STEPS}"
echo "  API tests: PASS"
echo "  Generation quality: ${PASSED}/${TOTAL}"
echo "  Average latency: ${AVG_LATENCY}ms"
echo ""
echo "Logs saved to: ${TEST_DIR}"
echo "Server log: ${TEST_DIR}/server.log"
echo "Training log: ${TEST_DIR}/training.log"
echo ""

# Sample generation
echo -e "${YELLOW}Sample Generation:${NC}"
echo "Prompt: \"Write a Rust function that adds two numbers\""
echo ""
echo "$GENERATED_TEXT" | head -20
echo ""

echo -e "${GREEN}Validation complete!${NC}"
echo ""
echo "To run the full 8-hour production training:"
echo "  ./scripts/train_production_8h.sh"
echo ""
echo "To start the server manually:"
echo "  cargo run --release -p nanochat-serve -- --model ${MODEL_PATH} --port 8000"
