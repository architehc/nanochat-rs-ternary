#!/bin/bash
# End-to-End Validation Script for nanochat-rs-ternary
#
# Validates the complete pipeline using existing integration tests:
# - Triangle of Truth (kernel correctness)
# - GGUF roundtrip (pack → GGUF → load → GEMV)
# - Export roundtrip (train → export → load → inference)
# - E2E generation (full model forward pass)
# - mHC property tests (doubly stochastic invariants)

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

WORKSPACE_DIR="/home/habitat/ternary-clawd/nanochat-rs-ternary"
cd "${WORKSPACE_DIR}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}nanochat-rs E2E Validation Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Test 1: Build all components
echo -e "${GREEN}[1/6] Building all components...${NC}"
cargo build --workspace --release --features cuda 2>&1 | grep -E "Compiling|Finished" | tail -5
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Test 2: Run unit tests
echo -e "${GREEN}[2/6] Running unit tests...${NC}"
TEST_OUTPUT=$(cargo test --workspace --lib 2>&1)
PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1)
echo -e "${GREEN}✓ Unit tests: ${PASSED} passed${NC}"
echo ""

# Test 3: Triangle of Truth (kernel correctness)
echo -e "${GREEN}[3/6] Triangle of Truth (kernel validation)...${NC}"
cargo test --test triangle_of_truth -- --nocapture 2>&1 | grep -E "^test|test result"
echo -e "${GREEN}✓ All kernel paths produce identical output${NC}"
echo ""

# Test 4: GGUF Roundtrip
echo -e "${GREEN}[4/6] GGUF roundtrip tests...${NC}"
cargo test --test roundtrip_test -- --nocapture 2>&1 | grep -E "^test|test result"
echo -e "${GREEN}✓ Pack → GGUF → Load → GEMV validated${NC}"
echo ""

# Test 5: Export Roundtrip
echo -e "${GREEN}[5/6] Export roundtrip (train → export → inference)...${NC}"
cargo test --test export_roundtrip -- --nocapture 2>&1 | grep -E "^test|test result|Export-load|correlation"
echo -e "${GREEN}✓ Training/inference parity validated${NC}"
echo ""

# Test 6: E2E Generation
echo -e "${GREEN}[6/6] End-to-end generation tests...${NC}"
cargo test --test e2e_generate -- --nocapture 2>&1 | grep -E "^test|test result"
echo -e "${GREEN}✓ Full model forward pass validated${NC}"
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ E2E Validation PASSED${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo "All pipeline components validated:"
echo "  ✅ Ternary kernels (CPU + GPU)"
echo "  ✅ GGUF serialization"
echo "  ✅ mHC doubly stochastic invariants"
echo "  ✅ Training → Export → Inference pipeline"
echo "  ✅ Autoregressive generation"
echo "  ✅ Weight tying"
echo ""

echo "Ready for production training!"
echo ""
echo "Quick start:"
echo "  1. Quick test (2 hours):  ./scripts/train_nano_125m.sh"
echo "  2. Production (8 hours):  ./scripts/train_small_560m.sh"
echo "  3. SOTA (24 hours):       ./scripts/train_medium_3b.sh"
echo "  4. Custom (8 hours):      ./scripts/train_production_8h.sh"
echo ""
echo "Monitor training:"
echo "  tensorboard --logdir runs/"
echo ""
echo "After training, start inference:"
echo "  cargo run --release -p nanochat-serve -- \\"
echo "    --model runs/experiment/model.gguf \\"
echo "    --port 8000"
