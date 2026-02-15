#!/bin/bash
# Quick E3 Benchmark using predefined configs
# Compares baseline vs E3-full
# Runtime: ~10 minutes (500 steps each)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

WORKSPACE_DIR="/home/habitat/ternary-clawd/nanochat-rs-ternary"
RESULTS_DIR="${WORKSPACE_DIR}/benchmark_results_quick_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}E3 Quick Benchmark${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

mkdir -p "${RESULTS_DIR}"
cd "${WORKSPACE_DIR}"

# Build
echo -e "${BLUE}Building...${NC}"
cargo build --release -p nanochat-train --features cuda 2>&1 | grep -E "Compiling|Finished" | tail -3
echo ""

# Configs to test
declare -A configs
configs["Baseline"]="d20"
configs["MTP_Only"]="d20-mtp"

# Run benchmarks
for name in "${!configs[@]}"; do
    config="${configs[$name]}"

    echo -e "${GREEN}Benchmarking: ${name} (${config})${NC}"

    output_dir="${RESULTS_DIR}/${name}"
    mkdir -p "${output_dir}"

    # Determine batch size based on config (E3-full is larger, needs smaller batch)
    batch_size=4
    if [[ "${config}" == *"e3"* ]]; then
        batch_size=1  # Large model needs smaller batch to fit in GPU
    fi

    # Run 500 steps (~2-3 minutes on GPU)
    RUST_LOG=info timeout 900 target/release/nanochat-train train \
        --config "${config}" \
        --dataset synthetic \
        --n-samples 10000 \
        --batch-size ${batch_size} \
        --epochs 1 \
        --checkpoint-interval 10000 \
        --log-interval 50 \
        --device cuda \
        --checkpoint-dir "${output_dir}/checkpoints" \
        > "${output_dir}/training.log" 2>&1 || {
            echo -e "${YELLOW}⚠ ${name} timed out or failed${NC}"
        }

    echo -e "${GREEN}✓ ${name} complete${NC}"
    echo ""
done

# Generate report
echo -e "${BLUE}Generating comparison...${NC}"
echo ""

cat > "${RESULTS_DIR}/REPORT.md" <<EOF
# E3 Quick Benchmark Results

**Date**: $(date)
**Configs**: Baseline (d20) vs E3 Full (d20-e3-full)

---

## Results

EOF

for name in "${!configs[@]}"; do
    config="${configs[$name]}"
    log="${RESULTS_DIR}/${name}/training.log"

    if [ ! -f "${log}" ]; then
        echo "### ${name} (${config})" >> "${RESULTS_DIR}/REPORT.md"
        echo "❌ Failed to run" >> "${RESULTS_DIR}/REPORT.md"
        echo "" >> "${RESULTS_DIR}/REPORT.md"
        continue
    fi

    # Extract metrics
    final_loss=$(grep -oP 'loss=\K[0-9.]+' "${log}" | tail -1 || echo "N/A")
    steps=$(grep -c "step=" "${log}" || echo "0")

    # Check if MTP/Collider were enabled
    mtp_status="❌"
    collider_status="❌"
    grep -q "MTP enabled" "${log}" && mtp_status="✅"
    grep -q "Collider enabled" "${log}" && collider_status="✅"

    cat >> "${RESULTS_DIR}/REPORT.md" <<RESULT
### ${name} (${config})

- **Steps**: ${steps}
- **Final Loss**: ${final_loss}
- **MTP**: ${mtp_status}
- **Collider**: ${collider_status}

RESULT

    echo "\`\`\`" >> "${RESULTS_DIR}/REPORT.md"
    tail -10 "${log}" | grep -E "step=|loss=" || echo "No training logs found"
    echo "\`\`\`" >> "${RESULTS_DIR}/REPORT.md"
    echo "" >> "${RESULTS_DIR}/REPORT.md"
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results: ${RESULTS_DIR}/REPORT.md"
echo ""
cat "${RESULTS_DIR}/REPORT.md"
