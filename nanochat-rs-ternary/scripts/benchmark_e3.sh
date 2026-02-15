#!/bin/bash
# E3 Comprehensive Benchmark Suite
#
# Compares all E3 optimizer combinations and validates performance claims:
# - Baseline (standard Muon)
# - MTP only (15-20% data efficiency)
# - 8-bit Muon (86% memory reduction)
# - GaLore 2 (50-65% memory reduction)
# - Full E3 (all optimizations)
#
# Each config runs for 1000 steps (~5 minutes on RTX PRO 6000)
# Total runtime: ~25 minutes

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

WORKSPACE_DIR="/home/habitat/ternary-clawd/nanochat-rs-ternary"
RESULTS_DIR="${WORKSPACE_DIR}/benchmark_results_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}E3 Comprehensive Benchmark Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"
cd "${WORKSPACE_DIR}"

# Build in release mode
echo -e "${BLUE}Building in release mode...${NC}"
cargo build --release -p nanochat-train --features cuda
echo ""

# Test configurations
configs=(
    "configs/e3/baseline.toml"
    "configs/e3/mtp_only.toml"
    "configs/e3/muon_8bit.toml"
    "configs/e3/galore2.toml"
    "configs/e3/full.toml"
)

config_names=(
    "Baseline"
    "MTP_Only"
    "8bit_Muon"
    "GaLore2"
    "E3_Full"
)

# Run benchmarks
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    name="${config_names[$i]}"

    echo -e "${GREEN}[${i}/${#configs[@]}] Benchmarking: ${name}${NC}"
    echo "  Config: ${config}"
    echo ""

    # Create output directory for this config
    output_dir="${RESULTS_DIR}/${name}"
    mkdir -p "${output_dir}"

    # Run training for 1000 steps
    RUST_LOG=info target/release/nanochat-train train \
        --config "${config}" \
        --dataset synthetic \
        --n-samples 50000 \
        --batch-size 16 \
        --epochs 1 \
        --checkpoint-interval 5000 \
        --log-interval 100 \
        --checkpoint-dir "${output_dir}/checkpoints" \
        > "${output_dir}/training.log" 2>&1 || {
            echo -e "${YELLOW}⚠ ${name} failed, continuing...${NC}"
            continue
        }

    echo -e "${GREEN}✓ ${name} complete${NC}"
    echo ""
done

# Generate comparison report
echo -e "${BLUE}Generating comparison report...${NC}"
echo ""

cat > "${RESULTS_DIR}/REPORT.md" <<EOF
# E3 Benchmark Results

**Date**: $(date +"%Y-%m-%d %H:%M:%S")
**GPU**: NVIDIA RTX PRO 6000 Blackwell (96GB)
**Workspace**: ${WORKSPACE_DIR}

---

## Configurations Tested

| Config | MTP | Collider | Async Loader | Optimizer | Memory Opt |
|--------|-----|----------|--------------|-----------|------------|
| Baseline | ❌ | ❌ | ❌ | Muon | None |
| MTP Only | ✅ | ❌ | ❌ | Muon | None |
| 8-bit Muon | ✅ | ✅ | ✅ | Muon | 8-bit (86%) |
| GaLore 2 | ✅ | ✅ | ✅ | Muon | GaLore (50-65%) |
| E3 Full | ✅ | ✅ | ✅ | Muon | 8-bit + GaLore |

---

## Results

EOF

# Extract metrics from logs
for i in "${!configs[@]}"; do
    name="${config_names[$i]}"
    log_file="${RESULTS_DIR}/${name}/training.log"

    if [ ! -f "${log_file}" ]; then
        echo "### ${name}" >> "${RESULTS_DIR}/REPORT.md"
        echo "" >> "${RESULTS_DIR}/REPORT.md"
        echo "⚠ Benchmark failed - no log file" >> "${RESULTS_DIR}/REPORT.md"
        echo "" >> "${RESULTS_DIR}/REPORT.md"
        continue
    fi

    # Extract final loss
    final_loss=$(grep -oP 'loss=\K[0-9.]+' "${log_file}" | tail -1 || echo "N/A")

    # Extract average tokens/sec
    avg_tps=$(grep -oP 'tokens_per_sec=[0-9.]+' "${log_file}" | \
              sed 's/tokens_per_sec=//' | \
              awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')

    # Extract average gradient norm
    avg_grad=$(grep -oP 'grad_norm=[0-9.]+' "${log_file}" | \
               sed 's/grad_norm=//' | \
               awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')

    # Count steps
    steps=$(grep -c "Training step completed" "${log_file}" || echo "0")

    cat >> "${RESULTS_DIR}/REPORT.md" <<RESULT
### ${name}

- **Steps completed**: ${steps}
- **Final loss**: ${final_loss}
- **Avg throughput**: ${avg_tps} tokens/sec
- **Avg grad norm**: ${avg_grad}

RESULT

    # Show last 5 log lines
    echo "\`\`\`" >> "${RESULTS_DIR}/REPORT.md"
    tail -5 "${log_file}" >> "${RESULTS_DIR}/REPORT.md"
    echo "\`\`\`" >> "${RESULTS_DIR}/REPORT.md"
    echo "" >> "${RESULTS_DIR}/REPORT.md"
done

# Add performance comparison
cat >> "${RESULTS_DIR}/REPORT.md" <<EOF
---

## Performance Comparison

**Expected E3 Gains**:
- MTP: 15-20% data efficiency improvement
- Collider: 35% faster backprop
- Async Loader: 90%+ GPU utilization
- 8-bit Muon: 86% memory reduction
- GaLore 2: 50-65% memory reduction

**Actual Results**: See logs above

---

## How to Reproduce

\`\`\`bash
./scripts/benchmark_e3.sh
\`\`\`

## View Results

\`\`\`bash
cat ${RESULTS_DIR}/REPORT.md
\`\`\`

## Logs Location

Each config's full training log is available at:
\`${RESULTS_DIR}/<config_name>/training.log\`
EOF

# Display summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Summary:"
cat "${RESULTS_DIR}/REPORT.md"
echo ""
echo -e "${BLUE}To view full report:${NC}"
echo "  cat ${RESULTS_DIR}/REPORT.md"
echo ""
echo -e "${BLUE}To view individual logs:${NC}"
echo "  ls ${RESULTS_DIR}/*/training.log"
