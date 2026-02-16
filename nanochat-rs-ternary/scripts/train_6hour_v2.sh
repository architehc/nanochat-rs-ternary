#!/bin/bash
# Simplified 6-hour continuous training

set -eo pipefail

CHECKPOINT_DIR="${1:-checkpoints/rust-6hour}"
BATCH_SIZE=2
DATA_PATH="data/rust_tokens.bin"

echo "═══════════════════════════════════════════════════════════"
echo "  6-Hour Continuous Training (Simplified)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Start time: $(date)"
echo "End time:   $(date -d '+6 hours')"
echo ""

mkdir -p "$CHECKPOINT_DIR"

# Use timeout to enforce 6 hour limit
timeout 6h bash -c '
set -o pipefail
CHECKPOINT_DIR="'"$CHECKPOINT_DIR"'"
RUN=0
while true; do
    RUN=$((RUN + 1))
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "Training Run #$RUN - $(date +%H:%M:%S)"
    echo "═══════════════════════════════════════════════════════════"

    cd /home/habitat/ternary-clawd/nanochat-rs-ternary

    # Train until OOM
    cargo run --release --example train_rust_maxgpu --features nanochat-train/cuda -- \
        --data data/rust_tokens.bin \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --device cuda:0 \
        --batch-size 2 \
        --total-steps 100000 \
        --checkpoint-interval 500 \
        --log-interval 100 \
        2>&1 | tee -a "$CHECKPOINT_DIR/training.log"

    EXIT=$?
    echo "Exit code: $EXIT"

    # Brief pause for GPU cleanup
    sleep 3
done
' || echo "6-hour time limit reached"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Training Complete - $(date)"
echo "═══════════════════════════════════════════════════════════"

# Show final stats
LAST_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null | sort -V | tail -1)
if [ -n "$LAST_CHECKPOINT" ]; then
    FINAL_STEP=$(basename "$LAST_CHECKPOINT" | sed 's/step_//')
    echo "Final checkpoint: step_$FINAL_STEP"
    echo "Total checkpoints: $(ls -d "$CHECKPOINT_DIR"/step_* | wc -l)"
    echo "Storage used: $(du -sh "$CHECKPOINT_DIR" | cut -f1)"
fi
