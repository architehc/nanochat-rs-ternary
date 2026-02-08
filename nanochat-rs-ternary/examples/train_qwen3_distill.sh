#!/usr/bin/env bash
#
# Example training script for Qwen3-Coder-80B hybrid ternary distillation
#
# This script demonstrates recommended settings for training a hybrid ternary
# student model using a remote FP8 teacher endpoint.
#
# Hardware assumptions:
# - Dual AMD EPYC 9654 (224 threads, 1TB DDR5)
# - NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
# - 3.5TB SSD storage (1.9TB available)
#
# Memory usage:
# - Student model: ~5GB VRAM (3B active params)
# - Activations + gradients: ~3GB VRAM (batch_size=4, seq_len=2048)
# - KV cache: ~1GB VRAM
# - Total: ~10GB VRAM (well within 96GB capacity)
#
# Disk usage (with keep_last_checkpoints=3):
# - Each checkpoint: ~160GB (full model state)
# - Max checkpoint space: 480GB (last 3)
# - Training logs: ~1GB
# - Total: ~500GB (fits in 1.9TB available)
#
# Training time estimate:
# - 100k steps @ 4 samples/step = 400k samples
# - ~2s/step (includes remote teacher query)
# - Total: ~55 hours (~2.3 days)

set -euo pipefail

# Configuration
TEACHER_ENDPOINT="${TEACHER_ENDPOINT:-https://crazyshit.ngrok.io}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3-80b-hybrid}"
DEVICE="${DEVICE:-cuda:0}"
PARALLEL="${PARALLEL:-true}"
MICRO_BATCHES="${MICRO_BATCHES:-8}"

# Training hyperparameters (tuned for Qwen3-80B)
TOTAL_STEPS=100000
WARMUP_STEPS=2000
BATCH_SIZE=4
SEQ_LEN=2048
LR=0.02           # Muon for ternary weights
MHC_LR=0.0001     # Lion for router/norms/mHC

# Distillation hyperparameters
TEMPERATURE=2.0    # Higher = smoother teacher distribution
KL_WEIGHT=0.5      # 50% KL, 50% CE (balanced)

# Logging and checkpointing
LOG_INTERVAL=100
CHECKPOINT_INTERVAL=1000
KEEP_LAST_CHECKPOINTS=3

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Log configuration
cat > "$CHECKPOINT_DIR/config.txt" <<EOF
Training Configuration
=====================

Model: Qwen3-Coder-80B Hybrid Ternary
- Total parameters: 80B
- Active parameters: ~3B
- Architecture: 48 layers (12×[3 DeltaNet + 1 Gated Attn])
- Experts: 512 (top-10 routing + 1 shared)
- Vocab: 151,936 tokens

Teacher:
- Mode: Remote FP8 endpoint
- Endpoint: $TEACHER_ENDPOINT
- Timeout: 60s

Student:
- Ternary: MoE expert weights (w_gate, w_up, w_down)
- FP8/BF16: Router, gates, norms, embeddings, lm_head, DeltaNet state

Training:
- Total steps: $TOTAL_STEPS
- Warmup steps: $WARMUP_STEPS
- Batch size: $BATCH_SIZE
- Sequence length: $SEQ_LEN
- Tokens per step: $((BATCH_SIZE * SEQ_LEN))

Optimizer:
- Muon LR: $LR (ternary weights)
- Lion LR: $MHC_LR (router, norms, mHC)
- Momentum: 0.95 (Muon)
- Betas: (0.9, 0.99) (Lion)
- Weight decay: 0.1
- Grad clip: 1.0

Distillation:
- Temperature: $TEMPERATURE
- KL weight: $KL_WEIGHT (CE weight: $(echo "1 - $KL_WEIGHT" | bc))
- Load balance: 0.01
- Router aux: 0.001

Schedule:
- Warmup: 0 → $WARMUP_STEPS (linear)
- Stable: $WARMUP_STEPS → $((TOTAL_STEPS * 80 / 100)) (constant LR)
- Decay: $((TOTAL_STEPS * 80 / 100)) → $TOTAL_STEPS (cosine to 0.1×LR)

Checkpointing:
- Directory: $CHECKPOINT_DIR
- Interval: $CHECKPOINT_INTERVAL steps
- Keep last: $KEEP_LAST_CHECKPOINTS checkpoints
- Auto-cleanup: Yes

Parallelization:
- Enabled: $PARALLEL
- Micro-batches: $MICRO_BATCHES
- Concurrent requests: $MICRO_BATCHES
- Expected speedup: ~$((MICRO_BATCHES < 8 ? MICRO_BATCHES : 8))x / 2 = ~$((MICRO_BATCHES / 2))x

Device: $DEVICE
Started: $(date)
EOF

echo "═══════════════════════════════════════════════════════════"
echo "  Qwen3-Coder-80B Hybrid Ternary Distillation"
echo "═══════════════════════════════════════════════════════════"
echo
cat "$CHECKPOINT_DIR/config.txt"
echo
echo "═══════════════════════════════════════════════════════════"
echo

# Check disk space
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Disk space available: ${AVAILABLE_GB}GB"
if [ "$AVAILABLE_GB" -lt 500 ]; then
    echo "WARNING: Less than 500GB available. Training may fail."
    echo "Expected usage: ~500GB (480GB checkpoints + logs)"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo

# Check if teacher endpoint is reachable
echo "Testing teacher endpoint connectivity..."
if curl -s --max-time 5 "$TEACHER_ENDPOINT" > /dev/null 2>&1; then
    echo "✓ Teacher endpoint reachable: $TEACHER_ENDPOINT"
else
    echo "✗ Cannot reach teacher endpoint: $TEACHER_ENDPOINT"
    echo "  Make sure the FP8 inference server is running."
    exit 1
fi
echo

# Run training
echo "Starting training..."
echo "  Logs: $CHECKPOINT_DIR/training.log"
echo "  Press Ctrl+C to stop (checkpoints will be saved)"
echo

cargo run --release --example distill_qwen3 -- \
    --teacher-endpoint "$TEACHER_ENDPOINT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS" \
    --total-steps "$TOTAL_STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    --log-interval "$LOG_INTERVAL" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --lr "$LR" \
    --mhc-lr "$MHC_LR" \
    --kl-weight "$KL_WEIGHT" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE" \
    --parallel "$PARALLEL" \
    --micro-batches "$MICRO_BATCHES" \
    2>&1 | tee "$CHECKPOINT_DIR/training.log"

echo
echo "═══════════════════════════════════════════════════════════"
echo "Training complete!"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Log: $CHECKPOINT_DIR/training.log"
echo "═══════════════════════════════════════════════════════════"
