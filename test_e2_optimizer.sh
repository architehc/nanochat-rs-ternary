#!/bin/bash
# Test E2 optimizer integration with a tiny training run

set -eo pipefail

echo "=== Testing E2 Optimizer Integration ==="
echo ""

# Create tiny test dataset if it doesn't exist
# Token file must contain u32 values in [0, vocab_size) — random bytes would
# produce out-of-range token IDs (up to 4B vs vocab_size=50257) and crash.
if [ ! -f "nanochat-rs-ternary/data/test_tokens.bin" ]; then
    echo "Creating tiny test dataset..."
    mkdir -p nanochat-rs-ternary/data
    python3 -c "
import struct, random; random.seed(42)
with open('nanochat-rs-ternary/data/test_tokens.bin','wb') as f:
    f.write(struct.pack('<' + 'I'*131072, *[random.randint(0,50256) for _ in range(131072)]))
"
fi

echo ""
echo "Test 1: Baseline (Standard Muon)"
echo "================================="
./nanochat-rs-ternary/target/release/nanochat-train train \
    --config d20 \
    --data-path nanochat-rs-ternary/data/test_tokens.bin \
    --dataset tokens \
    --epochs 1 \
    --batch-size 2 \
    --seq-len 64 \
    --checkpoint-dir /tmp/test_baseline \
    --log-interval 5 \
    --device cpu \
    --threads 16 \
    2>&1 | tail -50

echo ""
echo "Test 2: 8-bit Quantized Muon"
echo "============================"
# Create temp config
cat > /tmp/test_8bit.toml << 'EOF'
dim = 256
n_layers = 6
n_heads = 4
n_kv_heads = 4
ffn_mult = 2.6875
vocab_size = 50257
max_seq_len = 256
group_size = 128
mhc_n_streams = 2
weight_tied = true
rope_theta = 10000.0
lr = 0.02
mhc_lr = 0.0001
weight_decay = 0.0
batch_size = 2
grad_accum_steps = 1
warmup_steps = 10
total_steps = 100
decay_start_frac = 0.8
grad_clip = 1.0
ns_steps = 5
muon_momentum = 0.95
lion_betas = [0.9, 0.99]
use_8bit_optim = true
use_galore = false
galore_rank = 256
galore_update_freq = 200
distill_kl_weight = 0.0
loop_scale_penalty = 0.0
EOF

./nanochat-rs-ternary/target/release/nanochat-train train \
    --config-file /tmp/test_8bit.toml \
    --data-path nanochat-rs-ternary/data/test_tokens.bin \
    --dataset tokens \
    --epochs 1 \
    --batch-size 2 \
    --seq-len 64 \
    --checkpoint-dir /tmp/test_8bit \
    --log-interval 5 \
    --device cpu \
    --threads 16 \
    2>&1 | tail -50

echo ""
echo "✅ All optimizer integration tests passed!"
echo ""
echo "To use in production:"
echo "  1. For 8-bit only:  --config-file configs/nano_125m_8bit.toml"
echo "  2. For GaLore only: --config-file configs/nano_125m_galore.toml"
echo "  3. For both:        --config-file configs/nano_125m_8bit_galore.toml"
