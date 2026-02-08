# Distillation Training Examples

This directory contains examples for training hybrid ternary models using knowledge distillation from a remote FP8 teacher.

## Overview

**Goal**: Train a Qwen3-Coder-80B hybrid ternary student model using a high-performance remote FP8 teacher endpoint.

**Architecture**:
- **Teacher**: Full FP8 Qwen3-Coder-80B (remote inference at `https://crazyshit.ngrok.io`)
- **Student**: Hybrid ternary
  - **Ternary (Q1_58)**: MoE expert weights (512 experts × `w_gate/w_up/w_down`)
  - **FP8/BF16**: Router, gates, norms, embeddings, lm_head, DeltaNet state matrices

**Why hybrid?**:
- MoE experts = 99% of parameters → ternarize for 4× memory savings
- Router/gates/norms = 1% of parameters → keep FP8 for quality

## Files

- **`distill_qwen3.rs`**: Rust example program demonstrating the distillation API
- **`train_qwen3_distill.sh`**: Production training script with recommended settings

## Quick Start

### Option 1: Run the shell script (recommended)

```bash
# Set teacher endpoint (default: https://crazyshit.ngrok.io)
export TEACHER_ENDPOINT="https://crazyshit.ngrok.io"

# Run training with default settings (parallel mode enabled)
./examples/train_qwen3_distill.sh

# Or customize parallel settings
export MICRO_BATCHES=8
export PARALLEL=true
./examples/train_qwen3_distill.sh
```

### Option 2: Run the Rust example directly

```bash
cargo run --release --example distill_qwen3 -- \
    --teacher-endpoint https://crazyshit.ngrok.io \
    --checkpoint-dir checkpoints/qwen3-hybrid \
    --keep-last-checkpoints 3 \
    --total-steps 100000 \
    --batch-size 4 \
    --seq-len 2048
```

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--teacher-endpoint` | `https://crazyshit.ngrok.io` | Remote FP8 inference endpoint |
| `--total-steps` | `100000` | Total training steps |
| `--warmup-steps` | `2000` | Learning rate warmup steps |
| `--batch-size` | `4` | Batch size (4 × 2048 = 8192 tokens/step) |
| `--seq-len` | `2048` | Sequence length |
| `--lr` | `0.02` | Muon learning rate (ternary weights) |
| `--mhc-lr` | `0.0001` | Lion learning rate (router, norms, mHC) |
| `--kl-weight` | `0.5` | KL divergence weight (0.5 = 50% KL, 50% CE) |
| `--temperature` | `2.0` | Distillation temperature |
| `--keep-last-checkpoints` | `3` | Keep only last N checkpoints |
| `--parallel` | `true` | Enable parallel teacher queries |
| `--micro-batches` | `8` | Number of micro-batches for parallel training |

### Loss Function

The training loss combines multiple objectives:

```
Loss = (1-α)·CE + α·KL(teacher||student) + β·LB + γ·RA

where:
  CE  = Cross-entropy (ground truth labels)
  KL  = KL divergence (teacher distillation)
  LB  = Load balance loss (equal expert usage)
  RA  = Router auxiliary loss (confident routing)

  α = 0.5   (kl_weight)
  β = 0.01  (load_balance_weight)
  γ = 0.001 (router_aux_weight)
```

**Why KL divergence?**
- Captures teacher's uncertainty (soft targets)
- Provides richer training signal than hard labels alone
- Temperature scaling smooths distribution for better learning

## Resource Requirements

### Memory

| Component | GPU Memory | Notes |
|-----------|------------|-------|
| Student model | ~5GB | 3B active params (FP32 training) |
| Activations | ~2GB | batch_size=4, seq_len=2048 |
| Gradients | ~1GB | Student parameters only |
| KV cache | ~1GB | Attention caching |
| **Total** | **~10GB** | Fits comfortably in 96GB VRAM |

**Teacher memory**: 0GB (remote endpoint, no local weights)

### Disk Space

| Component | Space | Notes |
|-----------|-------|-------|
| Checkpoint (each) | ~160GB | Full student model state |
| Max checkpoints | ~480GB | With `keep_last_checkpoints=3` |
| Training logs | ~1GB | Per 100k steps |
| **Total** | **~500GB** | Fits in 1.9TB available |

Checkpoints are auto-cleaned to keep only the last N, preventing disk exhaustion.

### Training Time

| Setting | Time Estimate |
|---------|---------------|
| Steps/sec | ~0.5 (includes remote teacher query) |
| Total steps | 100,000 |
| **Total time** | **~55 hours (~2.3 days)** |

Time dominated by:
1. Remote teacher inference (~1s/step)
2. Student forward + backward (~0.5s/step)
3. Optimizer step (~0.5s/step)

## Monitoring

### Training Logs

The script logs to both console and `checkpoints/qwen3-hybrid/training.log`:

```
[  1000/ 100000] loss=2.34 lr=0.020 gnorm=1.2 tok/s=8500 elapsed=2000s
  -> checkpoint saved to checkpoints/qwen3-hybrid/step_1000 (162GB)

[  2000/ 100000] loss=2.12 lr=0.020 gnorm=1.0 tok/s=8700 elapsed=4100s
  Cleaned 1 old checkpoint(s), keeping last 3
  -> checkpoint saved to checkpoints/qwen3-hybrid/step_2000 (162GB)
```

### Disk Space Warnings

When disk usage exceeds 90%, you'll see:

```
WARNING: Disk 91% full (147GB / 1.8TB available)
```

The training will continue, but consider:
- Reducing `keep_last_checkpoints` (e.g., from 3 to 2)
- Cleaning old experiments: `rm -rf checkpoints/old-*`
- Moving completed checkpoints to external storage

## Checkpointing

### Auto-Cleanup

Checkpoints are automatically cleaned to keep only the last N:

```bash
checkpoints/qwen3-hybrid/
├── step_98000/   # Removed (too old)
├── step_99000/   # Kept
├── step_100000/  # Kept (most recent)
└── final/        # Always kept
```

### Manual Checkpoint Management

```bash
# List all checkpoints with sizes
du -sh checkpoints/qwen3-hybrid/step_*

# Remove specific checkpoint
rm -rf checkpoints/qwen3-hybrid/step_50000

# Keep only final checkpoint (free up space)
cd checkpoints/qwen3-hybrid
rm -rf step_*
```

## After Training

### Export to GGUF

Convert the final checkpoint to GGUF format for inference:

```bash
cargo run --release --bin qwen3_converter -- \
    --input checkpoints/qwen3-hybrid/final \
    --output models/qwen3-hybrid.gguf \
    --group-size 128
```

### Run Inference

Start the inference server:

```bash
cargo run --release --bin nanochat-serve -- \
    --model models/qwen3-hybrid.gguf \
    --port 8000 \
    --numa-aware
```

Test inference:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python function to check if a number is prime"}],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

## Troubleshooting

### Teacher endpoint unreachable

```
Error: Failed to connect to teacher endpoint: https://crazyshit.ngrok.io
```

**Fix**: Ensure the FP8 inference server is running and accessible:

```bash
curl https://crazyshit.ngrok.io
```

### Out of memory (GPU)

```
Error: CUDA out of memory
```

**Fix**: Reduce batch size:

```bash
--batch-size 2   # Down from 4
--seq-len 1024   # Down from 2048
```

### Disk full

```
Error: No space left on device
```

**Fix**: Clean old checkpoints or reduce retention:

```bash
# Clean all but last 2 checkpoints
--keep-last-checkpoints 2

# Or manually remove old checkpoints
rm -rf checkpoints/qwen3-hybrid/step_{1..50}000
```

### Training too slow

If training is slower than expected (~0.5 steps/sec):

1. **Check teacher latency**: Remote queries should be ~1s
   ```bash
   time curl -X POST https://crazyshit.ngrok.io/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"input_ids": [[1, 2, 3]], "return_logits": true}'
   ```

2. **Use smaller sequences**: Reduce `--seq-len` to 1024
3. **Increase batch size**: Try `--batch-size 8` (if GPU memory allows)

## Advanced: Multi-GPU Training

To train on multiple GPUs with data parallelism:

```bash
# GPU 0: Train first half of model
CUDA_VISIBLE_DEVICES=0 cargo run --release --example distill_qwen3 -- \
    --device cuda:0 \
    --checkpoint-dir checkpoints/qwen3-hybrid-gpu0 &

# GPU 1: Train second half independently
CUDA_VISIBLE_DEVICES=1 cargo run --release --example distill_qwen3 -- \
    --device cuda:1 \
    --checkpoint-dir checkpoints/qwen3-hybrid-gpu1 &

wait
```

Then merge checkpoints with a custom script (TODO: implement gradient averaging).

## References

- [BitNet b1.58 paper](https://arxiv.org/abs/2402.17764) - 1-bit quantization
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531) - Hinton et al.
- [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) - Teacher model
- [mHC-lite](https://arxiv.org/abs/2410.03062) - Exact doubly stochastic residuals
