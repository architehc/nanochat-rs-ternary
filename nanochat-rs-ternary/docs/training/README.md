# Training

## Quick Start

```bash
# 1. Build with CUDA support
cargo build --release -p nanochat-train --features cuda

# 2. Prepare tokenized data (if starting from raw text)
cargo run --release -p nanochat-train -- prepare-data \
    --text data/your_corpus.txt \
    --vocab-size 50257 \
    --output data/

# 3. Train (nano_125m on RTX 4090)
cargo run --release --features cuda -p nanochat-train -- train \
    --config nano-125m \
    --dataset tokens \
    --data-path data/tokens.bin \
    --batch-size 2 \
    --seq-len 256 \
    --checkpoint-dir checkpoints/nano125m \
    --log-interval 50 \
    --checkpoint-interval 5000 \
    --keep-last-checkpoints 5 \
    --device cuda

# 4. Export to GGUF + mHC for inference
cargo run --release -p nanochat-train -- export \
    --checkpoint checkpoints/nano125m/final \
    --gguf model.gguf \
    --mhc model.mhc
```

## Model Configs

| Config | Params | dim | Layers | Heads | VRAM (batch=2, seq=256) | Notes |
|--------|--------|-----|--------|-------|-------------------------|-------|
| `d20` | 18M | 256 | 6 | 4 | ~2 GB | Debug/smoke test |
| `nano-125m` | 127M | 768 | 12 | 12 | ~21 GB | **RTX 4090 sweet spot** |
| `nano-1b` | 1.1B | 2048 | 20 | 16 | OOM on 24GB | Needs 48GB+ GPU |
| `medium-3b` | 1.8B | 2048 | 28 | 32 | OOM on 24GB | Needs 80GB+ GPU |
| `large-7b` | 7.0B | 4096 | 32 | 32 | OOM on 24GB | Needs 2×80GB |
| `d20-e3-full` | 244M | 768 | 24 | 12 | ~21 GB | With MTP + Collider |
| `tiny-cpu` | 2M | 256 | 4 | 4 | CPU only | Unit test config |

## RTX 4090 (24GB) — Tested Configuration

The following was validated on an RTX 4090 with 24GB VRAM:

```
Model:       nano-125m (127M params)
Batch size:  2
Seq length:  256
Grad accum:  4 (effective batch = 2048 tokens/step)
Throughput:  ~2,200 tok/s
VRAM usage:  ~21.4 / 24 GB
GPU util:    91-97%
Temperature: 53-57°C
```

**Important VRAM notes:**
- Candle's autograd tape is memory-hungry; batch=4 or seq=512 will OOM
- Gradient accumulation stores GPU tensors across micro-steps
- The Muon optimizer's Newton-Schulz iterations allocate extra matrices
- batch=2, seq=256 with grad_accum=4 is the sweet spot for 24GB

## CLI Reference

### `train` subcommand

```
--config <name>              Model config preset (see table above)
--dataset <type>             "synthetic" or "tokens"
--data-path <path>           Path to tokenized .bin file (required for --dataset=tokens)
--epochs <n>                 Number of epochs (default: 5)
--batch-size <n>             Micro-batch size (default: from config)
--seq-len <n>                Sequence length (default: max_seq_len/2)
--checkpoint-dir <dir>       Where to save checkpoints
--resume <dir>               Resume from checkpoint directory
--log-interval <n>           Print stats every N optimizer steps (default: 50)
--checkpoint-interval <n>    Save checkpoint every N steps (default: 1000)
--keep-last-checkpoints <n>  Keep only last N checkpoints (default: 3)
--threads <n>                CPU threads (default: all available)
--n-samples <n>              Synthetic dataset size (default: 100000)
--device <dev>               "cpu" or "cuda" (default: cpu)
```

### `prepare-data` subcommand

Trains a byte-level BPE tokenizer and encodes text to binary tokens.

```
--text <path>       Input text file
--vocab-size <n>    Target vocabulary size (default: 4096)
--output <dir>      Output directory (produces tokenizer.json + tokens.bin)
```

Output format: flat binary of u32 little-endian token IDs, 4 bytes per token.

### `export` subcommand

Converts a training checkpoint to inference format.

```
--checkpoint <dir>   Checkpoint directory (containing model.safetensors + meta.json)
--gguf <path>        Output GGUF file path
--mhc <path>         Output mHC binary file path
```

## Checkpoint Format

Each checkpoint directory contains:
- `model.safetensors` — model weights
- `meta.json` — config, step number, loss
- `optimizer_state.bin` — Muon + Lion optimizer state (for resume)
- `mtp.safetensors` — MTP head weights (if MTP enabled)

Checkpoint size: ~970MB for nano_125m.

## Resuming Training

```bash
cargo run --release --features cuda -p nanochat-train -- train \
    --config nano-125m \
    --dataset tokens \
    --data-path data/tokens.bin \
    --resume checkpoints/nano125m/step_15000 \
    --checkpoint-dir checkpoints/nano125m \
    --device cuda
```

Resume restores: model weights, optimizer state, global step counter, and MTP params.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/train_nano125m_4090.sh` | Production nano_125m on RTX 4090 (tested) |
| `scripts/train_nano_125m.sh` | Generic nano_125m training |
| `scripts/train_small_560m.sh` | 560M param training |
| `scripts/train_production_8h.sh` | 8-hour GPU training job |
| `scripts/validate_e2e.sh` | End-to-end validation |
| `scripts/benchmark_optimizers.sh` | Optimizer comparison benchmarks |

## Optimizer Details

Two optimizers run in parallel:
- **Muon** (2D+ weights): Newton-Schulz orthogonalization, lr=0.02, momentum=0.95
- **Lion** (1D params, embeddings, mHC): Sign-based updates, lr=1e-4, betas=(0.9, 0.99)

Learning rate schedule: **Warmup-Stable-Decay (WSD)**
- Warmup: linear ramp 0 → lr over warmup_steps
- Stable: constant lr for 80% of training
- Decay: cosine anneal to 0.1×lr

## Training Data Format

Token files are flat binary arrays of `u32` little-endian values:
```
[token_0: u32][token_1: u32][token_2: u32]...
```

Each token ID must be in `[0, vocab_size)`. File size must be divisible by 4.

Generate from raw text:
```bash
cargo run --release -p nanochat-train -- prepare-data \
    --text corpus.txt --vocab-size 50257 --output data/
```
