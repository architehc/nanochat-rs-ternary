# nanochat-rs-ternary

A full-stack ternary language model: training, export, and inference. All linear
layers use **1.58-bit ternary weights {-1, 0, +1}** with **mHC-lite doubly
stochastic residual connections**. The trained 162M-parameter model generates
coherent children's stories (perplexity 4.01 on TinyStories) and is served via
an OpenAI-compatible HTTP API.

```
Training (Python/PyTorch)  -->  Export (GGUF + mHC)  -->  Inference (Rust/AVX2)
     ternary QAT + STE          107 MB model              OpenAI-compatible API
     mHC-lite BvN residuals      880 B mHC params          SSE streaming
     ~13.5K tok/s GPU            5.8x compression          14-32 GOPS CPU kernel
```

---

## What Makes This Different

**Ternary weights.** Every linear layer (attention projections, FFN, LM head)
stores only {-1, 0, +1} per weight with a shared f32 scale per 128-weight group.
This compresses linear layers from 32 bits to ~2.25 bits per weight. The full
model (including FP16 embeddings and F32 norms) is 107 MB on disk.

**mHC-lite residual connections.** Standard residual connections (`x = x + f(x)`)
can amplify signal norms exponentially through deep networks. mHC replaces this
with a learned doubly stochastic mixing matrix, guaranteeing composite gain <= 1.0
by construction. This uses Birkhoff-von-Neumann decomposition (exact, no Sinkhorn
iterations) and adds only 216 FP32 parameters to the entire model (9 per sub-layer
x 24 sub-layers).

**AVX2-accelerated inference.** A custom PSHUFB-based GEMV kernel achieves 14-32
GOPS on the ternary matrix-vector products, 16-18x faster than the scalar
reference implementation.

**End-to-end system.** Training, quantization-aware optimization, export to a
custom GGUF format, and a production Rust inference server with KV-cache,
streaming, and an OpenAI-compatible API -- all in one repository.

---

## Architecture

```
NanochatTernary-125M (162M total params)
├── Token Embedding         38.6M params    FP16 (50,257 x 768)
├── 12x Transformer Block   ~10.2M each     Ternary linear + FP32 norms/mHC
│   ├── mHC-Attention
│   │   ├── RMSNorm (pre-norm)              FP32 (768)
│   │   ├── Q/K/V/O projections             Ternary BitLinear (768x768 each)
│   │   ├── RoPE positional encoding        theta=10000
│   │   └── mHC-lite N=2 residual           9 FP32 params
│   └── mHC-FFN
│       ├── RMSNorm (pre-norm)              FP32 (768)
│       ├── SwiGLU gate/up/down             Ternary BitLinear (768x2048, 2048x768)
│       └── mHC-lite N=2 residual           9 FP32 params
├── Final RMSNorm                           FP32 (768)
└── LM Head                                 Ternary BitLinear (768x50257)
```

| Component | Parameters | Storage | Notes |
|-----------|-----------|---------|-------|
| Linear (ternary) | 123,532,032 | 2 bits/weight + scales | Quantized {-1,0,+1} |
| Embeddings | 38,597,376 | FP16 | Not quantized |
| Norms | 19,200 | FP32 | RMSNorm weights |
| mHC | 216 | FP32 | Never quantized |
| **Total** | **162,148,824** | **107 MB GGUF** | |

---

## Repository Structure

```
nanochat-rs-ternary/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── ternary-core/                   # Bit packing, planar SoA, GGUF I/O
│   │   └── src/
│   │       ├── encode.rs               # {-1,0,+1} <-> 2-bit encoding (BitNet standard)
│   │       ├── pack.rs                 # Group quantization + packing
│   │       ├── planar.rs               # PlanarWeights SoA + 128-byte AlignedVec
│   │       ├── gguf.rs                 # GGUF reader/writer (custom Q1_58 type)
│   │       └── verify.rs               # Triangle of Truth verification
│   │
│   ├── ternary-kernels/                # CPU GEMV kernels (C FFI)
│   │   ├── csrc/
│   │   │   ├── ternary_gemv.c          # Scalar reference kernel
│   │   │   ├── ternary_gemv_avx2.c     # AVX2 PSHUFB kernel (14-32 GOPS)
│   │   │   └── ternary_gemv_avx2.h     # C API header
│   │   └── src/
│   │       ├── cpu.rs                  # Safe Rust FFI wrappers
│   │       └── dispatch.rs             # Runtime CPU feature detection
│   │
│   ├── mhc-lite/                       # mHC doubly stochastic residuals
│   │   └── src/
│   │       ├── n2.rs                   # N=2 streams (sigmoid-based alpha)
│   │       ├── n4.rs                   # N=4 streams (24 permutations, full BvN)
│   │       ├── verify.rs               # DS verification, composite gain
│   │       └── io.rs                   # Binary serialization
│   │
│   ├── nanochat-model/                 # Transformer model
│   │   └── src/
│   │       ├── config.rs               # Model configs (d20, 125M, 560M, 7B)
│   │       ├── embed.rs                # Token embeddings
│   │       ├── norm.rs                 # RMSNorm
│   │       ├── attention.rs            # Multi-head attention + KV cache + RoPE
│   │       ├── ffn.rs                  # SwiGLU feed-forward
│   │       ├── bitlinear.rs            # Ternary linear layer (quant + GEMV)
│   │       ├── block.rs                # Transformer block with mHC wiring
│   │       └── model.rs                # Full model + GGUF loader
│   │
│   └── nanochat-serve/                 # HTTP inference server
│       └── src/
│           ├── main.rs                 # CLI entry point
│           ├── server.rs               # Axum HTTP server + SSE streaming
│           ├── engine.rs               # Generation engine + sampling
│           └── api.rs                  # OpenAI-compatible request/response types
│
├── training/                           # PyTorch training pipeline
│   ├── model.py                        # NanochatTernary model definition
│   ├── train.py                        # Training loop (AdamW, WSD schedule)
│   ├── ternary_qat.py                  # BitLinearSTE (quantization-aware training)
│   ├── mhc_lite.py                     # mHC-lite module (BvN, exact DS)
│   ├── export.py                       # PyTorch -> GGUF + mHC binary
│   ├── evaluate.py                     # Validation perplexity + generation
│   └── requirements.txt
│
├── tests/                              # Integration tests
│   ├── triangle_of_truth.rs            # Cross-validate kernel paths
│   ├── mhc_property_tests.rs           # DS invariants (1000 random inputs)
│   ├── roundtrip_test.rs               # Pack -> GGUF -> load -> GEMV
│   └── e2e_generate.rs                 # Full model forward + generation
│
└── benches/                            # Criterion benchmarks
    ├── gemv_bench.rs                   # GEMV throughput (all shapes)
    └── mhc_overhead.rs                 # mHC compute overhead measurement
```

**Codebase size:** ~8,400 lines Rust + ~2,700 lines Python + ~1,400 lines C

---

## Quick Start

### Prerequisites

- **Rust** 1.70+ (with `cargo`)
- **Python** 3.9+ with PyTorch 2.0+ and CUDA (for training)
- **x86_64 CPU** with AVX2 (for optimized inference; scalar fallback available)
- GCC or Clang (for compiling C kernels)

### Build

```bash
cd nanochat-rs-ternary
cargo build --release
```

### Run Tests

```bash
cargo test --workspace
# 238 tests, 99.55% coverage
```

### Serve a Trained Model

```bash
cargo run --release -p nanochat-serve -- \
  --model training/checkpoints/nanochat_125m.gguf \
  --mhc training/checkpoints/nanochat_125m.mhc \
  --port 8080
```

### Query the API

**Non-streaming:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Once upon a time"}],
    "max_tokens": 200,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95
  }'
```

**Streaming (SSE):**

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Once upon a time"}],
    "stream": true,
    "max_tokens": 200,
    "temperature": 0.8
  }'
```

**Available endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List loaded model |
| POST | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |

**Sampling parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature (0 = greedy) |
| `top_k` | int | 0 | Top-k filtering (0 = disabled) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `max_tokens` | int | 256 | Maximum tokens to generate |
| `seed` | int | null | Random seed for reproducibility |
| `stream` | bool | false | Enable SSE streaming |

---

## Training

### Requirements

```bash
cd training
pip install -r requirements.txt
# torch>=2.0, numpy, tqdm, datasets, tiktoken
```

### Train from Scratch

```bash
python train.py \
  --config 125m \
  --dataset tinystories \
  --device cuda \
  --epochs 1 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --seq_len 256 \
  --lr 3e-4 \
  --mhc_lr 1e-3 \
  --warmup_steps 500 \
  --save_path checkpoints/nanochat_125m.pt \
  --log_interval 50 \
  --diag_interval 500
```

### Export to GGUF

```bash
python export.py \
  --checkpoint checkpoints/nanochat_125m.pt \
  --gguf checkpoints/nanochat_125m.gguf \
  --mhc checkpoints/nanochat_125m.mhc \
  --config 125m
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/nanochat_125m.pt
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | NanochatTernary-125M (162M params) |
| Dataset | TinyStories (474M tokens, 2.1M stories) |
| Epochs | 1 |
| Effective batch size | 32 (8 x 4 gradient accumulation) |
| Sequence length | 256 |
| Optimizer | AdamW |
| Linear LR | 3e-4 |
| mHC/embed LR | 1e-3 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| LR schedule | Warmup-Stable-Decay (warmup=500, stable=80%, cosine decay to 10%) |
| Total steps | 57,635 |

### Training Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |
| CPU | Dual AMD EPYC 9654 (96c/192t per socket, 384 threads) |
| RAM | 1 TB DDR5 |
| Training time | ~9.7 hours |
| GPU memory peak | 5.7 GB |
| Throughput | ~13,500 tok/s |

**Minimum requirements to reproduce:** A single GPU with 8+ GB VRAM (e.g., RTX
3070) and 32 GB system RAM would be sufficient for the 125M model. The large
hardware specification targets the bigger model configs (7B, 25B-MoE).

---

## Results

### Training Metrics

| Metric | Value |
|--------|-------|
| Initial loss | ~10.86 (ln(50257) = 10.82) |
| Final loss | 1.33 |
| **Validation perplexity** | **4.01** |
| mHC composite gain | 1.000000 (bounded) |
| DS violations | 0 (across all 57K steps) |

### Generated Text Samples

**Greedy (temperature=0):**

> **Once upon a time,** there was a little girl named Lily. She loved to play
> outside in the sunshine. One day, she saw a big, scary dog in the park. She
> was scared and ran to her mommy.
>
> "Mommy, mommy! There's a scary dog in the park!" Lily said.
>
> "Don't worry, sweetie. The dog is friendly. He just wants to play," her mommy
> replied.

> **Tom was a good boy who** liked to help his mom. He helped her clean the
> house, he helped her clean the dishes, and he helped her with the dishes. He
> was a good boy.
>
> One day, Tom saw a big box in the living room. He wondered what was inside.
> He asked his mom, "Mom, what is in the box?"
>
> His mom smiled and said, "It is a surprise for you, Tom. But you have to wait
> until your birthday..."

**Sampled (temperature=0.8, top_k=50):**

> **The little cat** tried to jump. It jumped very high, but it could not reach
> the butterfly. The little cat was sad. Then, a kind mouse came by. The mouse
> said, "I can help you get the butterfly. I will show you how."
>
> The mouse showed the little cat how to jump. The little cat tried very hard.
> At last, the little cat could jump high. It jumped higher than the butterfly.
> The little cat was so happy.

All text above is generated by a model where every linear layer uses only
**ternary weights {-1, 0, +1}**.

### Model Compression

| Format | Size | Notes |
|--------|------|-------|
| FP32 (training) | 648 MB | Shadow weights |
| PyTorch checkpoint | 1.9 GB | + optimizer state |
| **GGUF (inference)** | **107 MB** | Ternary + FP16 embed + F32 norms |
| mHC binary | 880 B | 24 layers x 36 bytes |
| **Compression ratio** | **5.8x** | FP32 -> GGUF |

### CPU Kernel Performance (AVX2 PSHUFB)

Single-thread GEMV benchmarks on AMD EPYC 9654 (Zen4):

| Shape | Scalar | AVX2 | Speedup | AVX2 GOPS |
|-------|--------|------|---------|-----------|
| 128x128 | 0.019 ms | 0.002 ms | 8.2x | 14.0 |
| 512x512 | 0.305 ms | 0.021 ms | 14.3x | 24.6 |
| 2048x2048 | 4.893 ms | 0.299 ms | 16.4x | 28.1 |
| 4096x4096 | 19.66 ms | 1.185 ms | 16.6x | 28.3 |
| 4096x11008 | 52.51 ms | 3.189 ms | 16.5x | 28.3 |
| 11008x4096 | 52.73 ms | 2.850 ms | 18.5x | 31.7 |

The AVX2 kernel uses 128-bit PSHUFB with 16-entry lookup tables held in
registers, processing 16 rows at a time. It avoids the 256-bit PSHUFB pitfall
(lane-local, cannot do int16 LUTs across 256 bits).

---

## Technical Details

### Ternary Quantization-Aware Training

Training uses `BitLinearSTE` -- the forward pass quantizes FP32 shadow weights
to {-1, 0, +1} via absmean scaling, and the backward pass uses a straight-through
estimator (STE) to propagate gradients through the non-differentiable
quantization:

```
Forward:  w_ternary = round(w / mean(|w|)) clamped to {-1, 0, +1}
Backward: grad(w) = grad(w_ternary)  (straight-through)
```

Activations are quantized to INT8 per-token with absmax scaling before each
linear layer. Group size is 128 (matching BitNet b1.58).

### mHC-lite Residual Connections

Each transformer sub-layer uses a learned 2x2 doubly stochastic mixing matrix
instead of a simple residual addition:

```
Standard:   x = x + sublayer(x)
mHC-lite:   x_exp = H_res @ x_exp + h_post * sublayer(h_pre @ x_exp)
```

Where `H_res = alpha * I + (1-alpha) * J` and `alpha = sigmoid(logit)`. This is
an exact BvN decomposition -- doubly stochastic by construction, no iterative
normalization needed. The composite gain across all 24 sub-layers stays bounded
at exactly 1.0 throughout training.

### GGUF Format

The model uses a custom GGUF type `Q1_58 = 100` for ternary tensors. On load,
interleaved GGUF data is repacked into planar Structure-of-Arrays format
(contiguous weight bytes + contiguous scales) with 128-byte alignment for
optimal SIMD access.

### Encoding

BitNet standard encoding (GGUF-compatible):
- `-1` -> `0b11`
- ` 0` -> `0b00`
- `+1` -> `0b01`
- `0b10` -> invalid (decoded as 0, flagged in verification)

Four trits pack into one byte (2 bits each).

---

## Test Suite

238 tests across the workspace with 99.55% line coverage:

| Crate | Tests | What's Tested |
|-------|-------|---------------|
| ternary-core | 86 | Encode/decode roundtrip, packing, planar layout, GGUF I/O, alignment |
| ternary-kernels | 3 | FFI linkage, GEMV correctness vs scalar reference |
| mhc-lite | 42 | DS property (1000 random inputs), composite gain, N=2/N=4, serialization |
| nanochat-model | 32 | BitLinear forward, attention, FFN, block, full model, GGUF loading |
| nanochat-serve | 17 | API serde, engine sampling, HTTP handlers, SSE streaming |
| Integration | 48 | Triangle of Truth, mHC properties, roundtrip, e2e generation |

```bash
cargo test --workspace        # Run all tests
cargo bench                   # Run criterion benchmarks
```

---

## Model Configs

Pre-defined configurations in `config.rs`:

| Config | Dim | Layers | Heads | FFN | Vocab | Params |
|--------|-----|--------|-------|-----|-------|--------|
| `d20` | 256 | 6 | 4 | 683 | 32K | ~20M |
| `nano_125m` | 768 | 12 | 12 | 2048 | 50.3K | 162M |
| `nano_560m` | 1024 | 24 | 16 | 2731 | 32K | ~560M |
| `nano_7b` | 4096 | 32 | 32/8 GQA | 10923 | 128K | ~7B |

---

## Design Constraints

1. **VPERMW/PSHUFB kernels are primary.** LUT-Grouped degrades at large K (L1 overflow).
2. **Planar SoA at runtime.** Never interleaved [weights|scale] for compute.
3. **mHC parameters stay FP32.** Quantizing them would break doubly stochastic property.
4. **128-byte alignment** for all weight buffers (required for AVX2/AVX-512).
5. **BitNet encoding: 11=-1, 01=+1, 00=0.** GGUF/bitnet.cpp compatible.
6. **Group size 128.** BitNet b1.58 standard.

---

## License

This project is for research and educational purposes.

---

## References

- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764) -- Ma et al., 2024
- [TinyStories](https://arxiv.org/abs/2305.07759) -- Eldan & Li, 2023
- [Birkhoff-von Neumann theorem](https://en.wikipedia.org/wiki/Birkhoff%E2%80%93von_Neumann_theorem) -- doubly stochastic matrices as convex combinations of permutations
