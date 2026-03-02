# nanochat-rs-ternary

A full-stack ternary language model: training, export, and inference. All linear
layers use **1.58-bit ternary weights {-1, 0, +1}** with **mHC-lite doubly
stochastic residual connections**. The trained 162M-parameter model generates
coherent children's stories (perplexity 4.01 on TinyStories) and is served via
an OpenAI-compatible HTTP API.

```
Training (Rust/Candle)  -->  Export (GGUF + mHC)  -->  Inference (Rust/AVX2)
    Muon/Lion optimizers       107 MB model              OpenAI-compatible API
    mHC-lite BvN residuals      880 B mHC params          SSE streaming
    Optional 8-bit/GaLore2      5.8x compression          14-32 GOPS CPU kernel
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
│   ├── nanochat-train/                 # Rust training + checkpoint/export CLI
│   │   └── src/
│   │       ├── main.rs                 # CLI: train / prepare-data / export
│   │       ├── train.rs                # Trainer loop + checkpoint management
│   │       ├── checkpoint.rs           # Checkpoint load/save (weights + metadata)
│   │       └── optim/                  # Muon, 8-bit Muon, GaLore2, Lion
│   │
│   └── nanochat-serve/                 # HTTP inference server
│       └── src/
│           ├── main.rs                 # CLI entry point
│           ├── server.rs               # Axum HTTP server + SSE streaming
│           ├── engine.rs               # Generation engine + sampling
│           └── api.rs                  # OpenAI-compatible request/response types
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

**Codebase size:** Rust-first codebase with native C kernels for hot GEMV paths

---

## Quick Start

### Prerequisites

- **Rust** 1.70+ (with `cargo`)
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
# 496 passing tests (+1 ignored), 99.55% coverage
```

### Serve a Trained Model

```bash
cargo run --release -p nanochat-serve -- \
  --model exported_models/nanochat_125m.gguf \
  --mhc exported_models/nanochat_125m.mhc \
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

### Train from Scratch

```bash
cargo run --release -p nanochat-train -- train \
  --config nano-125m \
  --dataset synthetic \
  --epochs 1 \
  --batch_size 8 \
  --seq_len 256 \
  --checkpoint_dir checkpoints/nanochat_125m \
  --checkpoint_interval 1000 \
  --keep_last_checkpoints 3 \
  --log_interval 50
```

### Export to GGUF

```bash
cargo run --release -p nanochat-train -- export \
  --checkpoint checkpoints/nanochat_125m/final \
  --gguf exported_models/nanochat_125m.gguf \
  --mhc exported_models/nanochat_125m.mhc
```

### Evaluate

```bash
cargo run --release -p nanochat-eval --example benchmark_model -- \
  --checkpoint checkpoints/nanochat_125m/final \
  --n-samples 100 \
  --output benchmark_results.json
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

496 passing tests (+1 ignored) across the workspace with 99.55% line coverage:

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

## Rust Code Generation Training (275M, Engram Architecture)

In addition to TinyStories, the model has been trained on ~97M tokens of Rust
source code (from clap, serde, tokio, and other crates) using a BPE tokenizer
with vocab=4096. The training uses the **Engram memory** architecture -- an
n-gram hash table memory attached to select transformer layers for improved
pattern recall.

### Architecture: nano-275m-engram

```
NanochatTernary-275M (dim=1024, 20 layers, 16 heads, 4 KV heads)
├── Token Embedding         4.2M params     FP16 (4096 x 1024)
├── 20x Transformer Block   ~13M each       Ternary BitLinear + FP32 norms
│   ├── GQA Attention (16Q/4KV heads)       Ternary BitLinear
│   ├── SwiGLU FFN                          Ternary BitLinear
│   └── Engram Memory (layers 0, 19)        n-gram hash table (d=128, table=10007)
├── Final RMSNorm                           FP32
└── LM Head                                 Ternary BitLinear (1024 x 4096)
```

### Training Configs and Results

| Config | LR | Steps | Decay Start | Loss | Gnorm | Notes |
|--------|-----|-------|-------------|------|-------|-------|
| **engram-v1** | 0.012 | 10K | 80% (8K) | **2.19** | stable | **Best generation quality** |
| engram-only | 0.012 | 20K | 80% (16K) | 3.09 | stable early | Longer but not better |
| engram-v5 | 0.012 | 30K | 80% (24K) | diverged | 300+ at 9K | LR too high for long runs |
| engram-v6 | 0.012 | 20K | 80%, clip=0.5 | diverged | 15+ at 10K | Grad clip delays, doesn't fix |
| **engram-v7** | 0.008 | 10K | 80% (8K) | **3.22** | 2.0-2.4 | Rock stable, lower LR |
| engram-v8 | 0.008 | 20K | 80% (16K) | 3.12 | stable | Extending v7 approach |
| engram-v9 | 0.010 | 15K | 80% (12K) | 3.32 | 4.5-5.0 | Middle ground, mild instability |
| engram-v10 | 0.012 | 15K | **50%** (7.5K) | in progress | TBD | Early decay hypothesis |
| baseline-v1 (MTP) | 0.012 | 10K | 80% | H=2.92 | stable | No engram, comparable |
| haar-v5 (wavefield) | 0.012 | 30K | 80% | **1.17** | stable | Bidirectional -- cannot generate |

### Key Finding: Learning Rate Stability

`lr=0.012` is unstable past ~9K steps at full learning rate -- gradient norms
blow up exponentially. The "early decay hypothesis" explains why engram-v1
(10K steps, decay at step 8K) succeeded: the model only saw ~6.5K steps at
full LR before cosine decay started reducing it.

| LR | Max Stable Steps | Gradient Norm |
|----|-----------------|---------------|
| 0.012 | ~8-9K | Blows up to 100+ after |
| 0.010 | ~12K+ | Elevated (4-5) but survivable |
| 0.008 | 20K+ (indefinite) | Rock stable (2.0-2.4) |

### Coherence Benchmark (Automated)

| Model | Composite | Syntax | Brackets | Repetition (3-gram) |
|-------|-----------|--------|----------|---------------------|
| **engram-v1** (loss 2.19) | **0.752** | **0.608** | 0.918 | **0.888** |
| engram-v2 (loss 2.94) | 0.740 | 0.617 | 0.951 | 0.818 |
| baseline-v1 (H=2.92) | 0.699 | 0.500 | 0.936 | 0.767 |

### Generation Quality vs Loss

- **loss ~3.5** (perplexity 33): gibberish, stuck on `{` repetition
- **loss ~3.0** (perplexity 20): recognizes Rust patterns (vec![], struct), incoherent flow
- **loss ~2.2** (perplexity 9): semi-coherent Rust with Cargo test patterns, struct/impl blocks
- **loss ~1.5-2.0** (estimated): needed for fully coherent code generation

### Sample Output (engram-v1, loss 2.19, temperature=0.8)

```rust
// Prompt: "pub struct Config {"
pub struct Config {
    pub id: String,
}
impl Config {
    pub id: Vec<Span>,
}
impl Config for Config {}
impl Config {
    pub fn new(s: &Path) -> Config {
        unsafe { Self(s) }
    }
}
impl Config {
    /// Config this is a `Serializer<Path>`.
    pub fn sample(&self) -> Config {
        self.0
    }
}
```

```rust
// Prompt: "async fn handle_request(req: Request) -> Response {"
async fn handle_request(req: Request) -> Response {
    let (req, _) = mock_request(req);
    let (req, _) = mock_request(req, service);
    match (req, Err(err) {
        Ok(err) => {
            let (req, _) = mock_request(req, client).unwrap();
            Ok(err.into_bytes())
        } else {
            None
        })
    }
    Ok(req)
}
```

The model generates syntactically plausible Rust code with correct use of
`impl`, `struct`, `fn`, `match`, `async`, `Result`, `Option`, and other Rust
idioms. While not compilable, it demonstrates that a 275M ternary-weight model
can learn meaningful code structure from ~97M tokens.

### Training Hardware (Rust Code Models)

| Component | Specification |
|-----------|--------------|
| GPUs | 2x NVIDIA GeForce RTX 4090 (24 GB VRAM each) |
| Training | Dual GPU concurrent runs (one model per GPU) |
| Throughput | ~420 tok/s per GPU (batch=2, seq=256, dim=1024) |
| VRAM usage | ~20-22 GB per model |

### Quick Start (Rust Training)

```bash
# Build with CUDA support
cd nanochat-rs-ternary
cargo build --release -p nanochat-train --features cuda

# Train engram-v1 config
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
./target/release/nanochat-train train \
    --config nano-275m-engram-v1 \
    --dataset tokens \
    --data-path data/rust_v2_prepared/tokens.bin \
    --batch-size 2 --seq-len 256 --epochs 999 \
    --total-steps 10000 \
    --checkpoint-dir checkpoints/nano-275m-engram-v1 \
    --checkpoint-interval 2000 \
    --log-interval 50 --device cuda

# Generate text from checkpoint
./target/release/nanochat-train generate \
    --checkpoint checkpoints/nano-275m-engram-v1/final \
    --tokenizer data/rust_v2_prepared/tokenizer.json \
    --device cuda \
    --prompt "fn main() {" \
    --max-tokens 200 --temperature 0.8 --top-k 50
```

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
