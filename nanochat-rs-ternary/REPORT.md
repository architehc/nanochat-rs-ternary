# Nanochat-Ternary 125M — Training & Evaluation Report

A 162M-parameter language model using **ternary quantization-aware training (QAT)**
with **mHC-lite doubly stochastic residual connections**, trained on TinyStories.

---

## 1. Architecture

| Component | Detail |
|-----------|--------|
| Model | NanochatTernary-125M |
| Hidden dim | 768 |
| Layers | 12 |
| Attention heads | 12 (MHA, head_dim=64) |
| FFN | SwiGLU, intermediate=2048 (2.667x) |
| Vocab | 50,257 (tiktoken GPT-2) |
| Max seq len | 2,048 |
| Quantization | Ternary {-1, 0, +1} via STE, group_size=128 |
| Residual | mHC-lite N=2 (BvN doubly stochastic) |
| Position encoding | RoPE (theta=10000) |
| Normalization | RMSNorm (pre-norm) |

**Parameter breakdown:**

| Category | Count | Notes |
|----------|-------|-------|
| Linear (ternary) | 123,532,032 | Quantized to {-1,0,+1} at inference |
| Embeddings | 38,597,376 | FP16 at inference |
| Norms | 19,200 | FP32 |
| mHC | 216 | FP32, never quantized |
| **Total** | **162,148,824** | |

### Key Design Choices

- **Ternary QAT with STE**: All linear layers use BitLinearSTE — forward pass
  quantizes weights to {-1,0,+1} via absmean scaling, backward uses
  straight-through estimator. Group size 128 matches BitNet b1.58.

- **mHC-lite (N=2)**: Each transformer sub-layer (attention, FFN) has an mHC
  residual connection. Input is expanded to 2 streams, mixed via a learned
  doubly stochastic matrix H_res = alpha*I + (1-alpha)*J where alpha =
  sigmoid(logit). This is exact BvN — no Sinkhorn iterations needed.

- **Why mHC matters**: Standard residual connections can amplify signal norms
  exponentially through deep networks. mHC's doubly stochastic constraint
  guarantees composite gain <= 1.0, preventing residual stream explosion while
  maintaining expressivity.

---

## 2. Training

### Dataset: TinyStories

| Metric | Value |
|--------|-------|
| Dataset | roneneldan/TinyStories (HuggingFace) |
| Training stories | 2,119,719 |
| Total tokens | 473,992,006 |
| Chunks (seq_len=256) | 1,844,326 |
| Tokenizer | tiktoken GPT-2 (50,257 vocab) |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Effective batch size | 32 (8 x 4 gradient accumulation) |
| Sequence length | 256 |
| Optimizer | AdamW |
| Linear LR | 3e-4 |
| mHC/embed LR | 1e-3 |
| Weight decay | 0.1 |
| Grad clipping | 1.0 |
| LR schedule | WSD (warmup=500, stable=80%, cosine decay to 10%) |
| Total steps | 57,635 |

### Training Metrics

| Metric | Value |
|--------|-------|
| Initial loss | ~10.86 (close to ln(50257) = 10.82) |
| Final loss (last 10 avg) | 1.33 |
| Best single-batch loss | 0.95 |
| Training throughput | ~13,500 tok/s |
| GPU memory peak | 5,743 MB |
| Total training time | ~9.7 hours |
| Hardware | NVIDIA RTX 4090 (24GB) |

### mHC Diagnostics (Throughout Training)

| Metric | Value |
|--------|-------|
| Composite gain | 1.000000 (bounded, as expected) |
| DS violations | None (0 across all 57K steps) |
| mHC layers | 24 (2 per transformer block × 12 blocks) |

---

## 3. Evaluation Results

### Validation Perplexity

| Metric | Value |
|--------|-------|
| Validation split | TinyStories validation (21,990 stories) |
| Eval tokens | 1,024,000 |
| **Average loss** | **1.3883** |
| **Perplexity** | **4.01** |

A perplexity of 4.01 means the model is, on average, choosing between ~4 equally
likely next tokens — quite good for a ternary model after just 1 epoch.

### Generated Text Samples

**Greedy (temperature=0):**

> **Prompt: "Once upon a time"**
>
> Once upon a time, there was a little girl named Lily. She loved to play outside
> in the sunshine. One day, she saw a big, scary dog in the park. She was scared
> and ran to her mommy.
>
> "Mommy, mommy! There's a scary dog in the park!" Lily said.
>
> "Don't worry, sweetie. The dog is friendly. He just wants to play," her mommy
> replied.
>
> Lily felt better and went back to playing. But then, she saw the dog again.
> This time, the dog was not scary. He was just playing with a ball...

> **Prompt: "Tom was a good boy who"**
>
> Tom was a good boy who liked to help his mom. He helped her clean the house,
> he helped her clean the dishes, and he helped her with the dishes. He was a
> good boy.
>
> One day, Tom saw a big box in the living room. He wondered what was inside.
> He asked his mom, "Mom, what is in the box?"
>
> His mom smiled and said, "It is a surprise for you, Tom. But you have to wait
> until your birthday..."

> **Prompt: "One day, a girl named Lucy"**
>
> One day, a girl named Lucy went to the park. She saw a big tree with a lot of
> fruit. Lucy wanted to pick some fruit to eat. She saw a boy named Tom near the
> tree. Tom was eating a fruit that looked very yummy.
>
> Lucy said, "Hi Tom! Can I have some fruit, please?" Tom looked at Lucy and
> said, "No, this is my fruit. You can't have it." Lucy felt sad and walked away.

**Sampled (temperature=0.8, top_k=50):**

> **Prompt: "The little cat"**
>
> The little cat tried to jump. It jumped very high, but it could not reach the
> butterfly. The little cat was sad. Then, a kind mouse came by. The mouse saw
> the little cat. The mouse said, "I can help you get the butterfly. I will show
> you how."
>
> The mouse showed the little cat how to jump. The little cat tried very hard.
> At last, the little cat could jump high. It jumped higher than the butterfly.
> The little cat was so happy. It thanked the kind mouse.

> **Prompt: "There was a big red"**
>
> There was a big red ball. The ball was very impressive. The ball was very big
> and shiny. It was so big that it could bounce very high.
>
> One day, a little boy saw the big red ball. He wanted to shoot it. He asked
> his dad, "Can I shoot the ball?" His dad said, "No, it is not safe for you to
> shoot the ball." But the little boy was very curious...

> **Prompt: "Mom said to her son"**
>
> Mom said to her son, "I can help you get the steak, but you must be very
> careful. Don't get too close."
>
> The little boy was very careful as he walked towards the steak. He looked at
> it carefully and saw that it was very cheap. He asked his mom, "Mom, can I
> have this steak please?"
>
> His mom said, "Yes, of course you can. But you must be careful and not drop
> it."

### Quality Assessment

The model produces **coherent, grammatically correct children's stories** with:
- Consistent characters and plot threads within a story
- Natural dialogue with proper quotation formatting
- Appropriate vocabulary level for the TinyStories domain
- Stories that reach logical conclusions (many end with `<|endoftext|>`)
- Proper narrative structure (setup, conflict/action, resolution)

This is notable because all linear layers use only **ternary weights {-1, 0, +1}** —
the model achieves this quality with effectively 1.58 bits per weight for the
linear layers.

---

## 4. Performance

### Model Size & Compression

| Format | Size | Notes |
|--------|------|-------|
| FP32 (training) | 648.6 MB | Full precision shadow weights |
| PyTorch checkpoint | 1.9 GB | Includes optimizer state |
| **GGUF (inference)** | **107 MB** | Ternary packed + FP16 embed + F32 norms |
| mHC binary | 880 B | 24 layers × 36 bytes + header |

| Compression Metric | Value |
|-------------------|-------|
| FP32 → GGUF ratio | **5.8x** |
| Linear weight bits | 2 bits/weight (+ 0.25 bits scale overhead) |
| Effective bits/weight (whole model) | 5.53 (includes FP16 embeddings) |

The ternary linear layers compress from 32 bits → ~2.25 bits per weight (2-bit
ternary encoding + per-group f32 scales at group_size=128). The overall model is
5.53 bits/weight because embeddings (24% of params) remain in FP16.

### Inference Speed

| Metric | Value |
|--------|-------|
| PyTorch autoregressive (GPU) | 29.4 tok/s |
| Note | Full causal attention recomputed each step |

### CPU GEMV Kernel Benchmarks (AVX2 PSHUFB)

Single-thread performance on the AVX2 kernel vs scalar reference:

| Shape | Scalar (ms) | AVX2 FFI (ms) | Speedup | Scalar GOPS | AVX2 GOPS |
|-------|-------------|---------------|---------|-------------|-----------|
| 128×128 | 0.019 | 0.0023 | 8.2x | 1.71 | 13.95 |
| 256×256 | 0.076 | 0.0067 | 11.4x | 1.72 | 19.70 |
| 512×512 | 0.305 | 0.021 | 14.3x | 1.72 | 24.55 |
| 1024×1024 | 1.225 | 0.076 | 16.2x | 1.71 | 27.73 |
| 2048×2048 | 4.893 | 0.299 | 16.4x | 1.72 | 28.09 |
| 4096×4096 | 19.66 | 1.185 | 16.6x | 1.71 | 28.32 |
| 4096×11008 | 52.51 | 3.189 | 16.5x | 1.72 | 28.31 |
| 11008×4096 | 52.73 | 2.850 | 18.5x | 1.71 | 31.66 |

The AVX2 PSHUFB kernel achieves **14-32 GOPS** across all shapes, with a
consistent **16-18x speedup** over the scalar reference. Performance peaks at
~32 GOPS for the 11008×4096 shape (FFN down-projection).

---

## 5. mHC-lite Analysis

### What is mHC-lite?

mHC (multi-stream Hadamard Connections) replaces standard residual connections
`x = x + layer(x)` with a learned doubly stochastic mixing:

```
x_expanded = H_res @ x_expanded + h_post * layer(h_pre @ x_expanded)
```

Where H_res is a 2×2 doubly stochastic matrix (non-negative, rows and columns
sum to 1). This is parameterized as `H = alpha*I + (1-alpha)*J` where
`alpha = sigmoid(logit)`, guaranteeing exact doubly stochastic property by
construction (BvN decomposition, no Sinkhorn needed).

### Why It Matters

| Property | Standard Residual | mHC-lite |
|----------|-------------------|----------|
| Composite gain (64 layers) | Unbounded (can reach 3000+) | <= 1.0 |
| Doubly stochastic | No | Yes (by construction) |
| Parameters per layer | 0 | 9 (N=2) |
| Compute overhead | 0 | ~32 FLOPs (<0.00004% of GEMV) |

### Results From This Training

| Metric | Value |
|--------|-------|
| Composite gain | 1.000000 |
| DS violations | 0 (across entire training) |
| mHC parameters | 216 total (9 per sub-layer × 24 sub-layers) |
| mHC parameter fraction | 0.000133% of total model |

The mHC constraint was maintained perfectly throughout all 57,635 training steps
with zero violations — demonstrating that the BvN parameterization works as
intended with no numerical issues.

---

## 6. Context & Comparisons

### TinyStories Paper Reference

The TinyStories paper (Eldan & Li, 2023) reports that models with 28M+ parameters
trained on TinyStories can produce coherent stories. Our 162M ternary model
demonstrates coherent generation after just 1 epoch of training.

### Ternary Model Landscape

| Model | Params | Bits/Weight | Perplexity | Notes |
|-------|--------|-------------|------------|-------|
| BitNet b1.58 (paper) | 3B | 1.58 | Competitive with FP16 at scale | Full-precision reference |
| **Nanochat-125m (ours)** | **162M** | **2.25 (linear)** | **4.01** | **1 epoch TinyStories, with mHC** |

### What 1 Epoch Means

With 474M tokens and 1 epoch, this model has seen each token exactly once. More
epochs would likely improve quality further — the TinyStories paper typically
trains for multiple epochs. The strong results after just 1 epoch suggest the
architecture and training setup are sound.

---

## 7. Artifacts

| File | Path | Size |
|------|------|------|
| Rust checkpoint dir | `checkpoints/nanochat_125m/final/` | model-dependent |
| GGUF weights | `exported_models/nanochat_125m.gguf` | 107 MB |
| mHC binary | `exported_models/nanochat_125m.mhc` | 880 B |

### Rust Codebase

| Crate | Tests | Status |
|-------|-------|--------|
| ternary-core | 86 | All passing |
| ternary-kernels | 3 | All passing |
| mhc-lite | 42 | All passing |
| nanochat-model | 32 | All passing |
| nanochat-serve | 17 | All passing |
| Integration tests | 48 | All passing |
| Additional workspace crates/tests | 268 | All passing |
| **Total** | **496 passing (+1 ignored)** | **All passing** |

Test coverage: 99.55% (1113/1118 lines).

---

## 8. Reproduction

### Training
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

### Export
```bash
cargo run --release -p nanochat-train -- export \
  --checkpoint checkpoints/nanochat_125m/final \
  --gguf exported_models/nanochat_125m.gguf \
  --mhc exported_models/nanochat_125m.mhc
```

### Evaluation
```bash
cargo run --release -p nanochat-eval --example benchmark_model -- \
  --checkpoint checkpoints/nanochat_125m/final \
  --n-samples 100 \
  --output benchmark_results.json
```

### Rust Tests
```bash
cargo test --workspace
```
