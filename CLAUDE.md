# CLAUDE.md — nanochat-rs Ternary + mHC-lite Implementation Plan

> **For Claude Code.** Execute phases in order. Each phase has a test gate —
> do not proceed to the next phase until the gate passes. All code lives in a
> Cargo workspace rooted at `nanochat-rs-ternary/`. Hardware target: dual AMD
> EPYC 9654 (224T, 1TB DDR5) + NVIDIA RTX PRO 6000 Blackwell (96GB).

---

## Repository Layout (create first)

```
nanochat-rs-ternary/
├── Cargo.toml                    # workspace root
├── CLAUDE.md                     # this file
├── crates/
│   ├── ternary-core/             # Phase 1: packing, planar SoA, GGUF
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── encode.rs         # BitNet encoding (11=-1)
│   │       ├── pack.rs           # pack/unpack, group quantization
│   │       ├── planar.rs         # PlanarWeights: SoA layout, aligned alloc
│   │       ├── gguf.rs           # GGUF reader/writer for ternary types
│   │       └── verify.rs         # Triangle of Truth (Rust side)
│   │
│   ├── ternary-kernels/          # Phase 2: CPU + GPU compute kernels
│   │   ├── Cargo.toml
│   │   ├── build.rs              # cc::Build for C kernels, cuda compilation
│   │   ├── csrc/
│   │   │   ├── ternary_gemv.c    # adapted from ternary_final.c (v3.3.1)
│   │   │   ├── ternary_gemv.h    # C API header
│   │   │   └── ternary_dp4a.cu   # GPU decode kernel
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── cpu.rs            # safe Rust wrappers over C FFI
│   │       ├── gpu.rs            # CUDA kernel launch wrappers
│   │       └── dispatch.rs       # runtime CPU feature detection + dispatch
│   │
│   ├── mhc-lite/                 # Phase 3: mHC-lite residual connections
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── n2.rs             # MhcLiteN2 (1 param, identity ↔ swap)
│   │       ├── n4.rs             # MhcLiteN4 (24 perms, full BvN)
│   │       ├── verify.rs         # doubly stochastic checks, composite gain
│   │       └── io.rs             # binary serialization (matches Python export)
│   │
│   ├── nanochat-model/           # Phase 4: transformer architecture
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs         # model configs (d20, 7B, 25B-MoE, 80B-MoE)
│   │       ├── embed.rs          # token + position embeddings
│   │       ├── norm.rs           # RMSNorm
│   │       ├── attention.rs      # MHA / GQA / MLA (with DeltaNet option)
│   │       ├── ffn.rs            # SwiGLU FFN (with MoE option)
│   │       ├── bitlinear.rs      # BitLinear: ternary GEMV dispatch
│   │       ├── block.rs          # TransformerBlock with mHC wiring
│   │       └── model.rs          # full model: embed → blocks → head
│   │
│   └── nanochat-serve/           # Phase 6: inference server
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs           # CLI + Axum HTTP server
│           ├── engine.rs         # KV-cache, sampling, batched decode
│           └── api.rs            # OpenAI-compatible /v1/chat/completions
│
├── training/                     # Phase 5: PyTorch training
│   ├── mhc_lite.py              # mHC-lite module (BvN, exact DS)
│   ├── ternary_qat.py           # BitLinear STE, absmean quantization
│   ├── model.py                 # nanochat architecture in PyTorch
│   ├── train.py                 # training loop (Muon+Lion, WSD schedule)
│   ├── export.py                # PyTorch → GGUF + mHC binary export
│   └── requirements.txt
│
├── tests/                        # Integration tests
│   ├── triangle_of_truth.rs     # cross-validate all kernel paths
│   ├── mhc_property_tests.rs    # doubly stochastic invariants
│   ├── roundtrip_test.rs        # pack → GGUF → load → GEMV → verify
│   └── e2e_generate.rs          # full model forward pass sanity
│
└── benches/
    ├── gemv_bench.rs            # criterion benchmarks for all kernel paths
    └── mhc_overhead.rs          # verify mHC adds <0.001% overhead
```

---

## Phase 1: Ternary Core (`ternary-core`)

### Goal
Bit-exact ternary packing, planar SoA storage, and GGUF I/O. This is the
foundation everything else depends on.

### 1.1 Create workspace and crate

```bash
mkdir -p nanochat-rs-ternary/crates/ternary-core/src
cd nanochat-rs-ternary
```

**`Cargo.toml`** (workspace root):
```toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.dependencies]
bytemuck = { version = "1.14", features = ["derive"] }
memmap2 = "0.9"
rayon = "1.8"
half = "2.3"
```

**`crates/ternary-core/Cargo.toml`**:
```toml
[package]
name = "ternary-core"
version = "0.1.0"
edition = "2021"

[dependencies]
bytemuck = { workspace = true }
memmap2 = { workspace = true }
rayon = { workspace = true }
```

### 1.2 Implement `encode.rs`

Encoding table (BitNet standard, GGUF-compatible):
```
-1 → 0b11
 0 → 0b00
+1 → 0b01
0b10 → invalid (decode as 0, but flag in verification)
```

Functions to implement:
- `encode_trit(val: i8) -> u8` — value to 2-bit code
- `decode_trit(bits: u8) -> i8` — 2-bit code to value
- `encode_trit_branchless(val: i8) -> u8` — branchless variant for SIMD prep

### 1.3 Implement `pack.rs`

Port from existing `ternary_pack.rs`. Functions:
- `pack_group(weights: &[f32]) -> (Vec<u8>, f32)` — quantize + pack 128 floats
- `unpack_group(packed: &[u8], scale: f32) -> Vec<f32>` — reconstruct
- `pack_matrix(weights: &[f32], rows: usize, cols: usize, group_size: usize) -> PackedMatrix`
- Internal: `quantize_one()`, `pack_4()`, `unpack_4()`

### 1.4 Implement `planar.rs` — THE critical data structure

This is the #1 production-reality fix from the review. **All runtime weight
access uses planar SoA layout, not interleaved.**

```rust
/// Planar SoA weight storage — THE runtime format.
///
/// WRONG: [32 packed bytes][2-byte scale][32 packed bytes][2-byte scale]...
///   → 34-byte stride, misaligns AVX-512, causes segfaults
///
/// RIGHT: weights[] contiguous + aligned, scales[] contiguous + aligned
///   → Maximum bandwidth, natural alignment
pub struct PlanarWeights {
    /// Packed ternary bytes, column-major, 128-byte aligned.
    /// Layout: weights[col * rows_padded + row_byte]
    /// rows_padded = round_up(rows / 4, 64) to enable NT-loads.
    pub weights: AlignedVec<u8>,

    /// Per-group scales, contiguous f32 array, 128-byte aligned.
    /// Layout: scales[col * n_groups + group_idx]
    /// where n_groups = ceil(rows / group_size)
    pub scales: AlignedVec<f32>,

    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub rows_padded: usize,
}
```

Implement:
- `AlignedVec<T>` — wrapper around `posix_memalign` / `aligned_alloc` with 128-byte alignment
- `PlanarWeights::from_row_major(data: &[f32], rows, cols, gs) -> Self`
- `PlanarWeights::from_packed(weights: &[u8], scales: &[f32], rows, cols, gs) -> Self`
- Column-major transpose during construction (critical for VPERMW kernel)
- `rows_padded = round_up(rows / 4, 64)` — enables 64-byte aligned NT-loads

### 1.5 Implement `gguf.rs`

Minimal GGUF reader that can:
- Parse header + metadata
- Extract tensor descriptors
- Memory-map tensor data
- Repack from GGUF interleaved → `PlanarWeights` SoA on load
- Write ternary tensors back to GGUF

Use `memmap2` for zero-copy loading. Register custom quant types:
```
GGUF_TYPE_Q1_58 = 100  (or whatever ID is available)
```

### 1.6 Implement `verify.rs`

Triangle of Truth verification harness (Rust side):
- `verify_gemv(pw: &PlanarWeights, x: &[i8], expected_y: &[f32], tol: f32) -> TestResult`
- Scalar reference implementation (bit-exact)
- Shape torture: test non-aligned M (M%32 != 0), various K
- Invalid codepoint injection: plant 0b10 patterns, assert decode=0
- Scale-only correctness: isolate per-group scaling, check error bounds

### Test Gate 1
```bash
cargo test -p ternary-core
```
Must pass:
- [x] Encode/decode roundtrip for all values {-1, 0, +1}
- [x] Invalid codepoint 0b10 decodes to 0
- [x] `pack_group` → `unpack_group` roundtrip within scale tolerance
- [x] PlanarWeights column-major layout correct (spot-check indices)
- [x] AlignedVec allocations are 128-byte aligned
- [x] GGUF write → read roundtrip preserves tensor data

---

## Phase 2: Ternary Kernels (`ternary-kernels`)

### Goal
Production CPU GEMV via FFI to C (v3.3.1 kernels), plus GPU decode kernel.
**VPERMW is primary.** Not LUT-Grouped.

### 2.1 Integrate `ternary_final.c` via FFI

Copy the uploaded `ternary_final.c` (v3.3.1) into `csrc/ternary_gemv.c`.
Create `csrc/ternary_gemv.h`:

```c
#pragma once
#include <stdint.h>

typedef struct {
    const uint8_t *weights;  // packed ternary, 128B aligned
    const float   *scales;   // per-group scales, 128B aligned
    int rows;                // M (output dim)
    int cols;                // K (input dim)  
    int group_size;          // typically 128
    int rows_padded;         // round_up(M, 64) for NT-loads
} PlanarWeightsC;

// Unified entrypoint — self-initializing, selects best kernel at first call
void ternary_gemv(
    const PlanarWeightsC *pw,
    const int8_t  *x,        // quantized activations [K]
    float act_scale,          // activation scale factor
    float *y                  // output [M], caller-allocated
);

// Expose for testing
void gemv_dp4a_ref(const PlanarWeightsC *pw, const int8_t *x, float act_scale, float *y);
```

**`build.rs`**:
```rust
fn main() {
    // CPU kernels — compile with native arch detection
    cc::Build::new()
        .file("csrc/ternary_gemv.c")
        .flag("-O3")
        .flag("-march=native")        // enables AVX-512 if available
        .flag("-fno-strict-aliasing")
        .flag("-DNDEBUG")
        .compile("ternary_gemv");

    println!("cargo:rerun-if-changed=csrc/ternary_gemv.c");
    println!("cargo:rerun-if-changed=csrc/ternary_gemv.h");

    // GPU kernel (optional — gate on CUDA toolkit presence)
    if std::env::var("CUDA_PATH").is_ok() || std::path::Path::new("/usr/local/cuda").exists() {
        // Use cc or custom nvcc invocation
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
}
```

### 2.2 Safe Rust FFI wrappers (`cpu.rs`)

```rust
extern "C" {
    fn ternary_gemv(pw: *const PlanarWeightsC, x: *const i8, act_scale: f32, y: *mut f32);
    fn gemv_dp4a_ref(pw: *const PlanarWeightsC, x: *const i8, act_scale: f32, y: *mut f32);
}

/// Safe wrapper — validates alignment, dimensions, dispatches to C
pub fn gemv(pw: &PlanarWeights, x: &[i8], act_scale: f32, y: &mut [f32]) {
    assert_eq!(x.len(), pw.cols);
    assert_eq!(y.len(), pw.rows);
    assert!(pw.weights.as_ptr() as usize % 128 == 0, "weights not 128B aligned");
    assert!(pw.scales.as_ptr() as usize % 128 == 0, "scales not 128B aligned");

    let c_pw = PlanarWeightsC { /* ... convert ... */ };
    unsafe { ternary_gemv(&c_pw, x.as_ptr(), act_scale, y.as_mut_ptr()); }
}
```

### 2.3 Dispatch chain documentation (`dispatch.rs`)

The dispatch chain (handled internally by the C code, but document for Rust callers):

```
Priority  Kernel           Requirement        Performance
────────  ───────────────  ─────────────────  ────────────────────
1         VPERMW fused     AVX-512 BW         19-36 GOPS (PRIMARY)
2         Dual-VPERMB      AVX-512 VBMI       16-26 GOPS
3         LUT-Grouped      Any x86_64         5-13 GOPS (degrades at large K)
4         Scalar dp4a-ref  Portable           16-18 GOPS (surprisingly good)
```

**VPERMW is primary even when VBMI is available** because the lo/hi byte
recombination overhead exceeds VPERMB's theoretical throughput advantage.

### 2.4 GPU kernel (`ternary_dp4a.cu`)

Port from existing `ternary_dp4a_decode.cu`. Two paths:

**Decode (N=1, autoregressive):**
- dp4a + constant-memory 256×i32 LUT
- Bandwidth-bound — simpler is better
- One thread per output row, warp-parallel reduction over K

**Prefill (N>16, prompt processing):**
- CUTLASS + custom ternary B-operand iterator → mma/wgmma
- Compute-bound when N is large enough
- Do NOT hand-write PTX — use CUTLASS abstractions

**Important:** LOP3 alone doesn't expand bits→bytes. The "INT8 masquerade via LOP3"
idea is insufficient as a complete solution. Use dp4a + LUT for decode.

### 2.5 Multi-threaded GEMV for CPU inference

```rust
/// Parallel GEMV: split M dimension across threads, each calls ternary_gemv
/// on its row slice. Uses rayon for work-stealing.
pub fn gemv_parallel(pw: &PlanarWeights, x: &[i8], act_scale: f32, y: &mut [f32]) {
    // For autoregressive (N=1): partition rows across threads
    // Each thread handles a contiguous row block
    // NUMA-aware: pin threads to socket that owns the weight slab
    y.par_chunks_mut(ROWS_PER_THREAD).enumerate().for_each(|(i, y_chunk)| {
        let row_start = i * ROWS_PER_THREAD;
        // ... call C kernel on row slice
    });
}
```

### Test Gate 2
```bash
cargo test -p ternary-kernels
```
Must pass:
- [x] FFI compiles and links (even without AVX-512 — falls back to scalar)
- [x] GEMV output matches scalar reference (bit-identical f32)
- [x] Shape torture: M=127, M=1, M=4096; K=128, K=11008
- [x] Invalid codepoint test (0b10 injected → output matches reference with 0)
- [x] Scale-only test: random weights/acts, max error < 1e-5 per group

```bash
cargo bench -p ternary-kernels
```
Record: GOPS at (2048², 4096², 4096×11008, 11008×4096, 8192²).
Compare against v3 spec measured numbers.

---

## Phase 3: mHC-lite (`mhc-lite`)

### Goal
Exact Birkhoff-von-Neumann doubly stochastic residual connections.
N=2 first (1 parameter per layer), N=4 later (24 permutations).
FP32 throughout — never quantize these to ternary.

### 3.1 Implement `n2.rs`

Port from `mhc_lite.rs` already created. Key struct:

```rust
pub struct MhcLiteN2 {
    pub alpha_logit: f32,       // sigmoid → α for identity↔swap interpolation
    pub pre_logits: [f32; 2],
    pub pre_bias: [f32; 2],
    pub post_logits: [f32; 2],
    pub post_bias: [f32; 2],
}
```

Functions:
- `h_res() -> [[f32; 2]; 2]` — doubly stochastic mixing matrix
- `h_pre() -> [f32; 2]` — non-negative pre-projection
- `h_post() -> [f32; 2]` — non-negative post-projection (2× scaled)
- `expand_input(x, dim) -> expanded` — [B, C] → [B, 2C]
- `prepare_input(expanded, dim) -> single` — [B, 2C] → [B, C] via h_pre
- `apply(expanded, layer_out, dim) -> expanded` — residual update
- `collapse_output(expanded, dim) -> single` — [B, 2C] → [B, C] (average)
- `from_bytes() / to_bytes()` — 36-byte serialization

### 3.2 Implement `n4.rs`

Full BvN parameterization with all 24 permutation matrices of S₄:

```rust
const PERMS_S4: [[usize; 4]; 24] = [ /* all 24 permutations */ ];

pub struct MhcLiteN4 {
    pub res_logits: [f32; 24],  // softmax → convex weights over 24 perms
    pub pre_logits: [f32; 4],
    pub pre_bias: [f32; 4],
    pub post_logits: [f32; 4],
    pub post_bias: [f32; 4],
}
```

`h_res()` computes: H = Σ_k softmax(logits)_k · P_k

This is **exact** doubly stochastic — no Sinkhorn iterations, no approximation gap.

### 3.3 Implement `verify.rs`

- `verify_doubly_stochastic(mat, tol)` — check non-neg, row/col sums = 1
- `composite_amax_gain(matrices)` — product of all H_res, measure Amax
  - For exact DS matrices, composite gain ≤ 1.0
  - mHC paper shows unconstrained HC reaches 3000+ at depth 64
- Property test: generate 1000 random logit vectors, verify all produce valid DS matrices

### 3.4 Implement `io.rs`

Binary serialization matching the Python `export_mhc_weights()` format:

```
Header: [magic:u32 = 0x6D484321][version:u32 = 1][n_layers:u32][n_streams:u32]
Per layer (N=2): 36 bytes — [alpha_logit][pre_logits×2][pre_bias×2][post_logits×2][post_bias×2]
Per layer (N=4): 160 bytes — [res_logits×24][pre_logits×4][pre_bias×4][post_logits×4][post_bias×4]
```

### Test Gate 3
```bash
cargo test -p mhc-lite
```
Must pass:
- [x] N=2: identity init produces H_res ≈ I (diagonal > 0.99)
- [x] N=2: all logits in [-10, 10] produce valid DS matrices (tol=1e-6)
- [x] N=4: random logits produce valid DS matrices (tol=1e-5)
- [x] N=4: composite gain over 64 random layers ≤ 1.0 + 1e-4
- [x] Serialization roundtrip (to_bytes → from_bytes) exact
- [x] Cross-validation: Rust output matches Python mhc_lite.py output for same inputs

---

## Phase 4: Model Architecture (`nanochat-model`)

### Goal
Full transformer model using ternary BitLinear layers with mHC-lite residual
connections. Supports d20 (560M), 7B, 25B-MoE, 80B-MoE configs.

### 4.1 Implement `config.rs`

```rust
pub struct ModelConfig {
    pub dim: usize,              // hidden dimension
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,       // for GQA
    pub ffn_mult: f32,           // SwiGLU intermediate = dim * ffn_mult
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub group_size: usize,       // ternary quantization group (128)
    pub mhc_n_streams: usize,    // 2 or 4
    // MoE config (optional)
    pub n_experts: Option<usize>,
    pub n_active_experts: Option<usize>,
    // Hybrid attention (optional)
    pub deltanet_ratio: Option<f32>,  // fraction of layers using DeltaNet
}
```

Presets: `ModelConfig::d20()`, `ModelConfig::nano_7b()`, `ModelConfig::moe_25b()`, `ModelConfig::moe_80b()`

### 4.2 Implement `bitlinear.rs`

The core ternary linear layer for inference:

```rust
pub struct BitLinear {
    pub pw: PlanarWeights,   // ternary packed, planar SoA
    pub rows: usize,         // output dim
    pub cols: usize,         // input dim
}

impl BitLinear {
    /// Forward pass: quantize activations → ternary GEMV → output
    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        // 1. Activation quantization: per-token absmax → INT8
        let (x_q, act_scale) = quantize_activations_i8(x);
        // 2. Ternary GEMV dispatch (calls into ternary-kernels)
        ternary_kernels::cpu::gemv(&self.pw, &x_q, act_scale, y);
    }
}
```

Activation quantization (per-token absmax):
```rust
fn quantize_activations_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let absmax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = absmax / 127.0;
    let inv_scale = if absmax > 0.0 { 127.0 / absmax } else { 0.0 };
    let x_q: Vec<i8> = x.iter().map(|v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8).collect();
    (x_q, scale)
}
```

### 4.3 Implement `block.rs` — Transformer block with mHC wiring

This is the critical integration point:

```rust
pub struct TransformerBlock {
    // mHC for attention sub-layer
    mhc_attn: MhcLiteN2,  // or N4
    // mHC for FFN sub-layer
    mhc_ffn: MhcLiteN2,

    // Pre-norms
    norm_attn: RMSNorm,
    norm_ffn: RMSNorm,

    // Attention projections (all BitLinear)
    wq: BitLinear,
    wk: BitLinear,
    wv: BitLinear,
    wo: BitLinear,

    // SwiGLU FFN (all BitLinear)
    w_gate: BitLinear,
    w_up: BitLinear,
    w_down: BitLinear,
}
```

Forward pass architecture per block:
```
x_expanded ──┬── [mhc_attn.prepare_input] → RMSNorm → Attention → [mhc_attn.apply]
             │                                                          │
             └── [mhc_attn.h_res residual] ────────────────────────────→ + → x_expanded

x_expanded ──┬── [mhc_ffn.prepare_input] → RMSNorm → SwiGLU_FFN → [mhc_ffn.apply]
             │                                                          │
             └── [mhc_ffn.h_res residual] ─────────────────────────────→ + → x_expanded
```

**mHC matrices stay FP32.** Everything else is ternary. The mHC overhead is
~32 FLOPs per layer (n=4) vs ~90M FLOPs for a single ternary GEMV at 4096×11008.
That's <0.00004% of layer compute.

### 4.4 Implement `model.rs`

```rust
pub struct NanochatModel {
    pub config: ModelConfig,
    pub tok_embed: Embedding,          // FP32 or FP16 (not ternary)
    pub blocks: Vec<TransformerBlock>,
    pub norm_final: RMSNorm,
    pub lm_head: BitLinear,            // or weight-tied with tok_embed
    // mHC expand/collapse happens at model boundaries
}

impl NanochatModel {
    pub fn forward(&self, token_ids: &[u32], pos: usize) -> Vec<f32> {
        // 1. Embed
        let mut x = self.tok_embed.forward(token_ids);

        // 2. Expand to multi-stream
        let mut x_exp = MhcLiteN2::expand_input(&x, self.config.dim);

        // 3. Transformer blocks
        for block in &self.blocks {
            x_exp = block.forward(&x_exp);
        }

        // 4. Collapse back to single stream
        x = MhcLiteN2::collapse_output(&x_exp, self.config.dim);

        // 5. Final norm + LM head
        self.norm_final.forward_inplace(&mut x);
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.lm_head.forward(&x, &mut logits);
        logits
    }
}
```

### 4.5 Weight loading

Load from GGUF + mHC binary:
```rust
impl NanochatModel {
    pub fn load(gguf_path: &str, mhc_path: &str) -> Self {
        let gguf = GgufFile::open(gguf_path);   // memory-mapped
        let mhc_params = MhcParams::load(mhc_path);

        // For each BitLinear: extract packed weights from GGUF,
        // repack to PlanarWeights SoA at load time
        // ...
    }
}
```

### Test Gate 4
```bash
cargo test -p nanochat-model
```
Must pass:
- [x] BitLinear forward produces same output as scalar reference GEMV
- [x] TransformerBlock with mHC produces finite, non-zero output
- [x] Full model forward pass (d20 config, random weights) produces valid logits
- [x] mHC expand → N blocks → collapse preserves tensor shape correctly
- [x] All mHC H_res matrices in model are doubly stochastic (runtime check)

---

## Phase 5: Training Pipeline (`training/`)

### Goal
PyTorch training with ternary QAT (STE) + mHC-lite, exporting to Rust format.
This is Python, runs on the RTX PRO 6000.

### 5.1 `mhc_lite.py`

Already created. Contains:
- `MhcLiteN2` — 2-stream, sigmoid-based α interpolation
- `MhcLiteN4` — 4-stream, full BvN with 24 perms
- `MhcLiteLayer` — generic wrapper
- `MhcTransformerBlock` — complete block with mHC wiring
- `measure_composite_gain()` — diagnostic
- `export_mhc_weights()` — binary export matching Rust `io.rs` format

### 5.2 `ternary_qat.py`

Ternary Quantization-Aware Training:

```python
class BitLinearSTE(nn.Module):
    """FP32 shadow weights → ternary via STE during training."""

    def __init__(self, in_features, out_features, group_size=128):
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.group_size = group_size

    def forward(self, x):
        # Quantize weight to ternary with STE
        w_ternary, scales = absmean_quantize(self.weight, self.group_size)
        w_q = self.weight + (w_ternary * expand_scales(scales) - self.weight).detach()
        # Quantize activations to INT8 with STE
        x_q, act_scale = per_token_absmax_quantize(x)
        x_ste = x + (x_q * act_scale - x).detach()
        return F.linear(x_ste, w_q)
```

### 5.3 `model.py`

PyTorch nanochat model with mHC integration:

```python
class NanochatTernary(nn.Module):
    def __init__(self, config):
        self.tok_embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            MhcTransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                n_streams=config.mhc_n_streams,
            ) for _ in range(config.n_layers)
        ])
        self.norm_final = RMSNorm(config.dim)
        self.lm_head = BitLinearSTE(config.dim, config.vocab_size)
```

### 5.4 `train.py`

Training loop with Muon + Lion optimizer split:

```python
# Parameter groups:
# - Linear weights (2D+) → Muon (lr=0.02, momentum=0.95)
# - mHC params + norms + biases → Lion (lr=1e-4, wd=0.1)
# - Embeddings → Lion (lr=1e-4, wd=0.0)

# Schedule: Warmup-Stable-Decay (WSD)
# - Warmup: 2000 steps, linear from 0 → lr
# - Stable: 80% of training at lr
# - Decay: cosine anneal to 0.1 * lr

# Logging:
# - Every 100 steps: loss, grad norms (separate for Muon/Lion groups)
# - Every 500 steps: measure_composite_gain() — alert if > 2.0
# - Every 1000 steps: checkpoint (FP32 shadows + mHC params)
```

### 5.5 `export.py`

Convert trained checkpoint to inference format:

```python
def export(checkpoint_path, gguf_path, mhc_path):
    model = load_checkpoint(checkpoint_path)

    # 1. Extract final ternary weights + scales
    for layer in model.modules():
        if isinstance(layer, BitLinearSTE):
            w_ternary, scales = absmean_quantize(layer.weight, 128)
            # Pack to 2-bit, write to GGUF

    # 2. Export mHC parameters
    export_mhc_weights(model, mhc_path)

    # 3. Export embeddings (FP16), norms (FP32)
    # ... append to GGUF
```

### Test Gate 5
```bash
cd training && python -m pytest test_training.py
```
Must pass:
- [x] `MhcLiteN2` + `MhcLiteN4` produce DS matrices for random logits
- [x] Composite gain ≤ 1.0 after 64 random layers
- [x] STE gradient flows through ternary quantization (grad is not None for all params)
- [x] Single training step runs without error (tiny model, 1 batch)
- [x] Export → load in Rust produces same logits (cross-language validation)

---

## Phase 6: Inference Server (`nanochat-serve`)

### Goal
CLI + HTTP API for inference, with KV-cache and sampling.

### 6.1 `engine.rs`

```rust
pub struct InferenceEngine {
    model: NanochatModel,
    kv_cache: Vec<KvCache>,  // per-layer
    // NUMA config for dual-socket EPYC
    thread_pools: [rayon::ThreadPool; 2],  // one per socket
}

impl InferenceEngine {
    pub fn generate(&mut self, prompt_ids: &[u32], max_tokens: usize) -> Vec<u32> {
        // Prefill: process all prompt tokens (batched)
        // Decode: one token at a time, autoregressive
    }
}
```

### 6.2 NUMA optimization (dual-socket EPYC)

```rust
// Two thread pools pinned per socket
// Weights allocated with first-touch on the socket that consumes them
// Activations duplicated per-socket (tiny: just 1 token × dim)
// Split model layers across sockets: layers 0..N/2 on socket 0, N/2..N on socket 1
```

### 6.3 `api.rs` — OpenAI-compatible API

```rust
// POST /v1/chat/completions
// Streaming SSE responses
// Temperature, top-p, top-k sampling
```

### 6.4 Huge pages + non-temporal loads (production optimization)

```rust
// Huge pages (2MB) for weight storage:
// madvise(ptr, len, MADV_HUGEPAGE) after mmap

// Non-temporal loads require 64-byte alignment:
// rows_padded = round_up(rows, 64) ensures every column access is aligned
// _mm512_stream_load_si512 for weight streaming (avoids polluting L1/L2)
```

### Test Gate 6
```bash
cargo test -p nanochat-serve
cargo run -p nanochat-serve -- --model test_model.gguf --mhc test_model.mhc --prompt "Hello"
```
Must pass:
- [x] CLI generates coherent text (even with random weights — should produce tokens)
- [x] HTTP API responds to /v1/chat/completions
- [x] KV-cache produces same output as full recompute (verify on short seq)

---

## Phase 7: Integration Tests + Benchmarks

### 7.1 `tests/triangle_of_truth.rs`

Cross-validate ALL kernel paths produce identical output:
```rust
#[test]
fn triangle_of_truth() {
    for (m, k) in [(128, 128), (2048, 2048), (4096, 11008), (127, 128), (1, 4096)] {
        let pw = random_planar_weights(m, k, 128);
        let x = random_activations(k);

        let y_scalar = gemv_scalar_ref(&pw, &x, 1.0);
        let y_ffi = gemv_ffi(&pw, &x, 1.0);

        assert_bitexact(&y_scalar, &y_ffi, m);
    }
}
```

### 7.2 `tests/mhc_property_tests.rs`

```rust
#[test]
fn mhc_n4_always_doubly_stochastic() {
    for seed in 0..1000 {
        let logits = random_logits_24(seed);
        let mhc = MhcLiteN4::from_random_logits(logits);
        let h = mhc.h_res();
        verify_doubly_stochastic(&h, 1e-5).unwrap();
    }
}

#[test]
fn mhc_composite_gain_bounded() {
    let matrices: Vec<_> = (0..64).map(|i| random_n4_h_res(i)).collect();
    let gain = composite_amax_gain(&matrices);
    assert!(gain <= 1.0 + 1e-4);
}
```

### 7.3 `tests/roundtrip_test.rs`

Full pipeline validation:
```rust
#[test]
fn python_to_rust_roundtrip() {
    // 1. Load test weights exported by Python export.py
    // 2. Load through GGUF reader → PlanarWeights
    // 3. Load mHC binary → MhcLiteN2
    // 4. Run forward pass
    // 5. Compare against Python's output (saved as .npy)
    // Tolerance: < 1e-4 (float rounding from pack/unpack)
}
```

### 7.4 Benchmarks

```bash
cargo bench
```

Expected results (single-thread, Zen4 EPYC):
```
Shape          VPERMW    Scalar-ref
2048²          ~30 GOPS  ~18 GOPS
4096²          ~25 GOPS  ~18 GOPS
4096×11008     ~20 GOPS  ~16 GOPS
11008×4096     ~20 GOPS  ~18 GOPS
```

mHC overhead benchmark: < 0.001% of total inference time.

---

## Execution Order Summary

```
Phase 1  ternary-core        2-3 days    Foundation: encoding, packing, planar SoA, GGUF
Phase 2  ternary-kernels     2-3 days    CPU GEMV (FFI to C), GPU kernel stub
Phase 3  mhc-lite            1-2 days    BvN doubly stochastic, N=2 + N=4
Phase 4  nanochat-model      3-4 days    Full transformer: BitLinear + mHC + attention + FFN
Phase 5  training/           3-5 days    PyTorch QAT + mHC, export pipeline
Phase 6  nanochat-serve      2-3 days    CLI + HTTP server, KV-cache, NUMA
Phase 7  integration tests   1-2 days    Triangle of Truth, property tests, benchmarks
```

Total: ~15-22 days for a solo developer with Claude Code assistance.

---

## Critical Constraints (Do Not Violate)

1. **VPERMW is primary CPU kernel.** Not LUT-Grouped. LUT-Grouped collapses at
   large K (1.3MB working set exceeds L1). VPERMW keeps 16-entry LUTs in registers.

2. **Planar SoA at runtime.** Never use interleaved [weights|scale] layout for
   compute. GGUF on disk can be interleaved; repack to planar on load.

3. **mHC matrices stay FP32.** They are tiny (768 params total for 48 layers, n=4).
   Quantizing them to ternary would break the doubly stochastic constraint.

4. **128-byte alignment for all weight buffers.** Required for AVX-512 NT-loads.
   Use `posix_memalign` or equivalent. `rows_padded = round_up(rows, 64)`.

5. **Encoding is BitNet standard: 11=-1, 01=+1, 00=0.** Not subtraction encoding.
   This is for GGUF/bitnet.cpp compatibility.

6. **Group size is 128.** This is the BitNet b1.58 standard. Don't change it
   without re-benchmarking the kernels (LUT sizes, accumulator overflow, etc.).

7. **Performance projections use 65% of peak bandwidth**, not 100%. Plan
   hardware needs and milestones around conservative numbers.

8. **Use `mhc-lite` (BvN), NOT Sinkhorn-Knopp.** BvN is exact by construction.
   Sinkhorn requires custom kernels and has an approximation gap.

9. **GPU decode uses dp4a + constant LUT.** GPU prefill uses CUTLASS. LOP3 alone
   does not expand bits→bytes.

10. **NUMA pinning is required for dual-socket EPYC.** First-touch allocation,
    thread pools per socket, weight sharding across NUMA domains.

---

## File Dependency Graph

```
ternary-core
    ├── encode.rs (standalone)
    ├── pack.rs (depends on encode.rs)
    ├── planar.rs (depends on pack.rs)
    ├── gguf.rs (depends on planar.rs)
    └── verify.rs (depends on all above)

ternary-kernels
    ├── csrc/ternary_gemv.c (standalone C, compiled by build.rs)
    ├── cpu.rs (FFI to C, depends on ternary-core::PlanarWeights)
    ├── gpu.rs (CUDA FFI, depends on ternary-core::PlanarWeights)
    └── dispatch.rs (wraps cpu.rs + gpu.rs)

mhc-lite
    ├── n2.rs (standalone)
    ├── n4.rs (standalone)
    ├── verify.rs (depends on n2, n4)
    └── io.rs (depends on n2, n4)

nanochat-model (depends on ternary-core, ternary-kernels, mhc-lite)
    ├── config.rs (standalone)
    ├── embed.rs, norm.rs (standalone)
    ├── bitlinear.rs (depends on ternary-kernels::cpu)
    ├── attention.rs (depends on bitlinear.rs)
    ├── ffn.rs (depends on bitlinear.rs)
    ├── block.rs (depends on attention, ffn, mhc-lite)
    └── model.rs (depends on all above)

nanochat-serve (depends on nanochat-model)
    ├── engine.rs (depends on nanochat-model)
    ├── api.rs (depends on engine.rs)
    └── main.rs (depends on all above)

training/ (Python, depends on nothing in Rust directly)
    ├── mhc_lite.py (standalone)
    ├── ternary_qat.py (standalone)
    ├── model.py (depends on mhc_lite, ternary_qat)
    ├── train.py (depends on model)
    └── export.py (depends on model — outputs GGUF + mHC binary for Rust)
```

---

## Per-Session Claude Code Instructions

When working on any phase, start each session by:

1. `cat CLAUDE.md` — re-read this plan
2. Check which test gates have passed: `cargo test --workspace 2>&1 | tail -20`
3. Work on the **lowest-numbered incomplete phase**
4. Run the phase's test gate before moving on
5. Commit after each passing test gate

When implementing a function:
- Write the function
- Write at least one test for it
- Run `cargo test` (or `pytest` for Python)
- Fix until green
- Move to next function

For C kernel integration:
- Do NOT modify `ternary_gemv.c` — it's battle-tested v3.3.1
- Only add `#include` guards and the header file for FFI
- Test via the Triangle of Truth harness
