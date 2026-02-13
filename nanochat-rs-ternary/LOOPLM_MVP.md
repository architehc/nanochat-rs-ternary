# LoopLM MVP Implementation Scope

**Paper:** arXiv:2510.25741 - LoopLM: Recurrent Loop Transformers

## MVP Goal
Implement LoopLM mechanics at **d20 scale** (~20M params) for proof-of-concept, not full 1.5B/7B training.

## Architecture Overview

### Core LoopLM Concept
Instead of stacking N unique layers, use a shallow "local" stack + a "shared" middle layer that loops multiple times:

```
Input → [Local Layer 1] → [Shared Loop Layer × L iterations] → [Local Layer 2] → Output
```

### MVP Configuration (d20 scale)
- **Total depth equivalent**: 6 layers (current d20 config)
- **Local layers**: 2 (1 before loop, 1 after loop)
- **Shared loop layer**: 1 layer, looped L=4 times
- **Effective depth**: 2 local + 4 loop iterations = 6 equivalent layers

### Key Components to Implement

#### 1. Loop-Aware Configuration
**Both** `TrainConfig` and `ModelConfig` need:
```rust
pub struct LoopConfig {
    /// Number of local (non-looped) layers before shared loop
    pub local_before: usize,
    /// Number of local (non-looped) layers after shared loop
    pub local_after: usize,
    /// Number of shared loop iterations (L in paper)
    pub loop_count: usize,
    /// Optional: adaptive loop control for inference
    pub adaptive_loop: Option<AdaptiveLoopConfig>,
}

pub struct AdaptiveLoopConfig {
    pub min_loops: usize,
    pub max_loops: usize,
    pub perplexity_threshold: f32,
}
```

#### 2. Shared Loop Block Architecture
Per paper Section 3.2:
- **Local transformations**: Q/K/V projections, FFN (unique per position)
- **Global recurrent updates**: Gate mechanisms (g_qk, g_ffn) that mix current with accumulated state

```rust
pub struct SharedLoopBlock {
    // Local projections (standard transformer sub-layers)
    local_qkv: BitLinear,
    local_ffn_gate: BitLinear,
    local_ffn_up: BitLinear,
    local_ffn_down: BitLinear,

    // Global gate projections (control recurrent mixing)
    global_gate_qk: BitLinear,   // g_qk gate
    global_gate_ffn: BitLinear,  // g_ffn gate

    // Norms
    norm_attn: RMSNorm,
    norm_ffn: RMSNorm,
}
```

#### 3. KV Cache Semantics Fix
**Problem**: Current code appends KV every attention call (line 159 in attention.rs)
**Solution**: Add `append_mode` flag:
- `PerToken`: Append KV once per token position (for autoregressive)
- `NoAppend`: Use existing cache (for inner loop iterations)

```rust
pub enum KVAppendMode {
    PerToken,    // Append KV to cache (normal autoregressive)
    NoAppend,    // Don't append, reuse cache (inner loop)
}
```

#### 4. Training Stages (Simplified for MVP)

**Stage 1 - Distillation** (Section 4.1):
- Train loop model to match teacher (standard transformer) outputs
- Loss: `L = KL(student_logits || teacher_logits) + loop_scale_penalty`
- Loop scale annealing: Start with high penalty, reduce to encourage deeper loops

**Stage 2 - SFT** (Section 4.2):
- Optional for MVP - focus on Stage 1 first
- Would require decomposition/reconstruction data format

**For MVP: Implement Stage 1 only**

#### 5. Export/Load Extensions
Extend GGUF metadata to include:
```
loop_config.local_before: u32
loop_config.local_after: u32
loop_config.loop_count: u32
loop_config.shared_layer_idx: u32  // Index of the shared loop layer
```

### Implementation Order (MVP)

1. ✅ **Define scope** (this document)
2. **Add loop config fields** to TrainConfig + ModelConfig
3. **Implement SharedLoopBlock** in both train and model crates
4. **Fix KV append semantics** with mode flag
5. **Add loop execution logic** in model forward pass
6. **Extend trainer** with basic Stage-1 distillation
7. **Update export/load** to serialize loop metadata
8. **Add tests**: loop parity, KV invariant, export roundtrip
9. **Run small training** (d20, 1K steps, verify convergence)
10. **Benchmark**: quality vs loop count, latency

### Success Criteria for MVP

- [ ] d20 model trains with loop config (local=2, shared=1, loops=4)
- [ ] KV cache length is correct (doesn't inflate per loop iteration)
- [ ] Export → Load → Inference produces consistent outputs
- [ ] Loop count can be varied at inference time (2-6 loops)
- [ ] Perplexity comparable to baseline 6-layer d20 model
- [ ] Tests pass: loop parity, KV invariant, export roundtrip

### Out of Scope for MVP

- Full 1.5B/7B scale training (too expensive for proof-of-concept)
- Stage-2 SFT with decomposition data
- Multi-loop layers (paper has complex nested loops)
- Chunk-based experts (Section 3.3)
- Production-grade adaptive stopping
- Long-context benchmarks (8K+ tokens)

### Timeline Estimate

- Steps 2-5 (core architecture): 2 days
- Steps 6-8 (training + export): 2 days
- Steps 9-10 (testing + benchmarks): 1 day
- **Total MVP**: ~5 days

---

## Next Steps

Start with **Step 2**: Add `LoopConfig` fields to both `TrainConfig` and `ModelConfig`.
