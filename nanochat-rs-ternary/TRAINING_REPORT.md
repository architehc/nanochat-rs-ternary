# Training Report — 2026-02-27/28

## Executive Summary

Dual-GPU training session on 2x RTX 4090 (23GB each). Three model architectures compared:

| Model | Architecture | Steps | Final Loss | Perplexity | Generation Quality |
|-------|-------------|-------|------------|-----------|-------------------|
| **haar-v5** | Wavefield Haar + mHC | 30,000 | **1.1714** | 3.22 | Garbled (wavefield bypass) |
| **engram-v1** | Engram N-gram + mHC | 10,000 | **2.1916** | 8.94 | Semi-coherent Rust code |
| **engram-v2** | Extended engram | 16,000 (in progress) | ~2.19 → improving | — | Expected better |
| **baseline-v1** | Standard + MTP | 10,000 (in progress) | — (just started) | — | — |

**Key finding**: Wavefield Haar achieves the lowest training loss but cannot generate coherent text because wavefield attention is bidirectional. The engram model with standard causal attention generates significantly better code despite 2x higher training loss.

## Training Configuration

### Hardware
- 2x NVIDIA RTX 4090 (23GB VRAM each)
- Host: WSL2 Linux, dual AMD EPYC / Intel platform

### Common Settings
- Dataset: ~97M tokens of Rust source code (BPE vocab=4096)
- Sequence length: 1024 tokens
- Batch size: 2
- Optimizer: Muon (2D+ weights) + Lion (1D/embedding params)
- LR schedule: Warmup-Stable-Decay (WSD)
- Label smoothing: 0.1
- Gradient clipping: 1.0

### Model Architectures (all ~275M params)

**haar-v5** (GPU 0):
- dim=1024, 20 layers, 16 heads, 4 KV heads (GQA 4:1)
- 50% wavefield Haar attention (10 wavefield + 10 standard layers)
- field_size=256, haar_levels=6, haar_direct=true
- Wavefield warmup delay: 1000 steps
- mHC-lite N=2 residual streams
- LR: 0.012, total_steps: 30000 (extended from 20000)
- Resumed from haar-v3 step 20000 (loss=1.1930)

**engram-v1** (GPU 1):
- dim=1024, 20 layers, 16 heads, 4 KV heads
- Standard causal attention (no wavefield)
- Engram N-gram memory on layers 0 and 19
  - d_mem=128, n_gram_orders=[2,3], table_size=10007
- mHC-lite N=2 residual streams
- LR: 0.012, total_steps: 10000

**baseline-v1** (GPU 0, Round 2):
- Same as haar-v5 but with use_wave_field=false
- MTP enabled (3 future tokens, weight=0.2)
- LR: 0.012, total_steps: 10000

## Training Progression

### haar-v5 (Wavefield Haar)

```
Step     Loss    H     LR        gnorm   Notes
20050    3.6955  3.77  0.012000  4.37    Resume from step 20000 (LR schedule fix)
20250    2.3986  2.74  0.012000  5.20    Rapid recovery
20500    1.2570  1.24  0.012000  0.80    Back to near-original loss
21000    1.2071  1.20  0.012000  0.24    Surpassed previous best (1.1930)
22000    1.2019  1.20  0.012000  0.29    Stable plateau
23000    1.1998  1.20  0.012000  0.23    Broke below 1.20
24000    1.1969  1.20  0.011993  0.41    Entered decay phase
25000    1.1925  1.19  0.010884  0.21    Steady improvement
26000    1.1863  1.18  0.008796  0.21
27000    1.1803  1.18  0.006317  0.15
28000    1.1763  1.18  0.003779  0.09
29000    1.1742  1.17  0.001559  0.06
30000    1.1714  1.17  0.001200  0.04    Final — all-time best
```

**Total training time**: 310 minutes (5.2 hours)
**Total steps this session**: 10,000 (20000 → 30000)
**Throughput**: ~555 tok/s average

### engram-v1 (Engram N-gram)

```
Step     Loss    H     LR        gnorm   Notes
50       8.3953  8.11  0.000400  17.14   Random init
500      5.6512  5.73  0.004000  5.13    Warmup phase
1000     4.6123  4.52  0.008000  3.02
1500     4.3562  4.02  0.010800  2.45    End of warmup
2000     4.1735  3.82  0.012000  2.10    Full LR
3000     3.5001  3.32  0.012000  2.43
4000     3.4568  3.36  0.012000  2.38
5000     3.2452  3.14  0.012000  2.66
6000     3.1680  3.09  0.012000  2.98
7000     3.1903  3.14  0.012000  3.59
8000     3.1920  3.14  0.011851  4.42    Entered decay
9000     2.8772  2.87  0.003426  3.79    Best non-final
10000    2.1916  2.19  0.001200  —       Final
```

**Total training time**: 295 minutes (4.9 hours)
**Total steps**: 10,000
**Throughput**: ~585 tok/s average

## Code Fixes Applied (This Session)

### Critical Fix: LR Schedule Extension
- `--total-steps` CLI flag now updates `config.total_steps` (not just `stop_at_step`)
- Previous behavior kept LR at minimum (0.0012) during extended training
- Fix allowed haar-v5 to train at full LR (0.012) in the stable phase

### Previous Session (18 Bug Fixes)
Key fixes included:
1. Entropy regularization tensor kept in computation graph
2. MTP parameters added to Lion optimizer group
3. QuantizedMuon detach to prevent memory leak
4. Device-aware checkpoint loading (CPU → CUDA)
5. Batch-size override on resume
6. NaN-safe sorting with f32::total_cmp()
7. ExitGate using post-FFN state
8. HASH_PRIMES replaced with actual primes
9. GQA validation (n_heads % n_kv_heads)
10. Final checkpoint loss tracking (was 0.0)

## Generation Quality Evaluation

### haar-v5 (loss=1.17) — Garbled output
```
fn main() { removed calledNErt L1.1 whereSE101>(&imest #[ED,ther128()?;
bit [_1direinkSelfth $ptr(agt down 12 bx_1_1_MM1 E(windows_fac...
```
**Root cause**: Wavefield attention is bidirectional. During causal generation,
wavefield layers are bypassed entirely (50% of attention capacity lost).
The model was trained with all 20 attention layers but only uses 10 at inference.

### engram-v1 (loss=2.19) — Semi-coherent Rust code
```rust
fn main() {
    let p = project()
        .file("Cargo.toml", r#"
            [package]
            name = "bar"
            version = "0.0.0"
            edition = "2015"
            authors = []
            [dependencies]
            bar = { path = "bar" }
        "#)
        .file("src/bar/src/main.rs", "")
        .build();
    p.cargo("check").run();
    p.cargo("check")
        .masquerade_as_nightly_cargo(&["build")
        .with_stderr_data(str![[r#"foo v0.0.1 ([ROOT]/...
```

The model recognizes:
- Cargo test harness patterns (project(), .file(), .cargo())
- Rust syntax (fn, pub, struct, impl, match, let, mut)
- Common patterns (Result, unwrap, Option, #[test])
- Lifetime annotations (&'static str)
- Derive macros (#[derive(Debug, Clone)])

Issues:
- Makes up types and method names
- Bracket matching not always correct
- Some repetition in output

## Key Findings

### 1. Wavefield Attention — High Training Efficiency, Zero Generation Utility
The wavefield Haar attention achieves the lowest training loss (1.17 vs 2.19) because
it can attend to all positions bidirectionally. However, this makes it completely
unsuitable for autoregressive generation. **Training loss is not comparable between
wavefield and standard attention models.**

Potential fixes:
- Train a causal variant of wavefield attention
- Use wavefield for representation learning, then distill to standard attention
- Implement KV-caching for wavefield (requires significant research)

### 2. Engram N-gram Memory — Promising for Code Generation
The engram model reached loss=2.19 in only 10,000 steps — significantly better than
previous non-wavefield models (best was 2.88 at 16,500 steps with the small 29.9M model).
The N-gram hash tables provide useful n-gram context at minimal compute cost.

### 3. WSD LR Schedule — Critical for Convergence
The cosine decay phase (last 20% of training) is responsible for the majority of
final loss improvement. Example from engram-v1:
- Step 8000 (start of decay): loss=3.19
- Step 10000 (end): loss=2.19
- **31% loss reduction in just 20% of training!**

### 4. Resume LR Discontinuity
When extending training with `--total-steps`, the LR schedule can jump from minimum
back to stable-phase LR. This causes a temporary loss spike (1.19 → 3.70 for haar,
2.19 → 3.00 for engram). The model recovers in ~250 steps. This is expected behavior
but could be smoothed with a gradual warmup on resume.

## Model Checkpoints

### Best Models
```
checkpoints/nano-275m-haar-v5/final/           # loss=1.1714 (best training loss)
checkpoints/nano-275m-engram-v1/final/         # loss=2.1916 (best for generation)
checkpoints/nano-275m-engram-v2/               # extended training (in progress)
checkpoints/nano-275m-baseline-v1/             # baseline comparison (in progress)
```

### Previous Best
```
checkpoints/nano-275m-haar-v3/step_20000/      # loss=1.1930 (previous best)
```

## Training Stability

- **No OOM errors** during the entire session
- **No NaN losses** detected
- **No crashes** — both processes ran to completion
- VRAM usage stable: 18.3GB (haar) and 20.5GB (engram)
- GPU utilization: 60-80% average
- Throughput: ~560-600 tok/s consistently

## Recommendations

1. **For coherent code generation**: Use the engram model, not wavefield haar.
   Continue training engram to 20,000+ steps for loss below 2.0.

2. **For wavefield research**: Investigate causal wavefield variants or
   distillation from wavefield to standard attention.

3. **For best training efficiency**: Use the WSD schedule with at least 20%
   decay phase. Set total_steps to be achievable in the training budget.

4. **Data scaling**: The current 97M token dataset is modest. Scaling to
   500M+ tokens of Rust code could improve generalization significantly.

5. **Architecture**: The LoopLM (loop_count=16) configuration is too slow
   for practical training (67 tok/s vs 585 tok/s). Reduce loop_count to 4
   or use standard deep architecture instead.
