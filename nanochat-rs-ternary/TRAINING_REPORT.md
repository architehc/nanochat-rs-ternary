# Training Report — 2026-02-27/28

## Executive Summary

Dual-GPU training session on 2x RTX 4090 (23GB each). Four model runs across two rounds:

| Model | Architecture | Steps | Final Loss | H (CE) | Generation Quality |
|-------|-------------|-------|------------|--------|-------------------|
| **haar-v5** | Wavefield Haar + mHC | 30,000 | **1.1714** | 1.17 | Garbled (wavefield bypass) |
| **engram-v1** | Engram N-gram + mHC | 10,000 | **2.1916** | 2.19 | Semi-coherent Rust (best) |
| **baseline-v1** | Standard + MTP | 10,000 | **4.1929** | **2.92** | Semi-coherent Rust |
| **engram-v2** | Extended engram (resumed) | 16,000 | **2.9363** | 2.95 | Degraded from v1 |

**Key findings**:
1. Wavefield Haar achieves the lowest training loss but cannot generate coherent text (bidirectional attention).
2. Engram-v1 remains the best model for code generation despite not having the lowest loss.
3. Baseline-v1 with MTP achieves H=2.92 — comparable to engram but the reported loss is inflated by MTP (~1.3 above pure CE).
4. Resuming from a converged checkpoint with full LR causes permanent destabilization (engram-v2 regression).

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

### engram-v2 (Extended Engram — Resumed from engram-v1)

```
Step     Loss    H     LR        gnorm   Notes
10050    2.9952  3.00  0.012000  3.62    Resume from step 10000 (LR spike from 0.0012→0.012)
10500    3.1106  3.09  0.012000  5.15    Loss WORSE than engram-v1 final (2.19)
11000    3.0274  3.00  0.012000  5.40    Not recovering
12000    3.1052  3.06  0.012000  7.20    gnorm escalating
13000    3.2042  3.13  0.011896  10.74   Entered decay, gnorm very high
14000    3.1659  3.17  0.008666  11.20   Decay not helping
15000    3.1983  3.21  0.003600  9.08    Late decay
15650    2.9235  2.94  0.001516  9.65    Best spot
16000    2.9363  2.95  0.001200  8.38    Final
```

**Total training time**: 175 minutes (2.9 hours)
**Total steps**: 6,000 (10000 → 16000)
**Throughput**: ~590 tok/s average

**Post-mortem**: The LR spike from 0.0012 (end of engram-v1 decay) back to 0.012
(full stable LR) destabilized the model permanently. Loss never recovered to
engram-v1's 2.19 despite 6000 additional steps. gnorm escalated to 10-12x,
indicating the optimizer was fighting against the disrupted weight landscape.
**Lesson**: Never resume a converged model at full LR. Use a warmup ramp or
lower peak LR.

### baseline-v1 (Standard Attention + MTP)

```
Step     Loss    H     LR        gnorm   Notes
50       11.6775 8.11  0.000300  20.73   Random init (MTP inflates reported loss)
500      8.1233  5.95  0.003000  7.65    Warmup phase
1000     6.6745  4.71  0.006000  3.32
2000     5.5735  3.68  0.012000  1.97    Full LR reached
3000     5.2819  3.53  0.012000  2.10
4000     4.8622  3.31  0.012000  1.93
5000     4.6131  3.14  0.012000  2.10
6000     4.6067  3.15  0.012000  2.44
7000     4.3884  3.02  0.012000  2.75
8000     4.3571  3.01  0.012000  3.27    Entered decay phase
8500     4.3033  3.00  0.010418  3.79
9000     4.0981  2.86  0.006600  3.38    Rapid improvement begins
9400     3.9496  2.77  0.003426  3.09    First loss < 4.0
9700     4.2061  2.92  0.001789  3.08
10000    4.1929  2.92  0.001200  3.06    Final
```

**Total training time**: 271 minutes (4.5 hours)
**Total steps**: 10,000
**Throughput**: ~660 tok/s average (600→720 after engram-v2 freed GPU 1)

**Note on MTP loss**: Reported loss includes Multi-Token Prediction auxiliary loss
(3 future tokens, weight=0.2). The H column shows pure cross-entropy for the primary
next-token prediction. H=2.92 is the comparable metric to engram-v1's loss=2.19.

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

### baseline-v1 (H=2.92, MTP) — Semi-coherent Rust code
```rust
fn main() {
    let data = base::var("dep.rs", "");
    let mut bar = b"hello world";
    for a in basic_root.build_dir(&mut b).unwrap();
    }
    let mut c = git::new();
    c.arg("root".parse(&mut a);
    let mut c = git::new(&mut c);
    c.arg("foo").arg("foo").arg("foo").arg("examples --option")
        .arg(&c);
    c.arg("build --lib")
        .arg("--option")
        .env("");
    cwd.arg("config")
        .arg("--option")
        .arg("build")
        .env("--option")
        .current_dir("src/bin/host/lib.rs")
        .assert()
        .stderr_eq(file!["stderr.term.svg"])
        .stdout_eq(file!["stderr.
```

The model recognizes:
- Variable bindings and mutation (let, let mut)
- Git/command builder patterns (.arg(), .env(), .current_dir())
- Test assertion patterns (.assert(), .stderr_eq())
- File paths and string literals

Issues:
- Repetitive .arg() chains
- Syntactic errors (unclosed parentheses, malformed for loop)
- Less structured than engram-v1

### engram-v2 (loss=2.94) — Degraded from v1
```rust
fn main() {
    let crate = std::io::dummy_dir().join("target")
        .file("RUSTFCXCECT.0.0.1")).await;
    let output = fs::read_dir().join("foo.1");
    tarp_path(["std-path",
        options,
        data_id:as_os::io::PathBuf,
        file_id:as_path_tto_string_t,
    }}
```

Clearly worse than engram-v1: more invented tokens (`tarp_path`, `RUSTFCXCECT`),
inconsistent syntax, but still recognizes basic Rust patterns.

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

### 4. Resume LR Discontinuity — A Critical Pitfall
When extending training with `--total-steps`, the LR schedule jumps from minimum
back to stable-phase LR. Results vary dramatically:

- **haar-v5**: LR spike (0.0012 → 0.012) caused loss 1.19 → 3.70. **Recovered
  in ~250 steps** to 1.27, eventually surpassed original. Success.
- **engram-v2**: Same LR spike (0.0012 → 0.012) caused loss 2.19 → 3.00.
  **Never recovered** — loss stayed at 2.94 after 6000 additional steps. Failure.

The difference: haar-v5 had wavefield attention (bidirectional) with very low gnorm
(0.04), while engram-v1 had standard attention with higher gnorm (2-4). The LR
spike was proportionally much larger relative to engram's gradient landscape.

**Recommendation**: When resuming from a converged checkpoint, either:
- Use a warmup ramp (100-200 steps from current LR to target)
- Use a lower peak LR (e.g., 0.003 instead of 0.012)
- Start fresh training with a different total_steps target instead

### 5. MTP Inflates Loss But Improves Hidden Representations
Baseline-v1 with MTP reports loss ~4.2 but achieves H=2.92 (pure next-token CE).
The ~1.3 loss inflation comes from predicting 3 additional future tokens. While
the MTP loss makes cross-model comparison tricky, MTP may improve the model's
internal representations through multi-step reasoning. Baseline-v1's generation
quality at H=2.92 is comparable to engram-v1 at loss=2.19, despite baseline having
trained without engram memory augmentation.

### 6. Engram vs Baseline — Minimal Difference at Same Scale
At 10,000 steps with the same architecture (dim=1024, 20 layers, 275M params):
- **engram-v1**: loss=2.19 (pure CE)
- **baseline-v1**: H=2.92 (pure CE, no engram)

The ~0.7 loss gap suggests engram provides measurable benefit, but the generation
quality difference is modest. Both produce semi-coherent Rust code. engram-v1
has slightly better syntactic structure while baseline-v1 has more varied vocabulary.

## Model Checkpoints

### Best Models
```
checkpoints/nano-275m-haar-v5/final/           # loss=1.1714 (best training loss, no generation)
checkpoints/nano-275m-engram-v1/final/         # loss=2.1916 (best for generation)
checkpoints/nano-275m-baseline-v1/final/       # H=2.92 (baseline comparison, MTP)
checkpoints/nano-275m-engram-v2/final/         # loss=2.9363 (failed resume, worse than v1)
```

### Previous Best
```
checkpoints/nano-275m-haar-v3/step_20000/      # loss=1.1930 (previous best)
```

## Training Stability

- **No OOM errors** during the entire session (~10 hours total GPU time)
- **No NaN losses** detected across all 4 runs
- **No crashes** — all processes ran to completion
- VRAM usage stable: 18-20GB per GPU
- GPU temperatures: 45-60C
- Throughput: ~585-700 tok/s (higher when single-GPU)

## Recommendations

1. **For coherent code generation**: Use engram-v1 (loss=2.19), the best model.
   For further improvement, train a fresh engram run with total_steps=20000
   (not resumed from a converged checkpoint).

2. **Do NOT resume converged checkpoints at full LR**: The engram-v2 experiment
   proves this destroys model quality. Either train fresh with more steps, or
   resume with a warmup ramp.

3. **For wavefield research**: Investigate causal wavefield variants or
   distillation from wavefield to standard attention. Current wavefield
   cannot generate coherent text.

4. **For best training efficiency**: Use the WSD schedule with at least 20%
   decay phase. The decay phase provides 31% of final loss improvement.

5. **Data scaling**: The current 97M token dataset is modest. Scaling to
   500M+ tokens of Rust code could improve generalization significantly.

6. **Architecture**: The LoopLM (loop_count=16) configuration is too slow
   for practical training (67 tok/s vs 585 tok/s). Reduce loop_count to 4
   or use standard deep architecture.

7. **MTP**: Baseline-v1 with MTP achieved comparable generation quality to
   engram-v1. MTP may be a low-cost way to improve representations without
   custom memory modules. Consider combining MTP + engram in future runs.
