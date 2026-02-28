# Training Report — 2026-02-27/28

## Executive Summary

Dual-GPU training session on 2x RTX 4090 (23GB each). Six model runs across three rounds:

| Model | Architecture | Steps | Final Loss | H (CE) | Coherence Score |
|-------|-------------|-------|------------|--------|----------------|
| **haar-v5** | Wavefield Haar + mHC | 30,000 | **1.1714** | 1.17 | N/A (garbled) |
| **engram-v1** | Engram N-gram + mHC | 10,000 | **2.1916** | 2.19 | **0.752** (best) |
| **engram-v2** | Extended engram (resumed) | 16,000 | **2.9363** | 2.95 | 0.740 |
| **engram-mtp** | Engram + MTP combined | 10,000 | **4.1924** | **2.91** | 0.734 |
| **baseline-v1** | Standard + MTP | 10,000 | **4.1929** | **2.92** | 0.699 |
| **engram-v3** | Fresh engram (20K) | *in progress* | — | — | (pending) |

**Key findings**:
1. **Coherence > Loss**: engram-v1 has the best generation quality (0.752) despite engram-v2 having better syntax coherence (0.617 vs 0.608). Repetition control is the key differentiator.
2. **MTP does NOT improve coherence**: engram-mtp (0.734) with MTP scores worse than engram-v1 (0.752) without MTP at the same step count. MTP overhead hurts single-token generation.
3. **Engram consistently helps**: All 3 engram variants beat baseline-v1 (0.699) for coherence.
4. Resuming from a converged checkpoint with full LR causes permanent destabilization (engram-v2 regression).
5. Wavefield Haar achieves the lowest training loss but cannot generate coherent text (bidirectional attention).

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

### engram-mtp (Engram + MTP Combined, Round 3)

```
Step     Loss    H     LR        gnorm   Notes
50       11.6289 8.11  0.000400  19.70   Random init (MTP inflates loss ~1.3)
500      7.8964  5.75  0.004000  6.63    Warmup phase
1000     6.5903  4.55  0.008000  2.90
1500     6.0641  4.03  0.012000  2.25    Full LR
2000     5.5013  3.64  0.012000  1.90
3000     5.2438  3.56  0.012000  2.21
4000     4.8369  3.30  0.012000  2.16
5000     4.5952  3.15  0.012000  2.46
6000     4.6118  3.16  0.012000  2.80
7000     4.3655  3.00  0.012000  3.29    First H < 3.0
8000     4.3392  3.00  0.012000  3.69    Entered decay phase
9000     4.0855  2.85  0.006600  3.90    Rapid improvement
9400     3.9448  2.77  0.003426  3.63    Best H
10000    4.1924  2.91  0.001200  3.50    Final
```

**Total training time**: 283.5 minutes (4.7 hours)
**Total steps**: 10,000
**Throughput**: ~610 tok/s average

**Analysis**: Combining Engram + MTP produced H=2.91, nearly identical to baseline-v1's
H=2.92. The MTP auxiliary loss adds gradient noise that appears to slow convergence of
the primary next-token prediction. The engram memory tables don't compensate for this
overhead at 10K steps.

### engram-v3 (Fresh 20K Engram, Round 3 — IN PROGRESS)

```
Step     Loss    H     LR        gnorm   Notes
50       8.3953  8.11  0.000400  17.14   Random init
500      5.6693  5.71  0.004000  5.96
1000     4.6947  4.50  0.008000  2.71
1500     4.3394  4.01  0.012000  2.18    Full LR
2000     3.9479  3.61  0.012000  1.81
3000     3.6810  3.51  0.012000  2.46
4000     3.4568  3.36  0.012000  2.38
5000     3.2549  3.20  0.012000  2.73
6000     3.3052  3.25  0.012000  3.05
7000     3.2862  3.25  0.012000  3.84
8000     3.2226  3.18  0.012000  5.83
9000     3.2133  3.18  0.012000  6.06
10000    3.2468  3.21  0.012000  6.76    50% done, gnorms elevated
```

Still running on GPU 1. Decay starts at step 16000. Grad norms are elevated
(6-7x) compared to earlier (2-3x) — possibly overfitting or training instability.
Will evaluate coherence when complete.

## Coherence Benchmark

Quantitative evaluation across 15 standard Rust prompts per model, using 6 metrics:
- **Bracket Balance** (w=0.15): Open/close parity of (), {}, [], <>
- **Repetition 3-gram** (w=0.15): Unique 3-gram ratio (higher = less repetitive)
- **Repetition 5-gram** (w=0.15): Unique 5-gram ratio
- **Rust Keywords** (w=0.15): Real Rust keyword density
- **Token Quality** (w=0.15): Absence of invented/garbage tokens
- **Syntax Coherence** (w=0.25): Regex-based pattern detection (fn, let, struct, impl, etc.)

### Results

| Rank | Model | Composite | Syntax | Rep(3g) | Rep(5g) | Brackets | Keywords | Tokens |
|------|-------|-----------|--------|---------|---------|----------|----------|--------|
| 1 | engram-v1 | **0.752** | 0.608 | **0.888** | **0.956** | 0.918 | 0.237 | 0.998 |
| 2 | engram-v2 | 0.740 | **0.617** | 0.818 | 0.905 | **0.951** | 0.235 | 0.993 |
| 3 | engram-mtp | 0.734 | 0.575 | 0.845 | 0.924 | 0.947 | 0.223 | 0.997 |
| 4 | baseline-v1 | 0.699 | 0.500 | 0.767 | 0.872 | 0.936 | **0.253** | 0.996 |

### Key Observations

1. **Repetition is the differentiator**: engram-v1 wins primarily because it's the least
   repetitive (3-gram: 0.888 vs 0.767-0.845 for others). This suggests the engram N-gram
   memory helps the model avoid repetitive patterns.

2. **MTP hurts repetition control**: engram-mtp (3-gram: 0.845) is worse than engram-v1
   (0.888) despite identical architecture except for MTP. The multi-token prediction
   objective may encourage the model to produce safe/repetitive patterns.

3. **Bracket balance correlates weakly with quality**: engram-v2 has the best bracket
   balance (0.951) but ranks 2nd overall. Good brackets don't guarantee good code.

4. **All models have similar token quality** (0.993-0.998), meaning invented tokens
   are rare across the board at this scale.

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

### 1. Coherence Scoring Reveals What Loss Cannot
Loss alone is misleading for generation quality assessment:
- haar-v5 has the best loss (1.17) but worst generation (garbled)
- engram-v1 (loss=2.19) and baseline-v1 (H=2.92) have different losses but
  both produce semi-coherent code
- **Coherence benchmark** (composite score) is a much better predictor of
  actual generation utility than raw loss

### 2. Engram N-gram Memory Reduces Repetition
The #1 differentiator between models is **repetition control**:
- engram-v1 3-gram uniqueness: 0.888 (best)
- engram-mtp 3-gram: 0.845
- engram-v2 3-gram: 0.818
- baseline-v1 3-gram: 0.767 (worst)

The engram hash tables appear to help the model track recent n-grams and avoid
repeating them. This is the module's most valuable contribution.

### 3. MTP Does Not Improve Generation Quality
Direct comparison at 10K steps:
- **engram-v1** (no MTP): H=2.19, coherence=0.752
- **engram-mtp** (with MTP): H=2.91, coherence=0.734

MTP adds ~1.3 to reported loss and achieves H=2.91 vs engram-v1's 2.19. Even
accounting for MTP overhead, the coherence benchmark shows MTP produces *worse*
generation quality. The multi-token prediction objective may encourage safe/repetitive
patterns that hurt diversity.

### 4. WSD LR Schedule — Critical for Convergence
The cosine decay phase (last 20% of training) is responsible for the majority of
final loss improvement. Example from engram-v1:
- Step 8000 (start of decay): loss=3.19
- Step 10000 (end): loss=2.19
- **31% loss reduction in just 20% of training!**

### 5. Resume LR Discontinuity — A Critical Pitfall
When extending training with `--total-steps`, the LR schedule jumps from minimum
back to stable-phase LR. Results vary dramatically:

- **haar-v5**: LR spike (0.0012 → 0.012) caused loss 1.19 → 3.70. **Recovered
  in ~250 steps** to 1.27, eventually surpassed original. Success.
- **engram-v2**: Same LR spike (0.0012 → 0.012) caused loss 2.19 → 3.00.
  **Never recovered** — loss stayed at 2.94 after 6000 additional steps. Failure.

**Recommendation**: When resuming from a converged checkpoint, either:
- Use a warmup ramp (100-200 steps from current LR to target)
- Use a lower peak LR (e.g., 0.003 instead of 0.012)
- Start fresh training with a different total_steps target instead

### 6. Wavefield Attention — High Training Efficiency, Zero Generation Utility
The wavefield Haar attention achieves the lowest training loss (1.17 vs 2.19) because
it can attend to all positions bidirectionally. However, this makes it completely
unsuitable for autoregressive generation.

## Model Checkpoints

### Best Models (ranked by coherence)
```
checkpoints/nano-275m-engram-v1/final/         # coherence=0.752, loss=2.1916 (BEST for generation)
checkpoints/nano-275m-engram-v2/final/         # coherence=0.740, loss=2.9363 (failed resume)
checkpoints/nano-275m-engram-mtp/final/        # coherence=0.734, H=2.91 (engram+MTP)
checkpoints/nano-275m-baseline-v1/final/       # coherence=0.699, H=2.92 (baseline+MTP)
checkpoints/nano-275m-haar-v5/final/           # loss=1.1714 (best loss, no generation)
```

### In Progress
```
checkpoints/nano-275m-engram-v3/               # fresh 20K engram, ~50% done
```

## Training Stability

- **One OOM error**: engram-v3 OOM at step 150 on GPU 1 (both GPUs initializing simultaneously). Restarted successfully.
- **No NaN losses** detected across all 6 runs
- **No other crashes** — all completed processes ran to final step
- VRAM usage stable: 18-20GB per GPU
- GPU temperatures: 43-64C
- Throughput: ~585-700 tok/s (600-660 typical, lower when both GPUs active)
- Total GPU hours: ~20 hours across 3 rounds

## Recommendations

1. **For coherent code generation**: Use engram-v1 (coherence=0.752), the best model.
   Wait for engram-v3 (fresh 20K) to potentially surpass it.

2. **Skip MTP for generation tasks**: MTP does not improve coherence scores and adds
   training overhead. Use pure CE loss with engram memory for best generation quality.

3. **Do NOT resume converged checkpoints at full LR**: The engram-v2 experiment
   proves this destroys model quality. Train fresh with more steps instead.

4. **Focus on repetition reduction**: The key quality gap between models is repetition.
   Techniques like frequency penalty, n-gram blocking, or contrastive decoding could
   help at inference time. The engram memory helps at training time.

5. **For best training efficiency**: Use the WSD schedule with at least 20%
   decay phase. The decay phase provides 31% of final loss improvement.

6. **Data scaling**: The current 97M token dataset is modest. Scaling to
   500M+ tokens of Rust code could improve generalization significantly.

7. **Use coherence benchmarks, not loss**: Always evaluate with the coherence
   benchmark (eval_coherence.sh + score_coherence.py) rather than relying on
   training loss or perplexity alone.
