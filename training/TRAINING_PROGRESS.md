# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: 130+ repos + synthetic CS algorithms + data structures
- **Schedule**: WSD lr=0.012, decay at 80% (step 120K), 150K steps total
- **Optimizer**: Muon (linear) + Lion (norms/mHC/embed)

## Latest Metrics (2026-03-06 01:46)
| Metric | Value |
|--------|-------|
| Step | 5200 / 150,000 |
| Loss | 3.5310 |
| Tokens/sec | 980 |
| Elapsed | 1h (5596s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 55C |
| Utilization | 79% |
| VRAM | 20349, 32607 MiB |

## Convergence Analysis
- Step 0-2000 (warmup): loss 8.35 -> 4.45
- Step 2000-5200 (full lr): loss 4.45 -> 3.53 (exploration phase)
- gnorm stable at ~2.4 (v1 had gnorms >130 = unstable)
- Decay starts at step 120K -> expect loss to drop below 3.0
- v13 went from ~3.5 -> 2.19 during its decay phase (steps 8K-10K)

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | **2.19** | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **5200/150K** | **3.5310** | **Running, healthy** |

## Loss Trajectory
```
[  3750/150000] loss=3.5985
[  3800/150000] loss=3.5930
[  3850/150000] loss=3.6702
[  3900/150000] loss=3.6669
[  3950/150000] loss=3.6022
[  4000/150000] loss=3.7783
[  4050/150000] loss=3.6011
[  4100/150000] loss=3.6016
[  4150/150000] loss=3.6307
[  4200/150000] loss=3.4648
[  4250/150000] loss=3.6874
[  4300/150000] loss=3.5666
[  4350/150000] loss=3.5088
[  4400/150000] loss=3.6234
[  4450/150000] loss=3.5278
[  4500/150000] loss=3.5932
[  4550/150000] loss=3.5585
[  4600/150000] loss=3.5625
[  4650/150000] loss=3.6414
[  4700/150000] loss=3.5512
[  4750/150000] loss=3.5289
[  4800/150000] loss=3.5534
[  4850/150000] loss=3.4991
[  4900/150000] loss=3.5642
[  4950/150000] loss=3.5306
[  5000/150000] loss=3.5253
[  5050/150000] loss=3.4936
[  5100/150000] loss=3.4969
[  5150/150000] loss=3.4652
[  5200/150000] loss=3.5310
```

## 7-Day Pipeline Status
1. [ACTIVE] Pre-training (150K steps, ~7 days at 960 tok/s)
2. [READY] GRPO RL with compiler feedback (launch from checkpoint)
3. [READY] Hourly monitoring daemon active with auto-restart

## Dataset v3 Composition
- 130+ production Rust repos (tokio, serde, ripgrep, bevy, etc.)
- Algorithm repos (TheAlgorithms/Rust, EbTech/rust-algorithms)
- Educational repos (rust-by-example, rustlings, book, patterns)
- 20 synthetic compiler-validated snippets (sorting, graphs, DP, BST, etc.)
- Total: 37,576 unique files, 353 MB raw, 106M tokens

---
*Auto-updated at 2026-03-06 01:46:06*
