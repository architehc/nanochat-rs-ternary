# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: 130+ repos + synthetic CS algorithms + data structures
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps
- **Optimizer**: Muon (linear) + Lion (norms/mHC/embed)

## Latest Metrics (2026-03-06 00:19)
| Metric | Value |
|--------|-------|
| Step | 350 / 150,000 |
| Loss | 6.5156 |
| Tokens/sec | 929 |
| Elapsed | 0h (378s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 53C |
| Utilization | 71% |
| VRAM | 20316, 32607 MiB |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | **2.19** | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **350/150K** | **6.5156** | **Running, strong convergence** |

## Loss Trajectory
```
[    50/150000] loss=8.3523
[   100/150000] loss=7.9572
[   150/150000] loss=7.4516
[   200/150000] loss=7.1054
[   250/150000] loss=6.8278
[   300/150000] loss=6.7247
[   350/150000] loss=6.5156
```

## 7-Day Pipeline
1. **Day 1-3**: Pre-training (current) — 150K steps on quality dataset
2. **Day 3-5**: Continue pre-training + evaluate checkpoints
3. **Day 5-7**: GRPO RL with compiler feedback — optimize for compilable Rust
4. Hourly monitoring with auto-restart and git push

## Dataset v3 Composition
- 130+ production Rust repos (tokio, serde, ripgrep, bevy, etc.)
- Algorithm repos (TheAlgorithms/Rust, EbTech/rust-algorithms)
- Educational repos (rust-by-example, rustlings, book, patterns)
- 20 synthetic compiler-validated snippets (sorting, graphs, DP, BST, etc.)
- Total: 37,576 unique files, 353 MB raw, 106M tokens

---
*Auto-updated at 2026-03-06 00:19:09*
