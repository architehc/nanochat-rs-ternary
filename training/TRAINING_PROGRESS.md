# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v4
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v4 (143M tokens, 65K files from 187 repos, vocab=4096)
- **Includes**: tokio, serde, bevy, rust-lang/rust stdlib, solana, reth, CS algorithms
- **Schedule**: WSD lr=0.006, decay at 40% (step 20K), 50K total steps
- **Resumed from**: v3/final (loss 2.72, 30K steps)

## Latest Metrics (2026-03-06 18:18)
| Metric | Value |
|--------|-------|
| Step | 0 / 50,000 |
| Loss | 3.4658 |
| Grad Norm | 1.91 |
| Learning Rate | 0.003722 |
| Tokens/sec | 257 |
| Elapsed | 3h (13103s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 47°C |
| Utilization | 82% |
| VRAM | 31954, 32607 MiB |
| Power | 163.10W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| 5090-v2 | 106M tok | 21K/150K | ~4.1 | gnorm blowup (lr=0.012), killed |
| 5090-v3 | 106M tok | 30K | 2.72 | lr=0.008, decay@20%, stable |
| **5090-v4** | **143M tok** | **0/50K** | **3.4658** | **Current (lr=0.006, 65K files, 187 repos)** |

## Loss Trajectory (last 20 readings)
```
No data yet
```

---
*Auto-updated by monitor_5090.sh every 30 min at 2026-03-06 18:18:33*
