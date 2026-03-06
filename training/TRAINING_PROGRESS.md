# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v3
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.008, decay at 20% (step 6K), 30K total steps
- **Resumed from**: v2/step_4000 (loss 3.5, gnorm 2.2)

## Latest Metrics (2026-03-06 07:20)
| Metric | Value |
|--------|-------|
| Step | 0 / 30,000 |
| Loss | 3.3279 |
| Grad Norm | 1.84 |
| Learning Rate | 0.008000 |
| Tokens/sec | 956 |
| Elapsed | 0h (1929s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 55°C |
| Utilization | 62% |
| VRAM | 20448, 32607 MiB |
| Power | 248.99W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| 5090-v2 | 106M tok | 21K/150K | ~4.1 | gnorm blowup (lr=0.012), killed |
| **5090-v3** | **106M tok** | **0/30K** | **3.3279** | **Current (lr=0.008, resumed from v2/4K)** |

## Loss Trajectory (last 20 readings)
```
No data yet
```

---
*Auto-updated by monitor_5090.sh every 30 min at 2026-03-06 07:20:02*
