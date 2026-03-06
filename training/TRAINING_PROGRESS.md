# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps

## Latest Metrics (2026-03-06 00:17)
| Metric | Value |
|--------|-------|
| Step | 250 / 150,000 |
| Loss | 6.8278 |
| Learning Rate | 0.001500 |
| Grad Norm | 4.77 |
| Tokens/sec | 963 |
| Elapsed | 0h (270s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 54°C |
| Utilization | 83% |
| VRAM | 20316, 32607 MiB |
| Power | 258.52W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **250/150K** | **6.8278** | **Current run** |

## Loss Trajectory (last 20 readings)
```
[    50/150000] loss=8.3523
[   100/150000] loss=7.9572
[   150/150000] loss=7.4516
[   200/150000] loss=7.1054
[   250/150000] loss=6.8278
```

---
*Auto-updated by monitor_5090.sh at 2026-03-06 00:17:03*
