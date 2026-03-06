# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps

## Latest Metrics (2026-03-06 01:17)
| Metric | Value |
|--------|-------|
| Step | 3600 / 150,000 |
| Loss | 3.7336 |
| Learning Rate | 0.012000 |
| Grad Norm | 2.21 |
| Tokens/sec | 966 |
| Elapsed | 1h (3874s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 57°C |
| Utilization | 82% |
| VRAM | 20349, 32607 MiB |
| Power | 255.37W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **3600/150K** | **3.7336** | **Current run** |

## Loss Trajectory (last 20 readings)
```
[  2650/150000] loss=3.9743
[  2700/150000] loss=3.9489
[  2750/150000] loss=3.9152
[  2800/150000] loss=4.0000
[  2850/150000] loss=3.8752
[  2900/150000] loss=3.8545
[  2950/150000] loss=3.7810
[  3000/150000] loss=3.8329
[  3050/150000] loss=3.8188
[  3100/150000] loss=3.7499
[  3150/150000] loss=3.8598
[  3200/150000] loss=3.7389
[  3250/150000] loss=3.7042
[  3300/150000] loss=3.7727
[  3350/150000] loss=3.6947
[  3400/150000] loss=3.6220
[  3450/150000] loss=3.6884
[  3500/150000] loss=3.6557
[  3550/150000] loss=3.7550
[  3600/150000] loss=3.7336
```

---
*Auto-updated by monitor_5090.sh at 2026-03-06 01:17:04*
