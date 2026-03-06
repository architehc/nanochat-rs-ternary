# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps

## Latest Metrics (2026-03-06 04:17)
| Metric | Value |
|--------|-------|
| Step | 13650 / 150,000 |
| Loss | 3.6048 |
| Learning Rate | 0.012000 |
| Grad Norm | 18.01 |
| Tokens/sec | 957 |
| Elapsed | 4h (14674s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 54°C |
| Utilization | 88% |
| VRAM | 20450, 32607 MiB |
| Power | 253.90W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **13650/150K** | **3.6048** | **Current run** |

## Loss Trajectory (last 20 readings)
```
[ 12700/150000] loss=3.5928
[ 12750/150000] loss=3.6589
[ 12800/150000] loss=3.5575
[ 12850/150000] loss=3.5387
[ 12900/150000] loss=3.5912
[ 12950/150000] loss=3.7072
[ 13000/150000] loss=3.5798
[ 13050/150000] loss=3.6585
[ 13100/150000] loss=3.5931
[ 13150/150000] loss=3.6637
[ 13200/150000] loss=3.5444
[ 13250/150000] loss=3.5939
[ 13300/150000] loss=3.5394
[ 13350/150000] loss=3.6471
[ 13400/150000] loss=3.6919
[ 13450/150000] loss=3.5605
[ 13500/150000] loss=3.6452
[ 13550/150000] loss=3.5898
[ 13600/150000] loss=3.5724
[ 13650/150000] loss=3.6048
```

---
*Auto-updated by monitor_5090.sh at 2026-03-06 04:17:06*
