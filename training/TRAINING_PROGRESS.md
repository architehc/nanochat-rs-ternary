# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps

## Latest Metrics (2026-03-06 03:17)
| Metric | Value |
|--------|-------|
| Step | 10300 / 150,000 |
| Loss | 3.4284 |
| Learning Rate | 0.012000 |
| Grad Norm | 8.16 |
| Tokens/sec | 955 |
| Elapsed | 3h (11075s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 54°C |
| Utilization | 81% |
| VRAM | 20363, 32607 MiB |
| Power | 252.13W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **10300/150K** | **3.4284** | **Current run** |

## Loss Trajectory (last 20 readings)
```
[  9400/150000] loss=3.4794
[  9450/150000] loss=3.4835
[  9500/150000] loss=3.5269
[  9550/150000] loss=3.5253
[  9600/150000] loss=3.3975
[  9650/150000] loss=3.5359
[  9700/150000] loss=3.4102
[  9750/150000] loss=3.4637
[  9800/150000] loss=3.5777
[  9850/150000] loss=3.4921
[  9900/150000] loss=3.4911
[  9950/150000] loss=3.4756
[ 10000/150000] loss=3.4602
[ 10050/150000] loss=3.4762
[ 10100/150000] loss=3.3735
[ 10150/150000] loss=3.4348
[ 10200/150000] loss=3.5256
[ 10250/150000] loss=3.3927
[ 10300/150000] loss=3.4284
```

---
*Auto-updated by monitor_5090.sh at 2026-03-06 03:17:05*
