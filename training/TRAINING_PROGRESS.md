# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps

## Latest Metrics (2026-03-06 06:17)
| Metric | Value |
|--------|-------|
| Step | 20350 / 150,000 |
| Loss | 3.9961 |
| Learning Rate | 0.012000 |
| Grad Norm | 78.19 |
| Tokens/sec | 952 |
| Elapsed | 6h (21868s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 52°C |
| Utilization | 83% |
| VRAM | 20450, 32607 MiB |
| Power | 256.98W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **20350/150K** | **3.9961** | **Current run** |

## Loss Trajectory (last 20 readings)
```
[ 19450/150000] loss=4.0735
[ 19500/150000] loss=4.0525
[ 19550/150000] loss=4.1132
[ 19600/150000] loss=4.0198
[ 19650/150000] loss=4.0680
[ 19700/150000] loss=4.0483
[ 19750/150000] loss=4.0053
[ 19800/150000] loss=4.0536
[ 19850/150000] loss=3.9735
[ 19900/150000] loss=4.0569
[ 19950/150000] loss=4.0853
[ 20000/150000] loss=4.0804
[ 20050/150000] loss=4.0421
[ 20100/150000] loss=4.1037
[ 20150/150000] loss=4.0457
[ 20200/150000] loss=4.1238
[ 20250/150000] loss=4.1108
[ 20300/150000] loss=4.0624
[ 20350/150000] loss=3.9961
```

---
*Auto-updated by monitor_5090.sh at 2026-03-06 06:17:08*
