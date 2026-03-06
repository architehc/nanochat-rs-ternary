# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps

## Latest Metrics (2026-03-06 05:17)
| Metric | Value |
|--------|-------|
| Step | 17000 / 150,000 |
| Loss | 3.7764 |
| Learning Rate | 0.012000 |
| Grad Norm | 39.02 |
| Tokens/sec | 951 |
| Elapsed | 5h (18271s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 57°C |
| Utilization | 89% |
| VRAM | 20450, 32607 MiB |
| Power | 261.25W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **17000/150K** | **3.7764** | **Current run** |

## Loss Trajectory (last 20 readings)
```
[ 16050/150000] loss=3.6406
[ 16100/150000] loss=3.8110
[ 16150/150000] loss=3.7177
[ 16200/150000] loss=3.7204
[ 16250/150000] loss=3.7150
[ 16300/150000] loss=3.7942
[ 16350/150000] loss=3.6791
[ 16400/150000] loss=3.8525
[ 16450/150000] loss=3.7782
[ 16500/150000] loss=3.7536
[ 16550/150000] loss=3.7757
[ 16600/150000] loss=3.8509
[ 16650/150000] loss=3.8750
[ 16700/150000] loss=3.8483
[ 16750/150000] loss=3.8162
[ 16800/150000] loss=3.7981
[ 16850/150000] loss=3.8860
[ 16900/150000] loss=3.7890
[ 16950/150000] loss=3.8438
[ 17000/150000] loss=3.7764
```

---
*Auto-updated by monitor_5090.sh at 2026-03-06 05:17:07*
