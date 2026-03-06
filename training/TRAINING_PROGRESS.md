# RTX 5090 Training Progress

## Current Run: nano-275m-engram-5090-v2
- **Config**: 275M params, dim=1024, 20 layers, Engram on [0,10,19]
- **Dataset**: rust_v3 (106M tokens, 37K files, vocab=4096)
- **Includes**: Synthetic CS algorithms, data structures, compiler-validated primitives
- **Schedule**: WSD lr=0.012, decay at 80%, 150K steps

## Latest Metrics (2026-03-06 02:17)
| Metric | Value |
|--------|-------|
| Step | 6950 / 150,000 |
| Loss | 3.3555 |
| Learning Rate | 0.012000 |
| Grad Norm | 3.09 |
| Tokens/sec | 953 |
| Elapsed | 2h (7472s) |

## GPU Status
| Metric | Value |
|--------|-------|
| Temperature | 55°C |
| Utilization | 85% |
| VRAM | 20349, 32607 MiB |
| Power | 256.57W |

## Training History
| Version | Dataset | Steps | Final Loss | Notes |
|---------|---------|-------|------------|-------|
| v13 (BEST) | 36M tok | 10K | 2.19 | lr=0.012, decay@80% |
| v14 | 36M tok | 15K | 2.98 | lr=0.010, decay@53% |
| 5090-v1 | 121M tok | 41K/100K | ~6.0 | Plateaued, killed |
| **5090-v2** | **106M tok** | **6950/150K** | **3.3555** | **Current run** |

## Loss Trajectory (last 20 readings)
```
[  6050/150000] loss=3.4166
[  6100/150000] loss=3.5530
[  6150/150000] loss=3.4920
[  6200/150000] loss=3.4488
[  6250/150000] loss=3.3674
[  6300/150000] loss=3.4498
[  6350/150000] loss=3.4495
[  6400/150000] loss=3.4202
[  6450/150000] loss=3.4996
[  6500/150000] loss=3.5081
[  6550/150000] loss=3.4790
[  6600/150000] loss=3.5282
[  6650/150000] loss=3.3892
[  6700/150000] loss=3.4534
[  6750/150000] loss=3.3977
[  6800/150000] loss=3.4372
[  6850/150000] loss=3.5007
[  6900/150000] loss=3.4498
[  6950/150000] loss=3.3555
```

---
*Auto-updated by monitor_5090.sh at 2026-03-06 02:17:05*
