# E3 Benchmark Results

**Date**: 2026-02-15 06:44:25
**GPU**: NVIDIA RTX PRO 6000 Blackwell (96GB)
**Workspace**: /home/habitat/ternary-clawd/nanochat-rs-ternary

---

## Configurations Tested

| Config | MTP | Collider | Async Loader | Optimizer | Memory Opt |
|--------|-----|----------|--------------|-----------|------------|
| Baseline | ❌ | ❌ | ❌ | Muon | None |
| MTP Only | ✅ | ❌ | ❌ | Muon | None |
| 8-bit Muon | ✅ | ✅ | ✅ | Muon | 8-bit (86%) |
| GaLore 2 | ✅ | ✅ | ✅ | Muon | GaLore (50-65%) |
| E3 Full | ✅ | ✅ | ✅ | Muon | 8-bit + GaLore |

---

## Results

