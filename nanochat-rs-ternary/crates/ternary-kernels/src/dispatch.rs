//! Runtime CPU feature detection and kernel dispatch documentation.
//!
//! The dispatch chain is handled internally by the C code (`ternary_gemv()`),
//! which is self-initializing and thread-safe. This module documents the
//! dispatch priority for Rust callers.
//!
//! ```text
//! Priority  Kernel           Requirement        Performance
//! ────────  ───────────────  ─────────────────  ────────────────────
//! 1         VPERMW fused     AVX-512 BW         19-36 GOPS (PRIMARY)
//! 2         Dual-VPERMB      AVX-512 VBMI       16-26 GOPS
//! 3         AVX2 PSHUFB      AVX2 + FMA         8-15 GOPS
//! 4         LUT-Grouped      Any x86_64         5-13 GOPS (degrades at large K)
//! 5         Scalar dp4a-ref  Portable           ~1.7 GOPS
//! ```
//!
//! **VPERMW is primary even when VBMI is available** because the lo/hi byte
//! recombination overhead exceeds VPERMB's theoretical throughput advantage.
//!
//! **AVX2 PSHUFB** uses 256-bit PSHUFB for LUT lookups with the same
//! COMPACT bit-extraction as VPERMW. Better than LUT-Grouped which degrades
//! at large K (1.3MB working set exceeds L1 at K=11008).

/// Re-export the main GEMV interface from cpu module.
pub use crate::cpu::gemv;
pub use crate::cpu::gemv_parallel;
pub use crate::cpu::gemv_scalar_ref;
