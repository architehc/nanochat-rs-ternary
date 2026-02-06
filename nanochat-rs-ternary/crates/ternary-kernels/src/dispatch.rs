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
//! 3         LUT-Grouped      Any x86_64         5-13 GOPS (degrades at large K)
//! 4         Scalar dp4a-ref  Portable           16-18 GOPS (surprisingly good)
//! ```
//!
//! **VPERMW is primary even when VBMI is available** because the lo/hi byte
//! recombination overhead exceeds VPERMB's theoretical throughput advantage.

/// Re-export the main GEMV interface from cpu module.
pub use crate::cpu::gemv;
pub use crate::cpu::gemv_scalar_ref;
