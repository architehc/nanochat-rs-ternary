//! Runtime CPU feature detection and kernel dispatch documentation.
//!
//! The dispatch chain is handled internally by the C code (`ternary_gemv()`),
//! which is self-initializing and thread-safe. This module documents the
//! dispatch priority for Rust callers.
//!
//! ```text
//! Priority  Kernel              Requirement        Platform
//! ────────  ──────────────────  ─────────────────  ──────────
//! 1         VPERMW fused        AVX-512 BW         x86_64
//! 2         Dual-VPERMB         AVX-512 VBMI       x86_64
//! 3         AVX2 Nibble-Split   AVX2 + FMA         x86_64
//! 4         SSSE3 PSIGNB        SSSE3              x86_64
//! 5         NEON Nibble-Split   AArch64 NEON       ARM64
//! 6         LUT-Grouped         Any x86_64         x86_64
//! 7         Scalar dp4a-ref     Portable           Any
//! ```
//!
//! **VPERMW is primary even when VBMI is available** because the lo/hi byte
//! recombination overhead exceeds VPERMB's theoretical throughput advantage.
//!
//! **AVX2 Nibble-Split** uses 256-bit PSHUFB for LUT lookups with 16-bit
//! vertical accumulation. Better than LUT-Grouped which degrades at large K.
//!
//! **SSSE3 PSIGNB** uses `_mm_sign_epi8` for zero-cost ternary multiplication
//! (sign(x, w) = x*w for w in {-1,0,+1}). Fallback for pre-AVX2 x86 CPUs.
//!
//! **NEON Nibble-Split** adapts the AVX2 nibble-split algorithm to ARM NEON
//! using `vtbl1q_u8` for 16-byte LUT lookup with 64-row blocking (leveraging
//! ARM64's 32 SIMD registers).

/// Re-export the main GEMV interface from cpu module.
pub use crate::cpu::gemv;
pub use crate::cpu::gemv_parallel;
pub use crate::cpu::gemv_scalar_ref;
pub use crate::cpu::selected_kernel_name;
