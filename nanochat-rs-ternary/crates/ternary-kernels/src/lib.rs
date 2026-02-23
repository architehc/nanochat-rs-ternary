/// Kernel auto-tuning (currently unused — C dispatcher handles kernel selection).
/// Gated behind `autotune` feature to avoid compiling 350 lines of dead code.
#[cfg(feature = "autotune")]
pub mod autotune;
pub mod cpu;
pub mod dispatch;
pub mod fwht;
pub mod haar;
#[cfg(any(feature = "cuda", has_cuda))]
pub mod gpu;
