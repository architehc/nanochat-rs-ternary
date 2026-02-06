//! GPU kernel wrappers (stub — requires CUDA toolkit).
//!
//! Two paths:
//! - Decode (N=1, autoregressive): dp4a + constant-memory 256×i32 LUT
//! - Prefill (N>16, prompt processing): CUTLASS + custom ternary B-operand iterator

// GPU kernel implementation will go here when CUDA is available.
// For now this is a stub.
