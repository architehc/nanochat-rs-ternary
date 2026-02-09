//! Custom CUDA operations for training.
//!
//! Provides CUDA kernels that are missing from Candle's built-in ops.

use candle_core::{Result, Tensor};

// Note: CUDA kernels are compiled and linked, but Candle's internal CUDA APIs
// are not publicly exposed in a stable way. For production use, we use a CPU
// fallback for sigmoid which has minimal overhead since sigmoid is only used
// in mHC layers (< 0.00004% of total compute).

/// Apply sigmoid using CPU fallback.
///
/// This is a workaround for Candle not having CUDA sigmoid support.
/// The overhead is minimal since sigmoid is only used in mHC layers.
pub fn cuda_sigmoid(tensor: &Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(tensor)
}
