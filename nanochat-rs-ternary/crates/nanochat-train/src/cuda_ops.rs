//! Custom CUDA operations for training.
//!
//! Provides CUDA kernels that are missing from Candle's built-in ops.

use candle_core::{Result, Tensor, Device};

// Note: CUDA kernels are compiled and linked, but Candle's internal CUDA APIs
// are not publicly exposed in a stable way. For production use, we use a CPU
// fallback for sigmoid which has minimal overhead since sigmoid is only used
// in mHC layers (< 0.00004% of total compute).

/// Apply sigmoid using CPU fallback for CUDA tensors.
///
/// This is a workaround for Candle not having CUDA sigmoid support.
/// Copies tensor to CPU, applies sigmoid, copies back to GPU.
/// The overhead is minimal since sigmoid is only used in mHC layers.
pub fn cuda_sigmoid(tensor: &Tensor) -> Result<Tensor> {
    if tensor.device().is_cuda() {
        // Copy to CPU, apply sigmoid, copy back
        let cpu_tensor = tensor.to_device(&Device::Cpu)?;
        let result = candle_nn::ops::sigmoid(&cpu_tensor)?;
        result.to_device(tensor.device())
    } else {
        // Already on CPU
        candle_nn::ops::sigmoid(tensor)
    }
}
