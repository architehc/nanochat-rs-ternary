//! Custom CUDA operations for training.
//!
//! Provides CUDA kernels that are missing from Candle's built-in ops.

use candle_core::{Device, Result, Tensor};

// Note: CUDA kernels are compiled and linked, but Candle's internal CUDA APIs
// are not publicly exposed in a stable way. For production use, we use a CPU
// fallback for sigmoid which has minimal overhead since sigmoid is only used
// in mHC layers (< 0.00004% of total compute).

/// Apply sigmoid using CPU fallback for CUDA tensors.
///
/// This is a workaround for Candle not having CUDA sigmoid support.
/// Copies tensor to CPU, applies sigmoid, copies back to GPU.
///
/// # Do not call this from a training path
///
/// Currently unused, and it should stay that way. The cost is not the sigmoid
/// arithmetic — it is that each call forces a **full device synchronize** plus
/// two PCIe transfers, which stalls the pipeline. Training steps here are
/// launch- and sync-bound (see `docs/BLACKWELL_LOW_PRECISION.md`), so a handful
/// of these per forward would dominate the step regardless of how little compute
/// they represent.
///
/// Every live call site instead expands sigmoid on-device as
/// `1 / (1 + exp(-x))` — see `mhc.rs`, `gated_deltanet.rs`, `engram.rs`,
/// `loop_block.rs`. Do the same in new code.
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
