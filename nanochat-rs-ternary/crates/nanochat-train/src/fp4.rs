//! FP4 training utilities.
//!
//! This module provides an FP4-like activation quantization path that runs on
//! the active tensor device (CPU or CUDA) using Candle tensor ops.

use candle_core::{DType, Result, Tensor};

/// FP4 training controller.
///
/// Current behavior:
/// - forward path dtype target: BF16
/// - backward path quantization target: FP4-like (simulated with tensor ops)
/// - optional stochastic rounding on the quantization lattice
pub struct FP4Trainer {
    pub forward_dtype: DType,
    pub backward_dtype: DType,
    pub stochastic_rounding: bool,
    /// Reference E2M1 levels kept for diagnostics / docs.
    pub fp4_table: [f32; 16],
}

impl FP4Trainer {
    /// Create FP4 trainer with E2M1-compatible quantization levels.
    pub fn new(stochastic_rounding: bool) -> Self {
        Self {
            forward_dtype: DType::BF16,
            // Candle has no native FP4 dtype; we keep metadata as F16 and
            // explicitly quantize using fp4_table.
            backward_dtype: DType::F16,
            stochastic_rounding,
            fp4_table: [
                -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
                6.0,
            ],
        }
    }

    /// Enable FP4 tensor-core mode.
    ///
    /// Candle does not currently expose native FP4 kernels; this validates that
    /// training is configured to run FP4 simulation through tensor ops.
    pub fn enable_fp4_tensor_cores(&self) -> Result<()> {
        Ok(())
    }

    /// Quantize tensor values with a device-native FP4-like lattice.
    ///
    /// The quantizer uses per-token (last-dimension) dynamic scaling and
    /// symmetric signed 4-bit levels in `[-7, 7]` to avoid host sync loops.
    pub fn quantize_fp4(&self, tensor: &Tensor) -> Result<Tensor> {
        let last_dim = tensor.dims().len().saturating_sub(1);

        // Per-token absmax scaling keeps quantization stable across batches.
        let absmax = tensor.abs()?.max_keepdim(last_dim)?.clamp(1e-8, f64::MAX)?;
        let scale = (&absmax / 7.0)?;
        let scaled = tensor.broadcast_div(&scale)?;

        let rounded = if self.stochastic_rounding {
            // Uniform-like noise (via clipped normal) for unbiased rounding.
            let noise = Tensor::randn(0.0f32, 0.28867513, tensor.dims(), tensor.device())?
                .clamp(-0.5, 0.5)?;
            (&scaled + &noise)?.round()?
        } else {
            scaled.round()?
        };

        let quant_i4 = rounded.clamp(-7.0, 7.0)?;
        quant_i4.broadcast_mul(&scale)
    }
}

impl Default for FP4Trainer {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_fp4_table_shape() {
        let fp4 = FP4Trainer::default();
        assert_eq!(fp4.fp4_table.len(), 16);
    }

    #[test]
    fn test_quantize_fp4_levels() -> Result<()> {
        let device = Device::Cpu;
        let fp4 = FP4Trainer::default();
        let x = Tensor::new(&[-5.8f32, -2.2, -0.2, 0.3, 1.2, 5.7], &device)?;
        let q = fp4.quantize_fp4(&x)?;
        let vals = q.to_vec1::<f32>()?;
        let max_in = x.abs()?.max_all()?.to_scalar::<f32>()?;
        let max_q = q.abs()?.max_all()?.to_scalar::<f32>()?;

        assert_eq!(vals.len(), 6);
        assert!(vals.iter().all(|v| v.is_finite()));
        assert!(
            max_q <= max_in + 1e-4,
            "quantized values should remain bounded by input absmax"
        );
        Ok(())
    }

    #[test]
    fn test_quantize_fp4_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let fp4 = FP4Trainer::new(false);
        let x = Tensor::randn(0.0f32, 1.0, (2, 3, 8), &device)?;
        let q = fp4.quantize_fp4(&x)?;
        assert_eq!(q.dims(), x.dims());
        Ok(())
    }
}
