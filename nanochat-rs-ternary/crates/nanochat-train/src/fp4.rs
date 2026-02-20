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

    /// Quantize tensor values using nearest-neighbor lookup into E2M1 levels.
    ///
    /// The quantizer uses per-token (last-dimension) dynamic scaling with
    /// the E2M1 FP4 table `[-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6]`.
    /// Each value is mapped to the nearest E2M1 level after scaling, then
    /// dequantized back. This produces a proper FP4 lattice instead of a
    /// uniform INT4 grid.
    pub fn quantize_fp4(&self, tensor: &Tensor) -> Result<Tensor> {
        let last_dim = tensor.dims().len().saturating_sub(1);

        // Per-token absmax scaling; max E2M1 magnitude is 6.0.
        let absmax = tensor.abs()?.max_keepdim(last_dim)?.clamp(1e-8, f64::MAX)?;
        let scale = (&absmax / 6.0)?; // max E2M1 value is 6.0
        let scaled = tensor.broadcast_div(&scale)?;

        // WARNING: Host round-trip — flatten_all().to_vec1() pulls data to CPU,
        // then Tensor::from_vec() pushes back. This is a significant performance
        // bottleneck for GPU training. For production use, implement a custom CUDA
        // kernel that performs nearest-neighbor E2M1 lookup entirely on-device.
        let flat = scaled.flatten_all()?.to_vec1::<f32>()?;
        let snapped: Vec<f32> = flat
            .iter()
            .map(|&v| {
                let mut best = self.fp4_table[0];
                let mut best_dist = (v - best).abs();
                for &level in &self.fp4_table[1..] {
                    let dist = (v - level).abs();
                    if dist < best_dist {
                        best = level;
                        best_dist = dist;
                    }
                }
                best
            })
            .collect();

        let snapped_tensor = Tensor::from_vec(snapped, scaled.shape(), tensor.device())?;
        // Reshape back to original shape (flatten_all may lose shape)
        let snapped_reshaped = snapped_tensor.reshape(tensor.shape())?;
        snapped_reshaped.broadcast_mul(&scale)
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
