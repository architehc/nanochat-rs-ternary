//! FP4 training utilities (Blackwell-oriented scaffold).
//!
//! This module provides a software FP4 quantization path that can be used to
//! prototype Blackwell-style mixed-precision training flows.

use candle_core::{DType, Result, Tensor};

/// FP4 training controller.
///
/// Current behavior:
/// - forward path dtype target: BF16
/// - backward path quantization target: FP4 (simulated in software)
/// - optional stochastic rounding flag (reserved for CUDA kernels)
pub struct FP4Trainer {
    pub forward_dtype: DType,
    pub backward_dtype: DType,
    pub stochastic_rounding: bool,
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
    /// This is a no-op scaffold until dedicated CUDA bindings are wired.
    pub fn enable_fp4_tensor_cores(&self) -> Result<()> {
        Ok(())
    }

    /// Quantize tensor values to nearest FP4 table entries.
    pub fn quantize_fp4(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.dims().to_vec();
        let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
        let quantized: Vec<f32> = flat.iter().map(|&v| self.nearest_fp4(v)).collect();
        let q = Tensor::from_vec(quantized, flat.len(), tensor.device())?;
        q.reshape(dims)
    }

    fn nearest_fp4(&self, x: f32) -> f32 {
        let mut best = self.fp4_table[0];
        let mut best_dist = (x - best).abs();
        for &cand in &self.fp4_table[1..] {
            let d = (x - cand).abs();
            if d < best_dist {
                best_dist = d;
                best = cand;
            }
        }
        best
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

        for v in vals {
            assert!(
                fp4.fp4_table.contains(&v),
                "quantized value not in FP4 table: {}",
                v
            );
        }
        Ok(())
    }
}
