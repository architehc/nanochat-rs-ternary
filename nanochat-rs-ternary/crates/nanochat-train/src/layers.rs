//! Trainable layers: BitLinearSTE, RMSNorm, Embedding (Candle versions).
//! Implemented in Phase B.

use candle_core::{Result, Tensor};
#[cfg(test)]
use candle_core::DType;
use candle_nn::VarBuilder;

use crate::quantize::{absmean_quantize, dequantize_ternary, per_token_absmax_quantize, dequantize_activations};

/// Ternary linear layer with STE for training.
///
/// Maintains FP32 shadow weights; quantizes to ternary in forward pass.
/// Gradients flow through via Straight-Through Estimator.
pub struct BitLinearSTE {
    weight: Tensor, // [out_features, in_features]
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: usize,
}

impl BitLinearSTE {
    pub fn new(in_f: usize, out_f: usize, gs: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints((out_f, in_f), "weight", candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 })?;
        Ok(Self {
            weight,
            in_features: in_f,
            out_features: out_f,
            group_size: gs,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Weight quantization with STE
        let (w_ternary, scales) = absmean_quantize(&self.weight, self.group_size)?;
        let w_deq = dequantize_ternary(&w_ternary, &scales, self.group_size)?;
        // STE: forward = quantized, backward = identity to shadow weight
        let w_ste = (&self.weight + (&w_deq - &self.weight)?.detach())?;

        // Activation quantization with STE
        let (x_q, act_scales) = per_token_absmax_quantize(x)?;
        let x_deq = dequantize_activations(&x_q, &act_scales)?;
        let x_ste = (x + (&x_deq - x)?.detach())?;

        // Linear: x_ste @ w_ste^T (handles 3D+ batched input)
        let w = w_ste.t()?;
        let x_dims = x_ste.dims().to_vec();
        if x_dims.len() == 3 {
            let (b, m, k) = (x_dims[0], x_dims[1], x_dims[2]);
            x_ste.reshape((b * m, k))?.matmul(&w)?.reshape((b, m, ()))
        } else {
            x_ste.matmul(&w)
        }
    }

    /// Get the weight tensor (for param groups, export, etc.).
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Extract ternary weights for export (no grad tracking).
    pub fn get_ternary_weights(&self) -> Result<(Tensor, Tensor)> {
        absmean_quantize(&self.weight.detach(), self.group_size)
    }
}

/// RMSNorm for training (differentiable).
pub struct RMSNormTrain {
    weight: Tensor, // [dim]
    eps: f64,
    pub dim: usize,
}

impl RMSNormTrain {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps: 1e-6, dim })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len() - 1;
        let variance = x.sqr()?.mean_keepdim(last_dim)?;
        let denominator = (variance + self.eps)?.sqrt()?;
        x.broadcast_div(&denominator)?.broadcast_mul(&self.weight)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_bitlinear_ste_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let layer = BitLinearSTE::new(128, 256, 128, vb.pp("test"))?;

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 128), &device)?;
        let y = layer.forward(&x)?;
        assert_eq!(y.dims(), &[2, 8, 256]);
        Ok(())
    }

    #[test]
    fn test_bitlinear_ste_gradient_flows() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let layer = BitLinearSTE::new(64, 32, 64, vb.pp("test"))?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 1, 64), &device)?;
        let y = layer.forward(&x)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        // Gradient should exist for weight
        let grad = grads.get(&layer.weight).expect("Gradient should exist for weight");
        let grad_norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(grad_norm > 0.0, "Gradient norm should be non-zero");
        Ok(())
    }

    #[test]
    fn test_bitlinear_ste_ternary_export() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let layer = BitLinearSTE::new(128, 64, 128, vb.pp("test"))?;

        let (w_t, _scales) = layer.get_ternary_weights()?;
        assert_eq!(w_t.dims(), &[64, 128]);
        // All values should be ternary
        let vals = w_t.flatten_all()?.to_vec1::<f32>()?;
        for &v in &vals {
            assert!(
                (v + 1.0).abs() < 1e-6 || v.abs() < 1e-6 || (v - 1.0).abs() < 1e-6,
                "Non-ternary: {}", v
            );
        }
        Ok(())
    }

    #[test]
    fn test_rmsnorm_output_normalized() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let norm = RMSNormTrain::new(64, vb.pp("test"))?;

        let x = Tensor::randn(0.0f32, 1.0, (2, 4, 64), &device)?;
        let y = norm.forward(&x)?;

        // Output should be finite
        let y_flat = y.flatten_all()?.to_vec1::<f32>()?;
        for &v in &y_flat {
            assert!(v.is_finite(), "Non-finite output: {}", v);
        }
        // RMS of output should be close to 1.0 (weight is 1.0)
        let rms = y.sqr()?.mean_keepdim(2)?.sqrt()?;
        let rms_vals = rms.flatten_all()?.to_vec1::<f32>()?;
        for &v in &rms_vals {
            assert!((v - 1.0).abs() < 0.1, "RMS should be ~1.0, got {}", v);
        }
        Ok(())
    }

    #[test]
    fn test_rmsnorm_gradient_flows() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let norm = RMSNormTrain::new(32, vb.pp("test"))?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 1, 32), &device)?;
        let y = norm.forward(&x)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let grad = grads.get(&norm.weight).expect("Gradient should exist");
        let grad_norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(grad_norm > 0.0, "Gradient should be non-zero");
        Ok(())
    }
}
