//! Differentiable mHC-lite module for training (Candle).

use candle_core::{Result, Tensor};
#[cfg(test)]
use candle_core::DType;
use candle_nn::VarBuilder;

/// Sigmoid helper that uses custom CUDA kernel when available.
#[inline]
fn sigmoid(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if t.device().is_cuda() {
            return crate::cuda_ops::cuda_sigmoid(t);
        }
    }
    candle_nn::ops::sigmoid(t)
}

/// Differentiable mHC-lite N=2 for training.
///
/// Mirrors `mhc_lite::MhcLiteN2` but uses Candle Vars for autograd.
pub struct MhcLiteN2Train {
    pub alpha_logit: Tensor,
    pub pre_logits: Tensor,
    pub pre_bias: Tensor,
    pub post_logits: Tensor,
    pub post_bias: Tensor,
}

impl MhcLiteN2Train {
    /// Identity-like initialization (alpha_logit=5.0 -> sigmoid~1.0).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let alpha_logit = vb.get_with_hints(1, "alpha_logit", candle_nn::Init::Const(5.0))?;
        let pre_logits = vb.get_with_hints(2, "pre_logits", candle_nn::Init::Const(0.0))?;
        let pre_bias = vb.get_with_hints(2, "pre_bias", candle_nn::Init::Const(0.5))?;
        let post_logits = vb.get_with_hints(2, "post_logits", candle_nn::Init::Const(0.0))?;
        let post_bias = vb.get_with_hints(2, "post_bias", candle_nn::Init::Const(0.5))?;
        Ok(Self { alpha_logit, pre_logits, pre_bias, post_logits, post_bias })
    }

    /// Compute 2x2 doubly stochastic residual matrix.
    pub fn h_res(&self) -> Result<(f32, f32)> {
        // alpha = sigmoid(alpha_logit)
        let alpha = sigmoid(&self.alpha_logit)?
            .to_vec1::<f32>()?[0];
        Ok((alpha, 1.0 - alpha))
    }

    /// Compute pre-projection weights (non-negative via sigmoid).
    pub fn h_pre(&self) -> Result<Tensor> {
        let raw = (&self.pre_logits + &self.pre_bias)?;
        sigmoid(&raw)
    }

    /// Compute post-projection weights (2x scaled sigmoid).
    pub fn h_post(&self) -> Result<Tensor> {
        let raw = (&self.post_logits + &self.post_bias)?;
        let sig = sigmoid(&raw)?;
        &sig * 2.0
    }

    /// Expand single stream to 2 streams: [batch, seq, dim] -> [batch, seq, 2*dim]
    pub fn expand_input(x: &Tensor, _dim: usize) -> Result<Tensor> {
        Tensor::cat(&[x, x], x.dims().len() - 1)
    }

    /// Collapse 2 streams back to 1: [batch, seq, 2*dim] -> [batch, seq, dim]
    pub fn collapse_output(x: &Tensor, dim: usize) -> Result<Tensor> {
        let last = x.dims().len() - 1;
        let s0 = x.narrow(last, 0, dim)?;
        let s1 = x.narrow(last, dim, dim)?;
        (&s0 + &s1)? / 2.0
    }

    /// Prepare input: mix 2 streams into 1 for sub-layer processing.
    /// x: [batch, seq, 2*dim] -> [batch, seq, dim]
    pub fn prepare_input(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        let last = x.dims().len() - 1;
        let s0 = x.narrow(last, 0, dim)?;
        let s1 = x.narrow(last, dim, dim)?;
        let h_pre = self.h_pre()?; // [2]
        let w0 = h_pre.get(0)?;
        let w1 = h_pre.get(1)?;
        let out = (s0.broadcast_mul(&w0)? + s1.broadcast_mul(&w1)?)?;
        Ok(out)
    }

    /// Apply residual: update expanded state with sub-layer output.
    /// x_exp: [batch, seq, 2*dim], layer_out: [batch, seq, dim]
    pub fn apply(&self, x_exp: &Tensor, layer_out: &Tensor, dim: usize) -> Result<Tensor> {
        let last = x_exp.dims().len() - 1;
        let s0 = x_exp.narrow(last, 0, dim)?;
        let s1 = x_exp.narrow(last, dim, dim)?;

        let alpha_t = sigmoid(&self.alpha_logit)?;
        let alpha = &alpha_t;
        let one_minus_alpha = (1.0 - &alpha_t)?;

        // H_res: [[alpha, 1-alpha], [1-alpha, alpha]]
        let new_s0 = (s0.broadcast_mul(alpha)? + s1.broadcast_mul(&one_minus_alpha)?)?;
        let new_s1 = (s0.broadcast_mul(&one_minus_alpha)? + s1.broadcast_mul(alpha)?)?;

        // H_post scaled layer output
        let h_post = self.h_post()?;
        let p0 = h_post.get(0)?;
        let p1 = h_post.get(1)?;
        let out_s0 = (new_s0 + layer_out.broadcast_mul(&p0)?)?;
        let out_s1 = (new_s1 + layer_out.broadcast_mul(&p1)?)?;

        Tensor::cat(&[&out_s0, &out_s1], last)
    }

    /// Collect all parameters for optimizer.
    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.alpha_logit, &self.pre_logits, &self.pre_bias, &self.post_logits, &self.post_bias]
    }

    /// Export to inference-format values.
    pub fn to_inference_values(&self) -> Result<mhc_lite::MhcLiteN2> {
        let alpha = self.alpha_logit.to_vec1::<f32>()?[0];
        let pre_l = self.pre_logits.to_vec1::<f32>()?;
        let pre_b = self.pre_bias.to_vec1::<f32>()?;
        let post_l = self.post_logits.to_vec1::<f32>()?;
        let post_b = self.post_bias.to_vec1::<f32>()?;
        Ok(mhc_lite::MhcLiteN2 {
            alpha_logit: alpha,
            pre_logits: [pre_l[0], pre_l[1]],
            pre_bias: [pre_b[0], pre_b[1]],
            post_logits: [post_l[0], post_l[1]],
            post_bias: [post_b[0], post_b[1]],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_mhc_n2_identity_init_ds() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mhc = MhcLiteN2Train::new(vb.pp("test"))?;

        let (alpha, one_m_alpha) = mhc.h_res()?;
        // With alpha_logit=5.0, sigmoid(5.0)≈0.9933 -> nearly identity
        assert!(alpha > 0.99, "Alpha should be ~1.0, got {}", alpha);
        assert!((alpha + one_m_alpha - 1.0).abs() < 1e-6, "Should sum to 1.0");
        Ok(())
    }

    #[test]
    fn test_mhc_n2_expand_collapse_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 4, 64), &device)?;
        let x_exp = MhcLiteN2Train::expand_input(&x, 64)?;
        assert_eq!(x_exp.dims(), &[2, 4, 128]);
        let x_col = MhcLiteN2Train::collapse_output(&x_exp, 64)?;
        assert_eq!(x_col.dims(), &[2, 4, 64]);

        // Since expand duplicates, collapse (average) should recover original
        let diff = (&x - &x_col)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "Collapse should recover original, diff={}", diff);
        Ok(())
    }

    #[test]
    fn test_mhc_n2_prepare_apply_shapes() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mhc = MhcLiteN2Train::new(vb.pp("test"))?;

        let x_exp = Tensor::randn(0.0f32, 1.0, (2, 4, 128), &device)?;
        let prepared = mhc.prepare_input(&x_exp, 64)?;
        assert_eq!(prepared.dims(), &[2, 4, 64]);

        let layer_out = Tensor::randn(0.0f32, 1.0, (2, 4, 64), &device)?;
        let result = mhc.apply(&x_exp, &layer_out, 64)?;
        assert_eq!(result.dims(), &[2, 4, 128]);
        Ok(())
    }

    #[test]
    fn test_mhc_n2_gradient_flows() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mhc = MhcLiteN2Train::new(vb.pp("test"))?;

        let x_exp = Tensor::randn(0.0f32, 1.0, (1, 1, 128), &device)?;
        let prepared = mhc.prepare_input(&x_exp, 64)?;
        let loss = prepared.sum_all()?;
        let grads = loss.backward()?;

        // Gradient should flow to alpha_logit
        let g = grads.get(&mhc.pre_logits).expect("Gradient should exist for pre_logits");
        let gn = g.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(gn > 0.0, "Gradient should be non-zero");
        Ok(())
    }

    #[test]
    fn test_mhc_n2_to_inference() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mhc = MhcLiteN2Train::new(vb.pp("test"))?;

        let inf = mhc.to_inference_values()?;
        assert!((inf.alpha_logit - 5.0).abs() < 1e-5, "Alpha logit should be 5.0");
        assert_eq!(inf.pre_bias, [0.5, 0.5]);
        Ok(())
    }
}
