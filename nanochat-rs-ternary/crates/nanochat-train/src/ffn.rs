//! SwiGLU Feed-Forward Network for training.

use candle_core::{Result, Tensor};
#[cfg(test)]
use candle_core::DType;
use candle_nn::VarBuilder;

use crate::layers::BitLinearSTE;

/// SwiGLU FFN: gate * silu(w_gate(x)) * w_up(x), then w_down.
pub struct FeedForwardTrain {
    pub w_gate: BitLinearSTE,
    pub w_up: BitLinearSTE,
    pub w_down: BitLinearSTE,
}

impl FeedForwardTrain {
    pub fn new(dim: usize, ffn_dim: usize, group_size: usize, vb: VarBuilder) -> Result<Self> {
        let w_gate = BitLinearSTE::new(dim, ffn_dim, group_size, vb.pp("w_gate"))?;
        let w_up = BitLinearSTE::new(dim, ffn_dim, group_size, vb.pp("w_up"))?;
        let w_down = BitLinearSTE::new(ffn_dim, dim, group_size, vb.pp("w_down"))?;
        Ok(Self { w_gate, w_up, w_down })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        let hidden = (gate * up)?;
        self.w_down.forward(&hidden)
    }

    pub fn linear_params(&self) -> Vec<&Tensor> {
        vec![self.w_gate.weight(), self.w_up.weight(), self.w_down.weight()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_ffn_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let ffn = FeedForwardTrain::new(64, 128, 64, vb.pp("test"))?;

        let x = Tensor::randn(0.0f32, 1.0, (2, 4, 64), &device)?;
        let y = ffn.forward(&x)?;
        assert_eq!(y.dims(), &[2, 4, 64]);
        Ok(())
    }

    #[test]
    fn test_ffn_gradient_flows() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let ffn = FeedForwardTrain::new(64, 128, 64, vb.pp("test"))?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 1, 64), &device)?;
        let y = ffn.forward(&x)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        for (name, w) in [("gate", ffn.w_gate.weight()), ("up", ffn.w_up.weight()), ("down", ffn.w_down.weight())] {
            let grad = grads.get(w).unwrap_or_else(|| panic!("{} should have gradient", name));
            let gn = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            assert!(gn > 0.0, "{} gradient should be non-zero", name);
        }
        Ok(())
    }
}
