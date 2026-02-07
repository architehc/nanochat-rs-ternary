//! Transformer block with mHC wiring for training.

use candle_core::{Result, Tensor};
#[cfg(test)]
use candle_core::DType;
use candle_nn::VarBuilder;

use crate::attention::AttentionTrain;
use crate::config::TrainConfig;
use crate::ffn::FeedForwardTrain;
use crate::layers::RMSNormTrain;
use crate::mhc::MhcLiteN2Train;

/// A single transformer block with mHC residual connections.
pub struct TransformerBlockTrain {
    pub mhc_attn: MhcLiteN2Train,
    pub mhc_ffn: MhcLiteN2Train,
    pub norm_attn: RMSNormTrain,
    pub norm_ffn: RMSNormTrain,
    pub attention: AttentionTrain,
    pub ffn: FeedForwardTrain,
    pub dim: usize,
}

impl TransformerBlockTrain {
    pub fn new(config: &TrainConfig, vb: VarBuilder) -> Result<Self> {
        let ffn_dim = config.ffn_dim();
        Ok(Self {
            mhc_attn: MhcLiteN2Train::new(vb.pp("mhc_attn"))?,
            mhc_ffn: MhcLiteN2Train::new(vb.pp("mhc_ffn"))?,
            norm_attn: RMSNormTrain::new(config.dim, vb.pp("norm_attn"))?,
            norm_ffn: RMSNormTrain::new(config.dim, vb.pp("norm_ffn"))?,
            attention: AttentionTrain::new(
                config.dim, config.n_heads, config.n_kv_heads, config.group_size,
                vb.pp("attn"),
            )?,
            ffn: FeedForwardTrain::new(config.dim, ffn_dim, config.group_size, vb.pp("ffn"))?,
            dim: config.dim,
        })
    }

    /// Forward: x_exp [batch, seq, 2*dim] -> [batch, seq, 2*dim]
    pub fn forward(&self, x_exp: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // Attention sub-layer with mHC
        let attn_in = self.mhc_attn.prepare_input(x_exp, self.dim)?;
        let attn_normed = self.norm_attn.forward(&attn_in)?;
        let attn_out = self.attention.forward(&attn_normed, cos, sin)?;
        let x_exp = self.mhc_attn.apply(x_exp, &attn_out, self.dim)?;

        // FFN sub-layer with mHC
        let ffn_in = self.mhc_ffn.prepare_input(&x_exp, self.dim)?;
        let ffn_normed = self.norm_ffn.forward(&ffn_in)?;
        let ffn_out = self.ffn.forward(&ffn_normed)?;
        self.mhc_ffn.apply(&x_exp, &ffn_out, self.dim)
    }

    /// Collect all linear weight parameters.
    pub fn linear_params(&self) -> Vec<&Tensor> {
        let mut params = self.attention.linear_params();
        params.extend(self.ffn.linear_params());
        params
    }

    /// Collect all mHC parameters.
    pub fn mhc_params(&self) -> Vec<&Tensor> {
        let mut params = self.mhc_attn.params().into_iter().collect::<Vec<_>>();
        params.extend(self.mhc_ffn.params());
        params
    }

    /// Collect norm parameters.
    pub fn norm_params(&self) -> Vec<&Tensor> {
        vec![self.norm_attn.weight(), self.norm_ffn.weight()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    use crate::attention::precompute_rope_freqs;

    fn test_config() -> TrainConfig {
        TrainConfig {
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.0,
            vocab_size: 100,
            max_seq_len: 32,
            group_size: 64,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 10,
            total_steps: 100,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 5,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
        }
    }

    #[test]
    fn test_block_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = test_config();
        let block = TransformerBlockTrain::new(&cfg, vb.pp("test"))?;

        let (cos, sin) = precompute_rope_freqs(16, 32, 10000.0, &device)?;
        let cos_s = cos.narrow(0, 0, 4)?;
        let sin_s = sin.narrow(0, 0, 4)?;

        let x_exp = Tensor::randn(0.0f32, 1.0, (2, 4, 128), &device)?; // 2*dim=128
        let y = block.forward(&x_exp, &cos_s, &sin_s)?;
        assert_eq!(y.dims(), &[2, 4, 128]);
        Ok(())
    }

    #[test]
    fn test_block_gradient_to_mhc() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = test_config();
        let block = TransformerBlockTrain::new(&cfg, vb.pp("test"))?;

        let (cos, sin) = precompute_rope_freqs(16, 32, 10000.0, &device)?;
        let cos_s = cos.narrow(0, 0, 4)?;
        let sin_s = sin.narrow(0, 0, 4)?;

        let x_exp = Tensor::randn(0.0f32, 1.0, (1, 4, 128), &device)?;
        let y = block.forward(&x_exp, &cos_s, &sin_s)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        // mHC params should have gradients
        let g = grads.get(&block.mhc_attn.pre_logits).expect("mHC attn pre_logits gradient");
        let gn = g.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(gn > 0.0, "mHC gradient should be non-zero");
        Ok(())
    }

    #[test]
    fn test_block_param_collection() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = test_config();
        let block = TransformerBlockTrain::new(&cfg, vb.pp("test"))?;

        let linear = block.linear_params();
        assert_eq!(linear.len(), 7); // 4 attn + 3 ffn

        let mhc = block.mhc_params();
        assert_eq!(mhc.len(), 10); // 5 per mHC * 2

        let norm = block.norm_params();
        assert_eq!(norm.len(), 2);
        Ok(())
    }
}
