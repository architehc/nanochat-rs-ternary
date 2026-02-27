//! Transformer block with mHC wiring for training.

#[cfg(test)]
use candle_core::DType;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::attention::AttentionTrain;
use crate::config::TrainConfig;
use crate::engram::EngramTrain;
use crate::ffn::FeedForwardTrain;
use crate::layers::RMSNormTrain;
use crate::mhc::MhcLiteN2Train;
use crate::wavefield::WaveFieldAttentionTrain;

/// Attention layer variant for training.
pub enum AttentionTrainLayer {
    Standard(AttentionTrain),
    WaveField(WaveFieldAttentionTrain),
}

/// A single transformer block with mHC residual connections.
pub struct TransformerBlockTrain {
    pub mhc_attn: MhcLiteN2Train,
    pub mhc_ffn: MhcLiteN2Train,
    pub norm_attn: RMSNormTrain,
    pub norm_ffn: RMSNormTrain,
    pub attention: AttentionTrainLayer,
    pub ffn: FeedForwardTrain,
    pub engram: Option<EngramTrain>,
    pub dim: usize,
    pub max_seq_len: usize,
}

impl TransformerBlockTrain {
    pub fn new(config: &TrainConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_standard(config, vb)
    }

    /// Create a standard attention block.
    pub fn new_standard(config: &TrainConfig, vb: VarBuilder) -> Result<Self> {
        let ffn_dim = config.ffn_dim();
        Ok(Self {
            mhc_attn: MhcLiteN2Train::new(vb.pp("mhc_attn"))?,
            mhc_ffn: MhcLiteN2Train::new(vb.pp("mhc_ffn"))?,
            norm_attn: RMSNormTrain::new(config.dim, vb.pp("norm_attn"))?,
            norm_ffn: RMSNormTrain::new(config.dim, vb.pp("norm_ffn"))?,
            attention: AttentionTrainLayer::Standard(AttentionTrain::new(
                config.dim,
                config.n_heads,
                config.n_kv_heads,
                config.group_size,
                vb.pp("attn"),
            )?),
            ffn: FeedForwardTrain::new(config.dim, ffn_dim, config.group_size, vb.pp("ffn"))?,
            engram: None,
            dim: config.dim,
            max_seq_len: config.max_seq_len,
        })
    }

    /// Create a wave field attention block.
    pub fn new_wavefield(config: &TrainConfig, vb: VarBuilder) -> Result<Self> {
        let ffn_dim = config.ffn_dim();
        let wf_heads = if config.wavefield_n_heads == 0 {
            config.n_heads
        } else {
            config.wavefield_n_heads
        };
        let wf_head_dim = config.dim / wf_heads;
        Ok(Self {
            mhc_attn: MhcLiteN2Train::new(vb.pp("mhc_attn"))?,
            mhc_ffn: MhcLiteN2Train::new(vb.pp("mhc_ffn"))?,
            norm_attn: RMSNormTrain::new(config.dim, vb.pp("norm_attn"))?,
            norm_ffn: RMSNormTrain::new(config.dim, vb.pp("norm_ffn"))?,
            attention: AttentionTrainLayer::WaveField(WaveFieldAttentionTrain::new(
                config.dim,
                wf_heads,
                wf_head_dim,
                config.wavefield_field_size,
                config.group_size,
                config.wavefield_head_coupling,
                match config.wavefield_convolve_mode.as_deref() {
                    Some("fwht") => crate::wavefield::ConvolveMode::Fwht,
                    Some("haar") => crate::wavefield::ConvolveMode::Haar,
                    Some("fft") | None => crate::wavefield::ConvolveMode::Fft,
                    Some(other) => {
                        return Err(candle_core::Error::Msg(format!(
                            "unknown wavefield_convolve_mode '{}'; expected 'fft', 'fwht', or 'haar'",
                            other
                        )));
                    }
                },
                config.wavefield_haar_levels,
                config.wavefield_haar_direct, // per_elem_gate: new code uses per-element gating
                vb.pp("wavefield"),
            )?),
            ffn: FeedForwardTrain::new(config.dim, ffn_dim, config.group_size, vb.pp("ffn"))?,
            engram: None,
            dim: config.dim,
            max_seq_len: config.max_seq_len,
        })
    }

    /// Attach an Engram module to this block.
    pub fn with_engram(mut self, engram: EngramTrain) -> Self {
        self.engram = Some(engram);
        self
    }

    /// Forward: x_exp [batch, seq, 2*dim] -> [batch, seq, 2*dim]
    ///
    /// `engram_indices` is optional — only needed when this block has an Engram module.
    /// Pass precomputed hash index tensors (from `EngramTrain::precompute_hash_indices()`).
    pub fn forward(
        &self,
        x_exp: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        engram_indices: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        // Attention sub-layer with mHC
        let attn_in = self.mhc_attn.prepare_input(x_exp, self.dim)?;

        // Apply Engram enrichment before attention (if present)
        let attn_in = if let (Some(engram), Some(indices)) = (&self.engram, engram_indices) {
            engram.forward(&attn_in, indices)?
        } else {
            attn_in
        };

        let attn_normed = self.norm_attn.forward(&attn_in)?;
        let attn_out = match &self.attention {
            AttentionTrainLayer::Standard(attn) => attn.forward(&attn_normed, cos, sin)?,
            AttentionTrainLayer::WaveField(wf) => wf.forward(&attn_normed, self.max_seq_len)?,
        };
        let x_exp = self.mhc_attn.apply(x_exp, &attn_out, self.dim)?;

        // FFN sub-layer with mHC
        let ffn_in = self.mhc_ffn.prepare_input(&x_exp, self.dim)?;
        let ffn_normed = self.norm_ffn.forward(&ffn_in)?;
        let ffn_out = self.ffn.forward(&ffn_normed)?;
        self.mhc_ffn.apply(&x_exp, &ffn_out, self.dim)
    }

    /// Collect all linear weight parameters.
    pub fn linear_params(&self) -> Vec<&Tensor> {
        let mut params = match &self.attention {
            AttentionTrainLayer::Standard(attn) => attn.linear_params(),
            AttentionTrainLayer::WaveField(wf) => wf.linear_params(),
        };
        params.extend(self.ffn.linear_params());
        if let Some(engram) = &self.engram {
            params.extend(engram.linear_params());
        }
        params
    }

    /// Collect all mHC parameters (includes wave field physics params).
    pub fn mhc_params(&self) -> Vec<&Tensor> {
        let mut params = self.mhc_attn.params().into_iter().collect::<Vec<_>>();
        params.extend(self.mhc_ffn.params());
        if let AttentionTrainLayer::WaveField(wf) = &self.attention {
            params.extend(wf.physics_params());
        }
        params
    }

    /// Collect norm parameters.
    pub fn norm_params(&self) -> Vec<&Tensor> {
        let mut params = vec![self.norm_attn.weight(), self.norm_ffn.weight()];
        if let Some(engram) = &self.engram {
            params.extend(engram.norm_params());
            params.extend(engram.conv_params());
        }
        params
    }

    /// Collect engram table (embedding) parameters for separate optimizer group.
    pub fn engram_table_params(&self) -> Vec<&Tensor> {
        if let Some(engram) = &self.engram {
            engram.table_params()
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::precompute_rope_freqs;
    use candle_core::Device;
    use candle_nn::VarMap;

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
            loop_config: None,
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
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,
            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
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
        let y = block.forward(&x_exp, &cos_s, &sin_s, None)?;
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
        let y = block.forward(&x_exp, &cos_s, &sin_s, None)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        // mHC params should have gradients
        let g = grads
            .get(&block.mhc_attn.pre_logits)
            .expect("mHC attn pre_logits gradient");
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
