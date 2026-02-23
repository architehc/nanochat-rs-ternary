//! Full training model: embedding -> blocks -> LM head.

#[cfg(test)]
use candle_core::DType;
use candle_core::{Result, Tensor};
use candle_nn::{self, Module, VarBuilder};

use crate::attention::precompute_rope_freqs;
use crate::block::TransformerBlockTrain;
use crate::config::TrainConfig;
use crate::layers::RMSNormTrain;
use crate::loop_block::SharedLoopBlock;
use crate::mhc::MhcLiteN2Train;

/// Parameter groups for split optimizers (Muon for linear, Lion for rest).
pub struct ParamGroups {
    pub linear: Vec<Tensor>,
    pub mhc: Vec<Tensor>,
    pub norm: Vec<Tensor>,
    pub embed: Vec<Tensor>,
}

/// Full nanochat training model.
pub struct NanochatTrainModel {
    pub config: TrainConfig,
    pub tok_embed: candle_nn::Embedding,
    // Standard architecture (used when loop_config is None)
    pub blocks: Vec<TransformerBlockTrain>,
    // LoopLM architecture (used when loop_config is Some)
    pub local_blocks_before: Vec<TransformerBlockTrain>,
    pub shared_loop_block: Option<SharedLoopBlock>,
    pub local_blocks_after: Vec<TransformerBlockTrain>,
    pub norm_final: RMSNormTrain,
    pub lm_head_weight: Option<Tensor>, // None if weight-tied
    freqs_cos: Tensor,
    freqs_sin: Tensor,
}

impl NanochatTrainModel {
    pub fn new(config: &TrainConfig, vb: VarBuilder) -> Result<Self> {
        // NOTE: candle_nn::embedding uses Init::Randn { stdev: 1.0 } by default,
        // which is way too large for weight-tied models (produces logits with absmax ~800).
        // Use GPT-2 standard: N(0, 0.02) for proper loss scale at initialization.
        let embed_weights = vb.pp("tok_embed").get_with_hints(
            (config.vocab_size, config.dim),
            "weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let tok_embed = candle_nn::Embedding::new(embed_weights, config.dim);

        // Build architecture based on loop_config
        let (blocks, local_blocks_before, shared_loop_block, local_blocks_after) =
            if let Some(loop_cfg) = &config.loop_config {
                // LoopLM architecture: local_before + shared_loop + local_after
                // Respect wavefield config for local layers (shared loop block stays standard)
                let before: Vec<_> = (0..loop_cfg.local_before)
                    .map(|i| {
                        if config.is_wavefield_layer(i) {
                            TransformerBlockTrain::new_wavefield(config, vb.pp(format!("local_before.{i}")))
                        } else {
                            TransformerBlockTrain::new(config, vb.pp(format!("local_before.{i}")))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                let shared = SharedLoopBlock::new(config, vb.pp("shared_loop"))?;

                let after_offset = loop_cfg.local_before + 1; // +1 for shared block
                let after: Vec<_> = (0..loop_cfg.local_after)
                    .map(|i| {
                        if config.is_wavefield_layer(after_offset + i) {
                            TransformerBlockTrain::new_wavefield(config, vb.pp(format!("local_after.{i}")))
                        } else {
                            TransformerBlockTrain::new(config, vb.pp(format!("local_after.{i}")))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                (Vec::new(), before, Some(shared), after)
            } else {
                // Standard architecture: n_layers blocks
                // Check each layer for wave field vs standard attention
                let blocks = (0..config.n_layers)
                    .map(|i| {
                        if config.is_wavefield_layer(i) {
                            TransformerBlockTrain::new_wavefield(config, vb.pp(format!("blocks.{i}")))
                        } else {
                            TransformerBlockTrain::new(config, vb.pp(format!("blocks.{i}")))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                (blocks, Vec::new(), None, Vec::new())
            };

        let norm_final = RMSNormTrain::new(config.dim, vb.pp("norm_final"))?;

        let lm_head_weight = if !config.weight_tied {
            let w = vb.get_with_hints(
                (config.vocab_size, config.dim),
                "lm_head.weight",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )?;
            Some(w)
        } else {
            None
        };

        let head_dim = config.dim / config.n_heads;
        let (freqs_cos, freqs_sin) =
            precompute_rope_freqs(head_dim, config.max_seq_len, config.rope_theta, vb.device())?;

        Ok(Self {
            config: config.clone(),
            tok_embed,
            blocks,
            local_blocks_before,
            shared_loop_block,
            local_blocks_after,
            norm_final,
            lm_head_weight,
            freqs_cos,
            freqs_sin,
        })
    }

    /// Forward: token_ids [batch, seq_len] -> logits [batch, seq_len, vocab_size]
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (logits, _hidden) = self.forward_with_hidden(token_ids)?;
        Ok(logits)
    }

    /// Forward pass returning the final hidden state before LM head.
    pub fn forward_hidden_only(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len) = token_ids.dims2()?;

        // 1. Embed
        let x = self.tok_embed.forward(token_ids)?; // [batch, seq, dim]

        // 2. Expand to 2 streams
        let mut x_exp = MhcLiteN2Train::expand_input(&x, self.config.dim)?;

        // 3. Slice RoPE freqs for this sequence length
        let cos = self.freqs_cos.narrow(0, 0, seq_len)?;
        let sin = self.freqs_sin.narrow(0, 0, seq_len)?;

        // 4. Transformer blocks (loop or standard)
        if let Some(ref loop_block) = self.shared_loop_block {
            let loop_cfg = self.config.loop_config.as_ref().unwrap();

            // 4a. Local layers before
            for block in &self.local_blocks_before {
                x_exp = block.forward(&x_exp, &cos, &sin)?;
            }

            // 4b. Shared loop (N iterations)
            let mut global_state: Option<Tensor> = None;
            for _ in 0..loop_cfg.loop_count {
                let (x_out, g_state) = loop_block.forward(&x_exp, global_state.as_ref())?;
                x_exp = x_out;
                global_state = Some(g_state);
            }

            // 4c. Local layers after
            for block in &self.local_blocks_after {
                x_exp = block.forward(&x_exp, &cos, &sin)?;
            }
        } else {
            // Standard architecture
            for block in &self.blocks {
                x_exp = block.forward(&x_exp, &cos, &sin)?;
            }
        }

        // 5. Collapse to single stream
        let x = MhcLiteN2Train::collapse_output(&x_exp, self.config.dim)?;

        // 6. Final norm
        self.norm_final.forward(&x)
    }

    /// Project hidden states to logits via LM head.
    ///
    /// Supports both 3D hidden states `[batch, seq, dim]` and compact 2D hidden states `[n, dim]`.
    pub fn project_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let lm_w = if self.config.weight_tied {
            self.tok_embed.embeddings().t()?
        } else {
            self.lm_head_weight.as_ref().unwrap().t()?
        };
        let x_dims = hidden.dims().to_vec();
        if x_dims.len() == 3 {
            let (b, m, k) = (x_dims[0], x_dims[1], x_dims[2]);
            hidden
                .reshape((b * m, k))?
                .matmul(&lm_w)?
                .reshape((b, m, ()))
        } else {
            hidden.matmul(&lm_w)
        }
    }

    /// Forward pass returning both logits and last hidden state (for MTP).
    pub fn forward_with_hidden(&self, token_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let hidden = self.forward_hidden_only(token_ids)?;
        let logits = self.project_hidden_to_logits(&hidden)?;
        Ok((logits, hidden))
    }

    /// Forward with cross-entropy loss.
    /// input_ids: [batch, seq_len]
    /// target_ids: [batch, seq_len]
    /// Returns: scalar loss tensor
    pub fn forward_loss(&self, input_ids: &Tensor, target_ids: &Tensor) -> Result<Tensor> {
        let logits = self.forward(input_ids)?; // [batch, seq, vocab]

        // Reshape for cross-entropy: [batch*seq, vocab] and [batch*seq]
        let (batch, seq_len, _vocab) = logits.dims3()?;
        let logits_flat = logits.reshape((batch * seq_len, self.config.vocab_size))?;
        let targets_flat = target_ids.flatten_all()?;

        // Cross-entropy loss
        candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)
    }

    /// Collect all parameters grouped by optimizer assignment.
    pub fn param_groups(&self) -> ParamGroups {
        let mut linear = Vec::new();
        let mut mhc = Vec::new();
        let mut norm = Vec::new();
        let embed = vec![self.tok_embed.embeddings().clone()];

        // Standard blocks
        for block in &self.blocks {
            linear.extend(block.linear_params().into_iter().cloned());
            mhc.extend(block.mhc_params().into_iter().cloned());
            norm.extend(block.norm_params().into_iter().cloned());
        }

        // LoopLM blocks
        for block in &self.local_blocks_before {
            linear.extend(block.linear_params().into_iter().cloned());
            mhc.extend(block.mhc_params().into_iter().cloned());
            norm.extend(block.norm_params().into_iter().cloned());
        }

        if let Some(ref loop_block) = self.shared_loop_block {
            linear.extend(loop_block.linear_params().into_iter().cloned());
            mhc.extend(loop_block.mhc_params().into_iter().cloned());
            norm.extend(loop_block.norm_params().into_iter().cloned());
        }

        for block in &self.local_blocks_after {
            linear.extend(block.linear_params().into_iter().cloned());
            mhc.extend(block.mhc_params().into_iter().cloned());
            norm.extend(block.norm_params().into_iter().cloned());
        }

        norm.push(self.norm_final.weight().clone());

        if let Some(ref w) = self.lm_head_weight {
            linear.push(w.clone());
        }

        ParamGroups {
            linear,
            mhc,
            norm,
            embed,
        }
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        let groups = self.param_groups();
        let count = |tensors: &[Tensor]| -> usize { tensors.iter().map(|t| t.elem_count()).sum() };
        count(&groups.linear) + count(&groups.mhc) + count(&groups.norm) + count(&groups.embed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainConfig;
    use candle_core::{Device, D};
    use candle_nn::VarMap;

    fn tiny_config() -> TrainConfig {
        TrainConfig {
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.0,
            vocab_size: 256,
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
        }
    }

    #[test]
    fn test_model_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let model = NanochatTrainModel::new(&cfg, vb)?;

        let ids = Tensor::zeros((2, 8), DType::U32, &device)?;
        let logits = model.forward(&ids)?;
        assert_eq!(logits.dims(), &[2, 8, 256]);
        Ok(())
    }

    #[test]
    fn test_model_loss_backward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let model = NanochatTrainModel::new(&cfg, vb)?;

        let ids = Tensor::zeros((1, 4), DType::U32, &device)?;
        let targets = Tensor::zeros((1, 4), DType::U32, &device)?;
        let logits = model.forward(&ids)?;

        let (batch, seq, vocab) = logits.dims3()?;
        let logits_flat = logits.reshape((batch * seq, vocab))?;
        let targets_flat = targets.reshape(batch * seq)?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        let _grads = loss.backward()?;

        let loss_val = loss.to_scalar::<f32>()?;
        assert!(loss_val.is_finite(), "Loss should be finite: {}", loss_val);
        Ok(())
    }

    #[test]
    fn test_model_weight_tied() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut cfg = tiny_config();
        cfg.weight_tied = true;
        let model = NanochatTrainModel::new(&cfg, vb)?;
        assert!(model.lm_head_weight.is_none());

        // Should still produce valid logits
        let ids = Tensor::zeros((1, 4), DType::U32, &device)?;
        let logits = model.forward(&ids)?;
        assert_eq!(logits.dim(D::Minus1)?, cfg.vocab_size);
        Ok(())
    }

    #[test]
    fn test_weight_tied_embeddings_receive_lm_head_gradients() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut cfg = tiny_config();
        cfg.weight_tied = true;
        let model = NanochatTrainModel::new(&cfg, vb)?;

        let ids = Tensor::zeros((1, 4), DType::U32, &device)?;
        let targets = Tensor::zeros((1, 4), DType::U32, &device)?;
        let loss = model.forward_loss(&ids, &targets)?;
        let grads = loss.backward()?;

        let emb = model.tok_embed.embeddings();
        let grad = grads
            .get(emb)
            .ok_or_else(|| candle_core::Error::Msg("missing embedding gradient".to_string()))?;
        let grad_norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(
            grad_norm.is_finite() && grad_norm > 0.0,
            "embedding grad norm should be finite and non-zero, got {}",
            grad_norm
        );
        Ok(())
    }

    #[test]
    fn test_model_param_groups_complete() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let model = NanochatTrainModel::new(&cfg, vb)?;
        let groups = model.param_groups();

        // 2 layers * 7 linear = 14 linear params (weight-tied, no lm_head)
        assert_eq!(groups.linear.len(), 14);
        // 2 layers * 10 mhc = 20
        assert_eq!(groups.mhc.len(), 20);
        // 2 layers * 2 norms + 1 final norm = 5
        assert_eq!(groups.norm.len(), 5);
        // 1 embedding
        assert_eq!(groups.embed.len(), 1);
        Ok(())
    }

    #[test]
    fn test_model_param_count() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let model = NanochatTrainModel::new(&cfg, vb)?;
        let count = model.param_count();
        // Should be > 0 and reasonable for tiny config
        assert!(count > 10_000, "Too few params: {}", count);
        assert!(
            count < 1_000_000,
            "Too many params for tiny config: {}",
            count
        );
        Ok(())
    }

    #[test]
    fn test_loop_architecture_construction() -> Result<()> {
        let device = Device::Cpu;

        // Test standard config (no loop)
        let cfg_std = TrainConfig::d20();
        assert!(cfg_std.loop_config.is_none());
        let varmap_std = VarMap::new();
        let vb_std = VarBuilder::from_varmap(&varmap_std, DType::F32, &device);
        let model_std = NanochatTrainModel::new(&cfg_std, vb_std)?;

        assert_eq!(
            model_std.blocks.len(),
            6,
            "d20 should have 6 standard blocks"
        );
        assert_eq!(model_std.local_blocks_before.len(), 0);
        assert_eq!(model_std.local_blocks_after.len(), 0);
        assert!(model_std.shared_loop_block.is_none());

        // Test loop config
        let cfg_loop = TrainConfig::d20_loop();
        assert!(cfg_loop.loop_config.is_some());
        let loop_cfg = cfg_loop.loop_config.as_ref().unwrap();

        let varmap_loop = VarMap::new();
        let vb_loop = VarBuilder::from_varmap(&varmap_loop, DType::F32, &device);
        let model_loop = NanochatTrainModel::new(&cfg_loop, vb_loop)?;

        assert_eq!(
            model_loop.blocks.len(),
            0,
            "d20_loop should have 0 standard blocks"
        );
        assert_eq!(model_loop.local_blocks_before.len(), loop_cfg.local_before);
        assert_eq!(model_loop.local_blocks_after.len(), loop_cfg.local_after);
        assert!(model_loop.shared_loop_block.is_some());

        Ok(())
    }
}
