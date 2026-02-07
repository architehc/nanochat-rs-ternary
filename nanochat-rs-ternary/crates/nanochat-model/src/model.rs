//! Full nanochat model: embed -> mHC expand -> blocks -> mHC collapse -> norm -> head.

use std::io;

use crate::attention::{Attention, RopeFreqs};
use crate::bitlinear::BitLinear;
use crate::block::{AttentionLayer, AttentionState, TransformerBlock};
use crate::config::ModelConfig;
use crate::deltanet::DeltaNetAttention;
use crate::embed::Embedding;
use crate::ffn::{FeedForward, FfnLayer, MoeExperts};
use crate::norm::RMSNorm;
use mhc_lite::MhcLiteN2;
use ternary_core::gguf::{GgufFile, GgufValue};

/// Determine whether a given layer should use DeltaNet attention based on the
/// deltanet_ratio config. Uses interleaved assignment: distributes DeltaNet layers
/// evenly among standard layers using stride-based placement.
///
/// For ratio=0.5 with 4 layers, layers 1 and 3 become DeltaNet (odd positions).
fn should_use_deltanet(layer_idx: usize, n_layers: usize, ratio: Option<f32>) -> bool {
    match ratio {
        None => false,
        Some(r) if r <= 0.0 => false,
        Some(r) if r >= 1.0 => true,
        Some(r) => {
            let n_deltanet = ((n_layers as f32) * r).round() as usize;
            if n_deltanet == 0 {
                return false;
            }
            // Stride-based interleaved assignment:
            // Place DeltaNet layers at evenly spaced positions.
            let stride = n_layers as f32 / n_deltanet as f32;
            for k in 0..n_deltanet {
                let dn_idx = (stride * k as f32 + stride / 2.0).floor() as usize;
                if dn_idx == layer_idx {
                    return true;
                }
            }
            false
        }
    }
}

/// Full nanochat-rs ternary model.
#[derive(Debug)]
pub struct NanochatModel {
    pub config: ModelConfig,
    pub tok_embed: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub norm_final: RMSNorm,
    pub lm_head: Option<BitLinear>,
    pub weight_tied: bool,
    pub rope: RopeFreqs,
    pub attn_states: Vec<AttentionState>,
}

impl NanochatModel {
    /// Create model with random weights (for testing / validation).
    pub fn new_random(config: ModelConfig) -> Self {
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);

        let blocks: Vec<_> = (0..config.n_layers)
            .map(|i| {
                if should_use_deltanet(i, config.n_layers, config.deltanet_ratio) {
                    TransformerBlock::new_random_deltanet(&config)
                } else {
                    TransformerBlock::new_random(&config)
                }
            })
            .collect();

        // Create attention states matching each block's type
        let attn_states: Vec<_> = blocks.iter()
            .map(|block| block.create_attn_state(&config))
            .collect();

        // LM head: vocab_size x dim (None if weight_tied)
        let weight_tied = config.weight_tied;
        let lm_head = if weight_tied {
            None
        } else {
            let lm_weights: Vec<f32> = (0..config.vocab_size * config.dim)
                .map(|i| {
                    let v = ((i as u32).wrapping_mul(2654435761) >> 16) % 200;
                    v as f32 / 100.0 - 1.0
                })
                .collect();
            Some(BitLinear::from_float(&lm_weights, config.vocab_size, config.dim, config.group_size))
        };

        Self {
            tok_embed: Embedding::new_random(config.vocab_size, config.dim, 42),
            blocks,
            norm_final: RMSNorm::new(config.dim),
            lm_head,
            weight_tied,
            rope,
            attn_states,
            config,
        }
    }

    /// Load model from GGUF weights + mHC binary file.
    pub fn from_gguf(gguf_path: &str, mhc_path: &str) -> io::Result<Self> {
        let gguf = GgufFile::open(gguf_path)?;

        // Extract config from GGUF metadata
        let config = Self::config_from_gguf(&gguf)?;
        let group_size = config.group_size;

        // Load embeddings (FP16 -> F32)
        let tok_embed = Self::load_embedding(&gguf, "tok_embed.weight")?;

        // Load mHC parameters
        let (_, mhc_layers) = mhc_lite::load_mhc_file(mhc_path)?;

        // Load transformer blocks
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("blocks.{i}");
            let use_deltanet = should_use_deltanet(i, config.n_layers, config.deltanet_ratio);

            // Attention weights
            let wq = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wq.weight"), group_size)?);
            let wk = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wk.weight"), group_size)?);
            let wv = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wv.weight"), group_size)?);
            let wo = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wo.weight"), group_size)?);

            let attention = if use_deltanet {
                let w_beta = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.attention.w_beta.weight"), group_size)?);
                AttentionLayer::DeltaNet(DeltaNetAttention {
                    wq, wk, wv, wo, w_beta,
                    n_heads: config.n_heads,
                    head_dim: config.head_dim(),
                })
            } else {
                AttentionLayer::Standard(Attention {
                    wq, wk, wv, wo,
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim(),
                    n_rep: config.n_rep(),
                })
            };

            // FFN weights — dense or MoE
            let ffn = if let Some(n_experts) = config.n_experts {
                let n_active = config.n_active_experts.unwrap_or(2);

                // Router weights
                let router = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.ffn.router.weight"), group_size)?);

                // Per-expert FFN weights
                let mut experts = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    let w_gate = BitLinear::new(gguf.load_planar_weights(
                        &format!("{prefix}.ffn.experts.{e}.w_gate.weight"), group_size)?);
                    let w_up = BitLinear::new(gguf.load_planar_weights(
                        &format!("{prefix}.ffn.experts.{e}.w_up.weight"), group_size)?);
                    let w_down = BitLinear::new(gguf.load_planar_weights(
                        &format!("{prefix}.ffn.experts.{e}.w_down.weight"), group_size)?);
                    experts.push(FeedForward {
                        w_gate, w_up, w_down,
                        ffn_dim: config.ffn_dim(),
                    });
                }

                FfnLayer::Moe(MoeExperts { router, experts, n_active })
            } else {
                let w_gate = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.ffn.w_gate.weight"), group_size)?);
                let w_up = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.ffn.w_up.weight"), group_size)?);
                let w_down = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.ffn.w_down.weight"), group_size)?);

                FfnLayer::Dense(FeedForward {
                    w_gate, w_up, w_down,
                    ffn_dim: config.ffn_dim(),
                })
            };

            // Norms
            let norm_attn = Self::load_norm(&gguf, &format!("{prefix}.norm_attn.weight"))?;
            let norm_ffn = Self::load_norm(&gguf, &format!("{prefix}.norm_ffn.weight"))?;

            // mHC params (order: attn_0, ffn_0, attn_1, ffn_1, ...)
            let mhc_attn = Self::extract_mhc_n2(&mhc_layers, i * 2)?;
            let mhc_ffn = Self::extract_mhc_n2(&mhc_layers, i * 2 + 1)?;

            blocks.push(TransformerBlock {
                mhc_attn, mhc_ffn,
                norm_attn, norm_ffn,
                attention, ffn,
                dim: config.dim,
            });
        }

        // Final norm
        let norm_final = Self::load_norm(&gguf, "norm_final.weight")?;

        // LM head (skip if weight_tied)
        let weight_tied = config.weight_tied;
        let lm_head = if weight_tied {
            None
        } else {
            Some(BitLinear::new(
                gguf.load_planar_weights("lm_head.weight", group_size)?))
        };

        // RoPE and attention states
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let attn_states: Vec<_> = blocks.iter()
            .map(|block| block.create_attn_state(&config))
            .collect();

        Ok(Self {
            tok_embed, blocks, norm_final, lm_head, weight_tied, rope, attn_states, config,
        })
    }

    fn config_from_gguf(gguf: &GgufFile) -> io::Result<ModelConfig> {
        let get_u32 = |key: &str| -> io::Result<usize> {
            match gguf.metadata.get(key) {
                Some(GgufValue::U32(v)) => Ok(*v as usize),
                _ => Err(io::Error::new(io::ErrorKind::NotFound,
                    format!("missing GGUF metadata: {key}"))),
            }
        };
        // Optional fields with defaults
        let weight_tied = match gguf.metadata.get("nanochat.weight_tied") {
            Some(GgufValue::U32(v)) => *v != 0,
            _ => false,
        };
        let n_experts = match gguf.metadata.get("nanochat.n_experts") {
            Some(GgufValue::U32(v)) if *v > 0 => Some(*v as usize),
            _ => None,
        };
        let n_active_experts = match gguf.metadata.get("nanochat.n_active_experts") {
            Some(GgufValue::U32(v)) if *v > 0 => Some(*v as usize),
            _ => None,
        };

        // DeltaNet ratio: stored as percentage (0-100) in GGUF, converted to 0.0-1.0
        let deltanet_ratio = match gguf.metadata.get("nanochat.deltanet_ratio_pct") {
            Some(GgufValue::U32(v)) if *v > 0 => Some(*v as f32 / 100.0),
            _ => None,
        };

        Ok(ModelConfig {
            dim: get_u32("nanochat.dim")?,
            n_layers: get_u32("nanochat.n_layers")?,
            n_heads: get_u32("nanochat.n_heads")?,
            n_kv_heads: get_u32("nanochat.n_heads")?, // MHA: same as n_heads
            ffn_mult: 2.667,
            vocab_size: get_u32("nanochat.vocab_size")?,
            max_seq_len: 2048,
            group_size: get_u32("nanochat.group_size")?,
            mhc_n_streams: get_u32("nanochat.mhc_n_streams")?,
            rope_theta: 10000.0,
            n_experts,
            n_active_experts,
            deltanet_ratio,
            weight_tied,
        })
    }

    fn load_embedding(gguf: &GgufFile, name: &str) -> io::Result<Embedding> {
        let tensor = gguf.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound,
                format!("tensor '{name}' not found")))?;

        let vocab_size = tensor.dims[0] as usize;
        let dim = tensor.dims[1] as usize;
        let data = gguf.tensor_data(tensor);

        // FP16 -> F32 conversion
        let weight: Vec<f32> = data.chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();

        if weight.len() != vocab_size * dim {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("embedding size mismatch: {} vs {}", weight.len(), vocab_size * dim)));
        }

        Ok(Embedding { weight, vocab_size, dim })
    }

    fn load_norm(gguf: &GgufFile, name: &str) -> io::Result<RMSNorm> {
        let tensor = gguf.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound,
                format!("tensor '{name}' not found")))?;

        let data = gguf.tensor_data(tensor);
        let weight: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(RMSNorm { weight, eps: 1e-6 })
    }

    fn extract_mhc_n2(layers: &[mhc_lite::MhcLayerParams], idx: usize) -> io::Result<MhcLiteN2> {
        match layers.get(idx) {
            Some(mhc_lite::MhcLayerParams::N2(mhc)) => Ok(mhc.clone()),
            Some(_) => Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("expected N2 mHC at index {idx}"))),
            None => Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("mHC index {idx} out of range (have {})", layers.len()))),
        }
    }

    /// Reset attention states (KV caches or recurrent states) for a new sequence.
    pub fn reset_caches(&mut self) {
        for state in &mut self.attn_states {
            state.reset();
        }
    }

    /// Forward pass for a single token at position `pos`.
    ///
    /// Returns logits [vocab_size].
    pub fn forward_token(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let dim = self.config.dim;
        // 1. Embed
        let mut x = vec![0.0f32; dim];
        self.tok_embed.forward_token(token_id, &mut x);

        // 2. Expand to multi-stream
        let mut x_exp = MhcLiteN2::expand_input(&x, dim);

        // 3. Transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            block.forward(&mut x_exp, &mut self.attn_states[i], &self.rope, pos);
        }

        // 4. Collapse back to single stream
        let x_collapsed = MhcLiteN2::collapse_output(&x_exp, dim);

        // 5. Final norm
        let mut x_normed = vec![0.0f32; dim];
        self.norm_final.forward(&x_collapsed, &mut x_normed);

        // 6. LM head (use embedding weights if tied)
        let mut logits = vec![0.0f32; self.config.vocab_size];
        if self.weight_tied {
            self.tok_embed.forward_as_lm_head(&x_normed, &mut logits);
        } else {
            self.lm_head.as_ref().unwrap().forward(&x_normed, &mut logits);
        }

        logits
    }

    /// Forward pass for a sequence of tokens (prefill).
    /// Processes tokens one at a time (no batched prefill yet).
    ///
    /// Returns logits for the last token only.
    pub fn forward_sequence(&mut self, token_ids: &[u32]) -> Vec<f32> {
        self.reset_caches();
        let mut logits = vec![];
        for (pos, &tid) in token_ids.iter().enumerate() {
            logits = self.forward_token(tid, pos);
        }
        logits
    }

    /// Batched forward pass for a sequence of tokens (prefill).
    ///
    /// Processes all tokens through each layer in batch, with causal attention masking.
    /// Returns logits for the last token only.
    pub fn forward_sequence_batched(&mut self, token_ids: &[u32]) -> Vec<f32> {
        self.reset_caches();

        let seq_len = token_ids.len();
        if seq_len == 0 {
            return vec![];
        }

        let dim = self.config.dim;
        let n_streams = self.config.mhc_n_streams;
        let exp_dim = n_streams * dim;

        // 1. Embed all tokens: [seq_len * dim]
        let mut x_all = vec![0.0f32; seq_len * dim];
        for (t, &tid) in token_ids.iter().enumerate() {
            self.tok_embed.forward_token(tid, &mut x_all[t * dim..(t + 1) * dim]);
        }

        // 2. Expand all tokens to multi-stream: [seq_len * exp_dim]
        let mut x_exp_all = vec![0.0f32; seq_len * exp_dim];
        for t in 0..seq_len {
            let x = &x_all[t * dim..(t + 1) * dim];
            let expanded = MhcLiteN2::expand_input(x, dim);
            x_exp_all[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(&expanded);
        }

        // 3. Transformer blocks (batched)
        for (i, block) in self.blocks.iter().enumerate() {
            block.forward_batch(
                &mut x_exp_all,
                &mut self.attn_states[i],
                &self.rope,
                0, // start_pos for fresh prefill
                seq_len,
            );
        }

        // 4. Collapse last token back to single stream
        let last_exp = &x_exp_all[(seq_len - 1) * exp_dim..seq_len * exp_dim];
        let x_collapsed = MhcLiteN2::collapse_output(last_exp, dim);

        // 5. Final norm
        let mut x_normed = vec![0.0f32; dim];
        self.norm_final.forward(&x_collapsed, &mut x_normed);

        // 6. LM head
        let mut logits = vec![0.0f32; self.config.vocab_size];
        if self.weight_tied {
            self.tok_embed.forward_as_lm_head(&x_normed, &mut logits);
        } else {
            self.lm_head.as_ref().unwrap().forward(&x_normed, &mut logits);
        }

        logits
    }

    /// Verify all mHC matrices in the model are doubly stochastic.
    pub fn verify_mhc(&self) -> Result<(), String> {
        for (i, block) in self.blocks.iter().enumerate() {
            let h_attn = block.mhc_attn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_attn, 1e-6)
                .map_err(|e| format!("block {i} mhc_attn: {e}"))?;

            let h_ffn = block.mhc_ffn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_ffn, 1e-6)
                .map_err(|e| format!("block {i} mhc_ffn: {e}"))?;
        }
        Ok(())
    }

    /// Count model parameters by category.
    pub fn param_count(&self) -> ModelParamCount {
        let dim = self.config.dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim();
        let ffn_dim = self.config.ffn_dim();
        let n_layers = self.config.n_layers;

        let embed = self.config.vocab_size * dim;

        // Count attention params per layer -- standard and DeltaNet have different sizes
        let mut attn_total = 0usize;
        for block in &self.blocks {
            match &block.attention {
                AttentionLayer::Standard(_) => {
                    // Q(dim*dim), K(dim*kv_dim), V(dim*kv_dim), O(dim*dim)
                    attn_total += dim * dim + 2 * dim * kv_dim + dim * dim;
                }
                AttentionLayer::DeltaNet(_) => {
                    // Q, K, V, O (all dim*dim) + beta (n_heads * dim)
                    attn_total += 4 * dim * dim + self.config.n_heads * dim;
                }
            }
        }

        // FFN params: dense vs MoE
        let ffn_per_layer = if let Some(n_experts) = self.config.n_experts {
            let router_params = n_experts * dim; // router: [n_experts, dim]
            let expert_params = dim * ffn_dim + dim * ffn_dim + ffn_dim * dim; // gate, up, down per expert
            router_params + n_experts * expert_params
        } else {
            dim * ffn_dim + dim * ffn_dim + ffn_dim * dim // gate, up, down
        };

        let norm_per_layer = dim * 2; // attn_norm + ffn_norm
        let mhc_per_layer = 9 * 2; // 9 params each for attn + ffn mHC (N=2)
        let lm_head = if self.weight_tied { 0 } else { self.config.vocab_size * dim };
        let final_norm = dim;

        ModelParamCount {
            total: embed + attn_total + n_layers * (ffn_per_layer + norm_per_layer + mhc_per_layer) + lm_head + final_norm,
            embed,
            attn: attn_total,
            ffn: n_layers * ffn_per_layer,
            norm: n_layers * norm_per_layer + final_norm,
            mhc: n_layers * mhc_per_layer,
            lm_head,
        }
    }
}

/// Parameter count breakdown.
#[derive(Debug)]
pub struct ModelParamCount {
    pub total: usize,
    pub embed: usize,
    pub attn: usize,
    pub ffn: usize,
    pub norm: usize,
    pub mhc: usize,
    pub lm_head: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> NanochatModel {
        // Use a small config for fast testing
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: None,
            weight_tied: false,
        };
        NanochatModel::new_random(config)
    }

    fn make_test_model_tied() -> NanochatModel {
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: None,
            weight_tied: true,
        };
        NanochatModel::new_random(config)
    }

    #[test]
    fn test_model_forward_single_token() {
        let mut model = make_test_model();
        let logits = model.forward_token(42, 0);

        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        assert!(logits.iter().any(|&v| v != 0.0), "all-zero logits");
    }

    #[test]
    fn test_model_forward_sequence() {
        let mut model = make_test_model();
        let tokens = vec![1, 5, 10, 20];
        let logits = model.forward_sequence(&tokens);

        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    }

    #[test]
    fn test_model_mhc_verify() {
        let model = make_test_model();
        model.verify_mhc().expect("mHC verification failed");
    }

    #[test]
    fn test_model_shape_through_layers() {
        let model = make_test_model();
        let dim = model.config.dim;
        let n_streams = model.config.mhc_n_streams;

        // Manually trace shapes
        let mut x = vec![0.0f32; dim];
        model.tok_embed.forward_token(1, &mut x);
        assert_eq!(x.len(), dim);

        let x_exp = MhcLiteN2::expand_input(&x, dim);
        assert_eq!(x_exp.len(), n_streams * dim, "expand shape wrong");

        let collapsed = MhcLiteN2::collapse_output(&x_exp, dim);
        assert_eq!(collapsed.len(), dim, "collapse shape wrong");
    }

    #[test]
    fn test_model_d20_config() {
        let config = ModelConfig::d20();
        let model = NanochatModel::new_random(config);
        let counts = model.param_count();
        println!("d20 params: {:?}", counts);
        assert!(counts.total > 0);
        assert!(counts.mhc > 0);
    }

    #[test]
    fn test_model_reset_caches() {
        let mut model = make_test_model();

        // Process a few tokens
        model.forward_token(1, 0);
        model.forward_token(2, 1);
        for state in &model.attn_states {
            match state {
                AttentionState::Kv(cache) => assert_eq!(cache.len, 2),
                AttentionState::Recurrent(_) => {} // DeltaNet state doesn't track len
            }
        }

        // Reset
        model.reset_caches();
        for state in &model.attn_states {
            match state {
                AttentionState::Kv(cache) => assert_eq!(cache.len, 0),
                AttentionState::Recurrent(s) => assert!(s.s.iter().all(|&v| v == 0.0)),
            }
        }
    }

    #[test]
    fn test_expand_n_blocks_collapse_shape() {
        // Test gate from CLAUDE.md:
        // "mHC expand → N blocks → collapse preserves tensor shape correctly"
        let mut model = make_test_model();
        let dim = model.config.dim;

        let mut x = vec![0.1f32; dim];
        model.tok_embed.forward_token(5, &mut x);

        // Expand
        let mut x_exp = MhcLiteN2::expand_input(&x, dim);
        let exp_len = x_exp.len();

        // N blocks
        for (i, block) in model.blocks.iter().enumerate() {
            block.forward(&mut x_exp, &mut model.attn_states[i], &model.rope, 0);
            assert_eq!(x_exp.len(), exp_len, "shape changed after block {i}");
        }

        // Collapse
        let collapsed = MhcLiteN2::collapse_output(&x_exp, dim);
        assert_eq!(collapsed.len(), dim, "final collapse shape wrong");
    }

    #[test]
    fn test_weight_tied_model_forward() {
        let mut model = make_test_model_tied();
        assert!(model.weight_tied);
        assert!(model.lm_head.is_none());

        let logits = model.forward_token(42, 0);
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        assert!(logits.iter().any(|&v| v != 0.0), "all-zero logits");
    }

    #[test]
    fn test_weight_tied_param_count_reduced() {
        let model_untied = make_test_model();
        let model_tied = make_test_model_tied();

        let count_untied = model_untied.param_count();
        let count_tied = model_tied.param_count();

        // Tied model should have vocab_size * dim fewer params (no LM head)
        let expected_diff = model_untied.config.vocab_size * model_untied.config.dim;
        assert_eq!(count_untied.total - count_tied.total, expected_diff);
        assert_eq!(count_tied.lm_head, 0);
        assert!(count_untied.lm_head > 0);
    }

    #[test]
    fn test_weight_tied_sequence_forward() {
        let mut model = make_test_model_tied();
        let tokens = vec![1, 5, 10, 20];
        let logits = model.forward_sequence(&tokens);

        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    }

    // ============================================================
    // Batched prefill tests
    // ============================================================

    #[test]
    fn test_batched_matches_sequential() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: None,
            weight_tied: false,
        };
        let mut model_seq = NanochatModel::new_random(config.clone());
        let mut model_batch = NanochatModel::new_random(config.clone());

        // Copy weights (same random model, same seed — they're already identical)
        let tokens = vec![1u32, 5, 10, 20];

        let logits_seq = model_seq.forward_sequence(&tokens);
        let logits_batch = model_batch.forward_sequence_batched(&tokens);

        assert_eq!(logits_seq.len(), logits_batch.len());
        let max_diff: f32 = logits_seq.iter()
            .zip(logits_batch.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "batched vs sequential logits max_diff={} (should be < 1e-4)",
            max_diff
        );
    }

    #[test]
    fn test_batched_prefill_single_token() {
        let mut model = make_test_model();
        let logits_single = model.forward_sequence(&[42]);
        let logits_batched = model.forward_sequence_batched(&[42]);

        assert_eq!(logits_single.len(), logits_batched.len());
        let max_diff: f32 = logits_single.iter()
            .zip(logits_batched.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "single token batched vs sequential max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_batched_prefill_empty() {
        let mut model = make_test_model();
        let logits = model.forward_sequence_batched(&[]);
        assert!(logits.is_empty());
    }

    #[test]
    fn test_batched_prefill_engine_uses_batch() {
        // Verify the engine's prefill produces valid output
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: None,
            weight_tied: false,
        };
        let mut model = NanochatModel::new_random(config);

        // Batched prefill followed by decode step
        let tokens = vec![1, 5, 10];
        let logits = model.forward_sequence_batched(&tokens);

        assert_eq!(logits.len(), 256);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert!(logits.iter().any(|&v| v != 0.0));

        // Now do a decode step after batched prefill
        let next_logits = model.forward_token(20, 3);
        assert_eq!(next_logits.len(), 256);
        assert!(next_logits.iter().all(|v| v.is_finite()));
    }

    // ============================================================
    // DeltaNet hybrid model tests
    // ============================================================

    #[test]
    fn test_hybrid_model_forward_ratio_half() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: Some(0.5),
            weight_tied: false,
        };
        let mut model = NanochatModel::new_random(config.clone());

        // Verify we have a mix of standard and DeltaNet layers
        let n_standard = model.blocks.iter()
            .filter(|b| matches!(b.attention, AttentionLayer::Standard(_)))
            .count();
        let n_deltanet = model.blocks.iter()
            .filter(|b| matches!(b.attention, AttentionLayer::DeltaNet(_)))
            .count();

        assert_eq!(n_standard, 2, "expected 2 standard layers");
        assert_eq!(n_deltanet, 2, "expected 2 DeltaNet layers");

        // Verify corresponding states match layer types
        for (i, (block, state)) in model.blocks.iter().zip(model.attn_states.iter()).enumerate() {
            match (&block.attention, state) {
                (AttentionLayer::Standard(_), AttentionState::Kv(_)) => {}
                (AttentionLayer::DeltaNet(_), AttentionState::Recurrent(_)) => {}
                _ => panic!("layer {} attention/state type mismatch", i),
            }
        }

        // Forward pass should work
        let logits = model.forward_token(42, 0);
        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits from hybrid model");
        assert!(logits.iter().any(|&v| v != 0.0), "all-zero logits from hybrid model");
    }

    #[test]
    fn test_deltanet_layers_have_no_kv_cache() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: Some(0.5),
            weight_tied: false,
        };
        let model = NanochatModel::new_random(config);

        for (i, (block, state)) in model.blocks.iter().zip(model.attn_states.iter()).enumerate() {
            match &block.attention {
                AttentionLayer::DeltaNet(_) => {
                    assert!(
                        matches!(state, AttentionState::Recurrent(_)),
                        "DeltaNet layer {} should have Recurrent state, not KV cache",
                        i
                    );
                }
                AttentionLayer::Standard(_) => {
                    assert!(
                        matches!(state, AttentionState::Kv(_)),
                        "Standard layer {} should have KV cache, not Recurrent state",
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn test_hybrid_model_sequence_forward() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: Some(0.5),
            weight_tied: false,
        };
        let mut model = NanochatModel::new_random(config.clone());

        let tokens = vec![1, 5, 10, 20];
        let logits = model.forward_sequence(&tokens);

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    }

    #[test]
    fn test_hybrid_model_reset_and_reuse() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: Some(0.5),
            weight_tied: false,
        };
        let mut model = NanochatModel::new_random(config);

        // First run
        let logits1 = model.forward_sequence(&[1, 2, 3]);

        // Reset and run again
        model.reset_caches();
        let logits2 = model.forward_sequence(&[1, 2, 3]);

        // Should produce identical results after reset
        for i in 0..logits1.len() {
            assert!(
                (logits1[i] - logits2[i]).abs() < 1e-5,
                "logit[{}] differs after reset: {} vs {}",
                i, logits1[i], logits2[i]
            );
        }
    }

    #[test]
    fn test_should_use_deltanet() {
        // No ratio: all standard
        assert!(!should_use_deltanet(0, 4, None));
        assert!(!should_use_deltanet(1, 4, None));

        // Ratio 0.0: all standard
        assert!(!should_use_deltanet(0, 4, Some(0.0)));

        // Ratio 1.0: all DeltaNet
        assert!(should_use_deltanet(0, 4, Some(1.0)));
        assert!(should_use_deltanet(3, 4, Some(1.0)));

        // Ratio 0.5 with 4 layers: expect 2 DeltaNet layers
        let dn_count: usize = (0..4)
            .filter(|&i| should_use_deltanet(i, 4, Some(0.5)))
            .count();
        assert_eq!(dn_count, 2, "expected 2 DeltaNet layers for ratio=0.5");
    }

    #[test]
    fn test_deltanet_param_count_includes_beta() {
        let config_std = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: None,
            weight_tied: false,
        };
        let config_dn = ModelConfig {
            deltanet_ratio: Some(1.0),
            ..config_std.clone()
        };

        let model_std = NanochatModel::new_random(config_std);
        let model_dn = NanochatModel::new_random(config_dn);

        let count_std = model_std.param_count();
        let count_dn = model_dn.param_count();

        // DeltaNet layers have extra w_beta (n_heads * dim per layer) and use
        // full dim for K and V instead of kv_dim.
        // Since n_kv_heads == n_heads here (MHA), kv_dim == dim, so:
        // standard = 4*dim*dim, deltanet = 4*dim*dim + n_heads*dim
        // DeltaNet should have n_heads*dim extra params per layer
        let extra_per_layer = model_std.config.n_heads * model_std.config.dim;
        let expected_diff = extra_per_layer * model_std.config.n_layers;
        assert_eq!(
            count_dn.attn - count_std.attn,
            expected_diff,
            "DeltaNet should add n_heads*dim params per layer for w_beta"
        );
    }
}
