//! Full nanochat model: embed -> mHC expand -> blocks -> mHC collapse -> norm -> head.

use std::io;

use crate::attention::{Attention, KvCache, RopeFreqs};
use crate::bitlinear::BitLinear;
use crate::block::TransformerBlock;
use crate::config::ModelConfig;
use crate::embed::Embedding;
use crate::ffn::FeedForward;
use crate::norm::RMSNorm;
use mhc_lite::MhcLiteN2;
use ternary_core::gguf::{GgufFile, GgufValue};

/// Full nanochat-rs ternary model.
#[derive(Debug)]
pub struct NanochatModel {
    pub config: ModelConfig,
    pub tok_embed: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub norm_final: RMSNorm,
    pub lm_head: BitLinear,
    pub rope: RopeFreqs,
    pub kv_caches: Vec<KvCache>,
}

impl NanochatModel {
    /// Create model with random weights (for testing / validation).
    pub fn new_random(config: ModelConfig) -> Self {
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let kv_caches: Vec<_> = (0..config.n_layers)
            .map(|_| KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim()))
            .collect();

        let blocks: Vec<_> = (0..config.n_layers)
            .map(|_| TransformerBlock::new_random(&config))
            .collect();

        // LM head: vocab_size x dim
        let lm_weights: Vec<f32> = (0..config.vocab_size * config.dim)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761) >> 16) % 200;
                v as f32 / 100.0 - 1.0
            })
            .collect();
        let lm_head = BitLinear::from_float(&lm_weights, config.vocab_size, config.dim, config.group_size);

        Self {
            tok_embed: Embedding::new_random(config.vocab_size, config.dim, 42),
            blocks,
            norm_final: RMSNorm::new(config.dim),
            lm_head,
            rope,
            kv_caches,
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

            // Attention weights
            let wq = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wq.weight"), group_size)?);
            let wk = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wk.weight"), group_size)?);
            let wv = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wv.weight"), group_size)?);
            let wo = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wo.weight"), group_size)?);

            let attention = Attention {
                wq, wk, wv, wo,
                n_heads: config.n_heads,
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim(),
                n_rep: config.n_rep(),
            };

            // FFN weights
            let w_gate = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.ffn.w_gate.weight"), group_size)?);
            let w_up = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.ffn.w_up.weight"), group_size)?);
            let w_down = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.ffn.w_down.weight"), group_size)?);

            let ffn = FeedForward {
                w_gate, w_up, w_down,
                ffn_dim: config.ffn_dim(),
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

        // LM head
        let lm_head = BitLinear::new(
            gguf.load_planar_weights("lm_head.weight", group_size)?);

        // RoPE and KV caches
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let kv_caches: Vec<_> = (0..config.n_layers)
            .map(|_| KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim()))
            .collect();

        Ok(Self {
            tok_embed, blocks, norm_final, lm_head, rope, kv_caches, config,
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

    /// Reset KV caches for a new sequence.
    pub fn reset_caches(&mut self) {
        for cache in &mut self.kv_caches {
            cache.len = 0;
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
            block.forward(&mut x_exp, &mut self.kv_caches[i], &self.rope, pos);
        }

        // 4. Collapse back to single stream
        let x_collapsed = MhcLiteN2::collapse_output(&x_exp, dim);

        // 5. Final norm
        let mut x_normed = vec![0.0f32; dim];
        self.norm_final.forward(&x_collapsed, &mut x_normed);

        // 6. LM head
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.lm_head.forward(&x_normed, &mut logits);

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
        let attn_per_layer = dim * dim + 2 * dim * kv_dim + dim * dim; // Q, K, V, O
        let ffn_per_layer = dim * ffn_dim + dim * ffn_dim + ffn_dim * dim; // gate, up, down
        let norm_per_layer = dim * 2; // attn_norm + ffn_norm
        let mhc_per_layer = 9 * 2; // 9 params each for attn + ffn mHC (N=2)
        let lm_head = self.config.vocab_size * dim;
        let final_norm = dim;

        ModelParamCount {
            total: embed + n_layers * (attn_per_layer + ffn_per_layer + norm_per_layer + mhc_per_layer) + lm_head + final_norm,
            embed,
            attn: n_layers * attn_per_layer,
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
        assert!(model.kv_caches.iter().all(|c| c.len == 2));

        // Reset
        model.reset_caches();
        assert!(model.kv_caches.iter().all(|c| c.len == 0));
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
            block.forward(&mut x_exp, &mut model.kv_caches[i], &model.rope, 0);
            assert_eq!(x_exp.len(), exp_len, "shape changed after block {i}");
        }

        // Collapse
        let collapsed = MhcLiteN2::collapse_output(&x_exp, dim);
        assert_eq!(collapsed.len(), dim, "final collapse shape wrong");
    }
}
