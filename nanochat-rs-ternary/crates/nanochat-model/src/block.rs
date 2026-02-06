//! TransformerBlock with mHC-lite residual wiring.
//!
//! Each block has two sub-layers (attention + FFN), each with its own mHC instance.
//! The mHC manages the multi-stream residual connection while the sub-layers
//! operate on a single collapsed stream.

use crate::attention::{Attention, KvCache, RopeFreqs};
use crate::config::ModelConfig;
use crate::ffn::FeedForward;
use crate::norm::RMSNorm;
use mhc_lite::MhcLiteN2;

/// Transformer block with mHC-lite residual connections.
///
/// Architecture per block:
/// ```text
/// x_expanded --+-- [mhc_attn.prepare_input] -> RMSNorm -> Attn -> [mhc_attn.apply]
///              |                                                        |
///              +-- [mhc_attn.h_res residual] --------------------------+ -> x_expanded
///
/// x_expanded --+-- [mhc_ffn.prepare_input] -> RMSNorm -> FFN -> [mhc_ffn.apply]
///              |                                                      |
///              +-- [mhc_ffn.h_res residual] --------------------------+ -> x_expanded
/// ```
#[derive(Debug)]
pub struct TransformerBlock {
    pub mhc_attn: MhcLiteN2,
    pub mhc_ffn: MhcLiteN2,
    pub norm_attn: RMSNorm,
    pub norm_ffn: RMSNorm,
    pub attention: Attention,
    pub ffn: FeedForward,
    pub dim: usize,
}

impl TransformerBlock {
    /// Create a block with random weights (for testing).
    pub fn new_random(config: &ModelConfig) -> Self {
        Self {
            mhc_attn: MhcLiteN2::new_identity(),
            mhc_ffn: MhcLiteN2::new_identity(),
            norm_attn: RMSNorm::new(config.dim),
            norm_ffn: RMSNorm::new(config.dim),
            attention: Attention::new_random(config),
            ffn: FeedForward::new_random(config),
            dim: config.dim,
        }
    }

    /// Forward pass for a single token (autoregressive).
    ///
    /// x_expanded: [n_streams * dim] multi-stream state
    /// cache:      KV cache for this layer
    /// rope:       precomputed RoPE frequencies
    /// pos:        current token position
    ///
    /// Modifies x_expanded in-place.
    pub fn forward(
        &self,
        x_expanded: &mut [f32],
        cache: &mut KvCache,
        rope: &RopeFreqs,
        pos: usize,
    ) {
        let dim = self.dim;
        let n_streams = x_expanded.len() / dim;
        assert_eq!(x_expanded.len(), n_streams * dim);

        // === Attention sub-layer ===
        // 1. Prepare input: mix streams -> single
        let attn_in = self.mhc_attn.prepare_input(x_expanded, dim);

        // 2. RMSNorm
        let mut normed = vec![0.0f32; dim];
        self.norm_attn.forward(&attn_in, &mut normed);

        // 3. Attention
        let mut attn_out = vec![0.0f32; dim];
        self.attention.forward(&normed, cache, rope, pos, &mut attn_out);

        // 4. mHC residual update
        let x_new = self.mhc_attn.apply(x_expanded, &attn_out, dim);
        x_expanded.copy_from_slice(&x_new);

        // === FFN sub-layer ===
        // 1. Prepare input
        let ffn_in = self.mhc_ffn.prepare_input(x_expanded, dim);

        // 2. RMSNorm
        self.norm_ffn.forward(&ffn_in, &mut normed);

        // 3. FFN
        let mut ffn_out = vec![0.0f32; dim];
        self.ffn.forward(&normed, &mut ffn_out);

        // 4. mHC residual update
        let x_new = self.mhc_ffn.apply(x_expanded, &ffn_out, dim);
        x_expanded.copy_from_slice(&x_new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_forward_finite() {
        let config = ModelConfig::d20();
        let block = TransformerBlock::new_random(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let mut cache = KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim());

        // Expanded input (2 streams)
        let mut x_exp = vec![0.1f32; config.mhc_n_streams * config.dim];

        block.forward(&mut x_exp, &mut cache, &rope, 0);

        assert!(
            x_exp.iter().all(|v| v.is_finite()),
            "non-finite block output"
        );
        assert!(
            x_exp.iter().any(|&v| v != 0.0),
            "all-zero block output"
        );
    }

    #[test]
    fn test_block_mhc_doubly_stochastic() {
        let config = ModelConfig::d20();
        let block = TransformerBlock::new_random(&config);

        // Verify mHC matrices are doubly stochastic
        let h_attn = block.mhc_attn.h_res();
        let h_ffn = block.mhc_ffn.h_res();

        mhc_lite::verify_doubly_stochastic_2x2(&h_attn, 1e-6).unwrap();
        mhc_lite::verify_doubly_stochastic_2x2(&h_ffn, 1e-6).unwrap();
    }

    #[test]
    fn test_block_shape_preserved() {
        let config = ModelConfig::d20();
        let block = TransformerBlock::new_random(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let mut cache = KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim());

        let expanded_dim = config.mhc_n_streams * config.dim;
        let mut x_exp = vec![0.5f32; expanded_dim];

        block.forward(&mut x_exp, &mut cache, &rope, 0);

        assert_eq!(x_exp.len(), expanded_dim, "expanded shape changed after forward");
    }

    #[test]
    fn test_block_multi_token() {
        let config = ModelConfig::d20();
        let block = TransformerBlock::new_random(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let mut cache = KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim());

        let mut x_exp = vec![0.1f32; config.mhc_n_streams * config.dim];

        // Process 3 tokens
        for pos in 0..3 {
            block.forward(&mut x_exp, &mut cache, &rope, pos);
            assert!(x_exp.iter().all(|v| v.is_finite()), "non-finite at pos {}", pos);
        }
        assert_eq!(cache.len, 3);
    }
}
