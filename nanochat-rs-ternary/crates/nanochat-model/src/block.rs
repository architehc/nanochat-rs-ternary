//! TransformerBlock with mHC-lite residual wiring.
//!
//! Each block has two sub-layers (attention + FFN), each with its own mHC instance.
//! The mHC manages the multi-stream residual connection while the sub-layers
//! operate on a single collapsed stream.
//!
//! Supports both standard MHA/GQA attention (with KV cache) and DeltaNet
//! linear recurrent attention (with recurrent state).

use crate::attention::{Attention, KvCache, RopeFreqs};
use crate::config::ModelConfig;
use crate::deltanet::{DeltaNetAttention, DeltaNetState};
use crate::ffn::{FeedForward, FfnLayer, MoeExperts};
use crate::norm::RMSNorm;
use mhc_lite::MhcLiteN2;

/// Attention layer variant -- standard MHA/GQA or DeltaNet recurrent.
#[derive(Debug)]
pub enum AttentionLayer {
    /// Standard multi-head attention with KV cache.
    Standard(Attention),
    /// DeltaNet linear recurrent attention with recurrent state.
    DeltaNet(DeltaNetAttention),
}

/// Attention state variant -- matches the attention layer type.
#[derive(Debug, Clone)]
pub enum AttentionState {
    /// KV cache for standard attention.
    Kv(KvCache),
    /// Recurrent state for DeltaNet attention.
    Recurrent(DeltaNetState),
}

impl AttentionState {
    /// Reset the attention state (KV cache len or recurrent state to zeros).
    pub fn reset(&mut self) {
        match self {
            AttentionState::Kv(cache) => cache.len = 0,
            AttentionState::Recurrent(state) => state.reset(),
        }
    }
}

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
    pub attention: AttentionLayer,
    pub ffn: FfnLayer,
    pub dim: usize,
}

impl TransformerBlock {
    /// Create a block with standard attention and random weights (for testing).
    pub fn new_random(config: &ModelConfig) -> Self {
        let ffn = if config.n_experts.is_some() {
            FfnLayer::Moe(MoeExperts::new_random(config))
        } else {
            FfnLayer::Dense(FeedForward::new_random(config))
        };

        Self {
            mhc_attn: MhcLiteN2::new_identity(),
            mhc_ffn: MhcLiteN2::new_identity(),
            norm_attn: RMSNorm::new(config.dim),
            norm_ffn: RMSNorm::new(config.dim),
            attention: AttentionLayer::Standard(Attention::new_random(config)),
            ffn,
            dim: config.dim,
        }
    }

    /// Create a block with DeltaNet attention and random weights (for testing).
    pub fn new_random_deltanet(config: &ModelConfig) -> Self {
        let ffn = if config.n_experts.is_some() {
            FfnLayer::Moe(MoeExperts::new_random(config))
        } else {
            FfnLayer::Dense(FeedForward::new_random(config))
        };

        Self {
            mhc_attn: MhcLiteN2::new_identity(),
            mhc_ffn: MhcLiteN2::new_identity(),
            norm_attn: RMSNorm::new(config.dim),
            norm_ffn: RMSNorm::new(config.dim),
            attention: AttentionLayer::DeltaNet(DeltaNetAttention::new_random(config)),
            ffn,
            dim: config.dim,
        }
    }

    /// Create the appropriate attention state for this block's attention type.
    pub fn create_attn_state(&self, config: &ModelConfig) -> AttentionState {
        match &self.attention {
            AttentionLayer::Standard(_) => {
                AttentionState::Kv(KvCache::new(
                    config.max_seq_len,
                    config.n_kv_heads,
                    config.head_dim(),
                ))
            }
            AttentionLayer::DeltaNet(_) => {
                AttentionState::Recurrent(DeltaNetState::new(
                    config.n_heads,
                    config.head_dim(),
                ))
            }
        }
    }

    /// Forward pass for a single token (autoregressive).
    ///
    /// x_expanded: [n_streams * dim] multi-stream state
    /// attn_state: attention state for this layer (KV cache or DeltaNet state)
    /// rope:       precomputed RoPE frequencies (used only for standard attention)
    /// pos:        current token position (used only for standard attention)
    ///
    /// Modifies x_expanded in-place.
    pub fn forward(
        &self,
        x_expanded: &mut [f32],
        attn_state: &mut AttentionState,
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

        // 3. Attention (dispatch based on layer type)
        let mut attn_out = vec![0.0f32; dim];
        match (&self.attention, attn_state) {
            (AttentionLayer::Standard(attn), AttentionState::Kv(cache)) => {
                attn.forward(&normed, cache, rope, pos, &mut attn_out);
            }
            (AttentionLayer::DeltaNet(attn), AttentionState::Recurrent(state)) => {
                attn.forward(&normed, state, &mut attn_out);
            }
            _ => panic!("attention layer type and state type mismatch"),
        }

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

    /// Batched forward pass for prefill: process `seq_len` tokens.
    ///
    /// x_exp_batch: [seq_len * n_streams * dim] — flattened multi-stream state for each token
    /// attn_state:  attention state for this layer
    /// rope:        RoPE frequencies
    /// start_pos:   starting position offset
    /// seq_len:     number of tokens
    ///
    /// Modifies x_exp_batch in-place.
    pub fn forward_batch(
        &self,
        x_exp_batch: &mut [f32],
        attn_state: &mut AttentionState,
        rope: &RopeFreqs,
        start_pos: usize,
        seq_len: usize,
    ) {
        let dim = self.dim;
        let n_streams = x_exp_batch.len() / (seq_len * dim);
        let exp_dim = n_streams * dim;
        assert_eq!(x_exp_batch.len(), seq_len * exp_dim);

        // === Attention sub-layer (batched) ===
        // 1. Prepare input for each token: mix streams -> single
        let mut attn_in_batch = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let prepared = self.mhc_attn.prepare_input(x_exp, dim);
            attn_in_batch[t * dim..(t + 1) * dim].copy_from_slice(&prepared);
        }

        // 2. RMSNorm batch
        let mut normed_batch = vec![0.0f32; seq_len * dim];
        self.norm_attn.forward_batch(&attn_in_batch, seq_len, &mut normed_batch);

        // 3. Attention (dispatch based on layer type)
        let mut attn_out_batch = vec![0.0f32; seq_len * dim];
        match (&self.attention, attn_state) {
            (AttentionLayer::Standard(attn), AttentionState::Kv(cache)) => {
                attn.forward_batch(&normed_batch, seq_len, cache, rope, start_pos, &mut attn_out_batch);
            }
            (AttentionLayer::DeltaNet(attn), AttentionState::Recurrent(state)) => {
                // DeltaNet processes tokens one at a time (recurrent)
                for t in 0..seq_len {
                    let normed = &normed_batch[t * dim..(t + 1) * dim];
                    let out = &mut attn_out_batch[t * dim..(t + 1) * dim];
                    attn.forward(normed, state, out);
                }
            }
            _ => panic!("attention layer type and state type mismatch"),
        }

        // 4. mHC residual update for each token
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let attn_out = &attn_out_batch[t * dim..(t + 1) * dim];
            let x_new = self.mhc_attn.apply(x_exp, attn_out, dim);
            x_exp_batch[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(&x_new);
        }

        // === FFN sub-layer (batched) ===
        // 1. Prepare input for each token
        let mut ffn_in_batch = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let prepared = self.mhc_ffn.prepare_input(x_exp, dim);
            ffn_in_batch[t * dim..(t + 1) * dim].copy_from_slice(&prepared);
        }

        // 2. RMSNorm batch
        self.norm_ffn.forward_batch(&ffn_in_batch, seq_len, &mut normed_batch);

        // 3. FFN batch
        let mut ffn_out_batch = vec![0.0f32; seq_len * dim];
        self.ffn.forward_batch(&normed_batch, seq_len, &mut ffn_out_batch);

        // 4. mHC residual update for each token
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let ffn_out = &ffn_out_batch[t * dim..(t + 1) * dim];
            let x_new = self.mhc_ffn.apply(x_exp, ffn_out, dim);
            x_exp_batch[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(&x_new);
        }
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
        let mut state = block.create_attn_state(&config);

        // Expanded input (2 streams)
        let mut x_exp = vec![0.1f32; config.mhc_n_streams * config.dim];

        block.forward(&mut x_exp, &mut state, &rope, 0);

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
        let mut state = block.create_attn_state(&config);

        let expanded_dim = config.mhc_n_streams * config.dim;
        let mut x_exp = vec![0.5f32; expanded_dim];

        block.forward(&mut x_exp, &mut state, &rope, 0);

        assert_eq!(x_exp.len(), expanded_dim, "expanded shape changed after forward");
    }

    #[test]
    fn test_block_multi_token() {
        let config = ModelConfig::d20();
        let block = TransformerBlock::new_random(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let mut state = block.create_attn_state(&config);

        let mut x_exp = vec![0.1f32; config.mhc_n_streams * config.dim];

        // Process 3 tokens
        for pos in 0..3 {
            block.forward(&mut x_exp, &mut state, &rope, pos);
            assert!(x_exp.iter().all(|v| v.is_finite()), "non-finite at pos {}", pos);
        }
        match &state {
            AttentionState::Kv(cache) => assert_eq!(cache.len, 3),
            _ => panic!("expected KV cache state for standard attention block"),
        }
    }

    #[test]
    fn test_block_with_moe() {
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
            n_experts: Some(4),
            n_active_experts: Some(2),
            deltanet_ratio: None,
            weight_tied: false,
        };
        let block = TransformerBlock::new_random(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let mut state = block.create_attn_state(&config);

        // Verify the FFN is actually MoE
        assert!(matches!(block.ffn, FfnLayer::Moe(_)), "expected MoE FFN layer");

        let mut x_exp = vec![0.1f32; config.mhc_n_streams * config.dim];
        block.forward(&mut x_exp, &mut state, &rope, 0);

        assert!(
            x_exp.iter().all(|v| v.is_finite()),
            "non-finite MoE block output"
        );
        assert!(
            x_exp.iter().any(|&v| v != 0.0),
            "all-zero MoE block output"
        );
    }

    #[test]
    fn test_block_deltanet_forward() {
        let config = ModelConfig::d20();
        let block = TransformerBlock::new_random_deltanet(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let mut state = block.create_attn_state(&config);

        let mut x_exp = vec![0.1f32; config.mhc_n_streams * config.dim];

        block.forward(&mut x_exp, &mut state, &rope, 0);

        assert!(
            x_exp.iter().all(|v| v.is_finite()),
            "non-finite DeltaNet block output"
        );
        assert!(
            x_exp.iter().any(|&v| v != 0.0),
            "all-zero DeltaNet block output"
        );

        // Verify state type is recurrent
        match &state {
            AttentionState::Recurrent(s) => {
                assert!(s.s.iter().any(|&v| v != 0.0), "DeltaNet state should be non-zero");
            }
            AttentionState::Kv(_) => panic!("expected DeltaNet recurrent state"),
        }
    }

    #[test]
    fn test_attention_state_reset() {
        let config = ModelConfig::d20();

        // Test KV reset
        let mut kv_state = AttentionState::Kv(KvCache::new(
            config.max_seq_len,
            config.n_kv_heads,
            config.head_dim(),
        ));
        if let AttentionState::Kv(ref mut cache) = kv_state {
            cache.len = 5;
        }
        kv_state.reset();
        match &kv_state {
            AttentionState::Kv(cache) => assert_eq!(cache.len, 0),
            _ => panic!("expected KV state"),
        }

        // Test DeltaNet reset
        let mut dn_state = AttentionState::Recurrent(DeltaNetState::new(4, 64));
        if let AttentionState::Recurrent(ref mut s) = dn_state {
            s.s[0] = 1.0;
        }
        dn_state.reset();
        match &dn_state {
            AttentionState::Recurrent(s) => assert!(s.s.iter().all(|&v| v == 0.0)),
            _ => panic!("expected recurrent state"),
        }
    }
}
