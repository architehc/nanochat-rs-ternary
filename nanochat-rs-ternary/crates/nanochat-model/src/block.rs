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

#[derive(Debug, Default)]
struct BatchWorkspace {
    attn_in_batch: Vec<f32>,
    normed_batch: Vec<f32>,
    attn_out_batch: Vec<f32>,
    ffn_in_batch: Vec<f32>,
    ffn_out_batch: Vec<f32>,
    residual_tmp: Vec<f32>,
}

/// Attention layer variant -- standard MHA/GQA or DeltaNet recurrent.
#[allow(clippy::large_enum_variant)]
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
    batch_workspace: BatchWorkspace,
}

fn mhc_prepare_input_into(mhc: &MhcLiteN2, x: &[f32], dim_c: usize, out: &mut [f32]) {
    let h_pre = mhc.h_pre();
    let batch = x.len() / (2 * dim_c);
    debug_assert_eq!(x.len(), batch * 2 * dim_c);
    debug_assert_eq!(out.len(), batch * dim_c);

    for b in 0..batch {
        let s0 = &x[b * 2 * dim_c..b * 2 * dim_c + dim_c];
        let s1 = &x[b * 2 * dim_c + dim_c..b * 2 * dim_c + 2 * dim_c];
        let o = &mut out[b * dim_c..(b + 1) * dim_c];
        for i in 0..dim_c {
            o[i] = h_pre[0] * s0[i] + h_pre[1] * s1[i];
        }
    }
}

fn mhc_apply_into(mhc: &MhcLiteN2, x: &[f32], layer_output: &[f32], dim_c: usize, out: &mut [f32]) {
    let h_res = mhc.h_res();
    let h_post = mhc.h_post();
    let batch = x.len() / (2 * dim_c);
    debug_assert_eq!(x.len(), batch * 2 * dim_c);
    debug_assert_eq!(layer_output.len(), batch * dim_c);
    debug_assert_eq!(out.len(), batch * 2 * dim_c);

    for b in 0..batch {
        let s0 = &x[b * 2 * dim_c..b * 2 * dim_c + dim_c];
        let s1 = &x[b * 2 * dim_c + dim_c..b * 2 * dim_c + 2 * dim_c];
        let ly = &layer_output[b * dim_c..(b + 1) * dim_c];

        let (o0, o1) = out[b * 2 * dim_c..b * 2 * dim_c + 2 * dim_c].split_at_mut(dim_c);

        for i in 0..dim_c {
            let s0_with_res = s0[i] + h_post[0] * ly[i];
            let s1_with_res = s1[i] + h_post[1] * ly[i];
            o0[i] = h_res[0][0] * s0_with_res + h_res[0][1] * s1_with_res;
            o1[i] = h_res[1][0] * s0_with_res + h_res[1][1] * s1_with_res;
        }
    }
}

impl TransformerBlock {
    pub(crate) fn from_parts(
        mhc_attn: MhcLiteN2,
        mhc_ffn: MhcLiteN2,
        norm_attn: RMSNorm,
        norm_ffn: RMSNorm,
        attention: AttentionLayer,
        ffn: FfnLayer,
        dim: usize,
    ) -> Self {
        Self {
            mhc_attn,
            mhc_ffn,
            norm_attn,
            norm_ffn,
            attention,
            ffn,
            dim,
            batch_workspace: BatchWorkspace::default(),
        }
    }

    /// Create a block with standard attention and random weights (for testing).
    pub fn new_random(config: &ModelConfig) -> Self {
        let ffn = if config.n_experts.is_some() {
            FfnLayer::Moe(Box::new(MoeExperts::new_random(config)))
        } else {
            FfnLayer::Dense(Box::new(FeedForward::new_random(config)))
        };

        Self::from_parts(
            MhcLiteN2::new_identity(),
            MhcLiteN2::new_identity(),
            RMSNorm::new(config.dim),
            RMSNorm::new(config.dim),
            AttentionLayer::Standard(Attention::new_random(config)),
            ffn,
            config.dim,
        )
    }

    /// Create a block with DeltaNet attention and random weights (for testing).
    pub fn new_random_deltanet(config: &ModelConfig) -> Self {
        let ffn = if config.n_experts.is_some() {
            FfnLayer::Moe(Box::new(MoeExperts::new_random(config)))
        } else {
            FfnLayer::Dense(Box::new(FeedForward::new_random(config)))
        };

        Self::from_parts(
            MhcLiteN2::new_identity(),
            MhcLiteN2::new_identity(),
            RMSNorm::new(config.dim),
            RMSNorm::new(config.dim),
            AttentionLayer::DeltaNet(DeltaNetAttention::new_random(config)),
            ffn,
            config.dim,
        )
    }

    /// Create the appropriate attention state for this block's attention type.
    pub fn create_attn_state(&self, config: &ModelConfig) -> AttentionState {
        match &self.attention {
            AttentionLayer::Standard(_) => AttentionState::Kv(KvCache::new(
                config.max_seq_len,
                config.n_kv_heads,
                config.head_dim(),
            )),
            AttentionLayer::DeltaNet(_) => {
                AttentionState::Recurrent(DeltaNetState::new(config.n_heads, config.head_dim()))
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
            _ => unreachable!("attention layer type and state type mismatch"),
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
        &mut self,
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
        let batch_len = seq_len * dim;

        let mut ws = std::mem::take(&mut self.batch_workspace);
        ws.attn_in_batch.resize(batch_len, 0.0);
        ws.normed_batch.resize(batch_len, 0.0);
        ws.attn_out_batch.resize(batch_len, 0.0);
        ws.ffn_in_batch.resize(batch_len, 0.0);
        ws.ffn_out_batch.resize(batch_len, 0.0);
        ws.residual_tmp.resize(exp_dim, 0.0);

        // === Attention sub-layer (batched) ===
        // 1. Prepare input for each token: mix streams -> single
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let out = &mut ws.attn_in_batch[t * dim..(t + 1) * dim];
            mhc_prepare_input_into(&self.mhc_attn, x_exp, dim, out);
        }

        // 2. RMSNorm batch
        self.norm_attn
            .forward_batch(&ws.attn_in_batch, seq_len, &mut ws.normed_batch);

        // 3. Attention (dispatch based on layer type)
        match (&self.attention, attn_state) {
            (AttentionLayer::Standard(attn), AttentionState::Kv(cache)) => {
                attn.forward_batch(
                    &ws.normed_batch,
                    seq_len,
                    cache,
                    rope,
                    start_pos,
                    &mut ws.attn_out_batch,
                );
            }
            (AttentionLayer::DeltaNet(attn), AttentionState::Recurrent(state)) => {
                // DeltaNet processes tokens one at a time (recurrent)
                for t in 0..seq_len {
                    let normed = &ws.normed_batch[t * dim..(t + 1) * dim];
                    let out = &mut ws.attn_out_batch[t * dim..(t + 1) * dim];
                    attn.forward(normed, state, out);
                }
            }
            _ => unreachable!("attention layer type and state type mismatch"),
        }

        // 4. mHC residual update for each token
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let attn_out = &ws.attn_out_batch[t * dim..(t + 1) * dim];
            mhc_apply_into(&self.mhc_attn, x_exp, attn_out, dim, &mut ws.residual_tmp);
            x_exp_batch[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(&ws.residual_tmp);
        }

        // === FFN sub-layer (batched) ===
        // 1. Prepare input for each token
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let out = &mut ws.ffn_in_batch[t * dim..(t + 1) * dim];
            mhc_prepare_input_into(&self.mhc_ffn, x_exp, dim, out);
        }

        // 2. RMSNorm batch
        self.norm_ffn
            .forward_batch(&ws.ffn_in_batch, seq_len, &mut ws.normed_batch);

        // 3. FFN batch
        self.ffn
            .forward_batch(&ws.normed_batch, seq_len, &mut ws.ffn_out_batch);

        // 4. mHC residual update for each token
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let ffn_out = &ws.ffn_out_batch[t * dim..(t + 1) * dim];
            mhc_apply_into(&self.mhc_ffn, x_exp, ffn_out, dim, &mut ws.residual_tmp);
            x_exp_batch[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(&ws.residual_tmp);
        }

        self.batch_workspace = ws;
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
        assert!(x_exp.iter().any(|&v| v != 0.0), "all-zero block output");
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

        assert_eq!(
            x_exp.len(),
            expanded_dim,
            "expanded shape changed after forward"
        );
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
            assert!(
                x_exp.iter().all(|v| v.is_finite()),
                "non-finite at pos {}",
                pos
            );
        }
        match &state {
            AttentionState::Kv(cache) => assert_eq!(cache.len, 3),
            _ => panic!("expected KV cache state for standard attention block"),
        }
    }

    #[test]
    fn test_block_with_moe() {
        let mut config = ModelConfig::test_config(128, 2, 4, 256);
        config.n_experts = Some(4);
        config.n_active_experts = Some(2);
        let block = TransformerBlock::new_random(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let mut state = block.create_attn_state(&config);

        // Verify the FFN is actually MoE
        assert!(
            matches!(block.ffn, FfnLayer::Moe(_)),
            "expected MoE FFN layer"
        );

        let mut x_exp = vec![0.1f32; config.mhc_n_streams * config.dim];
        block.forward(&mut x_exp, &mut state, &rope, 0);

        assert!(
            x_exp.iter().all(|v| v.is_finite()),
            "non-finite MoE block output"
        );
        assert!(x_exp.iter().any(|&v| v != 0.0), "all-zero MoE block output");
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
                assert!(
                    s.s.iter().any(|&v| v != 0.0),
                    "DeltaNet state should be non-zero"
                );
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
