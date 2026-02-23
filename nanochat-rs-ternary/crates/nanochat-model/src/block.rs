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
use crate::wavefield::{HeadCoupling, WaveFieldAttention, WaveFieldState, WaveKernelCache, WavePhysicsParams};
use mhc_lite::MhcLiteN2;
use std::sync::Mutex;

#[derive(Debug, Default)]
struct BatchWorkspace {
    attn_in_batch: Vec<f32>,
    normed_batch: Vec<f32>,
    attn_out_batch: Vec<f32>,
    ffn_in_batch: Vec<f32>,
    ffn_out_batch: Vec<f32>,
    residual_tmp: Vec<f32>,
}

#[derive(Debug, Default)]
struct TokenWorkspace {
    attn_in: Vec<f32>,
    normed: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_in: Vec<f32>,
    ffn_out: Vec<f32>,
    residual_tmp: Vec<f32>,
}

/// Attention layer variant -- standard MHA/GQA, DeltaNet recurrent, or WaveField.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum AttentionLayer {
    /// Standard multi-head attention with KV cache.
    Standard(Attention),
    /// DeltaNet linear recurrent attention with recurrent state.
    DeltaNet(DeltaNetAttention),
    /// Wave field attention with FFT-based propagation.
    WaveField(WaveFieldAttention),
}

/// Attention state variant -- matches the attention layer type.
#[derive(Debug, Clone)]
pub enum AttentionState {
    /// KV cache for standard attention.
    Kv(KvCache),
    /// Recurrent state for DeltaNet attention.
    Recurrent(DeltaNetState),
    /// Wave field state (constant size, no KV cache).
    WaveField(WaveFieldState),
}

impl AttentionState {
    /// Rewind the attention state to a specific sequence length.
    ///
    /// For KV caches, this truncates the cache.
    /// For recurrent states (DeltaNet, WaveField), this currently resets
    /// if len is 0, otherwise it's a no-op (recurrent states cannot be easily rewound).
    pub fn rewind(&mut self, len: usize) {
        match self {
            AttentionState::Kv(cache) => {
                if len < cache.len {
                    cache.len = len;
                }
            }
            AttentionState::Recurrent(state) => {
                if len == 0 {
                    state.reset();
                }
                // Partial rewind not supported for recurrent state
            }
            AttentionState::WaveField(state) => {
                if len == 0 {
                    state.reset();
                }
                // Partial rewind not supported for wave field state
            }
        }
    }

    /// Reset the attention state (KV cache len or recurrent state to zeros).
    pub fn reset(&mut self) {
        self.rewind(0);
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
    token_workspace: Mutex<TokenWorkspace>,
    batch_workspace: Mutex<BatchWorkspace>,
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
            token_workspace: Mutex::new(TokenWorkspace::default()),
            batch_workspace: Mutex::new(BatchWorkspace::default()),
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

    /// Create a block with wave field attention and random weights (for testing).
    pub fn new_random_wavefield(config: &ModelConfig) -> Self {
        let wf_cfg = config
            .wavefield_config
            .as_ref()
            .expect("wavefield_config must be set for wave field layers");

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
            AttentionLayer::WaveField(WaveFieldAttention::new_random(
                wf_cfg,
                config.dim,
                config.max_seq_len,
            )),
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
                let dn_heads = config.deltanet_qk_heads.unwrap_or(config.n_heads);
                AttentionState::Recurrent(DeltaNetState::new(
                    dn_heads,
                    config.deltanet_qk_head_dim(),
                ))
            }
            AttentionLayer::WaveField(wf) => {
                AttentionState::WaveField(WaveFieldState::new(
                    wf.n_heads,
                    wf.field_size,
                    wf.head_dim,
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

        let mut ws = self
            .token_workspace
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let TokenWorkspace {
            attn_in,
            normed,
            attn_out,
            ffn_in,
            ffn_out,
            residual_tmp,
        } = &mut *ws;
        attn_in.resize(dim, 0.0);
        normed.resize(dim, 0.0);
        attn_out.resize(dim, 0.0);
        ffn_in.resize(dim, 0.0);
        ffn_out.resize(dim, 0.0);
        residual_tmp.resize(x_expanded.len(), 0.0);

        // === Attention sub-layer ===
        // 1. Prepare input: mix streams -> single
        self.mhc_attn.prepare_input_into(x_expanded, dim, attn_in);

        // 2. RMSNorm
        self.norm_attn.forward(attn_in, normed);

        // 3. Attention (dispatch based on layer type)
        match (&self.attention, attn_state) {
            (AttentionLayer::Standard(attn), AttentionState::Kv(cache)) => {
                attn.forward(normed, cache, rope, pos, attn_out);
            }
            (AttentionLayer::DeltaNet(attn), AttentionState::Recurrent(state)) => {
                attn.forward(normed, state, attn_out);
            }
            (AttentionLayer::WaveField(wf), AttentionState::WaveField(state)) => {
                wf.forward(normed, state, pos, attn_out);
            }
            _ => unreachable!("attention layer type and state type mismatch"),
        }

        // 4. mHC residual update
        self.mhc_attn.apply_into(x_expanded, attn_out, dim, residual_tmp);
        x_expanded.copy_from_slice(residual_tmp);

        // === FFN sub-layer ===
        // 1. Prepare input
        self.mhc_ffn.prepare_input_into(x_expanded, dim, ffn_in);

        // 2. RMSNorm
        self.norm_ffn.forward(ffn_in, normed);

        // 3. FFN
        self.ffn.forward(normed, ffn_out);

        // 4. mHC residual update
        self.mhc_ffn.apply_into(x_expanded, ffn_out, dim, residual_tmp);
        x_expanded.copy_from_slice(residual_tmp);
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
        let batch_len = seq_len * dim;

        let mut ws = self
            .batch_workspace
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let BatchWorkspace {
            attn_in_batch,
            normed_batch,
            attn_out_batch,
            ffn_in_batch,
            ffn_out_batch,
            residual_tmp,
        } = &mut *ws;
        attn_in_batch.resize(batch_len, 0.0);
        normed_batch.resize(batch_len, 0.0);
        attn_out_batch.resize(batch_len, 0.0);
        ffn_in_batch.resize(batch_len, 0.0);
        ffn_out_batch.resize(batch_len, 0.0);
        residual_tmp.resize(exp_dim, 0.0);

        // === Attention sub-layer (batched) ===
        // 1. Prepare input for each token: mix streams -> single
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let out = &mut attn_in_batch[t * dim..(t + 1) * dim];
            self.mhc_attn.prepare_input_into(x_exp, dim, out);
        }

        // 2. RMSNorm batch
        self.norm_attn
            .forward_batch(attn_in_batch, seq_len, normed_batch);

        // 3. Attention (dispatch based on layer type)
        match (&self.attention, attn_state) {
            (AttentionLayer::Standard(attn), AttentionState::Kv(cache)) => {
                attn.forward_batch(
                    normed_batch,
                    seq_len,
                    cache,
                    rope,
                    start_pos,
                    attn_out_batch,
                );
            }
            (AttentionLayer::DeltaNet(attn), AttentionState::Recurrent(state)) => {
                attn.forward_batch(normed_batch, seq_len, state, attn_out_batch);
            }
            (AttentionLayer::WaveField(wf), AttentionState::WaveField(state)) => {
                wf.forward_batch(normed_batch, state, start_pos, seq_len, attn_out_batch);
            }
            _ => unreachable!("attention layer type and state type mismatch"),
        }

        // 4. mHC residual update for each token
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let attn_out = &attn_out_batch[t * dim..(t + 1) * dim];
            self.mhc_attn.apply_into(x_exp, attn_out, dim, residual_tmp);
            x_exp_batch[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(residual_tmp);
        }

        // === FFN sub-layer (batched) ===
        // 1. Prepare input for each token
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let out = &mut ffn_in_batch[t * dim..(t + 1) * dim];
            self.mhc_ffn.prepare_input_into(x_exp, dim, out);
        }

        // 2. RMSNorm batch
        self.norm_ffn
            .forward_batch(ffn_in_batch, seq_len, normed_batch);

        // 3. FFN batch
        self.ffn.forward_batch(normed_batch, seq_len, ffn_out_batch);

        // 4. mHC residual update for each token
        for t in 0..seq_len {
            let x_exp = &x_exp_batch[t * exp_dim..(t + 1) * exp_dim];
            let ffn_out = &ffn_out_batch[t * dim..(t + 1) * dim];
            self.mhc_ffn.apply_into(x_exp, ffn_out, dim, residual_tmp);
            x_exp_batch[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(residual_tmp);
        }
    }

    /// Clone the block and place weight allocations on a specific NUMA node.
    pub fn clone_to_node(&self, node: usize) -> Self {
        Self::from_parts(
            self.mhc_attn.clone(),
            self.mhc_ffn.clone(),
            self.norm_attn.clone_to_node(node),
            self.norm_ffn.clone_to_node(node),
            match &self.attention {
                AttentionLayer::Standard(a) => AttentionLayer::Standard(Attention {
                    wq: a.wq.clone_to_node(node),
                    wk: a.wk.clone_to_node(node),
                    wv: a.wv.clone_to_node(node),
                    wo: a.wo.clone_to_node(node),
                    n_heads: a.n_heads,
                    n_kv_heads: a.n_kv_heads,
                    head_dim: a.head_dim,
                    n_rep: a.n_rep,
                }),
                AttentionLayer::DeltaNet(a) => AttentionLayer::DeltaNet(DeltaNetAttention {
                    wq: a.wq.clone_to_node(node),
                    wk: a.wk.clone_to_node(node),
                    wv: a.wv.clone_to_node(node),
                    wo: a.wo.clone_to_node(node),
                    w_beta: a.w_beta.clone_to_node(node),
                    n_heads: a.n_heads,
                    head_dim: a.head_dim,
                }),
                AttentionLayer::WaveField(a) => AttentionLayer::WaveField(WaveFieldAttention::from_parts(
                    a.scatter_proj.clone_to_node(node),
                    a.gate_proj.clone_to_node(node),
                    a.out_proj.clone_to_node(node),
                    WavePhysicsParams {
                        omega: a.physics.omega.clone(),
                        alpha: a.physics.alpha.clone(),
                        phi: a.physics.phi.clone(),
                    },
                    a.coupling.as_ref().map(|c| HeadCoupling {
                        weights: c.weights.clone(),
                        n_heads: c.n_heads,
                    }),
                    match &a.kernel_cache {
                        WaveKernelCache::Fft { kernels_freq, n_heads, field_size, fft_size } => WaveKernelCache::Fft {
                            kernels_freq: kernels_freq.clone(),
                            n_heads: *n_heads,
                            field_size: *field_size,
                            fft_size: *fft_size,
                        },
                        WaveKernelCache::Fwht { kernels_fwht, n_heads, field_size } => WaveKernelCache::Fwht {
                            kernels_fwht: kernels_fwht.clone(),
                            n_heads: *n_heads,
                            field_size: *field_size,
                        },
                        WaveKernelCache::Haar { kernels_haar, n_heads, field_size, levels } => WaveKernelCache::Haar {
                            kernels_haar: kernels_haar.clone(),
                            n_heads: *n_heads,
                            field_size: *field_size,
                            levels: *levels,
                        },
                    },
                    a.n_heads,
                    a.head_dim,
                    a.field_size,
                    a.stride,
                )),
            },
            match &self.ffn {
                FfnLayer::Dense(f) => FfnLayer::Dense(Box::new(FeedForward {
                    w_gate: f.w_gate.clone_to_node(node),
                    w_up: f.w_up.clone_to_node(node),
                    w_down: f.w_down.clone_to_node(node),
                    ffn_dim: f.ffn_dim,
                })),
                FfnLayer::Moe(m) => FfnLayer::Moe(Box::new(MoeExperts {
                    router: m.router.clone_to_node(node),
                    experts: m.experts.iter().map(|e| FeedForward {
                        w_gate: e.w_gate.clone_to_node(node),
                        w_up: e.w_up.clone_to_node(node),
                        w_down: e.w_down.clone_to_node(node),
                        ffn_dim: e.ffn_dim,
                    }).collect(),
                    n_active: m.n_active,
                    shared_expert: m.shared_expert.as_ref().map(|e| FeedForward {
                        w_gate: e.w_gate.clone_to_node(node),
                        w_up: e.w_up.clone_to_node(node),
                        w_down: e.w_down.clone_to_node(node),
                        ffn_dim: e.ffn_dim,
                    }),
                })),
            },
            self.dim,
        )
    }
}

impl Clone for TransformerBlock {
    fn clone(&self) -> Self {
        Self::from_parts(
            self.mhc_attn.clone(),
            self.mhc_ffn.clone(),
            self.norm_attn.clone(),
            self.norm_ffn.clone(),
            match &self.attention {
                AttentionLayer::Standard(a) => AttentionLayer::Standard(Attention {
                    wq: a.wq.clone(),
                    wk: a.wk.clone(),
                    wv: a.wv.clone(),
                    wo: a.wo.clone(),
                    n_heads: a.n_heads,
                    n_kv_heads: a.n_kv_heads,
                    head_dim: a.head_dim,
                    n_rep: a.n_rep,
                }),
                AttentionLayer::DeltaNet(a) => AttentionLayer::DeltaNet(DeltaNetAttention {
                    wq: a.wq.clone(),
                    wk: a.wk.clone(),
                    wv: a.wv.clone(),
                    wo: a.wo.clone(),
                    w_beta: a.w_beta.clone(),
                    n_heads: a.n_heads,
                    head_dim: a.head_dim,
                }),
                AttentionLayer::WaveField(a) => AttentionLayer::WaveField(WaveFieldAttention::from_parts(
                    a.scatter_proj.clone(),
                    a.gate_proj.clone(),
                    a.out_proj.clone(),
                    WavePhysicsParams {
                        omega: a.physics.omega.clone(),
                        alpha: a.physics.alpha.clone(),
                        phi: a.physics.phi.clone(),
                    },
                    a.coupling.as_ref().map(|c| HeadCoupling {
                        weights: c.weights.clone(),
                        n_heads: c.n_heads,
                    }),
                    match &a.kernel_cache {
                        WaveKernelCache::Fft { kernels_freq, n_heads, field_size, fft_size } => WaveKernelCache::Fft {
                            kernels_freq: kernels_freq.clone(),
                            n_heads: *n_heads,
                            field_size: *field_size,
                            fft_size: *fft_size,
                        },
                        WaveKernelCache::Fwht { kernels_fwht, n_heads, field_size } => WaveKernelCache::Fwht {
                            kernels_fwht: kernels_fwht.clone(),
                            n_heads: *n_heads,
                            field_size: *field_size,
                        },
                        WaveKernelCache::Haar { kernels_haar, n_heads, field_size, levels } => WaveKernelCache::Haar {
                            kernels_haar: kernels_haar.clone(),
                            n_heads: *n_heads,
                            field_size: *field_size,
                            levels: *levels,
                        },
                    },
                    a.n_heads,
                    a.head_dim,
                    a.field_size,
                    a.stride,
                )),
            },
            match &self.ffn {
                FfnLayer::Dense(f) => FfnLayer::Dense(Box::new(FeedForward {
                    w_gate: f.w_gate.clone(),
                    w_up: f.w_up.clone(),
                    w_down: f.w_down.clone(),
                    ffn_dim: f.ffn_dim,
                })),
                FfnLayer::Moe(m) => FfnLayer::Moe(Box::new(MoeExperts {
                    router: m.router.clone(),
                    experts: m.experts.clone(),
                    n_active: m.n_active,
                    shared_expert: m.shared_expert.clone(),
                })),
            },
            self.dim,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_forward_finite() {
        let config = ModelConfig::d20();
        let block = TransformerBlock::new_random(&config);
        let rope = RopeFreqs::new(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            config.rope_scale,
        );
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
        let rope = RopeFreqs::new(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            config.rope_scale,
        );
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
        let rope = RopeFreqs::new(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            config.rope_scale,
        );
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
        let rope = RopeFreqs::new(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            config.rope_scale,
        );
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
        let rope = RopeFreqs::new(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            config.rope_scale,
        );
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
            AttentionState::WaveField(_) => panic!("expected DeltaNet recurrent state"),
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
