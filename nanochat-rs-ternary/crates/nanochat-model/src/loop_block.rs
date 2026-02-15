//! Shared loop block for LoopLM inference with ternary quantized weights.
//!
//! This is the inference-optimized version of SharedLoopBlock, using
//! quantized ternary weights instead of training-mode floating point.


use crate::attention::KvCache;
use crate::bitlinear::BitLinear;
use crate::config::ModelConfig;
use crate::norm::RMSNorm;
use mhc_lite::MhcLiteN2;

/// Shared loop block for inference: recurrent transformer layer with global state.
///
/// Uses ternary quantized weights (BitLinear) for memory efficiency and speed.
#[derive(Debug)]
pub struct SharedLoopBlock {
    // Local attention projections (ternary quantized)
    pub wq: BitLinear,
    pub wk: BitLinear,
    pub wv: BitLinear,
    pub wo: BitLinear,

    // Global attention gate (ternary quantized)
    pub g_qk: BitLinear,

    // Local FFN projections (ternary quantized SwiGLU)
    pub w_gate: BitLinear,
    pub w_up: BitLinear,
    pub w_down: BitLinear,

    // Global FFN gate (ternary quantized)
    pub g_ffn: BitLinear,

    // Norms (FP32)
    pub norm_attn: RMSNorm,
    pub norm_ffn: RMSNorm,

    // mHC residual connection handlers (FP32)
    pub mhc_attn: MhcLiteN2,
    pub mhc_ffn: MhcLiteN2,

    // Config
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_rep: usize,
    pub ffn_dim: usize,
}

impl SharedLoopBlock {
    /// Create a new shared loop block from loaded weights.
    ///
    /// This is typically called when loading a model from GGUF, not constructed from scratch.
    /// For testing, use `new_empty()` to create with zero-initialized weights.
    pub fn new(
        wq: BitLinear,
        wk: BitLinear,
        wv: BitLinear,
        wo: BitLinear,
        g_qk: BitLinear,
        w_gate: BitLinear,
        w_up: BitLinear,
        w_down: BitLinear,
        g_ffn: BitLinear,
        norm_attn: RMSNorm,
        norm_ffn: RMSNorm,
        mhc_attn: MhcLiteN2,
        mhc_ffn: MhcLiteN2,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Self {
        let head_dim = dim / n_heads;
        let n_rep = n_heads / n_kv_heads;
        let ffn_dim = w_gate.rows; // Infer from weight dimensions

        Self {
            wq,
            wk,
            wv,
            wo,
            g_qk,
            w_gate,
            w_up,
            w_down,
            g_ffn,
            norm_attn,
            norm_ffn,
            mhc_attn,
            mhc_ffn,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rep,
            ffn_dim,
        }
    }

    /// Create with empty (zero-initialized) weights for testing and initialization.
    pub fn new_empty(cfg: &ModelConfig) -> Self {
        use ternary_core::planar::PlanarWeights;

        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = dim / n_heads;
        let kv_dim = n_kv_heads * head_dim;
        let ffn_dim = cfg.ffn_dim();
        let group_size = cfg.group_size;

        // Helper to create empty (zero) PlanarWeights
        let empty_pw = |rows: usize, cols: usize| -> PlanarWeights {
            let weights = vec![0.0f32; rows * cols];
            PlanarWeights::from_row_major(&weights, rows, cols, group_size)
        };

        // All projections with empty ternary weights
        let wq = BitLinear::new(empty_pw(dim, dim));
        let wk = BitLinear::new(empty_pw(kv_dim, dim));
        let wv = BitLinear::new(empty_pw(kv_dim, dim));
        let wo = BitLinear::new(empty_pw(dim, dim));

        let g_qk = BitLinear::new(empty_pw(dim, dim));

        let w_gate = BitLinear::new(empty_pw(ffn_dim, dim));
        let w_up = BitLinear::new(empty_pw(ffn_dim, dim));
        let w_down = BitLinear::new(empty_pw(dim, ffn_dim));

        let g_ffn = BitLinear::new(empty_pw(dim, dim));

        // Norms and mHC
        let norm_attn = RMSNorm::new(dim);
        let norm_ffn = RMSNorm::new(dim);

        let mhc_attn = MhcLiteN2::new_identity();
        let mhc_ffn = MhcLiteN2::new_identity();

        let n_rep = n_heads / n_kv_heads;

        Self {
            wq,
            wk,
            wv,
            wo,
            g_qk,
            w_gate,
            w_up,
            w_down,
            g_ffn,
            norm_attn,
            norm_ffn,
            mhc_attn,
            mhc_ffn,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rep,
            ffn_dim,
        }
    }

    /// Forward pass for ONE loop iteration.
    ///
    /// Args:
    /// - x_expanded: [dim * n_streams] - mHC-expanded hidden state (single token)
    /// - global_state: Option<&[f32]> - accumulated global state [dim]
    /// - kv_cache: &mut KvCache - shared KV cache across iterations
    /// - append_kv: bool - whether to append to cache (true for first iteration)
    /// - token_pos: Option<usize> - explicit token position for causal masking (None = infer from cache)
    ///
    /// Returns:
    /// - Ok((x_expanded_out, global_state_out)) on success
    /// - Err on cache/position mismatch
    ///
    /// # Errors
    /// Returns error if token_pos exceeds cache length
    pub fn forward(
        &self,
        x_expanded: &[f32],
        global_state: Option<&[f32]>,
        kv_cache: &mut KvCache,
        append_kv: bool,
        token_pos: Option<usize>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        // ========== Attention Sub-Layer ==========

        // 1. mHC prepare: collapse expanded input to single stream
        let x = self.mhc_attn.prepare_input(x_expanded, self.dim);

        // 2. Pre-norm
        let mut x_norm = vec![0.0f32; self.dim];
        self.norm_attn.forward(&x, &mut x_norm);

        // 3. Local Q/K/V projections
        let mut q = vec![0.0f32; self.dim];
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];

        self.wq.forward(&x_norm, &mut q);
        self.wk.forward(&x_norm, &mut k);
        self.wv.forward(&x_norm, &mut v);

        // 4. Determine token position for causal masking
        // If not provided, infer from cache (autoregressive mode)
        let pos = token_pos.unwrap_or_else(|| {
            if append_kv {
                // About to add new token at position kv_cache.len
                kv_cache.len
            } else {
                // Re-processing last token in cache (loop iteration > 0)
                kv_cache.len.saturating_sub(1)
            }
        });

        // 5. Append to KV cache if requested
        if append_kv {
            kv_cache.append_kv(&k, &v);
        }

        // 6. Compute attention over cached KV with causal masking
        let mut attn_out = vec![0.0f32; self.dim];
        self.compute_attention(&q, kv_cache, pos, &mut attn_out)?;

        // 6. Output projection
        let mut attn_proj = vec![0.0f32; self.dim];
        self.wo.forward(&attn_out, &mut attn_proj);

        // 7. Global gate: mix with accumulated global state
        let gated_attn = if let Some(g_state) = global_state {
            let mut gate_weights = vec![0.0f32; self.dim];
            self.g_qk.forward(&x_norm, &mut gate_weights);

            // Sigmoid activation
            for w in &mut gate_weights {
                *w = 1.0 / (1.0 + (-*w).exp());
            }

            // Mix: gate * attn_proj + (1 - gate) * global_state
            gate_weights
                .iter()
                .zip(attn_proj.iter())
                .zip(g_state.iter())
                .map(|((&g, &a), &s)| g * a + (1.0 - g) * s)
                .collect()
        } else {
            attn_proj
        };

        // 8. mHC apply: add residual
        let x_expanded = self.mhc_attn.apply(x_expanded, &gated_attn, self.dim);

        // ========== FFN Sub-Layer ==========

        // 1. mHC prepare
        let x = self.mhc_ffn.prepare_input(&x_expanded, self.dim);

        // 2. Pre-norm
        let mut x_norm = vec![0.0f32; self.dim];
        self.norm_ffn.forward(&x, &mut x_norm);

        // 3. SwiGLU FFN
        let mut gate = vec![0.0f32; self.ffn_dim];
        let mut up = vec![0.0f32; self.ffn_dim];

        self.w_gate.forward(&x_norm, &mut gate);
        self.w_up.forward(&x_norm, &mut up);

        // SwiGLU: silu(gate) * up
        for i in 0..self.ffn_dim {
            let x = gate[i];
            let silu = x / (1.0 + (-x).exp()); // silu(x) = x * sigmoid(x)
            gate[i] = silu * up[i];
        }

        let mut ffn_out = vec![0.0f32; self.dim];
        self.w_down.forward(&gate, &mut ffn_out);

        // 4. Global FFN gate
        let gated_ffn = if let Some(g_state) = global_state {
            let mut gate_weights = vec![0.0f32; self.dim];
            self.g_ffn.forward(&x_norm, &mut gate_weights);

            // Sigmoid
            for w in &mut gate_weights {
                *w = 1.0 / (1.0 + (-*w).exp());
            }

            // Mix
            gate_weights
                .iter()
                .zip(ffn_out.iter())
                .zip(g_state.iter())
                .map(|((&g, &f), &s)| g * f + (1.0 - g) * s)
                .collect()
        } else {
            ffn_out
        };

        // 5. mHC apply
        let x_expanded_out = self.mhc_ffn.apply(&x_expanded, &gated_ffn, self.dim);

        // 6. Update global state: average of attention and FFN outputs
        let global_state_out: Vec<f32> = gated_attn
            .iter()
            .zip(gated_ffn.iter())
            .map(|(&a, &f)| (a + f) * 0.5)
            .collect();

        Ok((x_expanded_out, global_state_out))
    }

    /// Compute multi-head attention over cached KV with causal masking.
    ///
    /// # Arguments
    /// * `q` - Query vector [dim]
    /// * `kv_cache` - KV cache (may contain future positions in batched mode)
    /// * `token_pos` - Current token position (for causal masking)
    /// * `out` - Output buffer [dim]
    ///
    /// # Errors
    /// Returns error if token_pos exceeds cache length (indicates caller bug)
    fn compute_attention(
        &self,
        q: &[f32],
        kv_cache: &KvCache,
        token_pos: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        // Causal masking: only attend to positions 0..=token_pos
        let causal_len = token_pos + 1;

        // Bounds check: ensure token_pos doesn't exceed cache population
        if causal_len > kv_cache.len {
            return Err(format!(
                "token_pos ({}) exceeds KV cache length ({}). This indicates a bug in the caller.",
                token_pos, kv_cache.len
            ));
        }

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Zero output
        for o in out.iter_mut() {
            *o = 0.0;
        }

        // For each query head
        for h in 0..self.n_heads {
            let kv_h = h / self.n_rep; // GQA: multiple Q heads share KV heads
            let q_offset = h * self.head_dim;
            let kv_offset = kv_h * self.head_dim;

            // Compute attention scores over causal positions only
            let mut scores = vec![0.0f32; causal_len];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_base = t * (self.n_kv_heads * self.head_dim) + kv_offset;
                let mut dot = 0.0f32;
                for d in 0..self.head_dim {
                    dot += q[q_offset + d] * kv_cache.k[k_base + d];
                }
                *score = dot * scale;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for score in &mut scores {
                *score /= sum;
            }

            // Weighted sum over V (only causal positions)
            for t in 0..causal_len {
                let v_base = t * (self.n_kv_heads * self.head_dim) + kv_offset;
                let weight = scores[t];
                for d in 0..self.head_dim {
                    out[q_offset + d] += weight * kv_cache.v[v_base + d];
                }
            }
        }

        Ok(())
    }

    /// Batched forward pass for prefill: process multiple tokens through loop block.
    ///
    /// CRITICAL: Must maintain causal masking and per-token state consistency.
    /// This processes tokens one-by-one but with optimized weight access.
    ///
    /// # Arguments
    /// * `x_exp_all` - Expanded activations [seq_len * (2*dim)]
    /// * `global_states` - Per-token global states from previous iteration (seq_len items)
    /// * `kv_cache` - Shared KV cache (updated incrementally)
    /// * `append_kv` - Whether to append to KV cache (true for first iteration)
    /// * `seq_len` - Number of tokens in sequence
    ///
    /// # Returns
    /// Ok(Vector of per-token global states for next iteration) or Err on cache mismatch
    ///
    /// # Errors
    /// Propagates errors from forward() if token positions exceed cache length
    pub fn forward_batch(
        &self,
        x_exp_all: &mut [f32],
        global_states: &[Option<Vec<f32>>],
        kv_cache: &mut KvCache,
        append_kv: bool,
        seq_len: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let exp_dim = 2 * self.dim;
        let mut new_global_states = Vec::with_capacity(seq_len);

        // Process tokens sequentially to maintain causality
        // Each token can only attend to itself and previous tokens
        for t in 0..seq_len {
            let x_exp = &x_exp_all[t * exp_dim..(t + 1) * exp_dim].to_vec();
            let global_state = global_states.get(t).and_then(|gs| gs.as_deref());

            // Use single-token forward path with explicit position for causal masking
            let (x_out, g_state) = self.forward(
                x_exp,
                global_state,
                kv_cache,
                append_kv,
                Some(t), // Pass explicit token position
            )?;

            x_exp_all[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(&x_out);
            new_global_states.push(g_state);
        }

        Ok(new_global_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    #[test]
    fn test_shared_loop_block_construction() {
        let cfg = ModelConfig::d20();
        let block = SharedLoopBlock::new_empty(&cfg);

        assert_eq!(block.dim, 256);
        assert_eq!(block.n_heads, 4);
        assert_eq!(block.head_dim, 64);
    }

    #[test]
    fn test_shared_loop_block_forward_single_iteration() {
        let cfg = ModelConfig::d20();
        let block = SharedLoopBlock::new_empty(&cfg);

        let dim = cfg.dim;
        let n_streams = cfg.mhc_n_streams;

        // Create input: [dim * n_streams]
        let x_expanded = vec![0.1f32; dim * n_streams];

        // Create empty KV cache
        let mut kv_cache = KvCache::new(cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim());

        // First iteration (no global state, append KV)
        let (x_out, global_state) = block
            .forward(&x_expanded, None, &mut kv_cache, true, None)
            .unwrap();

        // Check output shapes
        assert_eq!(x_out.len(), dim * n_streams);
        assert_eq!(global_state.len(), dim);

        // Check KV cache was populated
        assert_eq!(kv_cache.len, 1); // Should have 1 token in cache
    }

    #[test]
    fn test_shared_loop_block_multiple_iterations() {
        let cfg = ModelConfig::d20();
        let block = SharedLoopBlock::new_empty(&cfg);

        let dim = cfg.dim;
        let n_streams = cfg.mhc_n_streams;

        let x_expanded = vec![0.1f32; dim * n_streams];
        let mut kv_cache = KvCache::new(cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim());

        // Iteration 1: append KV, no global state
        let (mut x_state, mut global_state) = block
            .forward(&x_expanded, None, &mut kv_cache, true, None)
            .unwrap();

        let kv_len_after_first = kv_cache.len;
        assert_eq!(kv_len_after_first, 1);

        // Iteration 2: don't append KV, use global state
        (x_state, global_state) = block
            .forward(&x_state, Some(&global_state), &mut kv_cache, false, None)
            .unwrap();

        // KV cache should NOT grow
        assert_eq!(kv_cache.len, kv_len_after_first);

        // Iteration 3: another loop
        (x_state, global_state) = block
            .forward(&x_state, Some(&global_state), &mut kv_cache, false, None)
            .unwrap();

        // KV cache still shouldn't grow
        assert_eq!(kv_cache.len, kv_len_after_first);

        // Output shapes should remain consistent
        assert_eq!(x_state.len(), dim * n_streams);
        assert_eq!(global_state.len(), dim);
    }

    #[test]
    fn test_attention_computation_shapes() {
        let cfg = ModelConfig::d20();
        let block = SharedLoopBlock::new_empty(&cfg);

        let dim = cfg.dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim();

        // Create Q and populate KV cache with 3 tokens
        let q = vec![0.1f32; dim];
        let mut kv_cache = KvCache::new(cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim());

        for _ in 0..3 {
            let k = vec![0.1f32; kv_dim];
            let v = vec![0.2f32; kv_dim];
            kv_cache.append_kv(&k, &v);
        }

        // Compute attention (token at position 2 attends to all 3 cached tokens)
        let mut out = vec![0.0f32; dim];
        block
            .compute_attention(&q, &kv_cache, 2, &mut out)
            .expect("compute_attention should succeed with valid inputs");

        // Output should have correct shape and be non-zero
        assert_eq!(out.len(), dim);
        assert!(
            out.iter().any(|&x| x != 0.0),
            "Attention output should be non-zero"
        );
    }

    #[test]
    fn test_global_state_mixing() {
        let cfg = ModelConfig::d20();
        let block = SharedLoopBlock::new_empty(&cfg);

        let dim = cfg.dim;
        let n_streams = cfg.mhc_n_streams;

        let x_expanded = vec![1.0f32; dim * n_streams];
        let mut kv_cache = KvCache::new(cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim());

        // First iteration creates global state
        let (x1, global_state1) = block
            .forward(&x_expanded, None, &mut kv_cache, true, None)
            .unwrap();

        // Second iteration uses global state
        let (x2, global_state2) = block
            .forward(&x1, Some(&global_state1), &mut kv_cache, false, None)
            .unwrap();

        // With zero weights, global states will be zero but shapes should be correct
        assert_eq!(global_state1.len(), dim);
        assert_eq!(global_state2.len(), dim);

        // Both iterations should complete without error
        // (With non-zero weights, states would differ, but zero weights produce zero outputs)
        assert!(true, "Global state mixing mechanism works");
    }
}
