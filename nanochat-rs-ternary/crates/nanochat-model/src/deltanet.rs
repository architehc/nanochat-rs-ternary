//! DeltaNet hybrid attention — linear recurrent attention with delta rule updates.
//!
//! DeltaNet maintains per-head recurrent state S: [head_dim, head_dim] instead of
//! a KV cache. This makes it O(1) memory per token at inference time, suitable
//! for long-context scenarios.
//!
//! Update rule per token:
//! ```text
//! beta = sigmoid(W_beta @ x)
//! k_norm = normalize(W_k @ x)
//! S = S - beta * (S @ k_norm) @ k_norm^T + beta * (W_v @ x) @ k_norm^T
//! output = S @ (W_q @ x)
//! ```

use crate::bitlinear::BitLinear;
use crate::config::ModelConfig;

/// Recurrent state for DeltaNet attention (one per layer).
///
/// Each head maintains a [head_dim x head_dim] state matrix S.
/// Total state size: n_heads * head_dim * head_dim floats.
#[derive(Debug, Clone)]
pub struct DeltaNetState {
    /// Flattened state: [n_heads * head_dim * head_dim]
    /// Layout: s[h * head_dim * head_dim + i * head_dim + j] = S_h[i][j]
    pub s: Vec<f32>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl DeltaNetState {
    /// Create a new zero-initialized DeltaNet state.
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        Self {
            s: vec![0.0; n_heads * head_dim * head_dim],
            n_heads,
            head_dim,
        }
    }

    /// Reset state to zeros.
    pub fn reset(&mut self) {
        for v in self.s.iter_mut() {
            *v = 0.0;
        }
    }
}

/// DeltaNet attention layer.
///
/// Uses linear recurrent attention with delta rule updates instead of
/// softmax attention + KV cache. All projections are BitLinear (ternary).
#[derive(Debug)]
pub struct DeltaNetAttention {
    /// Query projection: dim -> dim
    pub wq: BitLinear,
    /// Key projection: dim -> dim
    pub wk: BitLinear,
    /// Value projection: dim -> dim
    pub wv: BitLinear,
    /// Output projection: dim -> dim
    pub wo: BitLinear,
    /// Beta gate projection: dim -> n_heads (one scalar beta per head)
    pub w_beta: BitLinear,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl DeltaNetAttention {
    /// Create DeltaNet attention with random weights (for testing).
    pub fn new_random(config: &ModelConfig) -> Self {
        let dim = config.dim;
        let hd = config.deltanet_qk_head_dim();
        let gs = config.group_size;
        let n_heads = config.deltanet_qk_heads.unwrap_or(config.n_heads);

        Self {
            wq: BitLinear::from_float(&random_weights(dim, dim, 20), dim, dim, gs),
            wk: BitLinear::from_float(&random_weights(dim, dim, 21), dim, dim, gs),
            wv: BitLinear::from_float(&random_weights(dim, dim, 22), dim, dim, gs),
            wo: BitLinear::from_float(&random_weights(dim, dim, 23), dim, dim, gs),
            w_beta: BitLinear::from_float(&random_weights(n_heads, dim, 24), n_heads, dim, gs),
            n_heads,
            head_dim: hd,
        }
    }

    /// Forward pass for a single token, updating the recurrent state.
    ///
    /// x:     [dim] input activations
    /// state: DeltaNetState for this layer (mutated in-place)
    /// out:   [dim] output buffer
    pub fn forward(&self, x: &[f32], state: &mut DeltaNetState, out: &mut [f32]) {
        let dim = self.wq.cols;
        let n_heads = self.n_heads;
        let hd = self.head_dim;
        assert_eq!(x.len(), dim);
        assert_eq!(out.len(), dim);
        assert_eq!(state.n_heads, n_heads);
        assert_eq!(state.head_dim, hd);

        // Project Q, K, V
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; dim];
        let mut v = vec![0.0f32; dim];
        let mut beta_raw = vec![0.0f32; n_heads];

        self.wq.forward(x, &mut q);
        self.wk.forward(x, &mut k);
        self.wv.forward(x, &mut v);
        self.w_beta.forward(x, &mut beta_raw);

        // Sigmoid for beta gates
        for b in beta_raw.iter_mut() {
            *b = sigmoid(*b);
        }

        // Normalize K per head (L2 normalization)
        let mut k_norm = k.clone();
        for h in 0..n_heads {
            let offset = h * hd;
            l2_normalize(&mut k_norm[offset..offset + hd]);
        }

        // Per-head recurrent update + output computation
        let mut attn_out = vec![0.0f32; dim];

        for (h, &beta) in beta_raw.iter().enumerate().take(n_heads) {
            let h_offset = h * hd;
            let s_offset = h * hd * hd;

            // Compute S @ k_norm for this head -> sk: [head_dim]
            let mut sk = vec![0.0f32; hd];
            for (i, sk_val) in sk.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for j in 0..hd {
                    sum += state.s[s_offset + i * hd + j] * k_norm[h_offset + j];
                }
                *sk_val = sum;
            }

            // Update S:
            // S = S - beta * sk @ k_norm^T + beta * v @ k_norm^T
            // = S + beta * (v - sk) @ k_norm^T
            for i in 0..hd {
                let diff_i = v[h_offset + i] - sk[i];
                for j in 0..hd {
                    state.s[s_offset + i * hd + j] += beta * diff_i * k_norm[h_offset + j];
                }
            }

            // Output: S @ q for this head
            for i in 0..hd {
                let mut sum = 0.0f32;
                for j in 0..hd {
                    sum += state.s[s_offset + i * hd + j] * q[h_offset + j];
                }
                attn_out[h_offset + i] = sum;
            }
        }

        // Output projection
        self.wo.forward(&attn_out, out);
    }

    /// Batched forward pass for DeltaNet recurrent attention.
    ///
    /// This remains recurrent across sequence positions (stateful by design),
    /// but reuses temporary buffers to avoid per-token allocations.
    pub fn forward_batch(
        &self,
        x_batch: &[f32],
        seq_len: usize,
        state: &mut DeltaNetState,
        out_batch: &mut [f32],
    ) {
        let dim = self.wq.cols;
        let n_heads = self.n_heads;
        let hd = self.head_dim;
        assert_eq!(x_batch.len(), seq_len * dim);
        assert_eq!(out_batch.len(), seq_len * dim);
        assert_eq!(state.n_heads, n_heads);
        assert_eq!(state.head_dim, hd);

        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; dim];
        let mut v = vec![0.0f32; dim];
        let mut k_norm = vec![0.0f32; dim];
        let mut beta_raw = vec![0.0f32; n_heads];
        let mut attn_out = vec![0.0f32; dim];
        let mut sk = vec![0.0f32; hd];

        for t in 0..seq_len {
            let x = &x_batch[t * dim..(t + 1) * dim];
            let out = &mut out_batch[t * dim..(t + 1) * dim];

            self.wq.forward(x, &mut q);
            self.wk.forward(x, &mut k);
            self.wv.forward(x, &mut v);
            self.w_beta.forward(x, &mut beta_raw);

            for b in beta_raw.iter_mut() {
                *b = sigmoid(*b);
            }

            k_norm.copy_from_slice(&k);
            for h in 0..n_heads {
                let offset = h * hd;
                l2_normalize(&mut k_norm[offset..offset + hd]);
            }

            attn_out.fill(0.0);
            for (h, &beta) in beta_raw.iter().enumerate().take(n_heads) {
                let h_offset = h * hd;
                let s_offset = h * hd * hd;

                for (i, sk_val) in sk.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for j in 0..hd {
                        sum += state.s[s_offset + i * hd + j] * k_norm[h_offset + j];
                    }
                    *sk_val = sum;
                }

                for i in 0..hd {
                    let diff_i = v[h_offset + i] - sk[i];
                    for j in 0..hd {
                        state.s[s_offset + i * hd + j] += beta * diff_i * k_norm[h_offset + j];
                    }
                }

                for i in 0..hd {
                    let mut sum = 0.0f32;
                    for j in 0..hd {
                        sum += state.s[s_offset + i * hd + j] * q[h_offset + j];
                    }
                    attn_out[h_offset + i] = sum;
                }
            }

            self.wo.forward(&attn_out, out);
        }
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// L2-normalize a vector in-place, with epsilon for numerical stability.
#[inline]
fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    if norm_sq <= 1e-8 {
        for x in v.iter_mut() {
            *x = 0.0;
        }
        return;
    }
    let norm = norm_sq.sqrt();
    let inv_norm = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

/// Generate deterministic pseudo-random weights for testing.
fn random_weights(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; rows * cols];
    for (i, val) in w.iter_mut().enumerate() {
        let v = ((i as u32).wrapping_add(seed * 7919)).wrapping_mul(2654435761) >> 16;
        *val = (v % 200) as f32 / 100.0 - 1.0;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn deltanet_test_config() -> ModelConfig {
        ModelConfig::test_config(128, 2, 4, 256)
    }

    #[test]
    fn test_deltanet_state_init_zero() {
        let state = DeltaNetState::new(4, 32);
        assert_eq!(state.s.len(), 4 * 32 * 32);
        assert!(
            state.s.iter().all(|&v| v == 0.0),
            "state should be all zeros"
        );
    }

    #[test]
    fn test_deltanet_single_token_forward() {
        let config = deltanet_test_config();
        let attn = DeltaNetAttention::new_random(&config);
        let mut state = DeltaNetState::new(config.n_heads, config.head_dim());

        let x = vec![0.1f32; config.dim];
        let mut out = vec![0.0f32; config.dim];

        attn.forward(&x, &mut state, &mut out);

        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite DeltaNet output"
        );
        // After one token, state should be non-zero (beta > 0, so update happens)
        assert!(
            state.s.iter().any(|&v| v != 0.0),
            "state should be non-zero after one token"
        );
    }

    #[test]
    fn test_deltanet_sequence_state_accumulation() {
        let config = deltanet_test_config();
        let attn = DeltaNetAttention::new_random(&config);
        let mut state = DeltaNetState::new(config.n_heads, config.head_dim());

        let mut out = vec![0.0f32; config.dim];

        // Process multiple tokens — state should change each time
        let mut prev_state_sum = 0.0f32;
        for i in 0..5 {
            let x: Vec<f32> = (0..config.dim)
                .map(|j| ((i * 37 + j * 13) % 200) as f32 / 100.0 - 1.0)
                .collect();
            attn.forward(&x, &mut state, &mut out);

            let state_sum: f32 = state.s.iter().map(|v| v.abs()).sum();
            assert!(
                out.iter().all(|v| v.is_finite()),
                "non-finite output at token {}",
                i
            );

            if i > 0 {
                // State should change between tokens
                assert!(
                    (state_sum - prev_state_sum).abs() > 1e-6,
                    "state not accumulating at token {}",
                    i
                );
            }
            prev_state_sum = state_sum;
        }
    }

    #[test]
    fn test_deltanet_state_reset() {
        let config = deltanet_test_config();
        let attn = DeltaNetAttention::new_random(&config);
        let mut state = DeltaNetState::new(config.n_heads, config.head_dim());

        let x = vec![0.1f32; config.dim];
        let mut out = vec![0.0f32; config.dim];

        // Process a token to make state non-zero
        attn.forward(&x, &mut state, &mut out);
        assert!(state.s.iter().any(|&v| v != 0.0));

        // Reset
        state.reset();
        assert!(
            state.s.iter().all(|&v| v == 0.0),
            "state should be zero after reset"
        );
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2 norm should be 1.0, got {}",
            norm
        );
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        // Should not panic, should produce zeros (or near-zero due to epsilon)
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_deltanet_forward_batch_matches_stepwise() {
        let config = deltanet_test_config();
        let attn = DeltaNetAttention::new_random(&config);
        let seq_len = 6usize;
        let dim = config.dim;

        let mut x_batch = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            for d in 0..dim {
                x_batch[t * dim + d] = ((t * 97 + d * 13) % 200) as f32 / 100.0 - 1.0;
            }
        }

        let mut state_step = DeltaNetState::new(config.n_heads, config.head_dim());
        let mut out_step = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            let x = &x_batch[t * dim..(t + 1) * dim];
            let out = &mut out_step[t * dim..(t + 1) * dim];
            attn.forward(x, &mut state_step, out);
        }

        let mut state_batch = DeltaNetState::new(config.n_heads, config.head_dim());
        let mut out_batch = vec![0.0f32; seq_len * dim];
        attn.forward_batch(&x_batch, seq_len, &mut state_batch, &mut out_batch);

        let max_diff = out_step
            .iter()
            .zip(out_batch.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "batch DeltaNet output deviates from stepwise output: {}",
            max_diff
        );
    }
}
