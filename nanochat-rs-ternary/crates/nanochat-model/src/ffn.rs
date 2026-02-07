//! SwiGLU Feed-Forward Network (all BitLinear projections) and MoE routing.

use crate::bitlinear::BitLinear;
use crate::config::ModelConfig;

/// SwiGLU FFN: out = w_down(silu(w_gate(x)) * w_up(x))
#[derive(Debug)]
pub struct FeedForward {
    pub w_gate: BitLinear,
    pub w_up: BitLinear,
    pub w_down: BitLinear,
    pub ffn_dim: usize,
}

impl FeedForward {
    /// Create FFN with random weights (for testing).
    pub fn new_random(config: &ModelConfig) -> Self {
        let ffn_dim = config.ffn_dim();
        let dim = config.dim;
        let gs = config.group_size;

        Self {
            w_gate: BitLinear::from_float(&random_weights(ffn_dim, dim, 10), ffn_dim, dim, gs),
            w_up: BitLinear::from_float(&random_weights(ffn_dim, dim, 11), ffn_dim, dim, gs),
            w_down: BitLinear::from_float(&random_weights(dim, ffn_dim, 12), dim, ffn_dim, gs),
            ffn_dim,
        }
    }

    /// Forward pass.
    ///
    /// x:   [dim] input
    /// out: [dim] output
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let dim = self.w_down.rows;

        // Gate and Up projections
        let mut gate = vec![0.0f32; self.ffn_dim];
        let mut up = vec![0.0f32; self.ffn_dim];
        self.w_gate.forward(x, &mut gate);
        self.w_up.forward(x, &mut up);

        // SwiGLU: silu(gate) * up
        for i in 0..self.ffn_dim {
            gate[i] = silu(gate[i]) * up[i];
        }

        // Down projection
        self.w_down.forward(&gate, out);
        let _ = dim; // silence unused
    }
}

/// Mixture of Experts with top-k gating.
///
/// Routes each token to the top-k experts based on router logits,
/// then computes a weighted sum of their outputs.
#[derive(Debug)]
pub struct MoeExperts {
    /// Router projection: [n_experts, dim] -- produces gating logits
    pub router: BitLinear,
    /// Individual FFN blocks, one per expert
    pub experts: Vec<FeedForward>,
    /// Number of active experts per token (top-k)
    pub n_active: usize,
}

impl MoeExperts {
    /// Create MoE with random weights (for testing).
    pub fn new_random(config: &ModelConfig) -> Self {
        let n_experts = config.n_experts.expect("n_experts must be set for MoE");
        let n_active = config.n_active_experts.expect("n_active_experts must be set for MoE");

        // Router: maps dim -> n_experts
        let router_weights: Vec<f32> = (0..n_experts * config.dim)
            .map(|i| {
                let v = ((i as u32).wrapping_add(99 * 7919)).wrapping_mul(2654435761) >> 16;
                (v % 200) as f32 / 100.0 - 1.0
            })
            .collect();
        let router = BitLinear::from_float(&router_weights, n_experts, config.dim, config.group_size);

        let experts: Vec<FeedForward> = (0..n_experts)
            .map(|_| FeedForward::new_random(config))
            .collect();

        Self { router, experts, n_active }
    }

    /// Forward: route token to top-k experts, weighted sum of outputs.
    ///
    /// x:   [dim] input
    /// out: [dim] output (accumulated weighted expert outputs)
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let n_experts = self.experts.len();

        // 1. Router logits: [n_experts]
        let mut logits = vec![0.0f32; n_experts];
        self.router.forward(x, &mut logits);

        // 2. Top-k selection
        let top_k = top_k_indices(&logits, self.n_active);

        // 3. Softmax over selected experts' logits (gating weights)
        let weights = softmax_selected(&logits, &top_k);

        // 4. For each selected expert: run FFN, accumulate weighted output
        let dim = x.len();
        for v in out.iter_mut() {
            *v = 0.0;
        }

        let mut expert_out = vec![0.0f32; dim];
        for (idx, &expert_id) in top_k.iter().enumerate() {
            for v in expert_out.iter_mut() {
                *v = 0.0;
            }
            self.experts[expert_id].forward(x, &mut expert_out);

            let w = weights[idx];
            for j in 0..dim {
                out[j] += w * expert_out[j];
            }
        }
    }
}

/// Unified FFN layer: either dense or MoE.
#[derive(Debug)]
pub enum FfnLayer {
    Dense(FeedForward),
    Moe(MoeExperts),
}

impl FfnLayer {
    /// Forward pass through either dense FFN or MoE.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        match self {
            FfnLayer::Dense(ffn) => ffn.forward(x, out),
            FfnLayer::Moe(moe) => moe.forward(x, out),
        }
    }

    /// Batched forward: process `seq_len` tokens.
    pub fn forward_batch(&self, x_batch: &[f32], seq_len: usize, out_batch: &mut [f32]) {
        let dim = match self {
            FfnLayer::Dense(ffn) => ffn.w_down.rows,
            FfnLayer::Moe(moe) => moe.router.cols,
        };
        assert_eq!(x_batch.len(), seq_len * dim);
        assert_eq!(out_batch.len(), seq_len * dim);

        for t in 0..seq_len {
            let x = &x_batch[t * dim..(t + 1) * dim];
            let out = &mut out_batch[t * dim..(t + 1) * dim];
            self.forward(x, out);
        }
    }
}

/// Select top-k indices from a slice of values (largest first).
pub fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    // Sort by value descending
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

/// Softmax over selected indices only.
///
/// Returns weights corresponding to the selected indices (same order as `selected`).
pub fn softmax_selected(logits: &[f32], selected: &[usize]) -> Vec<f32> {
    // Find max for numerical stability
    let max_val = selected
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = selected.iter().map(|&i| (logits[i] - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        // Fallback: uniform weights
        vec![1.0 / selected.len() as f32; selected.len()]
    }
}

/// SiLU activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Deterministic pseudo-random weights for testing.
fn random_weights(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; rows * cols];
    for i in 0..w.len() {
        let v = ((i as u32).wrapping_add(seed * 7919)).wrapping_mul(2654435761) >> 16;
        w[i] = (v % 200) as f32 / 100.0 - 1.0;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.0);
        assert!(silu(-1.0) < 0.0);
        // silu(x) ≈ x for large positive x
        assert!((silu(10.0) - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_ffn_forward() {
        let config = ModelConfig::d20();
        let ffn = FeedForward::new_random(&config);

        let x = vec![0.1f32; config.dim];
        let mut out = vec![0.0f32; config.dim];
        ffn.forward(&x, &mut out);

        assert!(out.iter().all(|v| v.is_finite()), "non-finite FFN output");
        assert!(out.iter().any(|&v| v != 0.0), "all-zero FFN output");
    }

    #[test]
    fn test_ffn_dim_matches_config() {
        let config = ModelConfig::d20();
        let ffn = FeedForward::new_random(&config);
        assert_eq!(ffn.ffn_dim, config.ffn_dim());
        assert_eq!(ffn.w_gate.rows, config.ffn_dim());
        assert_eq!(ffn.w_gate.cols, config.dim);
        assert_eq!(ffn.w_down.rows, config.dim);
        assert_eq!(ffn.w_down.cols, config.ffn_dim());
    }

    #[test]
    fn test_moe_forward_finite_output() {
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
            n_experts: Some(8),
            n_active_experts: Some(2),
            deltanet_ratio: None,
            weight_tied: false,
        };
        let moe = MoeExperts::new_random(&config);

        let x = vec![0.1f32; config.dim];
        let mut out = vec![0.0f32; config.dim];
        moe.forward(&x, &mut out);

        assert!(out.iter().all(|v| v.is_finite()), "non-finite MoE output");
        assert!(out.iter().any(|&v| v != 0.0), "all-zero MoE output");
    }

    #[test]
    fn test_router_top_k_selection() {
        // 8 experts, select top 2
        let logits = vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4];
        let top2 = top_k_indices(&logits, 2);

        assert_eq!(top2.len(), 2);
        // Highest is index 2 (0.9), second is index 4 (0.8)
        assert_eq!(top2[0], 2, "top-1 should be expert 2 (value 0.9)");
        assert_eq!(top2[1], 4, "top-2 should be expert 4 (value 0.8)");
    }

    #[test]
    fn test_softmax_weights_sum_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 0.5, 4.0, 1.5, 2.5, 0.1];
        let selected = vec![2, 4]; // select experts 2 and 4
        let weights = softmax_selected(&logits, &selected);

        assert_eq!(weights.len(), 2);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax weights should sum to 1.0, got {}",
            sum
        );
        // All weights should be positive
        assert!(weights.iter().all(|&w| w > 0.0), "all weights must be positive");
        // Weight for expert 4 (logit=4.0) should be larger than expert 2 (logit=3.0)
        assert!(weights[1] > weights[0], "expert 4 should have higher weight than expert 2");
    }

    #[test]
    fn test_ffn_layer_dense_and_moe() {
        let dense_config = ModelConfig::d20();
        let moe_config = ModelConfig {
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

        // Dense variant
        let dense_layer = FfnLayer::Dense(FeedForward::new_random(&dense_config));
        let x_dense = vec![0.1f32; dense_config.dim];
        let mut out_dense = vec![0.0f32; dense_config.dim];
        dense_layer.forward(&x_dense, &mut out_dense);
        assert!(out_dense.iter().all(|v| v.is_finite()), "non-finite dense output");

        // MoE variant
        let moe_layer = FfnLayer::Moe(MoeExperts::new_random(&moe_config));
        let x_moe = vec![0.1f32; moe_config.dim];
        let mut out_moe = vec![0.0f32; moe_config.dim];
        moe_layer.forward(&x_moe, &mut out_moe);
        assert!(out_moe.iter().all(|v| v.is_finite()), "non-finite MoE output");
    }

    #[test]
    fn test_moe_config_presets_valid() {
        // moe_25b
        let c25 = ModelConfig::moe_25b();
        assert_eq!(c25.n_experts, Some(8));
        assert_eq!(c25.n_active_experts, Some(2));
        assert!(c25.n_active_experts.unwrap() <= c25.n_experts.unwrap());

        // moe_80b
        let c80 = ModelConfig::moe_80b();
        assert_eq!(c80.n_experts, Some(16));
        assert_eq!(c80.n_active_experts, Some(2));
        assert!(c80.n_active_experts.unwrap() <= c80.n_experts.unwrap());

        // ffn_dim should be aligned for both
        assert!(c25.ffn_dim() % c25.group_size == 0);
        assert!(c80.ffn_dim() % c80.group_size == 0);
    }
}
