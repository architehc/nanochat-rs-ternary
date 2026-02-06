//! SwiGLU Feed-Forward Network (all BitLinear projections).

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
}
