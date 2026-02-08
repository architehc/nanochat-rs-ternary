//! Training configuration for nanochat ternary models.

use serde::{Deserialize, Serialize};

/// Model + training hyperparameter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    // Model architecture
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub ffn_mult: f32,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub group_size: usize,
    pub mhc_n_streams: usize,
    pub weight_tied: bool,
    pub rope_theta: f32,

    // Training hyperparams
    pub lr: f64,
    pub mhc_lr: f64,
    pub weight_decay: f64,
    pub batch_size: usize,
    pub grad_accum_steps: usize,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub decay_start_frac: f64,
    pub grad_clip: f64,
    pub ns_steps: usize,
    pub muon_momentum: f64,
    pub lion_betas: (f64, f64),
}

impl TrainConfig {
    /// Compute FFN hidden dimension from dim and ffn_mult, aligned to group_size.
    pub fn ffn_dim(&self) -> usize {
        let raw = (self.dim as f32 * self.ffn_mult) as usize;
        // Round up to multiple of group_size
        raw.div_ceil(self.group_size) * self.group_size
    }

    /// Estimate total parameter count.
    pub fn param_count_estimate(&self) -> usize {
        let ffn_dim = self.ffn_dim();
        let d = self.dim;
        let v = self.vocab_size;

        // Embedding
        let embed = v * d;

        // Per layer
        let attn = 4 * d * d; // wq, wk, wv, wo (simplified, ignoring GQA)
        let ffn = 3 * d * ffn_dim; // w_gate, w_up, w_down
        let norms = 2 * d;
        let mhc = if self.mhc_n_streams == 2 { 18 } else { 80 }; // 2 sub-layers * params_per
        let per_layer = attn + ffn + norms + mhc;

        // LM head
        let lm_head = if self.weight_tied { 0 } else { d * v };

        // Final norm
        let final_norm = d;

        embed + self.n_layers * per_layer + lm_head + final_norm
    }

    /// ~1.1B parameter model (GPT-2 scale).
    pub fn nano_1b() -> Self {
        Self {
            dim: 2048,
            n_layers: 20,
            n_heads: 16,
            n_kv_heads: 16,
            ffn_mult: 2.6875, // gives ffn_dim = 5504 aligned to 128
            vocab_size: 50257,
            max_seq_len: 1024,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,

            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 4,
            grad_accum_steps: 8,
            warmup_steps: 2000,
            total_steps: 100_000,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 5,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
        }
    }

    /// Small debug model (~20M params).
    pub fn d20() -> Self {
        Self {
            dim: 256,
            n_layers: 6,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.6875,
            vocab_size: 50257,
            max_seq_len: 256,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,

            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 32,
            grad_accum_steps: 1,
            warmup_steps: 100,
            total_steps: 5000,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 5,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
        }
    }

    /// Tiny CPU-friendly config (~2M params, vocab=4096).
    /// Designed for fast training iteration on CPU.
    pub fn tiny_cpu() -> Self {
        Self {
            dim: 256,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.0,
            vocab_size: 4096,
            max_seq_len: 256,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,

            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 8,
            grad_accum_steps: 1,
            warmup_steps: 50,
            total_steps: 5000,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 3,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
        }
    }

    /// Medium model (~125M params).
    pub fn nano_125m() -> Self {
        Self {
            dim: 768,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            ffn_mult: 2.6875,
            vocab_size: 50257,
            max_seq_len: 512,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,

            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 8,
            grad_accum_steps: 4,
            warmup_steps: 500,
            total_steps: 50_000,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 5,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_1b_ffn_dim_aligned() {
        let cfg = TrainConfig::nano_1b();
        let ffn_dim = cfg.ffn_dim();
        assert_eq!(ffn_dim % cfg.group_size, 0, "ffn_dim {} not aligned to {}", ffn_dim, cfg.group_size);
        assert!(ffn_dim >= 5000 && ffn_dim <= 6000, "ffn_dim {} out of expected range", ffn_dim);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = TrainConfig::nano_1b();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: TrainConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.dim, cfg2.dim);
        assert_eq!(cfg.n_layers, cfg2.n_layers);
        assert_eq!(cfg.vocab_size, cfg2.vocab_size);
        assert_eq!(cfg.weight_tied, cfg2.weight_tied);
        assert!((cfg.lr - cfg2.lr).abs() < 1e-10);
    }

    #[test]
    fn test_nano_1b_param_estimate() {
        let cfg = TrainConfig::nano_1b();
        let params = cfg.param_count_estimate();
        // Should be roughly 1B-1.2B
        assert!(params > 900_000_000, "Too few params: {}", params);
        assert!(params < 1_300_000_000, "Too many params: {}", params);
    }

    #[test]
    fn test_d20_param_estimate() {
        let cfg = TrainConfig::d20();
        let params = cfg.param_count_estimate();
        // d20 should be roughly 15M-30M
        assert!(params > 10_000_000, "Too few params: {}", params);
        assert!(params < 50_000_000, "Too many params: {}", params);
    }
}
