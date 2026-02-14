//! Training configuration for nanochat ternary models.

use serde::{Deserialize, Serialize};

fn default_galore_rank() -> usize {
    256 // Config B default (dual RTX 4090)
}

fn default_galore_update_freq() -> usize {
    200 // Update projections every 200 steps
}

fn default_mtp_n_tokens() -> usize {
    3 // Predict next 3 tokens: [1.0, 0.5, 0.25] weights
}

fn default_mtp_weight() -> f64 {
    0.2 // 20% weight for auxiliary MTP losses
}

fn default_collider_threshold() -> f64 {
    0.3 // Filter tokens with importance < 0.3
}

fn default_collider_sparsity() -> f64 {
    0.35 // Target 35% sparsity for 35% backprop speedup
}

fn default_async_n_workers() -> usize {
    4 // 4 preprocessing threads (balance CPU usage vs parallelism)
}

fn default_async_prefetch_size() -> usize {
    8 // Prefetch 8 batches (smooths variance, low memory overhead)
}

/// Adaptive loop control for inference (LoopLM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLoopConfig {
    /// Minimum number of loop iterations
    pub min_loops: usize,
    /// Maximum number of loop iterations
    pub max_loops: usize,
    /// Perplexity threshold for early stopping
    pub perplexity_threshold: f32,
}

/// LoopLM configuration: recurrent loop mechanics per arXiv:2510.25741.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopConfig {
    /// Number of local (non-looped) layers before the shared loop block
    pub local_before: usize,
    /// Number of local (non-looped) layers after the shared loop block
    pub local_after: usize,
    /// Number of shared loop iterations (L in paper)
    pub loop_count: usize,
    /// Optional: adaptive loop control for inference (can vary loop_count at runtime)
    pub adaptive_loop: Option<AdaptiveLoopConfig>,
}

impl LoopConfig {
    /// Total effective depth: local_before + loop_count + local_after
    pub fn effective_depth(&self) -> usize {
        self.local_before + self.loop_count + self.local_after
    }

    /// Whether this config uses looping (loop_count > 1)
    pub fn is_looped(&self) -> bool {
        self.loop_count > 1
    }
}

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

    /// LoopLM configuration (None = standard fixed-depth transformer)
    #[serde(default)]
    pub loop_config: Option<LoopConfig>,

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

    // Advanced optimizer options (E2 recommendations)
    /// Use 8-bit quantized optimizer states (arXiv:2509.23106)
    #[serde(default)]
    pub use_8bit_optim: bool,
    /// Use GaLore 2 low-rank gradient projection (arXiv:2504.20437)
    #[serde(default)]
    pub use_galore: bool,
    /// GaLore rank (Config A=512, Config B=256, Config C=384)
    #[serde(default = "default_galore_rank")]
    pub galore_rank: usize,
    /// GaLore projection update frequency
    #[serde(default = "default_galore_update_freq")]
    pub galore_update_freq: usize,

    // Multi-Token Prediction (arXiv:2204.05832)
    /// Use Multi-Token Prediction for denser training signals
    #[serde(default)]
    pub use_mtp: bool,
    /// Number of future tokens to predict (2-4 recommended)
    #[serde(default = "default_mtp_n_tokens")]
    pub mtp_n_tokens: usize,
    /// MTP auxiliary loss weight (0.1-0.3 recommended)
    #[serde(default = "default_mtp_weight")]
    pub mtp_weight: f64,

    // Collider Token Filtering (arXiv:2502.00340)
    /// Use Collider for token filtering (35% faster backprop)
    #[serde(default)]
    pub use_collider: bool,
    /// Importance threshold for filtering (0-1)
    #[serde(default = "default_collider_threshold")]
    pub collider_threshold: f64,
    /// Target sparsity ratio (0-1)
    #[serde(default = "default_collider_sparsity")]
    pub collider_sparsity: f64,

    // Async Data Loader (90%+ GPU utilization)
    /// Use async data loader with multi-threaded prefetching
    #[serde(default)]
    pub use_async_loader: bool,
    /// Number of preprocessing worker threads (2-8 recommended)
    #[serde(default = "default_async_n_workers")]
    pub async_n_workers: usize,
    /// Number of batches to prefetch (4-16 recommended)
    #[serde(default = "default_async_prefetch_size")]
    pub async_prefetch_size: usize,

    // Stage-1 distillation hyperparams (LoopLM training)
    /// Teacher model path for distillation (None = no distillation)
    #[serde(default)]
    pub distill_teacher: Option<String>,
    /// KL divergence weight for distillation loss
    #[serde(default)]
    pub distill_kl_weight: f64,
    /// Loop scale penalty weight (annealed during training)
    #[serde(default)]
    pub loop_scale_penalty: f64,
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
            loop_config: None,

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
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
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
            loop_config: None,

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
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
        }
    }

    /// Production d20 (560M) with full E3 optimization stack.
    /// Designed for RTX PRO 6000 Ada (96GB VRAM).
    pub fn d20_e3_full() -> Self {
        Self {
            // d20 architecture (560M params)
            dim: 768,
            n_layers: 24,
            n_heads: 12,
            n_kv_heads: 12,
            ffn_mult: 3.5,
            vocab_size: 50257,
            max_seq_len: 2048,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: false,
            rope_theta: 10000.0,
            loop_config: None,

            // WSD optimizer (optimized for Ada GPU)
            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 8,           // 8 × 2048 = 16K tokens/batch
            grad_accum_steps: 4,      // Effective: 65K tokens
            warmup_steps: 2000,
            total_steps: 50_000,      // 3.25B tokens total
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 3,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),

            // E2 Advanced (disabled for baseline)
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,

            // E3 P0: Multi-Token Prediction (ENABLED)
            use_mtp: true,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,

            // E3 P1: Collider Token Filtering (ENABLED)
            use_collider: true,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,

            // E3 P0: Async Data Loader (ENABLED)
            use_async_loader: true,
            async_n_workers: 6,
            async_prefetch_size: 12,

            // Distillation (disabled)
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
        }
    }

    /// Test config for 8-bit optimizer validation (small, fast).
    pub fn test_8bit() -> Self {
        Self {
            dim: 256,
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 8,
            ffn_mult: 2.0,
            vocab_size: 50257,
            max_seq_len: 256,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: false,
            rope_theta: 10000.0,
            loop_config: None,

            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 20,
            total_steps: 100,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 3,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
            use_8bit_optim: true, // ← 8-bit enabled for testing
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
        }
    }

    /// LoopLM variant of d20: 2 local + 4-iteration shared loop = 6 effective layers.
    pub fn d20_loop() -> Self {
        Self {
            dim: 256,
            n_layers: 3, // 2 local + 1 shared
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.6875,
            vocab_size: 50257,
            max_seq_len: 256,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: Some(LoopConfig {
                local_before: 1,
                local_after: 1,
                loop_count: 4,
                adaptive_loop: Some(AdaptiveLoopConfig {
                    min_loops: 2,
                    max_loops: 6,
                    perplexity_threshold: 5.0,
                }),
            }),

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
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            distill_teacher: None, // Can be set to teacher model path
            distill_kl_weight: 1.0,
            loop_scale_penalty: 0.1, // Annealed during training
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
            loop_config: None,

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
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
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
            loop_config: None,

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
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
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
        assert_eq!(
            ffn_dim % cfg.group_size,
            0,
            "ffn_dim {} not aligned to {}",
            ffn_dim,
            cfg.group_size
        );
        assert!(
            (5000..=6000).contains(&ffn_dim),
            "ffn_dim {} out of expected range",
            ffn_dim
        );
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
