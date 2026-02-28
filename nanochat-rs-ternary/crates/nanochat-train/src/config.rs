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

fn default_fp4_stochastic_rounding() -> bool {
    true
}

fn default_label_smooth_eps() -> f64 {
    0.1 // 10% label smoothing
}

fn default_entropy_weight() -> f64 {
    0.0 // disabled by default
}

fn default_engram_d_mem() -> usize {
    256
}

fn default_engram_n_heads() -> usize {
    4
}

fn default_engram_table_size() -> usize {
    50021 // prime, reduces hash collisions
}

fn default_engram_conv_kernel() -> usize {
    4
}

fn default_engram_lr_mult() -> f64 {
    5.0 // paper spec: 5× backbone lr for table params
}

fn default_wavefield_field_size() -> usize {
    1024
}

fn default_wavefield_ratio() -> f32 {
    1.0
}

fn default_wavefield_physics_lr() -> f64 {
    5e-4 // Higher than mhc_lr (1e-4) — physics params need faster learning
}

fn default_wavefield_warmup_delay() -> usize {
    0 // No delay by default; set >0 to freeze physics params during warmup
}

fn default_true() -> bool {
    true
}

/// Adaptive loop control for inference (LoopLM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLoopConfig {
    /// Minimum number of loop iterations
    pub min_loops: usize,
    /// Maximum number of loop iterations
    pub max_loops: usize,
    /// Perplexity threshold for early stopping (legacy, pre-exit-gate)
    pub perplexity_threshold: f32,
    /// Exit gate threshold: if exit_prob > this, stop looping (after min_loops).
    /// Default 0.5. Set to 1.0 to disable exit gate.
    #[serde(default = "default_exit_threshold")]
    pub exit_threshold: f32,
}

fn default_exit_threshold() -> f32 {
    0.5
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

    // Loss regularization
    /// Label smoothing epsilon: smooth_targets = (1-eps)*one_hot + eps/V.
    /// Default 0.1 (10%). Set to 0.0 to disable.
    #[serde(default = "default_label_smooth_eps")]
    pub label_smooth_eps: f64,
    /// Explicit entropy regularization weight: loss -= entropy_weight * H(p).
    /// Encourages diverse predictions, prevents logit collapse in ternary models.
    /// Default 0.0 (disabled). Typical values: 0.01-0.1.
    #[serde(default = "default_entropy_weight")]
    pub entropy_weight: f64,

    // FP4 mixed precision (Blackwell-oriented)
    /// Enable software-simulated FP4 activation quantization in training loop.
    #[serde(default)]
    pub use_fp4: bool,
    /// Use stochastic rounding behavior in FP4 module.
    #[serde(default = "default_fp4_stochastic_rounding")]
    pub fp4_stochastic_rounding: bool,

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

    // Wave Field Attention
    /// Enable wave field attention (O(n log n) FFT-based physics attention)
    #[serde(default)]
    pub use_wave_field: bool,
    /// Wave field spatial resolution (default: 1024)
    #[serde(default = "default_wavefield_field_size")]
    pub wavefield_field_size: usize,
    /// Number of wave field heads (0 = use n_heads)
    #[serde(default)]
    pub wavefield_n_heads: usize,
    /// Enable inter-head coupling matrix
    #[serde(default = "default_true")]
    pub wavefield_head_coupling: bool,
    /// Fraction of layers using wave field (1.0 = all layers when enabled)
    #[serde(default = "default_wavefield_ratio")]
    pub wavefield_ratio: f32,
    /// Convolution mode for wave field: "fft", "fwht", or "haar" (default: "fft")
    #[serde(default)]
    pub wavefield_convolve_mode: Option<String>,
    /// Number of Haar decomposition levels (only for haar mode, None = max)
    #[serde(default)]
    pub wavefield_haar_levels: Option<usize>,

    /// Separate learning rate for wavefield physics params (omega, alpha, phi, haar_coeffs).
    /// Higher than mhc_lr because these params start random and need to find spectral profile quickly.
    #[serde(default = "default_wavefield_physics_lr")]
    pub wavefield_physics_lr: f64,

    /// Delay (in steps) before wavefield physics params start learning.
    /// During this period, physics params are frozen (LR=0) while scatter/gate/out projections
    /// stabilize. Prevents kernel chasing a moving target during early training.
    #[serde(default = "default_wavefield_warmup_delay")]
    pub wavefield_warmup_delay: usize,

    /// Use direct Haar-domain kernel coefficients instead of time-domain damped oscillator.
    /// Only applies when wavefield_convolve_mode="haar". Gives the model independent control
    /// over every wavelet scale and position. Recommended for Haar training.
    /// Also enables per-element gate (n_heads*head_dim) vs legacy per-head gate (n_heads).
    /// Default false for backward compat with old checkpoints.
    #[serde(default)]
    pub wavefield_haar_direct: bool,

    // Engram: O(1) N-gram lookup memory (arXiv 2601.07372)
    /// Enable Engram N-gram memory tables
    #[serde(default)]
    pub use_engram: bool,
    /// Engram embedding dimension per head
    #[serde(default = "default_engram_d_mem")]
    pub engram_d_mem: usize,
    /// N-gram orders (e.g., [2, 3] for bigrams + trigrams)
    #[serde(default)]
    pub engram_n_gram_orders: Vec<usize>,
    /// Number of hash heads per order
    #[serde(default = "default_engram_n_heads")]
    pub engram_n_heads: usize,
    /// Hash table size per (order, head) — prime recommended
    #[serde(default = "default_engram_table_size")]
    pub engram_table_size: usize,
    /// Which unique layer indices get Engram (e.g., [0, 4] for first and last)
    #[serde(default)]
    pub engram_layers: Vec<usize>,
    /// Depthwise causal conv kernel size
    #[serde(default = "default_engram_conv_kernel")]
    pub engram_conv_kernel: usize,
    /// Learning rate multiplier for engram table params (paper: 5×)
    #[serde(default = "default_engram_lr_mult")]
    pub engram_lr_mult: f64,
}

impl TrainConfig {
    /// Validate configuration and return list of warnings/errors
    ///
    /// # Returns
    /// * `Ok(())` if configuration is valid
    /// * `Err(errors)` if configuration has critical errors
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Critical errors
        if self.dim == 0 {
            errors.push("dim must be greater than 0".to_string());
        }
        
        if self.n_layers == 0 {
            errors.push("n_layers must be greater than 0".to_string());
        }
        
        if self.n_heads == 0 {
            errors.push("n_heads must be greater than 0".to_string());
        }

        if !self.dim.is_multiple_of(self.n_heads) {
            errors.push(format!(
                "dim ({}) must be divisible by n_heads ({})",
                self.dim, self.n_heads
            ));
        }

        if self.n_kv_heads > 0 && !self.n_heads.is_multiple_of(self.n_kv_heads) {
            errors.push(format!(
                "n_heads ({}) must be divisible by n_kv_heads ({}) for GQA",
                self.n_heads, self.n_kv_heads
            ));
        }

        if !self.dim.is_multiple_of(self.group_size) {
            errors.push(format!(
                "dim ({}) must be divisible by group_size ({})",
                self.dim, self.group_size
            ));
        }

        if self.warmup_steps >= self.total_steps {
            errors.push(format!(
                "warmup_steps ({}) must be < total_steps ({})",
                self.warmup_steps, self.total_steps
            ));
        }

        if self.batch_size == 0 {
            errors.push("batch_size must be greater than 0".to_string());
        }

        if self.grad_accum_steps == 0 {
            errors.push("grad_accum_steps must be greater than 0".to_string());
        }

        if self.lr <= 0.0 {
            errors.push(format!("learning rate ({}) must be positive", self.lr));
        }

        // Warnings
        if self.batch_size * self.max_seq_len > 65536 {
            warnings.push(format!(
                "Large batch * seq_len ({}), may cause OOM",
                self.batch_size * self.max_seq_len
            ));
        }

        if self.use_galore && self.galore_rank > self.dim / 2 {
            warnings.push(format!(
                "GaLore rank ({}) > dim/2 ({}), memory savings minimal",
                self.galore_rank, self.dim / 2
            ));
        }

        if self.use_galore && self.galore_update_freq == 0 {
            errors.push("galore_update_freq must be greater than 0 when use_galore is true".to_string());
        }

        if self.use_async_loader && self.async_n_workers == 0 {
            warnings.push("async_n_workers is 0, async loader will not provide benefits".to_string());
        }

        if self.use_mtp && self.mtp_n_tokens == 0 {
            errors.push("mtp_n_tokens must be greater than 0 when use_mtp is true".to_string());
        }

        if self.use_collider && (self.collider_threshold < 0.0 || self.collider_threshold > 1.0) {
            errors.push(format!(
                "collider_threshold ({}) must be in [0, 1]",
                self.collider_threshold
            ));
        }

        if self.use_collider && (self.collider_sparsity < 0.0 || self.collider_sparsity > 1.0) {
            errors.push(format!(
                "collider_sparsity ({}) must be in [0, 1]",
                self.collider_sparsity
            ));
        }

        // Wave field validation (uses is_wavefield_layer below)
        if self.use_wave_field {
            if self.wavefield_field_size == 0 {
                errors.push("wavefield_field_size must be > 0".to_string());
            }
            if !(0.0..=1.0).contains(&self.wavefield_ratio) {
                errors.push(format!(
                    "wavefield_ratio ({}) must be in [0, 1]",
                    self.wavefield_ratio
                ));
            }
            let wf_heads = if self.wavefield_n_heads == 0 { self.n_heads } else { self.wavefield_n_heads };
            if self.dim > 0 && wf_heads > 0 && self.dim % wf_heads != 0 {
                errors.push(format!(
                    "dim ({}) must be divisible by wavefield n_heads ({})",
                    self.dim, wf_heads
                ));
            }
            // FWHT and Haar require power-of-2 field_size
            if let Some(ref mode) = self.wavefield_convolve_mode {
                match mode.as_str() {
                    "fwht" | "haar" => {
                        if !self.wavefield_field_size.is_power_of_two() {
                            errors.push(format!(
                                "wavefield_field_size ({}) must be power of 2 for {} mode",
                                self.wavefield_field_size, mode
                            ));
                        }
                    }
                    "fft" => {} // FFT pads to next power of 2 internally
                    other => {
                        errors.push(format!(
                            "unknown wavefield_convolve_mode '{}'; expected 'fft', 'fwht', or 'haar'",
                            other
                        ));
                    }
                }
            }
            // Validate haar_levels if specified
            if let Some(levels) = self.wavefield_haar_levels {
                let max_levels = (self.wavefield_field_size as f64).log2() as usize;
                if levels == 0 || levels > max_levels {
                    errors.push(format!(
                        "wavefield_haar_levels ({}) must be in [1, {}] for field_size {}",
                        levels, max_levels, self.wavefield_field_size
                    ));
                }
            }
        }

        // Log warnings
        for warning in &warnings {
            tracing::warn!("Config warning: {}", warning);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Determine whether a layer should use wave field attention.
    ///
    /// Uses the same interleaving logic as inference: wavefield_ratio controls
    /// the fraction of layers that become wave field, evenly distributed.
    pub fn is_wavefield_layer(&self, layer_idx: usize) -> bool {
        if !self.use_wave_field {
            return false;
        }
        let r = self.wavefield_ratio;
        if r <= 0.0 {
            return false;
        }
        if r >= 1.0 {
            return true;
        }
        if self.n_layers == 0 {
            return false;
        }
        let n_wavefield = ((self.n_layers as f32) * r).round() as usize;
        if n_wavefield == 0 {
            return false;
        }
        // Integer-safe interleaving: place wave field layers near bin centers
        for k in 0..n_wavefield {
            let num = (2u128 * k as u128 + 1) * self.n_layers as u128;
            let den = 2u128 * n_wavefield as u128;
            let wf_idx = usize::try_from(num / den).unwrap_or(usize::MAX);
            let wf_idx = wf_idx.min(self.n_layers - 1);
            if wf_idx == layer_idx {
                return true;
            }
        }
        false
    }

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
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
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
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// d20 with MTP enabled (for testing on 24GB GPUs).
    pub fn d20_mtp() -> Self {
        let mut cfg = Self::d20();
        cfg.use_mtp = true;
        cfg.mtp_n_tokens = 3;
        cfg.mtp_weight = 0.2;
        cfg
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
            batch_size: 8,       // 8 × 2048 = 16K tokens/batch
            grad_accum_steps: 4, // Effective: 65K tokens
            warmup_steps: 2000,
            total_steps: 50_000, // 3.25B tokens total
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 3,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),

            // E2 Advanced (enabled for full E3 profile)
            use_8bit_optim: true,
            use_galore: true,
            galore_rank: 256,
            galore_update_freq: 200,

            // E3 P0: Multi-Token Prediction (ENABLED)
            use_mtp: true,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,

            // E3 P1: Collider Token Filtering (sparse token compaction enabled)
            use_collider: true,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,

            // E3 P0: Async Data Loader (ENABLED)
            use_async_loader: true,
            async_n_workers: 6,
            async_prefetch_size: 12,
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,

            // Distillation (disabled)
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
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
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
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
                    exit_threshold: 0.5,
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
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None, // Can be set to teacher model path
            distill_kl_weight: 1.0,
            loop_scale_penalty: 0.1, // Annealed during training
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
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
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
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
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// Large model (~3B params).
    pub fn medium_3b() -> Self {
        Self {
            dim: 2048,
            n_layers: 28,
            n_heads: 32,
            n_kv_heads: 8,
            ffn_mult: 3.5,
            vocab_size: 50257,
            max_seq_len: 4096,
            group_size: 128,
            mhc_n_streams: 4,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,

            lr: 0.01,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 16,
            warmup_steps: 5000,
            total_steps: 200_000,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 5,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
            use_8bit_optim: true,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: true,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: true,
            async_n_workers: 8,
            async_prefetch_size: 16,
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// Extra-large model (~7B params) for multi-GPU training.
    pub fn large_7b() -> Self {
        Self {
            dim: 4096,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            ffn_mult: 2.6667,
            vocab_size: 128000,
            max_seq_len: 4096,
            group_size: 128,
            mhc_n_streams: 4,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,

            lr: 0.008,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 1,
            grad_accum_steps: 32,
            warmup_steps: 8000,
            total_steps: 300_000,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 5,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
            use_8bit_optim: true,
            use_galore: true,
            galore_rank: 512,
            galore_update_freq: 200,
            use_mtp: true,
            mtp_n_tokens: 4,
            mtp_weight: 0.2,
            use_collider: true,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: true,
            async_n_workers: 8,
            async_prefetch_size: 16,
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// 7B model tuned for a 6-day single-GPU training run (~1500 optimizer steps).
    ///
    /// Same architecture as `large_7b()` but with schedule/hyperparams adjusted for
    /// limited compute: fewer total steps, shorter warmup, gentler learning rate.
    /// All E3 features remain enabled (8-bit Muon, GaLore-512, MTP-4, Collider-35%).
    pub fn large_7b_6day() -> Self {
        let mut cfg = Self::large_7b();
        cfg.total_steps = 1_500;
        cfg.warmup_steps = 150;       // 10% of total (was 8000 for 300K run)
        cfg.decay_start_frac = 0.75;  // Start decay at step 1125
        cfg.lr = 0.004;               // Halved from 0.008 — fewer steps need gentler LR
        cfg
    }

    /// E3 profile with FP4 path enabled for Blackwell-oriented experimentation.
    pub fn d20_e3_fp4() -> Self {
        let mut cfg = Self::d20_e3_full();
        cfg.use_fp4 = true;
        cfg.fp4_stochastic_rounding = true;
        cfg
    }

    /// d20 with all wave field attention layers (for smoke testing).
    pub fn d20_wavefield() -> Self {
        let mut cfg = Self::d20();
        cfg.use_wave_field = true;
        cfg.wavefield_field_size = 256;
        cfg.wavefield_n_heads = 0; // use n_heads
        cfg.wavefield_head_coupling = true;
        cfg.wavefield_ratio = 1.0; // all layers wave field
        cfg
    }

    /// 125M param variant with all wave field attention.
    pub fn nano_125m_wavefield() -> Self {
        let mut cfg = Self::nano_125m();
        cfg.use_wave_field = true;
        cfg.wavefield_field_size = 512;
        cfg.wavefield_n_heads = 0;
        cfg.wavefield_head_coupling = true;
        cfg.wavefield_ratio = 1.0;
        cfg
    }

    /// d20 with wave field attention using FWHT (integer-only) convolution.
    pub fn d20_wavefield_fwht() -> Self {
        let mut cfg = Self::d20_wavefield();
        cfg.wavefield_convolve_mode = Some("fwht".to_string());
        cfg
    }

    /// 125M with wave field attention using FWHT (XOR convolution, integer-only).
    pub fn nano_125m_wavefield_fwht() -> Self {
        let mut cfg = Self::nano_125m_wavefield();
        cfg.wavefield_convolve_mode = Some("fwht".to_string());
        cfg
    }

    /// 125M with wave field attention using Haar (wavelet-basis scaling, integer-only).
    /// Uses direct Haar-domain coefficients for maximum expressiveness.
    pub fn nano_125m_wavefield_haar() -> Self {
        let mut cfg = Self::nano_125m_wavefield();
        cfg.wavefield_convolve_mode = Some("haar".to_string());
        // Partial decomposition — drops uninformative global DC levels
        let max_levels = (cfg.wavefield_field_size as f64).log2() as usize;
        cfg.wavefield_haar_levels = Some((max_levels - 2).max(1));
        cfg.wavefield_haar_direct = true;
        cfg.wavefield_physics_lr = 5e-4;
        cfg.wavefield_warmup_delay = 100;
        cfg
    }

    /// ~500M parameter hybrid: 50% Haar wave field + 50% standard attention (interleaved).
    /// Optimized for RTX 4090 (23GB). Uses GQA 3:1 to reduce KV memory.
    /// Haar layers provide bidirectional multi-scale context; standard attention handles causality.
    pub fn nano_500m_wave_haar() -> Self {
        Self {
            dim: 768,
            n_layers: 16,
            n_heads: 12,
            n_kv_heads: 4, // GQA 3:1
            ffn_mult: 2.6667, // ffn_dim = 2048, aligned to 128
            vocab_size: 4096, // BPE tokenizer trained on Rust corpus
            max_seq_len: 2048,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,

            lr: 0.015, // slightly lower for larger model
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1, // no grad_accum — Candle leaks memory with accum
            warmup_steps: 500,
            total_steps: 30_000, // ~1 epoch on 36M token dataset (batch=2, seq=512)
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 5,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),

            // Memory optimizations for 23GB VRAM
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,

            // Data efficiency — MTP gives 15-20% boost for free
            use_mtp: true,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,

            // Training speedups
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: true,
            async_n_workers: 4, // 4 workers for 8C/16T CPU
            async_prefetch_size: 8,

            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,

            // Wave field: 50% Haar + 50% standard attention (interleaved)
            use_wave_field: true,
            wavefield_field_size: 256, // power-of-2, reduced for VRAM
            wavefield_n_heads: 0, // use n_heads (12)
            wavefield_head_coupling: true,
            wavefield_ratio: 0.5, // 8 wavefield + 8 standard layers
            wavefield_convolve_mode: Some("haar".to_string()),
            wavefield_haar_levels: Some(6), // partial decomposition — drops uninformative global DC levels
            wavefield_physics_lr: 5e-4, // faster than mhc_lr for spectral profile learning
            wavefield_warmup_delay: 200, // freeze physics params for first 200 steps
            wavefield_haar_direct: true, // direct Haar-domain coefficients (not time-domain kernel)

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// Ablation: same as nano-500m-wave-haar but WITHOUT wavefield.
    /// Pure standard GQA attention for all 16 layers.
    /// Compare against nano_500m_wave_haar to measure wavefield contribution.
    pub fn nano_500m_baseline() -> Self {
        let mut cfg = Self::nano_500m_wave_haar();
        cfg.use_wave_field = false;
        cfg
    }

    /// nano-275m with Haar wavefield attention (dim=1024, 20 layers).
    /// ~277M params. Designed for dual-GPU A/B testing with nano_275m_baseline.
    /// Training: batch=2, seq=256 via CLI --seq-len 256 to fit 23GB VRAM.
    pub fn nano_275m_wave_haar() -> Self {
        Self {
            dim: 1024,
            n_layers: 20,
            n_heads: 16,     // 1024/16 = 64 head_dim
            n_kv_heads: 4,   // GQA 4:1
            ffn_mult: 3.0,   // ffn_dim = 3072, aligned to 128
            vocab_size: 4096,
            max_seq_len: 1024,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,

            lr: 0.012,       // scaled from 0.015 by sqrt(768/1024)
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 2000,
            total_steps: 134_764, // 2 epochs on ~34.5M tokens (batch=2, seq=256)
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 5,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),

            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,

            use_mtp: true,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,

            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: true,
            async_n_workers: 4,
            async_prefetch_size: 8,

            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,

            use_wave_field: true,
            wavefield_field_size: 256,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 0.5,  // 10 wavefield + 10 standard layers
            wavefield_convolve_mode: Some("haar".to_string()),
            wavefield_haar_levels: Some(6),
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 1000, // proportional to longer run
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// Ablation: same as nano-275m-wave-haar but WITHOUT wavefield.
    /// Pure standard GQA attention for all 20 layers.
    pub fn nano_275m_baseline() -> Self {
        let mut cfg = Self::nano_275m_wave_haar();
        cfg.use_wave_field = false;
        cfg
    }

    /// nano-275m with LoopLM + Wavefield + Engram.
    /// ~279M total params (79M GPU + 200M engram tables).
    /// Architecture: 2 local_before + 16-iteration shared loop + 2 local_after = 20 effective depth.
    /// Engram on layers 0 and 4 (first and last unique layers).
    pub fn nano_275m_wave_engram_loop() -> Self {
        Self {
            dim: 1024,
            n_layers: 5, // 2 local_before + 1 shared + 2 local_after
            n_heads: 16,
            n_kv_heads: 4,
            ffn_mult: 3.0,
            vocab_size: 4096,
            max_seq_len: 1024,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: Some(LoopConfig {
                local_before: 2,
                local_after: 2,
                loop_count: 16,
                adaptive_loop: None,
            }),

            lr: 0.012,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 2000,
            total_steps: 134_764,
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

            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,

            // Wavefield: layers 1 and 3 of 5 unique layers (50% ratio)
            use_wave_field: true,
            wavefield_field_size: 256,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 0.5,
            wavefield_convolve_mode: Some("haar".to_string()),
            wavefield_haar_levels: Some(6),
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 1000,
            wavefield_haar_direct: true,

            // Engram: N-gram memory on first and last unique layers
            use_engram: true,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![2, 3],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![0, 4], // first and last unique layers
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// nano-275m pure LoopLM (no wavefield, no engram).
    /// 5 unique layers (2 local_before + 1 shared + 2 local_after), 16 loop iterations = 20 effective depth.
    /// ~79M GPU params. Tests whether weight-sharing alone matches 20 unique layers.
    pub fn nano_275m_loop_only() -> Self {
        Self {
            dim: 1024,
            n_layers: 5,
            n_heads: 16,
            n_kv_heads: 4,
            ffn_mult: 3.0,
            vocab_size: 4096,
            max_seq_len: 1024,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: Some(LoopConfig {
                local_before: 2,
                local_after: 2,
                loop_count: 16,
                adaptive_loop: None,
            }),

            lr: 0.012,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 1500,
            total_steps: 20_000,
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
            use_async_loader: true,
            async_n_workers: 4,
            async_prefetch_size: 8,

            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,

            use_wave_field: false,
            wavefield_field_size: 256,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 0.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// nano-275m pure Engram (no wavefield, no loop).
    /// 20 standard GQA layers with N-gram memory on layers 0, 5, 10, 15, 19.
    /// ~277M GPU params + ~250M engram tables = ~527M total.
    pub fn nano_275m_engram_only() -> Self {
        Self {
            dim: 1024,
            n_layers: 20,
            n_heads: 16,
            n_kv_heads: 4,
            ffn_mult: 3.0,
            vocab_size: 4096,
            max_seq_len: 1024,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,

            lr: 0.012,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 1500,
            total_steps: 20_000,
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
            use_async_loader: true,
            async_n_workers: 4,
            async_prefetch_size: 8,

            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,

            use_wave_field: false,
            wavefield_field_size: 256,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 0.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,

            // Engram on 2 layers (first and last), smaller tables to fit 24GB VRAM
            use_engram: true,
            engram_d_mem: 128,
            engram_n_gram_orders: vec![2, 3],
            engram_n_heads: 4,
            engram_table_size: 10007,
            engram_layers: vec![0, 19],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// nano-275m Engram + MTP combo.
    /// Same as engram_only but with MTP enabled (3 future tokens).
    /// Total steps: 10000 for direct comparison with engram-v1 and baseline-v1.
    pub fn nano_275m_engram_mtp() -> Self {
        let mut cfg = Self::nano_275m_engram_only();
        cfg.use_mtp = true;
        cfg.mtp_n_tokens = 3;
        cfg.mtp_weight = 0.2;
        cfg.total_steps = 10_000;
        cfg
    }

    /// nano-275m Haar v3 — fresh run with matched hyperparams for fair comparison.
    /// 20 layers, 50% wavefield Haar, no MTP, total_steps=20000.
    pub fn nano_275m_haar_v3() -> Self {
        Self {
            dim: 1024,
            n_layers: 20,
            n_heads: 16,
            n_kv_heads: 4,
            ffn_mult: 3.0,
            vocab_size: 4096,
            max_seq_len: 1024,
            group_size: 128,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,

            lr: 0.012,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 1500,
            total_steps: 20_000,
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
            use_async_loader: true,
            async_n_workers: 4,
            async_prefetch_size: 8,

            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,

            use_wave_field: true,
            wavefield_field_size: 256,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 0.5,
            wavefield_convolve_mode: Some("haar".to_string()),
            wavefield_haar_levels: Some(6),
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 1000,
            wavefield_haar_direct: true,

            use_engram: false,
            engram_d_mem: 256,
            engram_n_gram_orders: vec![],
            engram_n_heads: 4,
            engram_table_size: 50021,
            engram_layers: vec![],
            engram_conv_kernel: 4,
            engram_lr_mult: 5.0,
        }
    }

    /// 125M hybrid: 50% standard attention + 50% wave field (interleaved).
    pub fn nano_125m_hybrid() -> Self {
        let mut cfg = Self::nano_125m();
        cfg.use_wave_field = true;
        cfg.wavefield_field_size = 512;
        cfg.wavefield_n_heads = 0;
        cfg.wavefield_head_coupling = true;
        cfg.wavefield_ratio = 0.5; // half the layers
        cfg
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

    #[test]
    fn test_medium_3b_param_estimate() {
        let cfg = TrainConfig::medium_3b();
        let params = cfg.param_count_estimate();
        assert!(params > 1_600_000_000, "Too few params: {}", params);
        assert!(params < 2_500_000_000, "Too many params: {}", params);
    }

    #[test]
    fn test_large_7b_6day_schedule() {
        let cfg = TrainConfig::large_7b_6day();
        assert_eq!(cfg.total_steps, 1_500);
        assert_eq!(cfg.warmup_steps, 150);
        assert!((cfg.decay_start_frac - 0.75).abs() < 1e-10);
        assert!((cfg.lr - 0.004).abs() < 1e-10);
        // Architecture matches large_7b
        assert_eq!(cfg.dim, 4096);
        assert_eq!(cfg.n_layers, 32);
        assert_eq!(cfg.vocab_size, 128000);
        // E3 features all enabled
        assert!(cfg.use_8bit_optim);
        assert!(cfg.use_galore);
        assert!(cfg.use_mtp);
        assert!(cfg.use_collider);
        assert!(cfg.use_async_loader);
        // Validation should pass
        cfg.validate().expect("large_7b_6day should validate");
    }

    #[test]
    fn test_large_7b_param_estimate() {
        let cfg = TrainConfig::large_7b();
        let params = cfg.param_count_estimate();
        assert!(params > 5_000_000_000, "Too few params: {}", params);
        assert!(params < 9_000_000_000, "Too many params: {}", params);
    }
}
