//! Centralized configuration management with TOML support.
//!
//! Provides structured configs for model, training, and serving with
//! load/save capabilities.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{NanoChatError, Result};

/// Model architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden dimension size.
    pub dim: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of key-value heads (for GQA).
    pub n_kv_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// FFN intermediate dimension multiplier.
    pub ffn_mult: f32,
    /// Quantization group size.
    pub group_size: usize,
    /// Number of mHC streams (2 or 4).
    pub mhc_n_streams: usize,
    /// RoPE theta parameter.
    pub rope_theta: f32,
    /// Whether weights are tied (embedding == lm_head).
    pub weight_tied: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dim: 768,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 50257,
            max_seq_len: 512,
            ffn_mult: 2.6875,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            weight_tied: false,
        }
    }
}

impl ModelConfig {
    /// Create config for d20 model (560M params).
    pub fn d20() -> Self {
        Self {
            dim: 768,
            n_layers: 20,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 50257,
            max_seq_len: 1024,
            ffn_mult: 2.6875,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            weight_tied: true,
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(NanoChatError::InvalidConfig("dim must be > 0".into()));
        }
        if self.n_heads == 0 || !self.dim.is_multiple_of(self.n_heads) {
            return Err(NanoChatError::InvalidConfig(
                "dim must be divisible by n_heads".into(),
            ));
        }
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(NanoChatError::InvalidConfig(
                "n_heads must be divisible by n_kv_heads".into(),
            ));
        }
        if !matches!(self.mhc_n_streams, 2 | 4) {
            return Err(NanoChatError::InvalidConfig(
                "mhc_n_streams must be 2 or 4".into(),
            ));
        }
        Ok(())
    }
}

/// Training hyperparameters configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size per GPU.
    pub batch_size: usize,
    /// Sequence length for training.
    pub seq_len: usize,
    /// Total number of training steps.
    pub total_steps: usize,
    /// Warmup steps for learning rate.
    pub warmup_steps: usize,
    /// Base learning rate.
    pub learning_rate: f64,
    /// Muon optimizer learning rate (for 2D+ params).
    pub muon_lr: f64,
    /// Lion optimizer learning rate (for 1D params, mHC).
    pub lion_lr: f64,
    /// Entropy regularization weight.
    pub entropy_weight: f64,
    /// Gradient clipping threshold.
    pub grad_clip: f64,
    /// Steps between checkpoints.
    pub checkpoint_interval: usize,
    /// Weight decay.
    pub weight_decay: f64,
    /// Gradient accumulation steps.
    pub grad_accum_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 256,
            total_steps: 10000,
            warmup_steps: 1000,
            learning_rate: 0.001,
            muon_lr: 0.02,
            lion_lr: 1e-4,
            entropy_weight: 0.01,
            grad_clip: 5.0,
            checkpoint_interval: 1000,
            weight_decay: 0.1,
            grad_accum_steps: 1,
        }
    }
}

impl TrainingConfig {
    /// Validate training configuration.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(NanoChatError::InvalidConfig(
                "batch_size must be > 0".into(),
            ));
        }
        if self.seq_len == 0 {
            return Err(NanoChatError::InvalidConfig("seq_len must be > 0".into()));
        }
        if self.total_steps == 0 {
            return Err(NanoChatError::InvalidConfig(
                "total_steps must be > 0".into(),
            ));
        }
        if self.grad_clip <= 0.0 {
            return Err(NanoChatError::InvalidConfig(
                "grad_clip must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Serving configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingConfig {
    /// Server host address.
    pub host: String,
    /// Server port.
    pub port: u16,
    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Enable Prometheus metrics endpoint.
    pub enable_metrics: bool,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            max_concurrent_requests: 100,
            timeout_secs: 30,
            enable_metrics: true,
        }
    }
}

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Model architecture config.
    pub model: ModelConfig,
    /// Training hyperparameters.
    pub training: TrainingConfig,
    /// Serving configuration.
    pub serving: ServingConfig,
    /// Logging level (debug, info, warn, error).
    pub log_level: String,
    /// Path to training data.
    pub data_path: String,
    /// Checkpoint directory.
    pub checkpoint_dir: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            serving: ServingConfig::default(),
            log_level: "info".into(),
            data_path: "./data".into(),
            checkpoint_dir: "./checkpoints".into(),
        }
    }
}

impl AppConfig {
    /// Load configuration from TOML file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            NanoChatError::Other(format!(
                "Failed to read config file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to TOML file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.validate()?;
        let content = toml::to_string_pretty(self)
            .map_err(|e| NanoChatError::Other(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(path.as_ref(), content)?;
        Ok(())
    }

    /// Validate all sub-configs.
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.training.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs_are_valid() {
        ModelConfig::default().validate().unwrap();
        TrainingConfig::default().validate().unwrap();
        AppConfig::default().validate().unwrap();
    }

    #[test]
    fn test_invalid_dim_division() {
        let cfg = ModelConfig {
            dim: 100,
            n_heads: 7, // 100 not divisible by 7
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_roundtrip() {
        let cfg = AppConfig::default();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        cfg.save(tmp.path()).unwrap();
        let loaded = AppConfig::from_file(tmp.path()).unwrap();
        assert_eq!(cfg.model.dim, loaded.model.dim);
        assert_eq!(cfg.training.total_steps, loaded.training.total_steps);
    }
}
