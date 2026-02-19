//! Centralized error types for nanochat.
//!
//! Uses thiserror for ergonomic error handling with context.

use thiserror::Error;

/// Main error type for nanochat operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum NanoChatError {
    /// Model file not found at specified path.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Invalid model configuration detected.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Training diverged (NaN or infinite loss).
    #[error("Training diverged at step {step}: loss={loss}")]
    TrainingDiverged { loss: f64, step: usize },

    /// Model collapse detected (entropy too low).
    #[error("Model collapse detected at step {step}: entropy={entropy} (threshold=5.0)")]
    ModelCollapse { entropy: f64, step: usize },

    /// CUDA out of memory error with context.
    #[error("CUDA out of memory: requested={requested_gb:.2}GB, available={available_gb:.2}GB")]
    CudaOutOfMemory {
        requested_gb: f64,
        available_gb: f64,
    },

    /// Gradient explosion detected.
    #[error("Gradient explosion at step {step}: grad_norm={grad_norm} (threshold=10.0)")]
    GradientExplosion { grad_norm: f64, step: usize },

    /// Checkpoint loading failed.
    #[error("Failed to load checkpoint from {path}: {reason}")]
    CheckpointLoadFailed { path: String, reason: String },

    /// Checkpoint saving failed.
    #[error("Failed to save checkpoint to {path}: {reason}")]
    CheckpointSaveFailed { path: String, reason: String },

    /// GGUF file parsing error.
    #[error("GGUF parse error: {0}")]
    GgufParseFailed(String),

    /// Tensor shape mismatch.
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// mHC doubly stochastic constraint violated.
    #[error("mHC matrix not doubly stochastic at layer {layer}: max_error={max_error}")]
    MhcNotDoublyStochastic { layer: usize, max_error: f64 },

    /// Quantization error.
    #[error("Quantization error: {0}")]
    QuantizationError(String),

    /// Tokenization error.
    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    /// IO error wrapper.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    /// TOML serialization/deserialization error.
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    /// Candle tensor library error.
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Generic error with context.
    #[error("{0}")]
    Other(String),
}

/// Convenient Result type alias.
pub type Result<T> = std::result::Result<T, NanoChatError>;

impl NanoChatError {
    /// Check if error is recoverable (can retry).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            NanoChatError::Io(_) | NanoChatError::CudaOutOfMemory { .. }
        )
    }

    /// Check if error indicates training should stop.
    pub fn should_stop_training(&self) -> bool {
        matches!(
            self,
            NanoChatError::TrainingDiverged { .. }
                | NanoChatError::ModelCollapse { .. }
                | NanoChatError::GradientExplosion { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NanoChatError::TrainingDiverged {
            loss: f64::NAN,
            step: 42,
        };
        assert!(err.to_string().contains("step 42"));
        assert!(err.should_stop_training());
    }

    #[test]
    fn test_model_collapse_detection() {
        let err = NanoChatError::ModelCollapse {
            entropy: 2.5,
            step: 100,
        };
        assert!(err.to_string().contains("entropy=2.5"));
        assert!(err.should_stop_training());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_cuda_oom() {
        let err = NanoChatError::CudaOutOfMemory {
            requested_gb: 12.5,
            available_gb: 8.0,
        };
        assert!(err.to_string().contains("12.50GB"));
        assert!(err.is_recoverable());
    }
}
