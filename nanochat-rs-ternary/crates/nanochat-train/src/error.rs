//! Custom error types for nanochat training
//!
//! Provides structured error handling with context, recoverability detection,
//! and user-friendly error messages.

use thiserror::Error;

/// Main error type for training operations
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum TrainError {
    /// Errors from the Candle tensor library
    #[error("Candle error: {0}")]
    Candle(String),

    /// Checkpoint save/load failures
    #[error("Checkpoint error at '{path}': {message}")]
    Checkpoint { message: String, path: String },

    /// Optimizer-specific errors
    #[error("Optimizer error: {0}")]
    Optimizer(String),

    /// Data loading failures
    #[error("Data loading error: {0}")]
    DataLoading(String),

    /// Configuration validation failures
    #[error("Configuration error: {0}")]
    Config(String),

    /// Out of memory errors with context
    #[error("Out of memory: {context} (requested ~{requested_gb:.2} GB)")]
    OutOfMemory {
        context: String,
        requested_gb: f64,
    },

    /// Training divergence detection
    #[error("Training diverged: loss={loss:.4} (threshold={threshold:.4})")]
    Divergence { loss: f64, threshold: f64 },

    /// I/O errors with path context
    #[error("IO error at '{path}': {message}")]
    Io { message: String, path: String },

    /// Async channel errors
    #[error("Channel error: {0}")]
    Channel(String),

    /// Worker thread panics
    #[error("Worker panic: {0}")]
    WorkerPanic(String),

    /// Timeout errors
    #[error("Operation timed out after {duration_secs}s: {context}")]
    Timeout {
        duration_secs: u64,
        context: String,
    },
}

/// Result type alias for training operations
pub type TrainResult<T> = std::result::Result<T, TrainError>;

impl TrainError {
    /// Check if error is recoverable (can retry operation)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            TrainError::DataLoading(_)
                | TrainError::Io { .. }
                | TrainError::Checkpoint { .. }
                | TrainError::Timeout { .. }
                | TrainError::Channel(_)
        )
    }

    /// Get suggested retry count for recoverable errors
    pub fn retry_count(&self) -> usize {
        match self {
            TrainError::DataLoading(_) => 3,
            TrainError::Io { .. } => 2,
            TrainError::Checkpoint { .. } => 1,
            TrainError::Timeout { .. } => 2,
            TrainError::Channel(_) => 3,
            _ => 0,
        }
    }

    /// Check if this is an OOM error
    pub fn is_oom(&self) -> bool {
        matches!(self, TrainError::OutOfMemory { .. })
    }

    /// Get the path associated with this error (if any)
    pub fn path(&self) -> Option<&str> {
        match self {
            TrainError::Checkpoint { path, .. } => Some(path),
            TrainError::Io { path, .. } => Some(path),
            _ => None,
        }
    }
}

impl From<candle_core::Error> for TrainError {
    fn from(err: candle_core::Error) -> Self {
        let err_str = err.to_string();
        
        // Detect OOM conditions from Candle errors
        if err_str.contains("out of memory")
            || err_str.contains("OOM")
            || err_str.contains("CUDA out of memory")
        {
            // Try to estimate requested memory from error message
            let requested_gb = 0.0; // Would need parsing
            return TrainError::OutOfMemory {
                context: err_str,
                requested_gb,
            };
        }

        TrainError::Candle(err_str)
    }
}

impl From<std::io::Error> for TrainError {
    fn from(err: std::io::Error) -> Self {
        TrainError::Io {
            message: err.to_string(),
            path: String::new(),
        }
    }
}

/// Helper trait for adding path context to IO operations
pub trait IoResultExt<T> {
    fn with_path<P: AsRef<std::path::Path>>(self, path: P) -> TrainResult<T>;
}

impl<T> IoResultExt<T> for std::io::Result<T> {
    fn with_path<P: AsRef<std::path::Path>>(self, path: P) -> TrainResult<T> {
        self.map_err(|e| TrainError::Io {
            message: e.to_string(),
            path: path.as_ref().display().to_string(),
        })
    }
}

/// Helper for creating checkpoint errors
pub fn checkpoint_error<P: AsRef<std::path::Path>>(message: impl Into<String>, path: P) -> TrainError {
    TrainError::Checkpoint {
        message: message.into(),
        path: path.as_ref().display().to_string(),
    }
}

/// Helper for creating OOM errors
pub fn oom_error(context: impl Into<String>, requested_gb: f64) -> TrainError {
    TrainError::OutOfMemory {
        context: context.into(),
        requested_gb,
    }
}

/// Helper for creating config errors
pub fn config_error(message: impl Into<String>) -> TrainError {
    TrainError::Config(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recoverability() {
        assert!(TrainError::DataLoading("test".to_string()).is_recoverable());
        assert!(TrainError::Io {
            message: "test".to_string(),
            path: "/tmp".to_string(),
        }
        .is_recoverable());
        assert!(!TrainError::Divergence {
            loss: 100.0,
            threshold: 10.0,
        }
        .is_recoverable());
    }

    #[test]
    fn test_oom_detection() {
        let candle_err = candle_core::Error::Msg("CUDA out of memory".to_string());
        let train_err: TrainError = candle_err.into();
        assert!(train_err.is_oom());
    }

    #[test]
    fn test_retry_counts() {
        assert_eq!(
            TrainError::DataLoading("test".to_string()).retry_count(),
            3
        );
        assert_eq!(
            TrainError::Io {
                message: "test".to_string(),
                path: "/tmp".to_string(),
            }
            .retry_count(),
            2
        );
        assert_eq!(
            TrainError::Divergence {
                loss: 100.0,
                threshold: 10.0,
            }
            .retry_count(),
            0
        );
    }

    #[test]
    fn test_path_extraction() {
        let err = checkpoint_error("failed", "/tmp/checkpoint");
        assert_eq!(err.path(), Some("/tmp/checkpoint"));

        let io_err = TrainError::Io {
            message: "failed".to_string(),
            path: "/tmp/data".to_string(),
        };
        assert_eq!(io_err.path(), Some("/tmp/data"));

        let other_err = TrainError::Optimizer("failed".to_string());
        assert_eq!(other_err.path(), None);
    }

    #[test]
    fn test_io_with_path() {
        let result: std::io::Result<()> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        let train_result: TrainResult<()> = result.with_path("/tmp/missing.txt");
        
        match train_result {
            Err(TrainError::Io { path, .. }) => assert_eq!(path, "/tmp/missing.txt"),
            _ => panic!("Expected IO error with path"),
        }
    }
}
