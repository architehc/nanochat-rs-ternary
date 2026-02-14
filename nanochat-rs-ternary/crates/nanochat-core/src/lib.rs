//! Core types and utilities shared across nanochat crates.
//!
//! Provides:
//! - Centralized error types via thiserror
//! - Configuration management with TOML support
//! - Common traits and utilities

pub mod config;
pub mod error;

// Re-export commonly used types
pub use config::{AppConfig, ModelConfig, ServingConfig, TrainingConfig};
pub use error::{NanoChatError, Result};
