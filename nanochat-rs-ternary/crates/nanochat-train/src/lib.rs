//! # Nanochat Training Library
//!
//! High-performance training infrastructure for ternary-quantized language models.
//!
//! ## Architecture Overview
//!
//! - **Optimizers**: Muon (orthogonalized momentum) + Lion hybrid with optional
//!   8-bit quantization and GaLore2 low-rank projection
//! - **Memory Efficiency**: Gradient checkpointing, tensor pooling, and efficient
//!   optimizer state management
//! - **Data Loading**: Async prefetching for 90%+ GPU utilization
//! - **Training Features**: Multi-Token Prediction, Collider token filtering, FP4
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use nanochat_train::{config::TrainConfig, train::Trainer};
//! use candle_core::Device;
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = TrainConfig::nano_125m();
//! let mut trainer = Trainer::new(config, Device::Cuda(0))?;
//!
//! // Train for one epoch
//! // let avg_loss = trainer.train_epoch(&dataset, 0)?;
//! # Ok(())
//! # }
//! ```

pub mod attention;
pub mod block;
pub mod checkpoint;
pub mod collider;
pub mod config;
pub mod data;
pub mod distill;
pub mod error;
pub mod export;
pub mod ffn;
pub mod fp4;
pub mod layers;
pub mod logging;
pub mod loop_block;
pub mod memory_pool;
pub mod mhc;
pub mod model;
pub mod mtp;
pub mod optim;
pub mod quantize;
pub mod sensitivity;
pub mod train;

#[cfg(feature = "cuda")]
pub mod cuda_ops;
