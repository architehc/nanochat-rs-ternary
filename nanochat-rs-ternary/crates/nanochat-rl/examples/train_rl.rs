//! Example: RL training with GRPO/GSPO and compiler feedback
//!
//! This demonstrates the complete reinforcement learning pipeline:
//! 1. Load base model from checkpoint
//! 2. Generate code samples for prompts
//! 3. Evaluate with compiler + AST analysis + optional Qwen3
//! 4. Compute rewards and update policy using GRPO
//!
//! Usage:
//!   cargo run --example train_rl -- --checkpoint checkpoints/rust-6hour/step_2000 --iterations 100

use anyhow::Result;
use clap::Parser;
use nanochat_rl::{RLConfig, RLTrainer};

#[derive(Parser, Debug)]
#[command(name = "train_rl")]
#[command(about = "Train Rust code generation model with RL")]
struct Args {
    /// Path to base model checkpoint
    #[arg(long, default_value = "checkpoints/rust-6hour/step_2000")]
    checkpoint: String,

    /// Number of RL iterations
    #[arg(long, default_value = "100")]
    iterations: usize,

    /// Number of code samples per prompt
    #[arg(long, default_value = "4")]
    n_samples: usize,

    /// Batch size (number of prompts per iteration)
    #[arg(long, default_value = "2")]
    batch_size: usize,

    /// Device for training
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Optional Qwen3 endpoint URL
    #[arg(long)]
    qwen_endpoint: Option<String>,

    /// Learning rate
    #[arg(long, default_value = "1e-5")]
    lr: f64,

    /// KL divergence coefficient
    #[arg(long, default_value = "0.1")]
    kl_coef: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Create RL configuration
    let mut config = RLConfig::default();
    #[allow(clippy::field_reassign_with_default)]
    {
        config.base_checkpoint = args.checkpoint;
        config.n_iterations = args.iterations;
        config.n_samples = args.n_samples;
        config.batch_size = args.batch_size;
        config.device = args.device;
        config.qwen_endpoint = args.qwen_endpoint;
        config.grpo.learning_rate = args.lr;
        config.grpo.kl_coef = args.kl_coef;
    }

    // Create trainer
    let mut trainer = RLTrainer::new(config)?;

    // Run training
    trainer.train().await?;

    Ok(())
}
