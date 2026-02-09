//! Simple training for nano-125M without distillation.
//!
//! Trains from scratch with synthetic data to demonstrate ternary quantization works.
//!
//! Usage:
//!   cargo run --release --example train_nano_simple -- \
//!     --total-steps 5000 \
//!     --checkpoint-dir checkpoints/nano-125m-simple

use nanochat_train::{
    config::TrainConfig,
    data::{Dataset, SyntheticDataset},
    train::Trainer,
};
use candle_core::Device;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train_nano_simple")]
#[command(about = "Train nano-125M from scratch (no distillation)")]
struct Args {
    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints/nano-125m-simple")]
    checkpoint_dir: String,

    /// Total training steps
    #[arg(long, default_value = "5000")]
    total_steps: usize,

    /// Warmup steps
    #[arg(long, default_value = "500")]
    warmup_steps: usize,

    /// Log interval
    #[arg(long, default_value = "100")]
    log_interval: usize,

    /// Checkpoint interval
    #[arg(long, default_value = "1000")]
    checkpoint_interval: usize,

    /// Keep last N checkpoints
    #[arg(long, default_value = "3")]
    keep_last_checkpoints: usize,

    /// Device (cpu, cuda:0, etc.)
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Batch size
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value = "512")]
    seq_len: usize,

    /// Learning rate (Muon for ternary weights)
    #[arg(long, default_value = "0.02")]
    lr: f32,

    /// Learning rate for other params (Lion)
    #[arg(long, default_value = "0.0001")]
    other_lr: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Training Nano-125M Model (GPU Accelerated)");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Parse device
    let device = if args.device.starts_with("cuda") {
        let gpu_id = args.device.strip_prefix("cuda:")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        match Device::new_cuda(gpu_id) {
            Ok(d) => {
                println!("✓ CUDA device initialized: GPU {}", gpu_id);
                d
            }
            Err(e) => {
                eprintln!("CUDA not available ({}), falling back to CPU", e);
                Device::Cpu
            }
        }
    } else {
        Device::Cpu
    };

    println!("Configuration:");
    println!("  Model: nano-125M (127M params)");
    println!("  Device: {:?}", device);
    println!("  Total steps: {}", args.total_steps);
    println!("  Batch size: {}", args.batch_size);
    println!("  Sequence length: {}", args.seq_len);
    println!("  Learning rate: {}", args.lr);
    println!("  Checkpoint dir: {}", args.checkpoint_dir);
    println!();

    // Create training config - use nano_125m for GPU training
    let mut config = TrainConfig::nano_125m();
    config.batch_size = args.batch_size;
    config.max_seq_len = args.seq_len;
    config.total_steps = args.total_steps;
    config.warmup_steps = args.warmup_steps;
    config.lr = args.lr as f64;

    println!("Model Architecture:");
    println!("  Dimension: {}", config.dim);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {}", config.n_heads);
    println!("  Estimated params: {}", format_count(config.param_count_estimate()));
    println!();

    // Create synthetic dataset
    println!("Creating synthetic code dataset...");
    let dataset = SyntheticDataset::new(
        config.vocab_size as u32,
        config.max_seq_len,
        args.total_steps * args.batch_size * 2,
        42,
    );
    println!("✓ Dataset created: {} samples", dataset.len());
    println!();

    // Start training
    println!("═══════════════════════════════════════════════════════════");
    println!("Starting training...");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Create trainer
    println!("Initializing trainer...");
    let mut trainer = Trainer::new(config, device)?;
    println!("✓ Trainer initialized");
    println!();

    // Calculate number of epochs from total_steps
    let steps_per_epoch = (dataset.len() + args.batch_size - 1) / args.batch_size;
    let epochs = (args.total_steps + steps_per_epoch - 1) / steps_per_epoch;
    println!("Training for {} epochs ({} steps per epoch)", epochs, steps_per_epoch);
    println!();

    trainer.train_loop(
        &dataset,
        epochs,
        args.log_interval,
        Some(&args.checkpoint_dir),
        args.checkpoint_interval,
        args.keep_last_checkpoints,
    )?;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("Training complete!");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Model saved to: {}", args.checkpoint_dir);
    println!();
    println!("Next steps:");
    println!("1. Export to GGUF for inference");
    println!("2. Start inference server");
    println!("3. Evaluate on HumanEval");

    Ok(())
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
