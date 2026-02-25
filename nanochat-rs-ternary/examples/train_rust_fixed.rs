//! Train Nanochat Model on Real Rust Code with Entropy Regularization.

use candle_core::Device;
use clap::Parser;
use nanochat_train::{
    config::TrainConfig,
    data::{Dataset, TokenFileDataset},
    train::Trainer,
};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "train_rust_fixed")]
struct Args {
    #[arg(long, default_value = "data/rust_tokens.bin")]
    data: String,

    #[arg(long, default_value = "checkpoints/rust-fixed")]
    checkpoint_dir: String,

    #[arg(long, default_value = "5000")]
    total_steps: usize,

    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(long, default_value = "4")]
    batch_size: usize,

    #[arg(long, default_value = "256")]
    seq_len: usize,

    #[arg(long, default_value = "0.01")]
    entropy_weight: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Training Nanochat on Real Rust Code - FIXED 🦀");
    println!("  Entropy Regularization: {}", args.entropy_weight);
    println!("═══════════════════════════════════════════════════════════
");

    let device = if args.device == "cpu" { Device::Cpu } else { Device::new_cuda(0)? };

    println!("Loading dataset...");
    let dataset = TokenFileDataset::from_binary_file(std::path::Path::new(&args.data), args.seq_len)?;
    println!("✓ Dataset loaded: {} samples
", dataset.len());

    let mut config = TrainConfig::d20();
    config.batch_size = args.batch_size;
    config.max_seq_len = args.seq_len;
    config.total_steps = args.total_steps;
    config.entropy_weight = args.entropy_weight;
    config.warmup_steps = 500;
    config.lr = 0.001; // Slower LR for better stability

    println!("Model Configuration:");
    println!("  Model: d20 (~18M params)");
    println!("  Entropy Weight: {}", config.entropy_weight);
    println!("  Learning Rate: {}", config.lr);
    println!();

    let mut trainer = Trainer::new(config, device)?;
    
    let steps_per_epoch = dataset.len().div_ceil(args.batch_size);
    let epochs = args.total_steps.div_ceil(steps_per_epoch);

    trainer.train_loop(
        &dataset,
        epochs,
        10, // log interval
        Some(&args.checkpoint_dir),
        500, // checkpoint interval
        3, // keep last
    )?;

    println!("
✓ Training complete!");
    Ok(())
}
