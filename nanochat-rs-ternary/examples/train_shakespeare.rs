//! Train on Tiny Shakespeare to showcase real text generation.
//!
//! This demonstrates the full pipeline with real data.

use nanochat_train::{
    config::TrainConfig,
    data::{Dataset, TokenFileDataset},
    train::Trainer,
};
use candle_core::Device;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train_shakespeare")]
#[command(about = "Train on Shakespeare for text generation showcase 🎭")]
struct Args {
    /// Path to tokenized data
    #[arg(long, default_value = "data/shakespeare_tokens.bin")]
    data: String,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints/shakespeare")]
    checkpoint_dir: String,

    /// Total training steps
    #[arg(long, default_value = "5000")]
    total_steps: usize,

    /// Warmup steps
    #[arg(long, default_value = "500")]
    warmup_steps: usize,

    /// Log interval
    #[arg(long, default_value = "50")]
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
    #[arg(long, default_value = "256")]
    seq_len: usize,

    /// Learning rate
    #[arg(long, default_value = "0.02")]
    lr: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Shakespeare Training Showcase 🎭");
    println!("═══════════════════════════════════════════════════════════\n");

    // Device setup
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
                eprintln!("⚠ CUDA not available ({}), falling back to CPU", e);
                Device::Cpu
            }
        }
    } else {
        Device::Cpu
    };

    // Load dataset
    println!("Loading Shakespeare dataset...");
    let dataset = TokenFileDataset::from_binary_file(
        std::path::Path::new(&args.data),
        args.seq_len,
    )?;
    println!("✓ Dataset loaded: {} samples\n", dataset.len());

    // Configuration - use tiny model with full GPT-2 vocab
    let mut config = TrainConfig::tiny_cpu();
    config.vocab_size = 50257; // GPT-2 vocabulary size
    config.batch_size = args.batch_size;
    config.max_seq_len = args.seq_len;
    config.total_steps = args.total_steps;
    config.warmup_steps = args.warmup_steps;
    config.lr = args.lr as f64;

    println!("Model Configuration:");
    println!("  Model: tiny (3.7M params for fast showcase)");
    println!("  Device: {:?}", device);
    println!("  Dimension: {}", config.dim);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {}", config.n_heads);
    println!("  Batch size: {}", args.batch_size);
    println!("  Sequence length: {}", args.seq_len);
    println!("  Total steps: {}", args.total_steps);
    println!("  Learning rate: {}\n", args.lr);

    // Create trainer
    println!("═══════════════════════════════════════════════════════════");
    println!("Starting training...");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Initializing trainer...");
    let mut trainer = Trainer::new(config, device)?;
    println!("✓ Trainer initialized\n");

    // Calculate epochs
    let steps_per_epoch = (dataset.len() + args.batch_size - 1) / args.batch_size;
    let epochs = (args.total_steps + steps_per_epoch - 1) / steps_per_epoch;
    println!("Training for {} epochs ({} steps per epoch)\n", epochs, steps_per_epoch);

    trainer.train_loop(
        &dataset,
        epochs,
        args.log_interval,
        Some(&args.checkpoint_dir),
        args.checkpoint_interval,
        args.keep_last_checkpoints,
    )?;

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Training Complete! 🎉");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Next steps:");
    println!("1. Export to GGUF:");
    println!("   cargo run --release -p nanochat-train --example export_checkpoint -- \\");
    println!("     --checkpoint {}/step_5000 \\", args.checkpoint_dir);
    println!("     --output models/shakespeare.gguf\n");

    println!("2. Test generation:");
    println!("   cargo run --release -p nanochat-serve -- \\");
    println!("     --model models/shakespeare.gguf \\");
    println!("     --mhc models/shakespeare.mhc \\");
    println!("     --tokenizer models/gpt2-tokenizer.json \\");
    println!("     --port 8082\n");

    println!("3. Try prompts like:");
    println!("   \"To be or not to be\"");
    println!("   \"Romeo, Romeo, wherefore art thou\"");
    println!("   \"All the world's a stage\"\n");

    Ok(())
}
