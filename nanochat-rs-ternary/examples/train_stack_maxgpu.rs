//! Train nano-125M on The Stack Python - MAX GPU UTILIZATION (24GB)
//!
//! Optimized for RTX 4090 24GB:
//! - nano-125M model (127M params)
//! - Large batch size for maximum throughput
//! - Real Python code dataset
//! - Target: Coherent code completion

use nanochat_train::{
    config::TrainConfig,
    data::{Dataset, TokenFileDataset},
    train::Trainer,
};
use candle_core::Device;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train_stack_maxgpu")]
#[command(about = "Train nano-125M on The Stack - MAX GPU! 🚀")]
struct Args {
    /// Path to tokenized Stack data
    #[arg(long, default_value = "data/stack_python_tokens.bin")]
    data: String,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints/nano-stack-maxgpu")]
    checkpoint_dir: String,

    /// Total training steps
    #[arg(long, default_value = "30000")]
    total_steps: usize,

    /// Warmup steps
    #[arg(long, default_value = "2000")]
    warmup_steps: usize,

    /// Log interval
    #[arg(long, default_value = "100")]
    log_interval: usize,

    /// Checkpoint interval
    #[arg(long, default_value = "2000")]
    checkpoint_interval: usize,

    /// Keep last N checkpoints
    #[arg(long, default_value = "3")]
    keep_last_checkpoints: usize,

    /// Device
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Batch size (tuned for 24GB GPU)
    #[arg(long, default_value = "16")]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value = "512")]
    seq_len: usize,

    /// Learning rate (Muon)
    #[arg(long, default_value = "0.02")]
    lr: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Training nano-125M on The Stack Python 💻");
    println!("  MAX GPU UTILIZATION (24GB) 🚀");
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
    println!("Loading The Stack Python dataset...");
    let dataset = TokenFileDataset::from_binary_file(
        std::path::Path::new(&args.data),
        args.seq_len,
    )?;
    println!("✓ Dataset loaded: {} samples\n", dataset.len());

    // Configuration - nano-125M with FULL vocab
    let mut config = TrainConfig::nano_125m();
    config.batch_size = args.batch_size;
    config.max_seq_len = args.seq_len;
    config.total_steps = args.total_steps;
    config.warmup_steps = args.warmup_steps;
    config.lr = args.lr as f64;

    println!("Model Configuration:");
    println!("  Model: nano-125M (127M params)");
    println!("  Device: {:?}", device);
    println!("  Dimension: {}", config.dim);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {}", config.n_heads);
    println!("  Batch size: {} (MAX GPU!)", args.batch_size);
    println!("  Sequence length: {}", args.seq_len);
    println!("  Effective batch: {} tokens/step", args.batch_size * args.seq_len);
    println!("  Total steps: {}", args.total_steps);
    println!("  Learning rate: {}\n", args.lr);

    // Memory estimate
    let model_params_gb = 127.0 * 4.0 / 1024.0;
    let batch_mem_gb = (args.batch_size * args.seq_len * 768 * 4 * 12) as f64 / 1024.0 / 1024.0 / 1024.0;
    println!("Estimated GPU Memory Usage:");
    println!("  Model + grads + optimizer: ~{:.1} GB", model_params_gb * 4.5);
    println!("  Activations (batch): ~{:.1} GB", batch_mem_gb);
    println!("  Total estimated: ~{:.1} GB / 24 GB\n", model_params_gb * 4.5 + batch_mem_gb);

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
    println!("     --checkpoint {}/step_30000 \\", args.checkpoint_dir);
    println!("     --output models/nano-stack.gguf\n");

    println!("2. Test code completion:");
    println!("   cargo run --release -p nanochat-serve -- \\");
    println!("     --model models/nano-stack.gguf \\");
    println!("     --mhc models/nano-stack.mhc \\");
    println!("     --tokenizer models/gpt2-tokenizer.json \\");
    println!("     --port 8084\n");

    println!("3. Try prompts like:");
    println!("   \"def fibonacci(n):\"");
    println!("   \"class BinaryTree:\"");
    println!("   \"import numpy as np\"\n");

    Ok(())
}
