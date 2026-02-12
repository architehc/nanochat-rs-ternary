//! Train nano-125M on Real Rust Code - MAX GPU UTILIZATION (24GB)
//!
//! Optimized for RTX 4090 24GB:
//! - nano-125M model (127M params)
//! - Large batch size for maximum throughput
//! - Real Rust code dataset (Tokio, Serde, Clap)
//! - Target: Coherent Rust code completion
//!
//! Dataset: 4.2M tokens from production Rust codebases
//! Expected training time: 3-7 hours for 30K steps

use nanochat_train::{
    config::TrainConfig,
    data::{Dataset, TokenFileDataset},
    train::Trainer,
};
use candle_core::Device;
use clap::Parser;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "train_rust_maxgpu")]
#[command(about = "Train nano-125M on Real Rust Code - MAX GPU! 🦀")]
struct Args {
    /// Path to tokenized Rust data
    #[arg(long, default_value = "data/rust_tokens.bin")]
    data: String,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints/rust-nano-125m")]
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

    /// Gradient clipping threshold
    #[arg(long, default_value = "1.0")]
    grad_clip: f32,

    /// Resume from latest checkpoint if available
    #[arg(long, default_value = "true")]
    resume: bool,
}

/// Find the latest checkpoint in a directory
fn find_latest_checkpoint(checkpoint_dir: &str) -> Option<(String, usize)> {
    let dir_path = Path::new(checkpoint_dir);
    if !dir_path.exists() {
        return None;
    }

    let mut checkpoints: Vec<(String, usize)> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            if let Ok(file_name) = entry.file_name().into_string() {
                if file_name.starts_with("step_") {
                    if let Ok(step) = file_name.strip_prefix("step_").unwrap().parse::<usize>() {
                        let checkpoint_path = entry.path();
                        // Verify checkpoint has required files
                        if checkpoint_path.join("model.safetensors").exists()
                            && checkpoint_path.join("meta.json").exists() {
                            checkpoints.push((checkpoint_path.to_string_lossy().to_string(), step));
                        }
                    }
                }
            }
        }
    }

    if checkpoints.is_empty() {
        return None;
    }

    // Return the checkpoint with the highest step number
    checkpoints.sort_by_key(|(_, step)| *step);
    checkpoints.last().cloned()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Training nano-125M on Real Rust Code 🦀");
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
    println!("Loading Rust code dataset...");
    let dataset = TokenFileDataset::from_binary_file(
        std::path::Path::new(&args.data),
        args.seq_len,
    )?;
    println!("✓ Dataset loaded: {} samples\n", dataset.len());

    // Configuration - d20 (20M params) - fits in 24GB GPU
    let mut config = TrainConfig::d20();
    config.batch_size = args.batch_size;
    config.max_seq_len = args.seq_len;
    config.total_steps = args.total_steps;
    config.warmup_steps = args.warmup_steps;
    config.lr = args.lr as f64;
    config.grad_clip = args.grad_clip as f64;

    // Calculate actual param count
    let param_count = {
        let embed = config.vocab_size * config.dim;
        let ffn_dim = (config.dim as f32 * config.ffn_mult) as usize;
        let per_layer = 4 * config.dim * config.dim  // Q,K,V,O
            + 3 * config.dim * ffn_dim;  // gate, up, down
        let total = embed + config.n_layers * per_layer + config.dim;  // +final norm
        total / 1_000_000  // in millions
    };

    println!("Model Configuration:");
    println!("  Model: d20 (~{}M params)", param_count);
    println!("  Device: {:?}", device);
    println!("  Dimension: {}", config.dim);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {}", config.n_heads);
    println!("  Batch size: {} (MAX GPU!)", args.batch_size);
    println!("  Sequence length: {}", args.seq_len);
    println!("  Effective batch: {} tokens/step", args.batch_size * args.seq_len);
    println!("  Total steps: {}", args.total_steps);
    println!("  Learning rate: {}", args.lr);
    println!("  Gradient clip: {}\n", args.grad_clip);

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

    // Check for existing checkpoint to resume from
    let resume_from_step = if args.resume {
        if let Some((checkpoint_path, step)) = find_latest_checkpoint(&args.checkpoint_dir) {
            println!("✓ Found checkpoint: {} (step {})", checkpoint_path, step);
            println!("  Resuming training from step {}\n", step);
            Some((checkpoint_path, step))
        } else {
            println!("No existing checkpoint found, starting from scratch\n");
            None
        }
    } else {
        println!("Resume disabled, starting from scratch\n");
        None
    };

    println!("Initializing trainer...");
    let (mut trainer, resumed) = if let Some((checkpoint_path, resume_step)) = resume_from_step {
        // Load from checkpoint
        let mut t = Trainer::from_checkpoint(&checkpoint_path, device)?;
        t.global_step = resume_step;
        println!("✓ Trainer loaded from checkpoint (step {})\n", resume_step);
        (t, true)
    } else {
        (Trainer::new(config, device)?, false)
    };

    if !resumed {
        println!("✓ Trainer initialized\n");
    }

    // Calculate epochs
    let steps_per_epoch = (dataset.len() + args.batch_size - 1) / args.batch_size;
    let epochs = (args.total_steps + steps_per_epoch - 1) / steps_per_epoch;
    println!("Training for {} epochs ({} steps per epoch)\n", epochs, steps_per_epoch);
    println!("Expected time: 3-7 hours for {}K steps\n", args.total_steps / 1000);

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
    println!("     --output models/rust-nano-125m.gguf\n");

    println!("2. Test Rust code completion:");
    println!("   cargo run --release -p nanochat-serve -- \\");
    println!("     --model models/rust-nano-125m.gguf \\");
    println!("     --mhc models/rust-nano-125m.mhc \\");
    println!("     --tokenizer models/gpt2-tokenizer.json \\");
    println!("     --port 8085\n");

    println!("3. Try prompts like:");
    println!("   \"fn fibonacci(n: usize) -> usize {{\"");
    println!("   \"struct BinaryTree<T> {{\"");
    println!("   \"use tokio::{{\"");
    println!("   \"impl Iterator for\"");
    println!("   \"async fn fetch_data() -> Result<\"");
    println!("   \"#[derive(\"");
    println!("   \"trait Parser {{\"");
    println!("   \"fn main() {{\"");
    println!();

    println!("4. Validate with rustc:");
    println!("   Generate code, save to file, run: rustc --check generated.rs\n");

    Ok(())
}
