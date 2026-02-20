//! Train a nano-125M model with distillation from Qwen3 endpoint.
//!
//! Quick demo to show ternary quantization working with a small model.
//!
//! Usage:
//!   cargo run --release --example train_nano_125m -- \
//!     --teacher-endpoint https://crazyshit.ngrok.io \
//!     --total-steps 10000 \
//!     --checkpoint-dir checkpoints/nano-125m

use candle_core::Device;
use clap::Parser;
use nanochat_train::{
    checkpoint::save_checkpoint,
    config::TrainConfig,
    data::{Dataset, SyntheticDataset},
    distill::{DistillConfig, DistillationTrainer, TeacherMode},
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "train_nano_125m")]
#[command(about = "Train nano-125M model with distillation")]
struct Args {
    /// Remote teacher endpoint
    #[arg(long, default_value = "https://crazyshit.ngrok.io")]
    teacher_endpoint: String,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints/nano-125m")]
    checkpoint_dir: String,

    /// Total training steps
    #[arg(long, default_value = "10000")]
    total_steps: usize,

    /// Warmup steps
    #[arg(long, default_value = "1000")]
    warmup_steps: usize,

    /// Checkpoint interval
    #[arg(long, default_value = "1000")]
    checkpoint_interval: usize,

    /// Keep last N checkpoints
    #[arg(long, default_value = "3")]
    keep_last_checkpoints: usize,

    /// Use parallel training
    #[arg(long, default_value = "true")]
    parallel: bool,

    /// Number of micro-batches for parallel training
    #[arg(long, default_value = "8")]
    micro_batches: usize,

    /// Device (cpu, cuda:0, etc.)
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// KL divergence weight
    #[arg(long, default_value = "0.5")]
    kl_weight: f64,

    /// Temperature
    #[arg(long, default_value = "2.0")]
    temperature: f64,

    /// Batch size
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value = "512")]
    seq_len: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Training Nano-125M with Distillation");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Parse device
    let device = if args.device.starts_with("cuda") {
        #[cfg(feature = "cuda")]
        {
            let gpu_id = args
                .device
                .strip_prefix("cuda:")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            Device::new_cuda(gpu_id).map_err(|e| format!("Failed to create CUDA device: {}", e))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("CUDA not available, using CPU");
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    println!("Configuration:");
    println!("  Model: nano-125M (125M params, 768 dim, 12 layers)");
    println!("  Teacher: {}", args.teacher_endpoint);
    println!("  Device: {:?}", device);
    println!("  Total steps: {}", args.total_steps);
    println!("  Warmup steps: {}", args.warmup_steps);
    println!("  Batch size: {}", args.batch_size);
    println!("  Sequence length: {}", args.seq_len);
    println!("  Parallel: {}", args.parallel);
    if args.parallel {
        println!("  Micro-batches: {}", args.micro_batches);
    }
    println!("  Checkpoint dir: {}", args.checkpoint_dir);
    println!();

    // Create nano-125M config
    let mut train_config = TrainConfig::nano_125m();
    train_config.batch_size = args.batch_size;
    train_config.max_seq_len = args.seq_len;
    train_config.total_steps = args.total_steps;
    train_config.warmup_steps = args.warmup_steps;

    // Save values before moving train_config
    let dim = train_config.dim;
    let n_layers = train_config.n_layers;
    let n_heads = train_config.n_heads;
    let n_kv_heads = train_config.n_kv_heads;
    let vocab_size = train_config.vocab_size;
    let max_seq_len = train_config.max_seq_len;
    let param_count = train_config.param_count_estimate();

    println!("Model Architecture:");
    println!("  Dimension: {}", dim);
    println!("  Layers: {}", n_layers);
    println!("  Heads: {}", n_heads);
    println!("  KV Heads: {}", n_kv_heads);
    println!("  Vocab size: {}", vocab_size);
    println!("  Estimated params: {}", format_count(param_count));
    println!();

    // Create distillation config
    let distill_config = DistillConfig {
        train_config,
        teacher_mode: TeacherMode::Remote {
            endpoint: args.teacher_endpoint,
            api_key: None,
            timeout_secs: 60,
            max_concurrent: 8,
        },
        temperature: args.temperature,
        kl_weight: args.kl_weight,
        load_balance_weight: 0.01,
        router_aux_weight: 0.001,
        freeze_student_fp8: false,
        micro_batches: if args.parallel { args.micro_batches } else { 1 },
    };

    // Create trainer
    println!("Initializing trainer...");
    let mut trainer = DistillationTrainer::new(distill_config, device.clone())?;
    println!("✓ Trainer initialized");
    println!();

    // Create synthetic dataset for quick demo
    println!("Creating synthetic code dataset...");
    let dataset = SyntheticDataset::new(
        vocab_size as u32,
        max_seq_len,
        args.total_steps * args.batch_size * 2, // 2x for variety
        42,
    );
    println!("✓ Dataset created: {} samples", dataset.len());
    println!();
    std::fs::create_dir_all(&args.checkpoint_dir)?;
    let checkpoint_interval = args.checkpoint_interval.max(1);
    let mut completed_steps = 0usize;
    let mut next_checkpoint_step = checkpoint_interval;
    let mut last_loss = 0.0f64;

    // Training loop
    println!("═══════════════════════════════════════════════════════════");
    println!("Starting training...");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    for epoch in 0..10 {
        println!("Epoch {}/10", epoch + 1);

        let start_time = std::time::Instant::now();
        let stats = trainer.train_epoch(&dataset, epoch)?;
        let epoch_time = start_time.elapsed().as_secs_f64();

        println!("  Avg loss: {:.4}", stats.avg_loss);
        println!("  Avg CE: {:.4}", stats.avg_ce);
        println!("  Avg KL: {:.4}", stats.avg_kl);
        println!("  Steps: {}", stats.steps);
        println!("  Time: {:.1}s", epoch_time);
        println!();
        completed_steps += stats.steps;
        last_loss = stats.avg_loss;

        while completed_steps >= next_checkpoint_step {
            let checkpoint_path =
                format!("{}/step_{:06}", args.checkpoint_dir, next_checkpoint_step);
            println!("  Saving checkpoint to {}...", checkpoint_path);
            save_checkpoint(
                &trainer.student_varmap,
                &trainer.config.train_config,
                trainer.global_step,
                stats.avg_loss,
                &checkpoint_path,
            )?;
            println!("  ✓ Checkpoint saved");

            if args.keep_last_checkpoints > 0 {
                let removed =
                    cleanup_old_checkpoints(&args.checkpoint_dir, args.keep_last_checkpoints)?;
                if removed > 0 {
                    println!("  ✓ Removed {} old checkpoint(s)", removed);
                }
            }

            next_checkpoint_step += checkpoint_interval;
        }
    }

    let final_checkpoint = format!("{}/final", args.checkpoint_dir);
    println!("Saving final checkpoint to {}...", final_checkpoint);
    save_checkpoint(
        &trainer.student_varmap,
        &trainer.config.train_config,
        trainer.global_step,
        last_loss,
        &final_checkpoint,
    )?;
    println!("✓ Final checkpoint saved");

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("Training complete!");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Next steps:");
    println!("1. Export to GGUF:");
    println!("   cargo run --bin export_model -- \\");
    println!("     --checkpoint {}/final \\", args.checkpoint_dir);
    println!("     --output nano-125m.gguf");
    println!();
    println!("2. Evaluate on HumanEval:");
    println!("   cargo run --release --example evaluate_codegen -- \\");
    println!("     --dataset humaneval \\");
    println!("     --data-path HumanEval.jsonl \\");
    println!("     --model-endpoint http://localhost:8080/v1/completions \\");
    println!("     --num-samples 10");

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

fn cleanup_old_checkpoints(
    checkpoint_dir: &str,
    keep_last: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut checkpoints: Vec<(PathBuf, usize)> = Vec::new();
    for entry in std::fs::read_dir(checkpoint_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(step_str) = name.strip_prefix("step_") else {
            continue;
        };
        let Ok(step) = step_str.parse::<usize>() else {
            continue;
        };
        checkpoints.push((path, step));
    }

    checkpoints.sort_by_key(|(_, step)| *step);
    let to_remove = checkpoints.len().saturating_sub(keep_last);
    let mut removed = 0usize;
    for (path, _) in checkpoints.into_iter().take(to_remove) {
        std::fs::remove_dir_all(path)?;
        removed += 1;
    }

    Ok(removed)
}
