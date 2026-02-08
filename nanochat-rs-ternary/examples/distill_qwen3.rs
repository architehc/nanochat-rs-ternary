//! Example: Distillation training for Qwen3-Coder-80B hybrid ternary model
//!
//! This example demonstrates training a hybrid ternary student model using a
//! remote FP8 teacher endpoint for knowledge distillation.
//!
//! Architecture:
//! - Teacher: Full FP8 Qwen3-Coder-80B (80B params, remote inference)
//! - Student: Hybrid ternary (3B active, 80B total)
//!   - Ternary: MoE expert weights (512 experts × w_gate/w_up/w_down)
//!   - FP8/BF16: Router, gates, norms, embeddings, lm_head, DeltaNet state
//!
//! Memory usage:
//! - Teacher: 0GB (remote endpoint, no local weights)
//! - Student: ~5GB (3B active params + gradients + optimizer states)
//! - Total GPU memory: ~10GB (includes activations + KV cache)
//!
//! Run:
//!   cargo run --release --example distill_qwen3 -- \
//!     --teacher-endpoint https://crazyshit.ngrok.io \
//!     --checkpoint-dir checkpoints/qwen3-hybrid \
//!     --keep-last-checkpoints 3

use candle_core::Device;
use nanochat_train::{
    config::TrainConfig,
    data::{Dataset, SyntheticDataset},
    distill::{DistillConfig, DistillationTrainer, TeacherMode},
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "distill_qwen3")]
#[command(about = "Distillation training for Qwen3-Coder hybrid ternary model")]
struct Args {
    /// Remote teacher endpoint (FP8 inference server)
    #[arg(long, default_value = "https://crazyshit.ngrok.io")]
    teacher_endpoint: String,

    /// API key for teacher endpoint (if required)
    #[arg(long)]
    teacher_api_key: Option<String>,

    /// Checkpoint directory for saving models
    #[arg(long, default_value = "checkpoints/qwen3-hybrid")]
    checkpoint_dir: String,

    /// Keep only last N checkpoints (0 = keep all)
    #[arg(long, default_value = "3")]
    keep_last_checkpoints: usize,

    /// Number of training steps
    #[arg(long, default_value = "100000")]
    total_steps: usize,

    /// Warmup steps
    #[arg(long, default_value = "2000")]
    warmup_steps: usize,

    /// Log interval (steps)
    #[arg(long, default_value = "100")]
    log_interval: usize,

    /// Checkpoint interval (steps)
    #[arg(long, default_value = "1000")]
    checkpoint_interval: usize,

    /// Batch size
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value = "2048")]
    seq_len: usize,

    /// Learning rate for Muon (ternary weights)
    #[arg(long, default_value = "0.02")]
    lr: f64,

    /// Learning rate for Lion (norms, mHC, router)
    #[arg(long, default_value = "0.0001")]
    mhc_lr: f64,

    /// KL divergence weight (0.0 = only CE, 1.0 = only KL)
    #[arg(long, default_value = "0.5")]
    kl_weight: f64,

    /// Distillation temperature
    #[arg(long, default_value = "2.0")]
    temperature: f64,

    /// Device (cpu, cuda:0, cuda:1, etc.)
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Use parallel teacher queries with gradient accumulation
    /// Leverages endpoint's concurrent capacity (8 requests) for 2-4x speedup
    #[arg(long, default_value_t = true)]
    parallel: bool,

    /// Number of micro-batches for parallel training (default: 8, matching endpoint capacity)
    #[arg(long, default_value = "8")]
    micro_batches: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Qwen3-Coder-80B Hybrid Ternary Distillation Training");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Parse device
    let device = if args.device.starts_with("cuda") {
        #[cfg(feature = "cuda")]
        {
            let gpu_id = args.device.strip_prefix("cuda:")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            Device::new_cuda(gpu_id)
                .map_err(|e| format!("Failed to create CUDA device: {}", e))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("CUDA not available, using CPU");
            Device::Cpu
        }
    } else {
        Device::Cpu
    };
    println!("Device: {:?}", device);

    // Create training config for Qwen3-Coder-80B
    // TODO: Add TrainConfig::qwen3_coder_80b() preset
    // For now, use a scaled-up config based on d20
    let mut train_config = TrainConfig {
        dim: 8192,
        n_layers: 48,
        n_heads: 64,
        n_kv_heads: 8,
        ffn_mult: 3.5,
        vocab_size: 151_936,
        max_seq_len: args.seq_len,
        group_size: 128,
        mhc_n_streams: 4,
        weight_tied: false,
        rope_theta: 10000.0,
        lr: args.lr,
        mhc_lr: args.mhc_lr,
        weight_decay: 0.1,
        batch_size: args.batch_size,
        grad_accum_steps: 1,
        warmup_steps: args.warmup_steps,
        total_steps: args.total_steps,
        decay_start_frac: 0.8,
        grad_clip: 1.0,
        ns_steps: 5,
        muon_momentum: 0.95,
        lion_betas: (0.9, 0.99),
    };
    // Save values before moving train_config
    let vocab_size = train_config.vocab_size;
    let total_steps = train_config.total_steps;

    // Create distillation config with remote teacher
    let distill_config = DistillConfig {
        train_config,
        teacher_mode: TeacherMode::Remote {
            endpoint: args.teacher_endpoint.clone(),
            api_key: args.teacher_api_key.clone(),
            timeout_secs: 60, // Generous timeout for 80B model
            max_concurrent: args.micro_batches,
        },
        temperature: args.temperature,
        kl_weight: args.kl_weight,
        load_balance_weight: 0.01,   // Encourage balanced expert usage
        router_aux_weight: 0.001,    // Improve routing decisions
        freeze_student_fp8: false,   // Train router/norms (can be true to freeze)
        micro_batches: args.micro_batches,
    };

    println!();
    println!("Configuration:");
    println!("  Model: Qwen3-Coder-80B (hybrid ternary)");
    println!("    - Total params: 80B");
    println!("    - Active params: ~3B");
    println!("    - Experts: 512 (top-10 routing)");
    println!("    - Layers: 48 (12×[3 DeltaNet + 1 Gated Attn])");
    println!();
    println!("  Teacher:");
    println!("    - Mode: Remote FP8 endpoint");
    println!("    - URL: {}", args.teacher_endpoint);
    println!("    - Timeout: 60s");
    println!();
    println!("  Training:");
    println!("    - Steps: {}", args.total_steps);
    println!("    - Warmup: {}", args.warmup_steps);
    println!("    - Batch size: {}", args.batch_size);
    println!("    - Seq len: {}", args.seq_len);
    println!("    - Tokens/step: {}", args.batch_size * args.seq_len);
    println!();
    println!("  Optimizer:");
    println!("    - Muon LR: {} (ternary weights)", args.lr);
    println!("    - Lion LR: {} (router, norms, mHC)", args.mhc_lr);
    println!();
    println!("  Distillation:");
    println!("    - Temperature: {}", args.temperature);
    println!("    - KL weight: {} (CE weight: {})", args.kl_weight, 1.0 - args.kl_weight);
    println!("    - Load balance: 0.01");
    println!("    - Router aux: 0.001");
    println!();
    println!("  Parallelization:");
    println!("    - Mode: {}", if args.parallel { "Parallel (gradient accumulation)" } else { "Sequential" });
    println!("    - Micro-batches: {}", args.micro_batches);
    println!("    - Concurrent requests: {}", args.micro_batches);
    println!("    - Expected speedup: {}x", if args.parallel { args.micro_batches.min(8) as f32 / 2.0 } else { 1.0 });
    println!();
    println!("  Checkpointing:");
    println!("    - Directory: {}", args.checkpoint_dir);
    println!("    - Interval: {} steps", args.checkpoint_interval);
    println!("    - Keep last: {}", args.keep_last_checkpoints);
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Create checkpoint directory
    std::fs::create_dir_all(&args.checkpoint_dir)?;

    // Initialize trainer
    println!("Initializing trainer...");
    let mut trainer = DistillationTrainer::new(distill_config, device)?;
    println!("✓ Trainer initialized");
    println!();

    // Create synthetic dataset for demonstration
    // In production, replace with real tokenized dataset
    println!("Creating dataset...");
    let n_samples = 10000; // Small for demo
    let dataset = SyntheticDataset::new(vocab_size as u32, args.seq_len, n_samples, 42);
    println!("✓ Dataset created: {} samples", dataset.len());
    println!();

    println!("Starting training...");
    if args.parallel && args.micro_batches > 1 {
        println!("✓ Parallel training enabled - teacher queries sent concurrently");
        println!("  Micro-batches: {}", args.micro_batches);
        println!("  Expected speedup: ~{:.1}x", args.micro_batches.min(8) as f32 / 2.0);
    } else {
        println!("Sequential training mode (micro_batches=1)");
    }
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Training loop with automatic parallel/sequential selection
    let epochs = (total_steps * args.batch_size) / n_samples + 1;

    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch + 1, epochs);

        match trainer.train_epoch(&dataset, epoch) {
            Ok(stats) => {
                println!("  Avg loss: {:.4}", stats.avg_loss);
                println!("  Avg CE: {:.4}", stats.avg_ce);
                println!("  Avg KL: {:.4}", stats.avg_kl);
            }
            Err(e) => {
                eprintln!("Training error: {}", e);
                break;
            }
        }

        if trainer.global_step >= total_steps {
            break;
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("Training complete!");
    println!("  Final step: {}", trainer.global_step);
    println!();

    // Save final checkpoint
    println!("Saving final checkpoint...");
    let final_path = format!("{}/final", args.checkpoint_dir);
    nanochat_train::checkpoint::save_checkpoint(
        &trainer.student_varmap,
        &trainer.config.train_config,
        trainer.global_step,
        0.0,
        &final_path,
    )
    .map_err(|e| format!("Failed to save checkpoint: {}", e))?;
    println!("✓ Final checkpoint saved to {}", final_path);
    println!();

    println!("Next steps:");
    println!("  1. Export to GGUF: cargo run --bin qwen3_converter -- \\");
    println!("       --input {} \\", final_path);
    println!("       --output qwen3-hybrid.gguf");
    println!();
    println!("  2. Run inference: cargo run --release --bin nanochat-serve -- \\");
    println!("       --model qwen3-hybrid.gguf \\");
    println!("       --port 8000");
    println!();

    Ok(())
}
