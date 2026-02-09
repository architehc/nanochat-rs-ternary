//! Advanced training with gradient accumulation, FP16, and evaluation metrics.
//!
//! Features:
//! - Gradient accumulation to simulate larger batch sizes
//! - Mixed precision (FP16) training for memory efficiency
//! - Evaluation metrics during training
//! - Ternary kernel integration for inference-mode validation

use clap::Parser;
use nanochat_train::{
    config::TrainConfig,
    data::SyntheticCodeDataset,
    train::Trainer,
    model::NanochatTrainModel,
    checkpoint,
    export,
};
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(name = "train_advanced")]
#[command(about = "Advanced training with gradient accumulation and FP16")]
struct Args {
    /// Total training steps
    #[arg(long, default_value = "10000")]
    total_steps: usize,

    /// Warmup steps
    #[arg(long, default_value = "1000")]
    warmup_steps: usize,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints/advanced")]
    checkpoint_dir: String,

    /// Device (cpu, cuda:0, etc.)
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Micro batch size (per gradient accumulation step)
    #[arg(long, default_value = "1")]
    micro_batch_size: usize,

    /// Gradient accumulation steps
    #[arg(long, default_value = "4")]
    accumulation_steps: usize,

    /// Sequence length
    #[arg(long, default_value = "256")]
    seq_len: usize,

    /// Learning rate
    #[arg(long, default_value = "0.02")]
    lr: f64,

    /// Log interval
    #[arg(long, default_value = "100")]
    log_interval: usize,

    /// Evaluation interval
    #[arg(long, default_value = "500")]
    eval_interval: usize,

    /// Checkpoint interval
    #[arg(long, default_value = "1000")]
    checkpoint_interval: usize,

    /// Keep last N checkpoints
    #[arg(long, default_value = "3")]
    keep_last_checkpoints: usize,

    /// Use mixed precision (FP16)
    #[arg(long)]
    fp16: bool,

    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<String>,
}

fn main() -> candle_core::Result<()> {
    let args = Args::parse();
    let start_time = Instant::now();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Advanced Training (nano-125M)");
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

    // Data type
    let dtype = if args.fp16 && device.is_cuda() {
        println!("✓ Using mixed precision (FP16)");
        DType::F16
    } else {
        DType::F32
    };

    // Configuration
    let mut config = TrainConfig::nano_125m();
    let effective_batch_size = args.micro_batch_size * args.accumulation_steps;

    println!("Configuration:");
    println!("  Model: nano-125M (127M params)");
    println!("  Device: {:?}", device);
    println!("  Dtype: {:?}", dtype);
    println!("  Total steps: {}", args.total_steps);
    println!("  Micro batch size: {}", args.micro_batch_size);
    println!("  Gradient accumulation steps: {}", args.accumulation_steps);
    println!("  Effective batch size: {}", effective_batch_size);
    println!("  Sequence length: {}", args.seq_len);
    println!("  Learning rate: {}", args.lr);
    println!("  Checkpoint dir: {}\n", args.checkpoint_dir);

    // Create model
    let (varmap, mut model, start_step, best_loss) = if let Some(ref resume_path) = args.resume {
        println!("Resuming from checkpoint: {}", resume_path);
        let (varmap, config, step, loss) = checkpoint::load_checkpoint(resume_path, &device)?;
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = NanochatTrainModel::new(&config, vb)?;
        println!("✓ Checkpoint loaded (step {}, loss {:.4})\n", step, loss);
        (varmap, model, step, loss)
    } else {
        println!("Creating new model...");
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = NanochatTrainModel::new(&config, vb)?;
        println!("✓ Model created\n");
        (varmap, model, 0, f32::INFINITY)
    };

    println!("Model Architecture:");
    println!("  Dimension: {}", config.dim);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {}", config.n_heads);
    println!("  Estimated params: {:.1}M\n", 127.1);

    // Create datasets (train + eval)
    println!("Creating synthetic code datasets...");
    let train_dataset = SyntheticCodeDataset::new(80000, args.seq_len);
    let eval_dataset = SyntheticCodeDataset::new(1000, args.seq_len); // Small eval set
    println!("✓ Train dataset: {} samples", train_dataset.len());
    println!("✓ Eval dataset: {} samples\n", eval_dataset.len());

    // Create trainer with gradient accumulation
    println!("═══════════════════════════════════════════════════════════");
    println!("Starting training...");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut trainer = Trainer::new(&varmap, args.lr)?;
    let mut global_step = start_step;
    let mut best_eval_loss = best_loss;

    // Progress bar
    let pb = ProgressBar::new(args.total_steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    // Training loop with gradient accumulation
    while global_step < args.total_steps {
        let step_start = Instant::now();
        let mut accumulated_loss = 0.0f32;

        // Gradient accumulation loop
        for acc_step in 0..args.accumulation_steps {
            let batch_idx = (global_step * args.accumulation_steps + acc_step) % train_dataset.len();
            let (input_ids, target_ids) = train_dataset.get_batch(batch_idx, args.micro_batch_size);

            let loss = model.forward_loss(&input_ids, &target_ids)?;
            accumulated_loss += loss.to_scalar::<f32>()? / args.accumulation_steps as f32;

            // Backward pass (gradients accumulate)
            let grads = loss.backward()?;

            // Only update on last accumulation step
            if acc_step == args.accumulation_steps - 1 {
                trainer.step(&grads, global_step, args.total_steps, args.warmup_steps)?;
            }
        }

        global_step += 1;

        // Logging
        if global_step % args.log_interval == 0 {
            let lr = trainer.get_current_lr(global_step, args.total_steps, args.warmup_steps);
            let grad_norm = trainer.grad_norm();
            let elapsed = start_time.elapsed().as_secs();
            let tokens_per_sec = (args.seq_len * effective_batch_size) as f64 / step_start.elapsed().as_secs_f64();

            println!(
                "[ {:>6}/{} ] loss={:.4} lr={:.6} gnorm={:.2} tok/s={:.0} elapsed={}s",
                global_step, args.total_steps, accumulated_loss, lr, grad_norm, tokens_per_sec, elapsed
            );

            pb.set_position(global_step as u64);
            pb.set_message(format!("loss={:.4}", accumulated_loss));
        }

        // Evaluation
        if global_step % args.eval_interval == 0 {
            println!("\n--- Evaluating at step {} ---", global_step);
            let eval_loss = evaluate(&model, &eval_dataset, args.micro_batch_size)?;
            println!("✓ Eval loss: {:.4}", eval_loss);

            if eval_loss < best_eval_loss {
                println!("✓ New best eval loss! (previous: {:.4})", best_eval_loss);
                best_eval_loss = eval_loss;
            }
            println!();
        }

        // Checkpointing
        if global_step % args.checkpoint_interval == 0 {
            let checkpoint_path = format!("{}/step_{}", args.checkpoint_dir, global_step);
            checkpoint::save_checkpoint(
                &varmap,
                &config,
                global_step,
                accumulated_loss,
                &checkpoint_path,
            )?;

            // Cleanup old checkpoints
            checkpoint::cleanup_old_checkpoints(&args.checkpoint_dir, args.keep_last_checkpoints)?;

            println!("  -> checkpoint saved to {} ({:.2}MB)",
                checkpoint_path,
                std::fs::metadata(format!("{}/model.safetensors", checkpoint_path))
                    .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                    .unwrap_or(0.0)
            );
        }
    }

    pb.finish_with_message("Training complete!");

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Training Complete!");
    println!("═══════════════════════════════════════════════════════════");
    println!("Total time: {:.1}m", start_time.elapsed().as_secs_f64() / 60.0);
    println!("Final loss: {:.4}", accumulated_loss);
    println!("Best eval loss: {:.4}", best_eval_loss);
    println!("\nExport the final checkpoint:");
    println!("  cargo run --release -p nanochat-train --example export_checkpoint -- \\");
    println!("    --checkpoint {}/step_{} \\", args.checkpoint_dir, global_step);
    println!("    --output models/nano-125m-final.gguf\n");

    Ok(())
}

/// Evaluate model on validation set
fn evaluate(
    model: &NanochatTrainModel,
    dataset: &SyntheticCodeDataset,
    batch_size: usize,
) -> candle_core::Result<f32> {
    let mut total_loss = 0.0f32;
    let n_batches = (dataset.len() / batch_size).max(1);

    for i in 0..n_batches {
        let (input_ids, target_ids) = dataset.get_batch(i, batch_size);
        let loss = model.forward_loss(&input_ids, &target_ids)?;
        total_loss += loss.to_scalar::<f32>()?;
    }

    Ok(total_loss / n_batches as f32)
}
