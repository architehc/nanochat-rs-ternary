//! Diagnostic training to identify the copying bug.
//!
//! Key features:
//! - Tiny model (d10 config) for fast iteration
//! - NO weight tying (separate LM head)
//! - Verbose prediction logging every 10 steps
//! - Activation magnitude tracking
//! - Only 500 steps for quick feedback

use clap::Parser;
use nanochat_train::{
    config::TrainConfig,
    train::Trainer,
    data::dataset::TokenFileDataset,
};
use candle_core::{Device, DType, Tensor};
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(name = "train_diagnostic")]
struct Args {
    /// Training data path
    #[arg(long, default_value = "data/rust_tokens.bin")]
    data: String,

    /// Device (cpu, cuda:0, etc.)
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints/diagnostic")]
    checkpoint_dir: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  DIAGNOSTIC TRAINING - Copying Bug Investigation");
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

    // TINY model config for fast debugging
    let config = TrainConfig {
        dim: 128,              // d10-ish (small)
        n_layers: 2,           // Minimal depth
        n_heads: 4,
        n_kv_heads: 4,
        ffn_mult: 2.0,
        vocab_size: 50257,     // GPT-2 vocab
        max_seq_len: 64,       // Short sequences
        group_size: 128,
        mhc_n_streams: 2,
        weight_tied: false,    // 🔑 NO WEIGHT TYING!
        rope_theta: 10000.0,
        loop_config: None,
        lr: 0.001,             // Conservative LR
        mhc_lr: 1e-4,
        weight_decay: 0.0,
        batch_size: 1,
        grad_accum_steps: 1,
        warmup_steps: 50,
        total_steps: 500,      // Just 500 steps for quick feedback
        decay_start_frac: 0.8,
        grad_clip: 1.0,
        ns_steps: 3,
        muon_momentum: 0.95,
        lion_betas: (0.9, 0.99),
        distill_teacher: None,
        distill_kl_weight: 0.0,
        loop_scale_penalty: 0.0,
    };

    println!("Model config:");
    println!("  dim: {}", config.dim);
    println!("  n_layers: {}", config.n_layers);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  weight_tied: {} 🔑 SEPARATE LM HEAD!", config.weight_tied);
    println!("  total_steps: {}\n", config.total_steps);

    // Load dataset
    println!("Loading training data: {}", args.data);
    let dataset = TokenFileDataset::from_binary_file(
        std::path::Path::new(&args.data),
        config.max_seq_len,
    )?;
    println!("✓ Loaded {} sequences\n", dataset.len());

    // Create trainer
    println!("Initializing trainer...");
    let mut trainer = Trainer::new(config.clone(), device.clone())?;
    println!("✓ Trainer initialized\n");

    println!("═══════════════════════════════════════════════════════════");
    println!("Starting diagnostic training loop...");
    println!("Every 10 steps: show what model ACTUALLY predicts");
    println!("═══════════════════════════════════════════════════════════\n");

    // Manual training loop with verbose diagnostics
    use nanochat_train::data::dataset::Dataset;
    use nanochat_train::data::dataset::DataLoader;

    let mut running_loss = 0.0;
    let mut interval_steps = 0;

    let loader = DataLoader::new(&dataset, 1, true, 42, &device);

    for (batch_idx, batch_result) in loader.enumerate() {
        if trainer.global_step >= config.total_steps {
            break;
        }

        let (input_ids, target_ids) = batch_result?;
        
        // Run training step
        let stats = trainer.train_step(&input_ids, &target_ids)?;
        running_loss += stats.loss;
        interval_steps += 1;

        // DIAGNOSTIC: Every 10 steps, show predictions
        if trainer.global_step % 10 == 0 && trainer.global_step > 0 {
            let avg_loss = running_loss / interval_steps as f64;
            println!("\n[Step {:>4}] loss={:.4} lr={:.6}", 
                     trainer.global_step, avg_loss, stats.lr);

            // Get current batch tokens for analysis
            let inp_vec = input_ids.to_vec2::<u32>()?;
            let tgt_vec = target_ids.to_vec2::<u32>()?;
            
            // Forward pass to see predictions
            let logits = trainer.model.forward(&input_ids)?;  // [1, seq, vocab]
            let seq_len = logits.dim(1)?;
            
            // Check first 5 positions
            let check_positions = std::cmp::min(5, seq_len);
            println!("  Predictions vs targets:");
            for pos in 0..check_positions {
                let pos_logits = logits.get(0)?.get(pos)?;
                let logits_vec = pos_logits.to_vec1::<f32>()?;
                
                let (pred_token, pred_logit) = logits_vec.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                
                let input_token = inp_vec[0][pos];
                let target_token = tgt_vec[0][pos];
                
                let pred_match = if pred_token as u32 == target_token {
                    "✓ CORRECT"
                } else if pred_token as u32 == input_token {
                    "✗ COPIES INPUT!"
                } else {
                    "✗ wrong"
                };
                
                println!("    pos {}: input={:>5} target={:>5} pred={:>5} (logit={:>6.1}) {}",
                         pos, input_token, target_token, pred_token, pred_logit, pred_match);
            }

            running_loss = 0.0;
            interval_steps = 0;
        }

        // Checkpoint every 100 steps
        if trainer.global_step % 100 == 0 && trainer.global_step > 0 {
            let path = format!("{}/step_{}", args.checkpoint_dir, trainer.global_step);
            std::fs::create_dir_all(&path)?;
            nanochat_train::checkpoint::save_checkpoint(
                &trainer.varmap,
                &trainer.config,
                trainer.global_step,
                running_loss / interval_steps.max(1) as f64,
                &path,
            ).map_err(|e| anyhow::anyhow!("checkpoint save: {}", e))?;
            println!("  💾 Checkpoint saved: {}", path);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Diagnostic training complete!");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Final checkpoint: {}/step_{}", args.checkpoint_dir, config.total_steps);
    println!("\nTest generation with:");
    println!("  cargo run --release --example debug_generation");
    println!("  (update checkpoint path to {})\n", args.checkpoint_dir);

    Ok(())
}
