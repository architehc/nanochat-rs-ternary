//! CLI entry point for nanochat-train.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "nanochat-train", about = "Train nanochat ternary models in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model
    Train {
        #[arg(long, default_value = "d20")]
        config: String,

        #[arg(long, default_value = "synthetic")]
        dataset: String,

        #[arg(long)]
        data_path: Option<String>,

        #[arg(long, default_value = "5")]
        epochs: usize,

        #[arg(long)]
        batch_size: Option<usize>,

        #[arg(long)]
        seq_len: Option<usize>,

        #[arg(long)]
        checkpoint_dir: Option<String>,

        #[arg(long)]
        resume: Option<String>,

        #[arg(long, default_value = "10")]
        log_interval: usize,

        #[arg(long, default_value = "cpu")]
        device: String,
    },

    /// Export trained model to GGUF + mHC
    Export {
        #[arg(long)]
        checkpoint: String,

        #[arg(long)]
        gguf: String,

        #[arg(long)]
        mhc: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            config,
            dataset,
            data_path,
            epochs,
            batch_size,
            seq_len,
            checkpoint_dir,
            resume: _,
            log_interval: _,
            device,
        } => {
            let device = match device.as_str() {
                "cpu" => candle_core::Device::Cpu,
                #[cfg(feature = "cuda")]
                "cuda" => candle_core::Device::new_cuda(0)?,
                other => {
                    eprintln!("Unknown device: {}. Use 'cpu' or 'cuda'.", other);
                    std::process::exit(1);
                }
            };

            let mut cfg = match config.as_str() {
                "d20" => nanochat_train::config::TrainConfig::d20(),
                "nano-125m" | "nano_125m" => nanochat_train::config::TrainConfig::nano_125m(),
                "nano-1b" | "nano_1b" => nanochat_train::config::TrainConfig::nano_1b(),
                other => {
                    eprintln!("Unknown config: {}. Use d20, nano-125m, or nano-1b.", other);
                    std::process::exit(1);
                }
            };

            if let Some(bs) = batch_size {
                cfg.batch_size = bs;
            }

            let effective_seq_len = seq_len.unwrap_or(cfg.max_seq_len / 2);

            println!("=== nanochat-train ===");
            println!("Config: {}", config);
            println!("Model params: ~{:.1}M", cfg.param_count_estimate() as f64 / 1e6);
            println!("FFN dim: {}", cfg.ffn_dim());
            println!("Device: {:?}", device);
            println!("Batch size: {}", cfg.batch_size);
            println!("Seq len: {}", effective_seq_len);
            println!("Epochs: {}", epochs);
            println!();

            let mut trainer = nanochat_train::train::Trainer::new(cfg.clone(), device)?;
            println!("Model initialized. Total params: {}", trainer.model.param_count());

            // Create dataset
            let ds: Box<dyn nanochat_train::data::Dataset> = match dataset.as_str() {
                "synthetic" => {
                    Box::new(nanochat_train::data::SyntheticDataset::new(
                        cfg.vocab_size as u32,
                        effective_seq_len,
                        10_000,
                        42,
                    ))
                }
                "tokens" => {
                    let path = data_path.expect("--data-path required for tokens dataset");
                    Box::new(nanochat_train::data::dataset::TokenFileDataset::from_binary_file(
                        std::path::Path::new(&path),
                        effective_seq_len,
                    )?)
                }
                other => {
                    eprintln!("Unknown dataset: {}. Use 'synthetic' or 'tokens'.", other);
                    std::process::exit(1);
                }
            };

            println!("Dataset: {} samples", ds.len());
            println!();

            for epoch in 0..epochs {
                let start = std::time::Instant::now();
                let avg_loss = trainer.train_epoch(ds.as_ref(), epoch)?;
                let elapsed = start.elapsed();
                println!(
                    "Epoch {}/{}: avg_loss={:.4} time={:.1}s step={}",
                    epoch + 1, epochs, avg_loss, elapsed.as_secs_f64(), trainer.global_step
                );

                if let Some(ref dir) = checkpoint_dir {
                    let path = format!("{}/epoch_{}", dir, epoch + 1);
                    nanochat_train::checkpoint::save_checkpoint(
                        &trainer.varmap, &cfg, trainer.global_step, avg_loss, &path,
                    )?;
                    println!("  Checkpoint saved to {}", path);
                }
            }

            println!("\nTraining complete! Final step: {}", trainer.global_step);
        }

        Commands::Export { checkpoint, gguf, mhc } => {
            let device = candle_core::Device::Cpu;

            println!("Loading checkpoint from {}...", checkpoint);
            let meta_json = std::fs::read_to_string(format!("{}/meta.json", checkpoint))?;
            let meta: nanochat_train::checkpoint::CheckpointMeta = serde_json::from_str(&meta_json)?;

            let mut varmap = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
            let model = nanochat_train::model::NanochatTrainModel::new(&meta.config, vb)?;

            // Load saved weights
            varmap.load(format!("{}/model.safetensors", checkpoint))?;

            println!("Exporting to {} and {}...", gguf, mhc);
            nanochat_train::export::export_model(&model, &meta.config, &gguf, &mhc)?;
            println!("Export complete!");
        }
    }

    Ok(())
}
