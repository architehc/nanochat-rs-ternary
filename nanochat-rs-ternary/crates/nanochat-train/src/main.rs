//! CLI entry point for nanochat-train.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "nanochat-train",
    about = "Train nanochat ternary models in Rust"
)]
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

        /// Resume training from a checkpoint directory
        #[arg(long)]
        resume: Option<String>,

        #[arg(long, default_value = "50")]
        log_interval: usize,

        /// Save checkpoint every N steps (0 = only at end)
        #[arg(long, default_value = "1000")]
        checkpoint_interval: usize,

        /// Keep only last N checkpoints (0 = keep all)
        #[arg(long, default_value = "3")]
        keep_last_checkpoints: usize,

        /// Number of CPU threads (default: all available)
        #[arg(long)]
        threads: Option<usize>,

        /// Number of synthetic samples (default: 100000)
        #[arg(long, default_value = "100000")]
        n_samples: usize,

        #[arg(long, default_value = "cpu")]
        device: String,
    },

    /// Train a BPE tokenizer on a text file and produce tokenizer.json + tokens.bin
    PrepareData {
        /// Path to input text file
        #[arg(long)]
        text: String,

        /// Target vocabulary size
        #[arg(long, default_value = "4096")]
        vocab_size: usize,

        /// Output directory for tokenizer.json and tokens.bin
        #[arg(long)]
        output: String,
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
    // Initialize structured logging
    tracing_subscriber::fmt::init();

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
            resume,
            log_interval,
            checkpoint_interval,
            keep_last_checkpoints,
            threads,
            n_samples,
            device,
        } => {
            // Set thread count before anything else
            if let Some(n) = threads {
                std::env::set_var("RAYON_NUM_THREADS", n.to_string());
                tracing::info!("Threads: {} (set via RAYON_NUM_THREADS)", n);
            } else {
                let n = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1);
                tracing::info!("Threads: {} (auto-detected)", n);
            }

            let device = match device.as_str() {
                "cpu" => candle_core::Device::Cpu,
                #[cfg(feature = "cuda")]
                "cuda" => candle_core::Device::new_cuda(0)?,
                other => {
                    tracing::error!("Unknown device: {}. Use 'cpu' or 'cuda'.", other);
                    std::process::exit(1);
                }
            };

            let mut cfg = match config.as_str() {
                "d20" => nanochat_train::config::TrainConfig::d20(),
                "d20-mtp" | "d20_mtp" => nanochat_train::config::TrainConfig::d20_mtp(),
                "d20-e3" | "d20_e3" | "d20-e3-full" | "d20_e3_full" => {
                    nanochat_train::config::TrainConfig::d20_e3_full()
                }
                "nano-125m" | "nano_125m" => nanochat_train::config::TrainConfig::nano_125m(),
                "nano-1b" | "nano_1b" => nanochat_train::config::TrainConfig::nano_1b(),
                "medium-3b" | "medium_3b" => nanochat_train::config::TrainConfig::medium_3b(),
                "tiny-cpu" | "tiny_cpu" => nanochat_train::config::TrainConfig::tiny_cpu(),
                "test-8bit" | "test_8bit" => nanochat_train::config::TrainConfig::test_8bit(),
                other => {
                    tracing::error!(
                        "Unknown config: {}. Use d20, d20-mtp, d20-e3-full, nano-125m, nano-1b, medium-3b, tiny-cpu, or test-8bit.",
                        other
                    );
                    std::process::exit(1);
                }
            };

            if let Some(bs) = batch_size {
                cfg.batch_size = bs;
            }

            let effective_seq_len = seq_len.unwrap_or(cfg.max_seq_len / 2);

            tracing::info!("=== nanochat-train ===");
            tracing::info!("Config: {}", config);
            tracing::info!(
                "Model params: ~{:.1}M",
                cfg.param_count_estimate() as f64 / 1e6
            );
            tracing::info!("FFN dim: {}", cfg.ffn_dim());
            tracing::info!("Device: {:?}", device);
            tracing::info!("Batch size: {}", cfg.batch_size);
            tracing::info!("Seq len: {}", effective_seq_len);
            tracing::info!("Epochs: {}", epochs);
            tracing::info!("Log interval: {} steps", log_interval);
            tracing::info!("Checkpoint interval: {} steps", checkpoint_interval);
            tracing::info!("");

            // Resume from checkpoint if specified
            let mut trainer = if let Some(ref ckpt_dir) = resume {
                tracing::info!("Resuming from checkpoint: {}", ckpt_dir);
                let meta_json = std::fs::read_to_string(format!("{}/meta.json", ckpt_dir))?;
                let meta: nanochat_train::checkpoint::CheckpointMeta =
                    serde_json::from_str(&meta_json)?;

                // Reconstruct model with same config
                let mut trainer = nanochat_train::train::Trainer::new(meta.config, device)?;
                trainer
                    .varmap
                    .load(format!("{}/model.safetensors", ckpt_dir))?;
                trainer.global_step = meta.step;
                tracing::info!("Resumed at step {} (loss={:.4})", meta.step, meta.loss);
                trainer
            } else {
                nanochat_train::train::Trainer::new(cfg.clone(), device)?
            };

            tracing::info!(
                "Model initialized. Total params: {}",
                trainer.model.param_count()
            );

            // Create dataset
            let ds: Box<dyn nanochat_train::data::Dataset> = match dataset.as_str() {
                "synthetic" => Box::new(nanochat_train::data::SyntheticDataset::new(
                    cfg.vocab_size as u32,
                    effective_seq_len,
                    n_samples,
                    42,
                )),
                "tokens" => {
                    let path = data_path.expect("--data-path required for tokens dataset");
                    Box::new(
                        nanochat_train::data::dataset::TokenFileDataset::from_binary_file(
                            std::path::Path::new(&path),
                            effective_seq_len,
                        )?,
                    )
                }
                other => {
                    tracing::error!("Unknown dataset: {}. Use 'synthetic' or 'tokens'.", other);
                    std::process::exit(1);
                }
            };

            tracing::info!("Dataset: {} samples", ds.len());
            let tokens_per_epoch = ds.len() as f64 * effective_seq_len as f64;
            tracing::info!("Tokens per epoch: {:.1}M", tokens_per_epoch / 1e6);
            tracing::info!("");

            // Run training loop with per-step logging and checkpoint management
            trainer.train_loop(
                ds.as_ref(),
                epochs,
                log_interval,
                checkpoint_dir.as_deref(),
                checkpoint_interval,
                keep_last_checkpoints,
            )?;
        }

        Commands::PrepareData {
            text,
            vocab_size,
            output,
        } => {
            nanochat_train::data::prepare_data(
                std::path::Path::new(&text),
                vocab_size,
                std::path::Path::new(&output),
            )
            .map_err(|e| e as Box<dyn std::error::Error>)?;
        }

        Commands::Export {
            checkpoint,
            gguf,
            mhc,
        } => {
            let device = candle_core::Device::Cpu;

            tracing::info!("Loading checkpoint from {}...", checkpoint);
            let meta_json = std::fs::read_to_string(format!("{}/meta.json", checkpoint))?;
            let meta: nanochat_train::checkpoint::CheckpointMeta =
                serde_json::from_str(&meta_json)?;

            let mut varmap = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
            let model = nanochat_train::model::NanochatTrainModel::new(&meta.config, vb)?;

            // Load saved weights
            varmap.load(format!("{}/model.safetensors", checkpoint))?;

            tracing::info!("Exporting to {} and {}...", gguf, mhc);
            nanochat_train::export::export_model(&model, &meta.config, &gguf, &mhc)?;
            tracing::info!("Export complete!");
        }
    }

    Ok(())
}
