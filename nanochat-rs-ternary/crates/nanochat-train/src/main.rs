//! CLI entry point for nanochat-train.

use clap::{Parser, Subcommand};
use nanochat_train::config::TrainConfig;

fn resolve_train_config(config: &str) -> Option<TrainConfig> {
    match config {
        "d20" => Some(TrainConfig::d20()),
        "d20-mtp" | "d20_mtp" => Some(TrainConfig::d20_mtp()),
        "d20-e3" | "d20_e3" | "d20-e3-full" | "d20_e3_full" => {
            Some(TrainConfig::d20_e3_full())
        }
        "d20-e3-fp4" | "d20_e3_fp4" => Some(TrainConfig::d20_e3_fp4()),
        "nano-125m" | "nano_125m" => Some(TrainConfig::nano_125m()),
        "nano-1b" | "nano_1b" => Some(TrainConfig::nano_1b()),
        "medium-3b" | "medium_3b" => Some(TrainConfig::medium_3b()),
        "large-7b" | "large_7b" => Some(TrainConfig::large_7b()),
        "tiny-cpu" | "tiny_cpu" => Some(TrainConfig::tiny_cpu()),
        "test-8bit" | "test_8bit" => Some(TrainConfig::test_8bit()),
        _ => None,
    }
}

fn resolve_device(device: &str) -> Result<candle_core::Device, String> {
    match device {
        "cpu" => Ok(candle_core::Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => candle_core::Device::new_cuda(0)
            .map_err(|e| format!("Failed to initialize CUDA device: {}", e)),
        other => Err(format!("Unknown device: {}. Use 'cpu' or 'cuda'.", other)),
    }
}

fn apply_batch_size_override(cfg: &mut TrainConfig, batch_size: Option<usize>) -> Result<(), String> {
    if let Some(bs) = batch_size {
        if bs == 0 {
            return Err("error: --batch-size must be > 0".to_string());
        }
        cfg.batch_size = bs;
    }
    Ok(())
}

fn effective_seq_len(seq_len: Option<usize>, cfg: &TrainConfig) -> Result<usize, String> {
    let effective = seq_len.unwrap_or(cfg.max_seq_len / 2);
    if effective == 0 {
        return Err("error: --seq-len must be > 0".to_string());
    }
    Ok(effective)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DatasetSpec {
    Synthetic,
    Tokens(String),
}

fn resolve_dataset_spec(dataset: &str, data_path: Option<String>) -> Result<DatasetSpec, String> {
    match dataset {
        "synthetic" => Ok(DatasetSpec::Synthetic),
        "tokens" => match data_path {
            Some(path) => Ok(DatasetSpec::Tokens(path)),
            None => Err("error: --data-path is required when --dataset=tokens".to_string()),
        },
        other => Err(format!(
            "Unknown dataset: {}. Use 'synthetic' or 'tokens'.",
            other
        )),
    }
}

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

            let device = match resolve_device(&device) {
                Ok(device) => device,
                Err(message) => {
                    tracing::error!("{}", message);
                    std::process::exit(1);
                }
            };
            let device_str = format!("{:?}", device);

            let mut cfg = match resolve_train_config(&config) {
                Some(cfg) => cfg,
                None => {
                    tracing::error!(
                        "Unknown config: {}. Use d20, d20-mtp, d20-e3-full, d20-e3-fp4, nano-125m, nano-1b, medium-3b, large-7b, tiny-cpu, or test-8bit.",
                        config
                    );
                    std::process::exit(1);
                }
            };

            if let Err(message) = apply_batch_size_override(&mut cfg, batch_size) {
                eprintln!("{}", message);
                std::process::exit(1);
            }

            // Resume from checkpoint if specified
            let mut trainer = if let Some(ref ckpt_dir) = resume {
                tracing::info!("Resuming from checkpoint: {}", ckpt_dir);
                let trainer = nanochat_train::train::Trainer::from_checkpoint(ckpt_dir, device)?;
                let meta_json = std::fs::read_to_string(format!("{}/meta.json", ckpt_dir))?;
                let meta: nanochat_train::checkpoint::CheckpointMeta =
                    serde_json::from_str(&meta_json)?;
                tracing::info!("Resumed at step {} (loss={:.4})", meta.step, meta.loss);
                // Use checkpoint's config for dataset/logging to avoid mismatch
                // (e.g. CLI --config d20 but checkpoint was trained with nano-125m)
                cfg = trainer.config.clone();
                // Re-apply CLI overrides on top of checkpoint config
                if let Err(message) = apply_batch_size_override(&mut cfg, batch_size) {
                    eprintln!("{}", message);
                    std::process::exit(1);
                }
                tracing::info!("Using checkpoint config (vocab_size={}, dim={})", cfg.vocab_size, cfg.dim);
                trainer
            } else {
                nanochat_train::train::Trainer::new(cfg.clone(), device)?
            };

            let effective_seq_len = match effective_seq_len(seq_len, &cfg) {
                Ok(v) => v,
                Err(message) => {
                    eprintln!("{}", message);
                    std::process::exit(1);
                }
            };

            tracing::info!("=== nanochat-train ===");
            tracing::info!("Config: {}", config);
            tracing::info!(
                "Model params: ~{:.1}M",
                cfg.param_count_estimate() as f64 / 1e6
            );
            tracing::info!("FFN dim: {}", cfg.ffn_dim());
            tracing::info!("Device: {}", device_str);
            tracing::info!("Batch size: {}", cfg.batch_size);
            tracing::info!("Seq len: {}", effective_seq_len);
            tracing::info!("Epochs: {}", epochs);
            tracing::info!("Log interval: {} steps", log_interval);
            tracing::info!("Checkpoint interval: {} steps", checkpoint_interval);
            tracing::info!("");

            tracing::info!(
                "Model initialized. Total params: {}",
                trainer.model.param_count()
            );

            // Create dataset
            let ds: Box<dyn nanochat_train::data::Dataset> = match resolve_dataset_spec(
                &dataset,
                data_path,
            ) {
                Ok(DatasetSpec::Synthetic) => Box::new(nanochat_train::data::SyntheticDataset::new(
                    cfg.vocab_size as u32,
                    effective_seq_len,
                    n_samples,
                    42,
                )),
                Ok(DatasetSpec::Tokens(path)) => {
                    Box::new(
                        nanochat_train::data::dataset::TokenFileDataset::from_binary_file(
                            std::path::Path::new(&path),
                            effective_seq_len,
                        )?,
                    )
                }
                Err(message) => {
                    if message.starts_with("Unknown dataset:") {
                        tracing::error!("{}", message);
                    } else {
                        eprintln!("{}", message);
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_train_config_aliases() {
        let aliases = [
            "d20",
            "d20-mtp",
            "d20_mtp",
            "d20-e3",
            "d20_e3",
            "d20-e3-full",
            "d20_e3_full",
            "d20-e3-fp4",
            "d20_e3_fp4",
            "nano-125m",
            "nano_125m",
            "nano-1b",
            "nano_1b",
            "medium-3b",
            "medium_3b",
            "large-7b",
            "large_7b",
            "tiny-cpu",
            "tiny_cpu",
            "test-8bit",
            "test_8bit",
        ];
        for alias in aliases {
            assert!(
                resolve_train_config(alias).is_some(),
                "expected alias {} to resolve",
                alias
            );
        }
        assert!(resolve_train_config("unknown").is_none());
    }

    #[test]
    fn test_resolve_device_cpu_and_unknown() {
        let device = resolve_device("cpu").expect("cpu device");
        assert!(matches!(device, candle_core::Device::Cpu));
        let err = resolve_device("weird").expect_err("unknown device should fail");
        assert!(err.contains("Unknown device"));
    }

    #[test]
    fn test_apply_batch_size_override() {
        let mut cfg = TrainConfig::d20();
        apply_batch_size_override(&mut cfg, Some(8)).expect("batch size override");
        assert_eq!(cfg.batch_size, 8);

        let err = apply_batch_size_override(&mut cfg, Some(0)).expect_err("zero should fail");
        assert_eq!(err, "error: --batch-size must be > 0");
    }

    #[test]
    fn test_effective_seq_len_defaults_and_validation() {
        let mut cfg = TrainConfig::d20();
        cfg.max_seq_len = 128;
        assert_eq!(effective_seq_len(None, &cfg).expect("default seq"), 64);
        assert_eq!(effective_seq_len(Some(32), &cfg).expect("explicit seq"), 32);

        cfg.max_seq_len = 0;
        let err = effective_seq_len(None, &cfg).expect_err("zero seq should fail");
        assert_eq!(err, "error: --seq-len must be > 0");
    }

    #[test]
    fn test_resolve_dataset_spec_paths() {
        assert_eq!(
            resolve_dataset_spec("synthetic", None).expect("synthetic dataset"),
            DatasetSpec::Synthetic
        );
        assert_eq!(
            resolve_dataset_spec("tokens", Some("data.bin".to_string()))
                .expect("tokens dataset"),
            DatasetSpec::Tokens("data.bin".to_string())
        );
        let missing = resolve_dataset_spec("tokens", None).expect_err("missing data path");
        assert_eq!(
            missing,
            "error: --data-path is required when --dataset=tokens"
        );
        let unknown = resolve_dataset_spec("other", None).expect_err("unknown dataset");
        assert_eq!(
            unknown,
            "Unknown dataset: other. Use 'synthetic' or 'tokens'."
        );
    }
}
