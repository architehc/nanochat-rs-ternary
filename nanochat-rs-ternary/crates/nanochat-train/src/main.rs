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
        "d20-loop" | "d20_loop" => Some(TrainConfig::d20_loop()),
        "nano-125m" | "nano_125m" => Some(TrainConfig::nano_125m()),
        "nano-1b" | "nano_1b" => Some(TrainConfig::nano_1b()),
        "medium-3b" | "medium_3b" => Some(TrainConfig::medium_3b()),
        "large-7b" | "large_7b" => Some(TrainConfig::large_7b()),
        "large-7b-6day" | "large_7b_6day" => Some(TrainConfig::large_7b_6day()),
        "tiny-cpu" | "tiny_cpu" => Some(TrainConfig::tiny_cpu()),
        "test-8bit" | "test_8bit" => Some(TrainConfig::test_8bit()),
        "d20-wave" | "d20_wave" | "d20-wavefield" | "d20_wavefield" => {
            Some(TrainConfig::d20_wavefield())
        }
        "nano-125m-wave" | "nano_125m_wave" | "nano-125m-wavefield" | "nano_125m_wavefield" => {
            Some(TrainConfig::nano_125m_wavefield())
        }
        "d20-wave-fwht" | "d20_wave_fwht" | "d20-wavefield-fwht" | "d20_wavefield_fwht" => {
            Some(TrainConfig::d20_wavefield_fwht())
        }
        "nano-125m-wave-fwht" | "nano_125m_wave_fwht" => {
            Some(TrainConfig::nano_125m_wavefield_fwht())
        }
        "nano-125m-wave-haar" | "nano_125m_wave_haar" => {
            Some(TrainConfig::nano_125m_wavefield_haar())
        }
        "nano-125m-hybrid" | "nano_125m_hybrid" => Some(TrainConfig::nano_125m_hybrid()),
        "nano-500m-wave-haar" | "nano_500m_wave_haar" | "nano-500m" | "nano_500m" => {
            Some(TrainConfig::nano_500m_wave_haar())
        }
        "nano-500m-baseline" | "nano_500m_baseline" => {
            Some(TrainConfig::nano_500m_baseline())
        }
        "nano-275m-wave-haar" | "nano_275m_wave_haar" | "nano-275m" | "nano_275m" => {
            Some(TrainConfig::nano_275m_wave_haar())
        }
        "nano-275m-baseline" | "nano_275m_baseline" => {
            Some(TrainConfig::nano_275m_baseline())
        }
        "nano-275m-wave-engram-loop" | "nano_275m_wave_engram_loop"
        | "nano-275m-engram-loop" | "nano_275m_engram_loop" => {
            Some(TrainConfig::nano_275m_wave_engram_loop())
        }
        "nano-275m-loop-only" | "nano_275m_loop_only" | "nano-275m-loop" | "nano_275m_loop" => {
            Some(TrainConfig::nano_275m_loop_only())
        }
        "nano-275m-engram-only" | "nano_275m_engram_only" | "nano-275m-engram" | "nano_275m_engram" => {
            Some(TrainConfig::nano_275m_engram_only())
        }
        "nano-275m-engram-mtp" | "nano_275m_engram_mtp" => {
            Some(TrainConfig::nano_275m_engram_mtp())
        }
        "nano-275m-engram-v4" | "nano_275m_engram_v4" => {
            Some(TrainConfig::nano_275m_engram_v4())
        }
        "nano-275m-engram-v5" | "nano_275m_engram_v5" => {
            Some(TrainConfig::nano_275m_engram_v5())
        }
        "nano-275m-engram-v6" | "nano_275m_engram_v6" => {
            Some(TrainConfig::nano_275m_engram_v6())
        }
        "nano-275m-engram-v7" | "nano_275m_engram_v7" => {
            Some(TrainConfig::nano_275m_engram_v7())
        }
        "nano-275m-engram-wide" | "nano_275m_engram_wide" => {
            Some(TrainConfig::nano_275m_engram_wide())
        }
        "nano-275m-haar-v3" | "nano_275m_haar_v3" => {
            Some(TrainConfig::nano_275m_haar_v3())
        }
        _ => None,
    }
}

fn resolve_device(device: &str) -> Result<candle_core::Device, String> {
    match device {
        "cpu" => Ok(candle_core::Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => candle_core::Device::new_cuda(0)
            .map_err(|e| format!("Failed to initialize CUDA device 0: {}", e)),
        #[cfg(feature = "cuda")]
        s if s.starts_with("cuda:") => {
            let id: usize = s.strip_prefix("cuda:").unwrap().parse()
                .map_err(|_| format!("Invalid CUDA device id in '{}'", s))?;
            candle_core::Device::new_cuda(id)
                .map_err(|e| format!("Failed to initialize CUDA device {}: {}", id, e))
        }
        other => Err(format!("Unknown device: {}. Use 'cpu', 'cuda', or 'cuda:N'.", other)),
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

        /// Override total training steps from config
        #[arg(long)]
        total_steps: Option<usize>,

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

    /// Generate text from a trained checkpoint
    Generate {
        /// Checkpoint directory
        #[arg(long)]
        checkpoint: String,

        /// Path to tokenizer.json
        #[arg(long)]
        tokenizer: String,

        /// Prompt text
        #[arg(long, default_value = "fn main() {")]
        prompt: String,

        /// Read prompt from file instead
        #[arg(long)]
        prompt_file: Option<String>,

        /// Max tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value = "0.8")]
        temperature: f64,

        /// Top-k sampling (0 = disabled)
        #[arg(long, default_value = "50")]
        top_k: usize,

        /// Device (cpu or cuda)
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Bypass wavefield attention layers (use for wavefield models).
        /// Wavefield layers are bidirectional and produce garbage during
        /// autoregressive generation. This flag disables them, keeping
        /// only the causal standard-attention layers active.
        #[arg(long, default_value = "false")]
        bypass_wavefield: bool,
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
            total_steps,
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

            // Trim all CUDA memory pools before device init.
            // Prevents OOM when GPU 0 is full and we target GPU N>0.
            nanochat_train::train::trim_all_cuda_memory_pools();

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
                        "Unknown config: {}. Use d20, d20-loop, d20-mtp, d20-e3-full, d20-e3-fp4, d20-wave, nano-125m, nano-125m-wave, nano-125m-hybrid, nano-275m-wave-haar, nano-275m-baseline, nano-275m-wave-engram-loop, nano-275m-loop-only, nano-275m-engram-only, nano-275m-haar-v3, nano-500m-wave-haar, nano-1b, medium-3b, large-7b, large-7b-6day, tiny-cpu, or test-8bit.",
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
                let mut trainer = nanochat_train::train::Trainer::from_checkpoint(ckpt_dir, device)?;
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
                // Sync override back to trainer so the actual training loop uses it
                trainer.config.batch_size = cfg.batch_size;
                tracing::info!("Using checkpoint config (vocab_size={}, dim={})", cfg.vocab_size, cfg.dim);
                trainer
            } else {
                nanochat_train::train::Trainer::new(cfg.clone(), device)?
            };

            // Apply --total-steps override.
            // Updates BOTH the stop point AND config.total_steps so the LR schedule
            // extends properly (warmup stays fixed, stable phase extends, decay shifts).
            if let Some(ts) = total_steps {
                trainer.stop_at_step = Some(ts);
                let old_total = trainer.config.total_steps;
                trainer.config.total_steps = ts;
                cfg.total_steps = ts;
                tracing::info!("Extended total_steps: {} → {} (LR schedule updated)", old_total, ts);
            }

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

        Commands::Generate {
            checkpoint,
            tokenizer,
            prompt,
            prompt_file,
            max_tokens,
            temperature,
            top_k,
            device,
            bypass_wavefield,
        } => {
            let prompt = if let Some(pf) = prompt_file {
                std::fs::read_to_string(&pf)?
            } else {
                prompt
            };
            let device = match resolve_device(&device) {
                Ok(d) => d,
                Err(msg) => { eprintln!("{}", msg); std::process::exit(1); }
            };

            // Load tokenizer
            let tok = nanochat_train::data::tokenizer::NanochatTokenizer::from_file(
                std::path::Path::new(&tokenizer),
            ).map_err(|e| -> Box<dyn std::error::Error> { e })?;

            // Load model from checkpoint
            tracing::info!("Loading checkpoint from {}...", checkpoint);
            let meta_json = std::fs::read_to_string(format!("{}/meta.json", checkpoint))?;
            let meta: nanochat_train::checkpoint::CheckpointMeta =
                serde_json::from_str(&meta_json)?;

            let varmap = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
            let model = nanochat_train::model::NanochatTrainModel::new(&meta.config, vb)?;

            // Load weights onto the correct device
            let tensors = candle_core::safetensors::load(
                format!("{}/model.safetensors", checkpoint), &device
            )?;
            let mut data = varmap.data().lock().unwrap();
            let mut loaded = 0usize;
            let mut missing = Vec::new();
            for (name, var) in data.iter_mut() {
                if let Some(t) = tensors.get(name) {
                    var.set(t).map_err(|e| format!("Failed to set {}: {}", name, e))?;
                    loaded += 1;
                } else {
                    missing.push(name.clone());
                }
            }
            let total_model = data.len();
            let total_ckpt = tensors.len();
            drop(data);

            tracing::info!("Loaded {}/{} model params from checkpoint ({} tensors in file)", loaded, total_model, total_ckpt);
            if !missing.is_empty() {
                tracing::warn!("MISSING {} weights (still random!): {:?}",
                    missing.len(),
                    if missing.len() <= 10 { &missing[..] } else { &missing[..10] }
                );
            }

            tracing::info!("Model loaded: {} params", model.param_count());

            // Auto-detect wavefield and enable bypass if needed
            let use_causal = bypass_wavefield || meta.config.use_wave_field;
            if use_causal && meta.config.use_wave_field {
                let n_wf = model.blocks.iter().filter(|b| b.is_wavefield()).count();
                let n_std = model.blocks.len() - n_wf;
                tracing::info!(
                    "Wavefield model detected ({} wavefield + {} standard layers). \
                     Bypassing wavefield attention for causal generation.",
                    n_wf, n_std
                );
            }

            // Encode prompt
            let mut token_ids: Vec<u32> = tok.encode(&prompt)
                .map_err(|e| -> Box<dyn std::error::Error> { e })?;
            tracing::info!("Prompt: {} ({} tokens)", prompt.chars().take(60).collect::<String>(), token_ids.len());

            // Context padding: wavefield layers are bidirectional and need a
            // well-populated field (~512 tokens) to produce good outputs. With short
            // prompts, the wavefield field is too sparse. Pad with training data prefix.
            let train_seq_len = meta.config.max_seq_len;
            let _prefix_len = if false && token_ids.len() < train_seq_len && meta.config.use_wave_field {
                let pad_needed = train_seq_len - token_ids.len();
                let data_path = "data/rust_v2_prepared/tokens.bin";
                if std::path::Path::new(data_path).exists() {
                    let token_data = std::fs::read(data_path)?;
                    let all_tokens: Vec<u32> = token_data.chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    // Use a random offset for diverse context
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    let max_offset = all_tokens.len().saturating_sub(pad_needed + 1);
                    let offset = if max_offset > 0 { rng.gen_range(0..max_offset) } else { 0 };
                    let pad_tokens: Vec<u32> = all_tokens[offset..offset + pad_needed].to_vec();
                    tracing::info!("Wavefield context padding: {} prefix tokens (total: {})",
                        pad_needed, train_seq_len);
                    let mut padded = pad_tokens;
                    padded.extend_from_slice(&token_ids);
                    token_ids = padded;
                    pad_needed
                } else {
                    0
                }
            } else {
                0
            };
            let prompt_len = token_ids.len();

            // Print prompt (only the user's actual prompt, not padding)
            print!("{}", prompt);
            use std::io::Write;
            std::io::stdout().flush()?;

            // Autoregressive generation with sliding window
            let max_ctx = train_seq_len; // cap context to training window size
            for _ in 0..max_tokens {
                // Use last max_ctx tokens as context window
                let start = if token_ids.len() > max_ctx { token_ids.len() - max_ctx } else { 0 };
                let input = candle_core::Tensor::new(
                    &token_ids[start..],
                    &device,
                )?.unsqueeze(0)?; // [1, min(seq, max_ctx)]

                let hidden = if use_causal {
                    model.forward_hidden_only_causal(&input)?
                } else {
                    model.forward_hidden_only(&input)?
                };
                let logits = model.project_hidden_to_logits(&hidden)?;

                // Get logits for last position: [1, seq, vocab] -> [vocab]
                let last_logits = logits.squeeze(0)?; // [seq, vocab]
                let seq_len = last_logits.dim(0)?;
                let last_logits = last_logits.get(seq_len - 1)?; // [vocab]
                let mut logits_vec: Vec<f32> = last_logits.to_vec1()?;

                // Sample next token
                let next_token = if temperature <= 0.0 || temperature < 1e-8 {
                    // Greedy (NaN-safe: total_cmp treats NaN as less than all values)
                    logits_vec
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                } else {
                    // Temperature + top-k sampling
                    for v in logits_vec.iter_mut() {
                        *v /= temperature as f32;
                    }

                    // Top-k: zero out everything outside top-k
                    if top_k > 0 && top_k < logits_vec.len() {
                        let mut indexed: Vec<(usize, f32)> =
                            logits_vec.iter().copied().enumerate().collect();
                        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
                        let threshold = indexed[top_k].1;
                        for v in logits_vec.iter_mut() {
                            if *v < threshold {
                                *v = f32::NEG_INFINITY;
                            }
                        }
                    }

                    // Softmax
                    let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = logits_vec.iter().map(|v| (v - max_val).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

                    // Sample from distribution
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    let r: f32 = rng.gen();
                    let mut cumsum = 0.0f32;
                    let mut chosen = (probs.len() - 1) as u32; // fallback to last token
                    for (i, &p) in probs.iter().enumerate() {
                        cumsum += p;
                        if r < cumsum {
                            chosen = i as u32;
                            break;
                        }
                    }
                    chosen
                };

                token_ids.push(next_token);

                // Decode and print just the new token
                let text_result: std::result::Result<String, _> = tok.decode(&[next_token]);
                if let Ok(text) = text_result {
                    print!("{}", text);
                    std::io::stdout().flush()?;
                }
            }

            println!();
            tracing::info!(
                "Generated {} tokens (prompt={}, new={})",
                token_ids.len(),
                prompt_len,
                token_ids.len() - prompt_len
            );
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
            "d20-loop",
            "d20_loop",
            "nano-125m",
            "nano_125m",
            "nano-1b",
            "nano_1b",
            "medium-3b",
            "medium_3b",
            "large-7b",
            "large_7b",
            "large-7b-6day",
            "large_7b_6day",
            "tiny-cpu",
            "tiny_cpu",
            "test-8bit",
            "test_8bit",
            "d20-wave",
            "d20_wave",
            "d20-wavefield",
            "d20_wavefield",
            "nano-125m-wave",
            "nano_125m_wave",
            "nano-125m-wavefield",
            "nano_125m_wavefield",
            "nano-125m-hybrid",
            "nano_125m_hybrid",
            "nano-500m-wave-haar",
            "nano_500m_wave_haar",
            "nano-500m",
            "nano_500m",
            "nano-275m-wave-haar",
            "nano_275m_wave_haar",
            "nano-275m",
            "nano_275m",
            "nano-275m-baseline",
            "nano_275m_baseline",
            "nano-275m-wave-engram-loop",
            "nano_275m_wave_engram_loop",
            "nano-275m-engram-loop",
            "nano_275m_engram_loop",
            "nano-275m-loop-only",
            "nano_275m_loop_only",
            "nano-275m-loop",
            "nano_275m_loop",
            "nano-275m-engram-only",
            "nano_275m_engram_only",
            "nano-275m-engram",
            "nano_275m_engram",
            "nano-275m-engram-mtp",
            "nano_275m_engram_mtp",
            "nano-275m-haar-v3",
            "nano_275m_haar_v3",
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
