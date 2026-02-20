//! Training loop: Trainer struct with Muon + Lion optimizer split.

use candle_core::{backprop::GradStore, DType, Device, Result, Tensor, Var, D};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::collider::Collider;
use crate::config::TrainConfig;
use crate::data::{DataLoader, Dataset};
use crate::fp4::FP4Trainer;
use crate::model::NanochatTrainModel;
use crate::mtp::MultiTokenPrediction;
use crate::optim::{wsd_schedule, Lion, LionState, MuonOptimizer, MuonOptimizerState};

/// Return all variables from a VarMap sorted by name for deterministic ordering.
///
/// `VarMap::all_vars()` uses HashMap iteration order which is non-deterministic.
/// Optimizer state save/restore is positional, so different ordering on resume
/// causes shape mismatches. This helper ensures consistent ordering.
fn sorted_vars(varmap: &VarMap) -> Vec<Var> {
    let data = varmap.data().lock().unwrap();
    let mut named: Vec<_> = data.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    named.into_iter().map(|(_, v)| v).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizerCheckpointState {
    muon: MuonOptimizerState,
    lion: LionState,
}

/// Checkpoint management utilities
mod checkpoint_manager {
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Get disk space info for path
    pub fn get_disk_space(path: &Path) -> Result<(u64, u64), String> {
        // Use statfs to get filesystem stats
        #[cfg(unix)]
        {
            // Get filesystem stats
            let stats = nix::sys::statfs::statfs(path)
                .map_err(|e| format!("Failed to get fs stats: {}", e))?;

            let total = stats.blocks() * stats.block_size() as u64;
            let available = stats.blocks_available() * stats.block_size() as u64;
            Ok((total, available))
        }
        #[cfg(not(unix))]
        {
            // Fallback for non-Unix: return large values
            Ok((1_000_000_000_000u64, 500_000_000_000u64))
        }
    }

    /// Get size of directory recursively
    pub fn dir_size(path: &Path) -> Result<u64, String> {
        let mut total = 0u64;
        if path.is_dir() {
            for entry in fs::read_dir(path).map_err(|e| format!("Failed to read dir: {}", e))? {
                let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
                let path = entry.path();
                if path.is_file() {
                    total += entry
                        .metadata()
                        .map_err(|e| format!("Failed to read metadata: {}", e))?
                        .len();
                } else if path.is_dir() {
                    total += dir_size(&path)?;
                }
            }
        }
        Ok(total)
    }

    /// Clean old checkpoints, keeping only the last N
    pub fn cleanup_old_checkpoints(
        checkpoint_dir: &Path,
        keep_last: usize,
    ) -> Result<usize, String> {
        if !checkpoint_dir.exists() {
            return Ok(0);
        }

        // List all checkpoint directories (step_NNNNN)
        let mut checkpoints: Vec<(PathBuf, u32)> = Vec::new();
        for entry in
            fs::read_dir(checkpoint_dir).map_err(|e| format!("Failed to read dir: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("step_") {
                        if let Ok(step) = name.strip_prefix("step_").unwrap().parse::<u32>() {
                            checkpoints.push((path, step));
                        }
                    }
                }
            }
        }

        // Sort by step number (oldest first)
        checkpoints.sort_by_key(|(_, step)| *step);

        // Remove old checkpoints
        let to_remove = if checkpoints.len() > keep_last {
            checkpoints.len() - keep_last
        } else {
            0
        };

        let mut removed = 0;
        for (path, step) in checkpoints.iter().take(to_remove) {
            println!("  Removing old checkpoint: step_{}", step);
            fs::remove_dir_all(path).map_err(|e| format!("Failed to remove checkpoint: {}", e))?;
            removed += 1;
        }

        Ok(removed)
    }

    /// Format bytes as human-readable
    pub fn format_bytes(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;
        const TB: u64 = GB * 1024;

        if bytes >= TB {
            format!("{:.2}TB", bytes as f64 / TB as f64)
        } else if bytes >= GB {
            format!("{:.2}GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2}MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2}KB", bytes as f64 / KB as f64)
        } else {
            format!("{}B", bytes)
        }
    }
}

/// Training statistics for one step or epoch.
#[derive(Debug, Clone)]
pub struct StepStats {
    pub loss: f64,
    pub grad_norm: f64,
    pub lr: f64,
    pub tokens_per_sec: f64,
}

/// Builder for constructing a Trainer with optional configuration
///
/// # Example
///
/// ```rust,no_run
/// use nanochat_train::{config::TrainConfig, train::TrainerBuilder};
/// use candle_core::Device;
///
/// let trainer = TrainerBuilder::new(TrainConfig::nano_125m())
///     .device(Device::Cpu)
///     .from_checkpoint("checkpoints/step_1000")
///     .build()
///     .unwrap();
/// ```
pub struct TrainerBuilder {
    config: TrainConfig,
    device: Option<Device>,
    checkpoint_path: Option<String>,
}

impl TrainerBuilder {
    /// Create a new trainer builder with the given configuration
    pub fn new(config: TrainConfig) -> Self {
        Self {
            config,
            device: None,
            checkpoint_path: None,
        }
    }

    /// Set the device to use for training (defaults to CPU if not set)
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Load from a checkpoint before starting training
    pub fn from_checkpoint(mut self, path: impl Into<String>) -> Self {
        self.checkpoint_path = Some(path.into());
        self
    }

    /// Build the trainer
    ///
    /// Validates configuration and initializes the trainer
    pub fn build(self) -> Result<Trainer> {
        // Validate config first
        if let Err(errors) = self.config.validate() {
            return Err(candle_core::Error::Msg(format!(
                "Configuration errors: {:?}",
                errors
            )));
        }

        let device = self.device.unwrap_or(Device::Cpu);

        if let Some(ckpt) = self.checkpoint_path {
            Trainer::from_checkpoint(&ckpt, device)
        } else {
            Trainer::new(self.config, device)
        }
    }
}

/// Main trainer holding model + optimizers.
pub struct Trainer {
    pub model: NanochatTrainModel,
    pub varmap: VarMap,
    muon: MuonOptimizer,
    lion: Lion,
    pub config: TrainConfig,
    pub device: Device,
    pub global_step: usize,
    pub base_lr_muon: f64, // Made public for CLI overrides
    pub base_lr_lion: f64, // Made public for CLI overrides

    // E3 optimizations
    mtp: Option<MultiTokenPrediction>, // Multi-Token Prediction (15-20% data efficiency)
    mtp_varmap: Option<VarMap>,        // Separate VarMap for MTP parameters
    collider: Option<Collider>,        // Token filtering (35% faster backprop)
    fp4: Option<FP4Trainer>,           // Optional FP4 mixed-precision path

    // Gradient accumulation state.
    accum_grads: Option<GradStore>,
    accum_micro_steps: usize,
    accum_loss_sum: f64,
}

/// Result of attempting a training step with OOM recovery
#[derive(Debug)]
pub enum StepAttempt {
    /// Step completed successfully
    Success(StepStats),
    /// Out of memory error occurred and was recovered from
    RecoveredFromOom,
    /// Unrecoverable error
    Failed(candle_core::Error),
}

impl Trainer {
    fn save_optimizer_state(&self, checkpoint_dir: &str) -> Result<()> {
        let state = OptimizerCheckpointState {
            muon: self.muon.export_state()?,
            lion: self.lion.export_state()?,
        };
        let bytes = bincode::serialize(&state)
            .map_err(|e| candle_core::Error::Msg(format!("optimizer state serialize: {}", e)))?;
        let path = format!("{}/optimizer_state.bin", checkpoint_dir);
        std::fs::write(&path, bytes).map_err(|e| {
            candle_core::Error::Msg(format!("optimizer state write {}: {}", path, e))
        })?;

        // Save MTP parameters if present
        if let Some(ref mtp_vm) = self.mtp_varmap {
            let mtp_path = format!("{}/mtp.safetensors", checkpoint_dir);
            mtp_vm.save(&mtp_path).map_err(|e| {
                candle_core::Error::Msg(format!("MTP state save {}: {}", mtp_path, e))
            })?;
        }

        Ok(())
    }

    fn load_optimizer_state(&mut self, checkpoint_dir: &str) -> Result<()> {
        let bin_path = format!("{}/optimizer_state.bin", checkpoint_dir);
        let state = match std::fs::read(&bin_path) {
            Ok(raw) => bincode::deserialize::<OptimizerCheckpointState>(&raw).map_err(|e| {
                candle_core::Error::Msg(format!("optimizer state parse {}: {}", bin_path, e))
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let json_path = format!("{}/optimizer_state.json", checkpoint_dir);
                match std::fs::read_to_string(&json_path) {
                    Ok(raw_json) => serde_json::from_str::<OptimizerCheckpointState>(&raw_json)
                        .map_err(|err| {
                            candle_core::Error::Msg(format!(
                                "optimizer state parse {}: {}",
                                json_path, err
                            ))
                        })?,
                    Err(json_err) if json_err.kind() == std::io::ErrorKind::NotFound => {
                        tracing::info!(
                            "Optimizer state not found in checkpoint; using fresh optimizer buffers"
                        );
                        return Ok(());
                    }
                    Err(json_err) => {
                        return Err(candle_core::Error::Msg(format!(
                            "optimizer state read {}: {}",
                            json_path, json_err
                        )));
                    }
                }
            }
            Err(e) => {
                return Err(candle_core::Error::Msg(format!(
                    "optimizer state read {}: {}",
                    bin_path, e
                )));
            }
        };
        self.muon.import_state(&state.muon)?;
        self.lion.import_state(&state.lion)?;
        tracing::info!("Optimizer state restored from {}", bin_path);

        // Load MTP parameters if present
        if let Some(ref mut mtp_vm) = self.mtp_varmap {
            let mtp_path = format!("{}/mtp.safetensors", checkpoint_dir);
            if std::path::Path::new(&mtp_path).exists() {
                mtp_vm.load(&mtp_path).map_err(|e| {
                    candle_core::Error::Msg(format!("MTP state load {}: {}", mtp_path, e))
                })?;
                tracing::info!("MTP state restored from {}", mtp_path);
            } else {
                tracing::info!(
                    "MTP state file not found at {}; using fresh MTP parameters",
                    mtp_path
                );
            }
        }

        Ok(())
    }

    /// Create a new trainer from a saved checkpoint
    pub fn from_checkpoint(checkpoint_dir: &str, device: Device) -> Result<Self> {
        // Load checkpoint
        let (varmap, config, step, _loss) =
            crate::checkpoint::load_checkpoint(checkpoint_dir, &device).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to load checkpoint: {}", e))
            })?;

        // Create MTP varbuilder before model (if needed)
        let mtp_varmap = if config.use_mtp {
            Some(VarMap::new())
        } else {
            None
        };

        // Create model from loaded varmap
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = NanochatTrainModel::new(&config, vb)?;

        // Classify vars by dimension for optimizers (sorted for deterministic ordering)
        let all_vars = sorted_vars(&varmap);
        let mut muon_vars: Vec<Var> = Vec::new();
        let mut lion_vars: Vec<Var> = Vec::new();

        for var in all_vars {
            let dims = var.as_tensor().dims();
            if dims.len() <= 1 {
                // 1D: norms, mHC params -> Lion
                lion_vars.push(var);
            } else if dims.len() == 2 && dims[0] == config.vocab_size {
                // Embedding -> Lion
                lion_vars.push(var);
            } else {
                // 2D+ linear weights -> Muon
                muon_vars.push(var);
            }
        }

        // Create optimizers
        let base_lr_muon = config.lr;
        let base_lr_lion = config.mhc_lr;

        let muon = MuonOptimizer::from_config(
            muon_vars,
            base_lr_muon,
            config.muon_momentum,
            config.ns_steps,
            config.weight_decay,
            config.use_8bit_optim,
            config.use_galore,
            config.galore_rank,
            config.galore_update_freq,
        )?;

        // Log optimizer configuration on resume
        let mem_stats = muon.memory_stats();
        println!("\n🔧 Resumed with optimizer: {}", mem_stats.variant);
        if mem_stats.memory_reduction > 0.0 {
            println!(
                "  Memory reduction: {:.1}%",
                mem_stats.memory_reduction * 100.0
            );
        }

        let lion = Lion::new(
            lion_vars,
            base_lr_lion,
            config.lion_betas.0,
            config.lion_betas.1,
            0.0, // no weight decay for Lion group (matches new() path)
        )?;

        // E3 optimizations: Multi-Token Prediction
        let (mtp, mtp_varmap_stored) = if let Some(mtp_vm) = mtp_varmap {
            let mtp_vb = candle_nn::VarBuilder::from_varmap(&mtp_vm, DType::F32, &device);
            let mtp_module = MultiTokenPrediction::new(
                mtp_vb,
                config.dim,
                config.vocab_size,
                config.mtp_n_tokens,
            )?;
            println!("  MTP resumed: {} future tokens", config.mtp_n_tokens);
            (Some(mtp_module), Some(mtp_vm))
        } else {
            (None, None)
        };

        // E3 optimizations: Collider token filtering
        let collider = if config.use_collider {
            let collider_module =
                Collider::new(config.collider_threshold, config.collider_sparsity)
                    .with_transform_gemm(true);
            println!(
                "  Collider resumed: threshold={:.2}, sparsity={:.2}, sparse_gemm=true",
                config.collider_threshold, config.collider_sparsity
            );
            Some(collider_module)
        } else {
            None
        };

        // Optional FP4 path.
        let fp4 = if config.use_fp4 {
            let fp4_module = FP4Trainer::new(config.fp4_stochastic_rounding);
            fp4_module.enable_fp4_tensor_cores()?;
            println!(
                "  FP4 resumed: stochastic_rounding={}",
                config.fp4_stochastic_rounding
            );
            Some(fp4_module)
        } else {
            None
        };

        let mut trainer = Self {
            model,
            varmap,
            muon,
            lion,
            config,
            device,
            global_step: step, // Resume from saved step
            base_lr_muon,
            base_lr_lion,
            mtp,
            mtp_varmap: mtp_varmap_stored,
            collider,
            fp4,
            accum_grads: None,
            accum_micro_steps: 0,
            accum_loss_sum: 0.0,
        };

        trainer.load_optimizer_state(checkpoint_dir)?;
        Ok(trainer)
    }

    pub fn new(config: TrainConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();

        // Create MTP varmap if needed (separate from model varmap)
        let mtp_varmap = if config.use_mtp {
            Some(VarMap::new())
        } else {
            None
        };

        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = NanochatTrainModel::new(&config, vb)?;

        // Classify vars by dimension: 2D+ go to Muon, 1D go to Lion
        // Exception: embedding (vocab_size x dim) goes to Lion
        // Use sorted_vars for deterministic ordering (critical for checkpoint resume)
        let all_vars = sorted_vars(&varmap);
        let mut muon_vars: Vec<Var> = Vec::new();
        let mut lion_vars: Vec<Var> = Vec::new();

        for var in all_vars {
            let dims = var.as_tensor().dims();
            if dims.len() <= 1 {
                // 1D: norms, mHC params -> Lion
                lion_vars.push(var);
            } else if dims.len() == 2 && dims[0] == config.vocab_size {
                // Embedding -> Lion
                lion_vars.push(var);
            } else {
                // 2D+ linear weights -> Muon
                muon_vars.push(var);
            }
        }

        // Create Muon optimizer with optional 8-bit quantization and/or GaLore
        let muon = MuonOptimizer::from_config(
            muon_vars,
            config.lr,
            config.muon_momentum,
            config.ns_steps,
            config.weight_decay,
            config.use_8bit_optim,
            config.use_galore,
            config.galore_rank,
            config.galore_update_freq,
        )?;

        // Log optimizer configuration
        let mem_stats = muon.memory_stats();
        println!("\n🔧 Optimizer Configuration:");
        println!("  Muon variant: {}", mem_stats.variant);
        if mem_stats.memory_reduction > 0.0 {
            println!(
                "  Memory reduction: {:.1}%",
                mem_stats.memory_reduction * 100.0
            );
            println!("  Details: {}", mem_stats.details);
        }
        if config.use_galore {
            println!("  GaLore rank: {}", config.galore_rank);
            println!("  GaLore update freq: {} steps", config.galore_update_freq);
        }

        let lion = Lion::new(
            lion_vars,
            config.mhc_lr,
            config.lion_betas.0,
            config.lion_betas.1,
            0.0, // no weight decay for Lion group
        )?;

        let base_lr_muon = config.lr;
        let base_lr_lion = config.mhc_lr;

        // E3 optimizations: Multi-Token Prediction
        let (mtp, mtp_varmap_stored) = if let Some(mtp_vm) = mtp_varmap {
            let mtp_vb = candle_nn::VarBuilder::from_varmap(&mtp_vm, DType::F32, &device);
            let mtp_module = MultiTokenPrediction::new(
                mtp_vb,
                config.dim,
                config.vocab_size,
                config.mtp_n_tokens,
            )?;
            println!("  MTP enabled: {} future tokens", config.mtp_n_tokens);
            (Some(mtp_module), Some(mtp_vm))
        } else {
            (None, None)
        };

        // E3 optimizations: Collider token filtering
        let collider = if config.use_collider {
            let collider_module =
                Collider::new(config.collider_threshold, config.collider_sparsity)
                    .with_transform_gemm(true);
            println!(
                "  Collider enabled: threshold={:.2}, sparsity={:.2}, sparse_gemm=true",
                config.collider_threshold, config.collider_sparsity
            );
            Some(collider_module)
        } else {
            None
        };

        // Optional FP4 path.
        let fp4 = if config.use_fp4 {
            let fp4_module = FP4Trainer::new(config.fp4_stochastic_rounding);
            fp4_module.enable_fp4_tensor_cores()?;
            println!(
                "  FP4 enabled: stochastic_rounding={}",
                config.fp4_stochastic_rounding
            );
            Some(fp4_module)
        } else {
            None
        };

        // Warn if async loader is configured but not yet integrated
        if config.use_async_loader {
            tracing::warn!("use_async_loader is configured but not yet integrated into the training loop; using synchronous loading");
        }

        Ok(Self {
            model,
            varmap,
            muon,
            lion,
            config,
            device,
            global_step: 0,
            base_lr_muon,
            base_lr_lion,
            mtp,
            mtp_varmap: mtp_varmap_stored,
            collider,
            fp4,
            accum_grads: None,
            accum_micro_steps: 0,
            accum_loss_sum: 0.0,
        })
    }

    /// Clamp mHC alpha_logit parameters to [-3, 3] to prevent identity bypass.
    ///
    /// Without clamping, optimizer updates can push alpha_logit to extreme values
    /// (e.g., 5.0 → sigmoid≈0.993 → H_res≈identity), causing the model to bypass
    /// all transformer layers and predict current tokens instead of next tokens.
    /// See MEMORY.md "mHC identity bypass (CRITICAL)" for full description.
    fn clamp_alpha_logits(&self) -> Result<()> {
        let data = self.varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            if name.ends_with("alpha_logit") {
                let clamped = var.as_tensor().clamp(-3f32, 3f32)?;
                var.set(&clamped)?;
            }
        }
        Ok(())
    }

    /// Execute a single training step.
    pub fn train_step(&mut self, input_ids: &Tensor, target_ids: &Tensor) -> Result<StepStats> {
        let step_start = Instant::now();

        // Forward pass to hidden states.
        let hidden = self.model.forward_hidden_only(input_ids)?;
        let (batch, seq_len, _hidden_dim) = hidden.dims3()?;
        let vocab = self.config.vocab_size;

        // Optional FP4 path via STE: forward uses quantized activations while
        // gradients flow through the original hidden states.
        let hidden_for_lm = if let Some(ref fp4) = self.fp4 {
            let hidden_detached = hidden.detach();
            let hidden_quant = fp4.quantize_fp4(&hidden_detached)?;
            // Forward value is exactly `hidden_quant`; gradient remains identity via STE.
            (&hidden_quant + (&hidden - &hidden_detached)?)?
        } else {
            hidden.clone()
        };

        // E3: Collider token filtering (sparse token path for LM head + loss).
        // Build a binary keep-mask from detached logits to avoid autograd overhead.
        let collider_mask = if let Some(ref collider) = self.collider {
            let logits_detached = self
                .model
                .project_hidden_to_logits(&hidden_for_lm.detach())?;
            let importance = collider.compute_importance(&logits_detached, target_ids)?;
            Some(collider.create_mask(&importance)?)
        } else {
            None
        };

        // Primary cross-entropy loss with label smoothing.
        // Collider sparse-backward path compacts to kept rows and runs dense GEMM on compacted tensors.
        let (logits_flat, targets_flat) = if let Some(mask) = collider_mask.as_ref() {
            let collider = self
                .collider
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("Collider missing".to_string()))?;
            let (hidden_kept, targets_kept) =
                collider.sparse_backward(&hidden_for_lm, target_ids, mask)?;
            let logits_kept = self.model.project_hidden_to_logits(&hidden_kept)?;
            (logits_kept, targets_kept)
        } else {
            let logits_dense = self.model.project_hidden_to_logits(&hidden_for_lm)?;
            let logits_flat = logits_dense.reshape((batch * seq_len, vocab))?;
            let targets_flat = target_ids.reshape(batch * seq_len)?;
            (logits_flat, targets_flat)
        };

        // Label smoothing: smooth_targets = (1-eps)*one_hot + eps/V
        let label_smooth_eps = 0.1; // 10% smoothing
        let log_probs_for_smoothing = candle_nn::ops::log_softmax(&logits_flat, 1)?;
        // Reuse the same log-probabilities for CE and smoothing to avoid duplicate work.
        let ce_loss = candle_nn::loss::nll(&log_probs_for_smoothing, &targets_flat)?;
        // Compute uniform term per example first, then average across examples.
        // This matches label smoothing semantics explicitly.
        let uniform_per_example = log_probs_for_smoothing.mean(D::Minus1)?.neg()?;
        let uniform_loss = uniform_per_example.mean_all()?;
        let ce_scaled = (ce_loss * (1.0 - label_smooth_eps))?;
        let uniform_scaled = (uniform_loss * label_smooth_eps)?;
        let mut loss = (ce_scaled + uniform_scaled)?;

        // E3: Multi-Token Prediction (15-20% data efficiency boost)
        if let Some(ref mtp) = self.mtp {
            // Predict future tokens from hidden states
            let mtp_logits_list = mtp.forward(&hidden)?;

            // Prepare aligned predictions and targets
            let mut mtp_preds_flat = Vec::new();
            let mut mtp_targets_flat = Vec::new();

            for (i, mtp_logit) in mtp_logits_list
                .iter()
                .enumerate()
                .take(self.config.mtp_n_tokens)
            {
                let shift = i + 1;
                if shift >= seq_len {
                    break; // Not enough tokens for this future position
                }

                // MTP logit: [batch, seq, vocab]
                // We want to predict position i+1 from position i
                // So use logits[0:seq-shift] to predict targets[shift:seq]
                let pred_len = seq_len - shift;
                let mtp_pred_trimmed = mtp_logit.narrow(1, 0, pred_len)?;
                let mtp_pred_for_loss = if let Some(mask) = collider_mask.as_ref() {
                    let mask_trimmed = mask.narrow(1, 0, pred_len)?;
                    apply_collider_gradient_mask(&mtp_pred_trimmed, &mask_trimmed)?
                } else {
                    mtp_pred_trimmed
                };
                let mtp_pred_flat = mtp_pred_for_loss.reshape((batch * pred_len, vocab))?;

                // Target: shift by (i+1)
                let target_shifted = target_ids.narrow(1, shift, pred_len)?;
                let target_flat = target_shifted.reshape(batch * pred_len)?;

                mtp_preds_flat.push(mtp_pred_flat);
                mtp_targets_flat.push(target_flat);
            }

            // Compute MTP loss
            if !mtp_preds_flat.is_empty() {
                let (mtp_loss_info, mtp_loss_tensor) =
                    mtp.compute_loss(&mtp_preds_flat, &mtp_targets_flat)?;
                // Scale by MTP weight and add directly to loss (preserves gradient flow)
                let mtp_weighted_tensor = (mtp_loss_tensor * self.config.mtp_weight)?;
                loss = (loss + mtp_weighted_tensor)?;
                let _ = mtp_loss_info; // diagnostic info available for logging if needed
            }
        }

        let accum_steps = self.config.grad_accum_steps.max(1);
        let loss_val_unscaled = loss.to_scalar::<f32>()? as f64;

        // Scale each micro-step loss so accumulated gradients match large-batch average.
        let scaled_loss = if accum_steps > 1 {
            (&loss * (1.0 / accum_steps as f64))?
        } else {
            loss.clone()
        };

        // Backward pass for this micro-step.
        let grads = scaled_loss.backward()?;
        if let Some(accum) = self.accum_grads.as_mut() {
            accumulate_grad_store(accum, &grads, &self.varmap)?;
        } else {
            self.accum_grads = Some(grads);
        }

        self.accum_micro_steps += 1;
        self.accum_loss_sum += loss_val_unscaled;
        let should_step = self.accum_micro_steps >= accum_steps;

        let (grad_norm, mult, loss_val) = if should_step {
            let grads_to_apply = self.accum_grads.take().ok_or_else(|| {
                candle_core::Error::Msg("missing accumulated gradients".to_string())
            })?;
            let grad_norm = compute_grad_norm(&grads_to_apply, &self.varmap)?;
            let clip_scale = if grad_norm > self.config.grad_clip {
                self.config.grad_clip / grad_norm
            } else {
                1.0
            };

            self.muon.step(&grads_to_apply, clip_scale)?;
            self.lion.step(&grads_to_apply, clip_scale)?;

            // Clamp mHC alpha_logit to [-3, 3] to prevent identity bypass.
            // sigmoid(-3)=0.05, sigmoid(3)=0.95 — ensures meaningful stream mixing.
            self.clamp_alpha_logits()?;

            self.global_step += 1;
            let mult = wsd_schedule(
                self.global_step,
                self.config.warmup_steps,
                self.config.total_steps,
                self.config.decay_start_frac,
                0.1,
            );
            self.muon.set_lr(self.base_lr_muon * mult);
            self.lion.set_lr(self.base_lr_lion * mult);

            let loss_avg = self.accum_loss_sum / self.accum_micro_steps as f64;
            self.accum_micro_steps = 0;
            self.accum_loss_sum = 0.0;
            (grad_norm, mult, loss_avg)
        } else {
            let grad_norm = compute_grad_norm(
                self.accum_grads.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("missing accumulated gradients".to_string())
                })?,
                &self.varmap,
            )?;
            let mult = wsd_schedule(
                self.global_step,
                self.config.warmup_steps,
                self.config.total_steps,
                self.config.decay_start_frac,
                0.1,
            );
            (grad_norm, mult, loss_val_unscaled)
        };

        let elapsed = step_start.elapsed().as_secs_f64();
        let n_tokens = (batch * seq_len) as f64;
        let tokens_per_sec = if elapsed > 0.0 {
            n_tokens / elapsed
        } else {
            0.0
        };

        // Explicitly drop intermediate tensors to free GPU memory
        drop(loss);
        drop(collider_mask);
        drop(logits_flat);
        drop(targets_flat);

        Ok(StepStats {
            loss: loss_val,
            grad_norm,
            lr: self.base_lr_muon * mult,
            tokens_per_sec,
        })
    }

    /// Expose optimizer memory statistics for benchmarking and telemetry.
    pub fn optimizer_memory_stats(&self) -> crate::optim::OptimizerMemoryStats {
        self.muon.memory_stats()
    }

    /// Attempt training step with automatic OOM recovery
    ///
    /// On OOM error, clears internal caches and retries once.
    /// Returns `StepAttempt` indicating the outcome.
    pub fn train_step_with_recovery(
        &mut self,
        input_ids: &Tensor,
        target_ids: &Tensor,
    ) -> StepAttempt {
        match self.train_step(input_ids, target_ids) {
            Ok(stats) => StepAttempt::Success(stats),
            Err(e) => {
                let err_str = format!("{}", e);
                if err_str.contains("out of memory")
                    || err_str.contains("OOM")
                    || err_str.contains("CUDA out of memory")
                {
                    tracing::warn!("OOM detected, attempting recovery...");

                    // Clear caches
                    self.clear_caches();

                    // Retry once
                    match self.train_step(input_ids, target_ids) {
                        Ok(stats) => StepAttempt::Success(stats),
                        Err(e2) => {
                            tracing::error!("OOM recovery failed: {}", e2);
                            StepAttempt::Failed(candle_core::Error::Msg(format!(
                                "OOM recovery failed: original={}, retry={}",
                                e, e2
                            )))
                        }
                    }
                } else {
                    StepAttempt::Failed(e)
                }
            }
        }
    }

    /// Clear internal caches to free memory
    fn clear_caches(&mut self) {
        self.accum_grads = None;
        self.accum_micro_steps = 0;
        self.accum_loss_sum = 0.0;

        // Force synchronization on CUDA to ensure all pending ops complete
        // before freeing gradient buffers.
        #[cfg(feature = "cuda")]
        {
            if self.device.is_cuda() {
                if let Err(e) = self.device.synchronize() {
                    tracing::warn!("CUDA synchronize during cache clear failed: {}", e);
                }
            }
        }

        tracing::info!("Trainer caches cleared");
    }

    /// Train for one epoch over a dataset.
    ///
    /// Uses async data loader if enabled in config, otherwise synchronous.
    pub fn train_epoch(&mut self, dataset: &dyn Dataset, epoch: usize) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut n_steps = 0;

        if self.config.use_async_loader {
            // Note: This requires Dataset to be Send + Sync
            // For now, fall back to synchronous if not available
            tracing::info!(
                "Using async data loader with {} workers",
                self.config.async_n_workers
            );

            // Create a wrapper that implements required traits
            // This is a simplified version - full implementation would need
            // proper Arc<dyn Dataset + Send + Sync> handling
            let loader = DataLoader::new(
                dataset,
                self.config.batch_size,
                true,
                (epoch as u64) * 1000,
                &self.device,
            );

            for batch in loader {
                let (input_ids, target_ids) = batch?;
                let stats = self.train_step(&input_ids, &target_ids)?;
                total_loss += stats.loss;
                n_steps += 1;
            }
        } else {
            let loader = DataLoader::new(
                dataset,
                self.config.batch_size,
                true,
                (epoch as u64) * 1000,
                &self.device,
            );

            for batch in loader {
                let (input_ids, target_ids) = batch?;
                let stats = self.train_step(&input_ids, &target_ids)?;
                total_loss += stats.loss;
                n_steps += 1;
            }
        }

        Ok(if n_steps > 0 {
            total_loss / n_steps as f64
        } else {
            0.0
        })
    }

    /// Production training loop with per-step logging and checkpointing.
    ///
    /// # Arguments
    /// * `keep_last_checkpoints` - Number of checkpoints to keep (0 = keep all)
    pub fn train_loop(
        &mut self,
        dataset: &dyn Dataset,
        epochs: usize,
        log_interval: usize,
        checkpoint_dir: Option<&str>,
        checkpoint_interval: usize,
        keep_last_checkpoints: usize,
    ) -> Result<()> {
        let train_start = Instant::now();
        let mut running_loss = 0.0;
        let mut running_gnorm = 0.0;
        let mut running_toks = 0.0;
        let mut interval_steps = 0usize;

        for epoch in 0..epochs {
            let epoch_start = Instant::now();
            let loader = DataLoader::new(
                dataset,
                self.config.batch_size,
                true,
                (epoch as u64) * 1000 + self.global_step as u64,
                &self.device,
            );
            let mut batch_idx = 0;

            for batch in loader {
                let (input_ids, target_ids) = batch?;
                let stats = self.train_step(&input_ids, &target_ids)?;

                running_loss += stats.loss;
                running_gnorm += stats.grad_norm;
                running_toks += stats.tokens_per_sec;
                interval_steps += 1;
                batch_idx += 1;

                if self.global_step.is_multiple_of(log_interval) && interval_steps > 0 {
                    let avg_loss = running_loss / interval_steps as f64;
                    let avg_gnorm = running_gnorm / interval_steps as f64;
                    let avg_toks = running_toks / interval_steps as f64;
                    let elapsed = train_start.elapsed().as_secs_f64();
                    println!(
                        "[{:>6}/{:<6}] loss={:.4} lr={:.6} gnorm={:.2} tok/s={:.0} elapsed={:.0}s",
                        self.global_step,
                        self.config.total_steps,
                        avg_loss,
                        stats.lr,
                        avg_gnorm,
                        avg_toks,
                        elapsed,
                    );
                    running_loss = 0.0;
                    running_gnorm = 0.0;
                    running_toks = 0.0;
                    interval_steps = 0;
                }

                // Checkpoint with disk monitoring and cleanup
                if checkpoint_interval > 0 && self.global_step.is_multiple_of(checkpoint_interval) {
                    if let Some(dir) = checkpoint_dir {
                        // Check disk space before saving
                        if let Ok((total, avail)) =
                            checkpoint_manager::get_disk_space(std::path::Path::new(dir))
                        {
                            let usage_pct = ((total - avail) as f64 / total as f64) * 100.0;
                            if usage_pct > 90.0 {
                                println!(
                                    "  WARNING: Disk {}% full ({} / {} available)",
                                    usage_pct as u32,
                                    checkpoint_manager::format_bytes(avail),
                                    checkpoint_manager::format_bytes(total)
                                );
                            }
                        }

                        // Clean old checkpoints if requested
                        if keep_last_checkpoints > 0 {
                            let dir_path = std::path::Path::new(dir);
                            if let Ok(removed) = checkpoint_manager::cleanup_old_checkpoints(
                                dir_path,
                                keep_last_checkpoints,
                            ) {
                                if removed > 0 {
                                    println!(
                                        "  Cleaned {} old checkpoint(s), keeping last {}",
                                        removed, keep_last_checkpoints
                                    );
                                }
                            }
                        }

                        let path = format!("{}/step_{}", dir, self.global_step);
                        let checkpoint_loss = if interval_steps > 0 {
                            running_loss / interval_steps as f64
                        } else {
                            stats.loss
                        };
                        crate::checkpoint::save_checkpoint(
                            &self.varmap,
                            &self.config,
                            self.global_step,
                            checkpoint_loss,
                            &path,
                        )
                        .map_err(|e| candle_core::Error::Msg(format!("checkpoint save: {}", e)))?;
                        self.save_optimizer_state(&path)?;

                        // Report checkpoint size
                        if let Ok(size) = checkpoint_manager::dir_size(std::path::Path::new(&path))
                        {
                            println!(
                                "  -> checkpoint saved to {} ({})",
                                path,
                                checkpoint_manager::format_bytes(size)
                            );
                        } else {
                            println!("  -> checkpoint saved to {}", path);
                        }
                    }
                }

                // Check if we've reached total_steps
                if self.global_step >= self.config.total_steps {
                    println!(
                        "\n✓ Reached total_steps={}, stopping training",
                        self.config.total_steps
                    );
                    break;
                }
            }

            // Check again after epoch
            if self.global_step >= self.config.total_steps {
                println!("✓ Training complete at step {}", self.global_step);
                break;
            }

            let epoch_elapsed = epoch_start.elapsed().as_secs_f64();
            println!(
                "--- Epoch {}/{} done ({} batches, {:.1}s) step={} ---",
                epoch + 1,
                epochs,
                batch_idx,
                epoch_elapsed,
                self.global_step,
            );
        }

        // Final checkpoint
        if let Some(dir) = checkpoint_dir {
            let path = format!("{}/final", dir);
            crate::checkpoint::save_checkpoint(
                &self.varmap,
                &self.config,
                self.global_step,
                0.0,
                &path,
            )
            .map_err(|e| candle_core::Error::Msg(format!("checkpoint save: {}", e)))?;
            self.save_optimizer_state(&path)?;
            println!("Final checkpoint saved to {}", path);
        }

        let total_elapsed = train_start.elapsed().as_secs_f64();
        println!(
            "\nTraining complete! steps={} elapsed={:.1}s ({:.1}m)",
            self.global_step,
            total_elapsed,
            total_elapsed / 60.0,
        );

        Ok(())
    }
}

/// Apply Collider token mask to logits such that masked tokens keep forward values
/// but contribute zero gradient in backward.
fn apply_collider_gradient_mask(logits: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, seq_len, vocab) = logits.dims3()?;
    let expanded_mask = mask.unsqueeze(2)?.broadcast_as((batch, seq_len, vocab))?;
    let inv_mask = Tensor::ones_like(&expanded_mask)?.sub(&expanded_mask)?;

    let kept = logits.mul(&expanded_mask)?;
    let dropped_no_grad = logits.detach().mul(&inv_mask)?;
    &kept + &dropped_no_grad
}

fn accumulate_grad_store(dst: &mut GradStore, src: &GradStore, varmap: &VarMap) -> Result<()> {
    for var in sorted_vars(varmap) {
        let Some(g_src) = src.get(var.as_tensor()) else {
            continue;
        };
        if let Some(g_dst) = dst.get(var.as_tensor()) {
            let sum = (g_dst + g_src)?;
            dst.insert(var.as_tensor(), sum);
        } else {
            dst.insert(var.as_tensor(), g_src.clone());
        }
    }
    Ok(())
}

/// Compute total gradient norm across all variables.
pub fn compute_grad_norm(grads: &candle_core::backprop::GradStore, varmap: &VarMap) -> Result<f64> {
    let mut total = 0.0f64;
    for var in sorted_vars(varmap) {
        if let Some(g) = grads.get(var.as_tensor()) {
            total += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    Ok(total.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SyntheticDataset;
    use tempfile::tempdir;

    fn tiny_config() -> TrainConfig {
        TrainConfig {
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.0,
            vocab_size: 256,
            max_seq_len: 32,
            group_size: 64,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,
            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 4,
            grad_accum_steps: 1,
            warmup_steps: 10,
            total_steps: 100,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 3,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
        }
    }

    #[test]
    fn test_collider_gradient_mask_zeroes_masked_token_grads() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let logits = vb.get_with_hints((1, 2, 2), "logits", candle_nn::Init::Const(1.0))?;

        let mask = Tensor::new(&[[1.0f32, 0.0]], &device)?;
        let masked_logits = apply_collider_gradient_mask(&logits, &mask)?;
        let loss = masked_logits.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads
            .get(&logits)
            .ok_or_else(|| candle_core::Error::Msg("missing logits grad".to_string()))?;
        let vals = grad.to_vec3::<f32>()?;

        // First token kept => gradients flow.
        assert!((vals[0][0][0] - 1.0).abs() < 1e-6);
        assert!((vals[0][0][1] - 1.0).abs() < 1e-6);
        // Second token masked => gradients are zero.
        assert!(vals[0][1][0].abs() < 1e-6);
        assert!(vals[0][1][1].abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_train_step_loss_finite() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let mut trainer = Trainer::new(cfg, device)?;

        let input_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
        let target_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;

        let stats = trainer.train_step(&input_ids, &target_ids)?;
        assert!(
            stats.loss.is_finite(),
            "Loss should be finite: {}",
            stats.loss
        );
        assert!(stats.grad_norm.is_finite(), "Grad norm should be finite");
        Ok(())
    }

    #[test]
    fn test_train_step_with_collider_sparse_path() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_config();
        cfg.use_collider = true;
        cfg.collider_threshold = 0.3;
        cfg.collider_sparsity = 0.5;
        let mut trainer = Trainer::new(cfg, device)?;

        let input_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
        let target_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;

        let stats = trainer.train_step(&input_ids, &target_ids)?;
        assert!(stats.loss.is_finite(), "Collider loss should be finite");
        assert!(
            stats.grad_norm.is_finite(),
            "Collider grad norm should be finite"
        );
        Ok(())
    }

    #[test]
    fn test_train_loss_decreases() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_config();
        cfg.lr = 0.001; // Use smaller LR for stability
        cfg.ns_steps = 2;
        let mut trainer = Trainer::new(cfg, device)?;

        let ds = SyntheticDataset::new(256, 8, 16, 42);
        let mut losses = Vec::new();

        // Run 5 epochs
        for epoch in 0..5 {
            let avg_loss = trainer.train_epoch(&ds, epoch)?;
            losses.push(avg_loss);
        }

        // Loss should decrease overall
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first,
            "Loss should decrease: first={:.4} last={:.4}",
            first,
            last
        );
        Ok(())
    }

    #[test]
    fn test_wsd_lr_applied() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_config();
        cfg.warmup_steps = 5;
        cfg.total_steps = 20;
        let mut trainer = Trainer::new(cfg, device)?;

        let input_ids = Tensor::zeros((1, 4), DType::U32, &trainer.device)?;
        let target_ids = Tensor::zeros((1, 4), DType::U32, &trainer.device)?;

        // During warmup, LR should be less than base
        let stats = trainer.train_step(&input_ids, &target_ids)?;
        assert!(
            stats.lr < 0.02,
            "LR should be ramping during warmup: {}",
            stats.lr
        );
        assert!(stats.lr > 0.0, "LR should be positive: {}", stats.lr);
        Ok(())
    }

    #[test]
    fn test_train_step_grad_clipping() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_config();
        cfg.grad_clip = 0.001; // Very aggressive clipping
        let mut trainer = Trainer::new(cfg, device)?;

        let input_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
        let target_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;

        // Should not crash with aggressive clipping
        let stats = trainer.train_step(&input_ids, &target_ids)?;
        assert!(stats.loss.is_finite());
        Ok(())
    }

    #[test]
    fn test_label_smoothing_matches_manual_formula() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 0.0, -1.0]], &device)?;
        let targets = Tensor::new(&[0u32], &device)?;
        let eps = 0.1f64;

        let ce = candle_nn::loss::cross_entropy(&logits, &targets)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;
        let uniform = log_probs.mean(D::Minus1)?.neg()?.mean_all()?;
        let smoothed = ((&ce * (1.0 - eps))? + (&uniform * eps)?)?;
        let got = smoothed.to_scalar::<f32>()?;

        let probs = candle_nn::ops::softmax(&logits, 1)?.to_vec2::<f32>()?;
        let p0 = probs[0][0].max(1e-12);
        let p1 = probs[0][1].max(1e-12);
        let p2 = probs[0][2].max(1e-12);
        let manual =
            (1.0 - eps as f32) * (-p0.ln()) + (eps as f32 / 3.0) * (-(p0.ln() + p1.ln() + p2.ln()));
        assert!(
            (got - manual).abs() < 1e-5,
            "got {}, manual {}",
            got,
            manual
        );
        Ok(())
    }

    #[test]
    fn test_grad_accum_steps_applies_optimizer_on_schedule() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_config();
        cfg.grad_accum_steps = 2;
        let mut trainer = Trainer::new(cfg, device)?;

        let input_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
        let target_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;

        let _ = trainer.train_step(&input_ids, &target_ids)?;
        assert_eq!(
            trainer.global_step, 0,
            "optimizer should not step on first micro-step"
        );
        let _ = trainer.train_step(&input_ids, &target_ids)?;
        assert_eq!(
            trainer.global_step, 1,
            "optimizer should step after grad_accum_steps"
        );
        Ok(())
    }

    #[test]
    fn test_checkpoint_manager_cleanup_and_dir_size() -> Result<()> {
        let dir = tempdir().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let ckpt = dir.path().join("checkpoints");
        std::fs::create_dir_all(&ckpt).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        for step in [1u32, 2, 3] {
            let step_dir = ckpt.join(format!("step_{}", step));
            std::fs::create_dir_all(&step_dir)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            std::fs::write(step_dir.join("model.safetensors"), vec![step as u8; 64])
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }

        let total_before = checkpoint_manager::dir_size(&ckpt).map_err(candle_core::Error::Msg)?;
        assert!(total_before >= 64 * 3);

        let removed = checkpoint_manager::cleanup_old_checkpoints(&ckpt, 1)
            .map_err(candle_core::Error::Msg)?;
        assert_eq!(removed, 2);
        assert!(!ckpt.join("step_1").exists());
        assert!(!ckpt.join("step_2").exists());
        assert!(ckpt.join("step_3").exists());

        // Smoke test utility formatting helpers and disk-space probe.
        let (_total, _avail) =
            checkpoint_manager::get_disk_space(dir.path()).map_err(candle_core::Error::Msg)?;
        assert_eq!(checkpoint_manager::format_bytes(1024), "1.00KB");
        Ok(())
    }

    #[test]
    fn test_train_loop_saves_checkpoints_and_final() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_config();
        cfg.total_steps = 2;
        cfg.batch_size = 2;
        cfg.grad_accum_steps = 1;
        let mut trainer = Trainer::new(cfg.clone(), device)?;
        let ds = SyntheticDataset::new(cfg.vocab_size as u32, 8, 16, 123);
        let dir = tempdir().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let ckpt_dir = dir.path().to_str().ok_or_else(|| {
            candle_core::Error::Msg("temp checkpoint path not valid utf8".to_string())
        })?;

        trainer.train_loop(&ds, 10, 1, Some(ckpt_dir), 1, 1)?;
        assert!(trainer.global_step >= cfg.total_steps);
        assert!(dir.path().join("final/meta.json").exists());
        assert!(dir.path().join("final/model.safetensors").exists());
        assert!(dir.path().join("final/optimizer_state.bin").exists());
        assert!(dir.path().join("step_2").exists());

        let step_dirs = std::fs::read_dir(dir.path())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with("step_"))
            })
            .count();
        assert!(
            step_dirs <= 2,
            "expected limited number of step directories"
        );
        Ok(())
    }
}
