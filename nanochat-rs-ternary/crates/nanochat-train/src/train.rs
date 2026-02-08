//! Training loop: Trainer struct with Muon + Lion optimizer split.

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::VarMap;
use std::time::Instant;

use crate::config::TrainConfig;
use crate::data::{Dataset, DataLoader};
use crate::model::NanochatTrainModel;
use crate::optim::{Muon, Lion, wsd_schedule};

/// Checkpoint management utilities
mod checkpoint_manager {
    use std::path::{Path, PathBuf};
    use std::fs;

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
                    total += entry.metadata()
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
    pub fn cleanup_old_checkpoints(checkpoint_dir: &Path, keep_last: usize) -> Result<usize, String> {
        if !checkpoint_dir.exists() {
            return Ok(0);
        }

        // List all checkpoint directories (step_NNNNN)
        let mut checkpoints: Vec<(PathBuf, u32)> = Vec::new();
        for entry in fs::read_dir(checkpoint_dir).map_err(|e| format!("Failed to read dir: {}", e))? {
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

/// Main trainer holding model + optimizers.
pub struct Trainer {
    pub model: NanochatTrainModel,
    pub varmap: VarMap,
    muon: Muon,
    lion: Lion,
    pub config: TrainConfig,
    pub device: Device,
    pub global_step: usize,
    base_lr_muon: f64,
    base_lr_lion: f64,
}

impl Trainer {
    pub fn new(config: TrainConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = NanochatTrainModel::new(&config, vb)?;

        // Classify vars by dimension: 2D+ go to Muon, 1D go to Lion
        // Exception: embedding (vocab_size x dim) goes to Lion
        let all_vars = varmap.all_vars();
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

        let muon = Muon::new(
            muon_vars,
            config.lr,
            config.muon_momentum,
            config.ns_steps,
            config.weight_decay,
        )?;

        let lion = Lion::new(
            lion_vars,
            config.mhc_lr,
            config.lion_betas.0,
            config.lion_betas.1,
            0.0, // no weight decay for Lion group
        )?;

        let base_lr_muon = config.lr;
        let base_lr_lion = config.mhc_lr;

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
        })
    }

    /// Execute a single training step.
    pub fn train_step(&mut self, input_ids: &Tensor, target_ids: &Tensor) -> Result<StepStats> {
        let step_start = Instant::now();

        // Forward
        let logits = self.model.forward(input_ids)?;

        // Cross-entropy loss
        let (batch, seq_len, vocab) = logits.dims3()?;
        let logits_flat = logits.reshape((batch * seq_len, vocab))?;
        let targets_flat = target_ids.reshape(batch * seq_len)?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;

        // Backward
        let grads = loss.backward()?;

        // Compute gradient norm for clipping
        let grad_norm = compute_grad_norm(&grads, &self.varmap)?;
        let clip_scale = if grad_norm > self.config.grad_clip {
            self.config.grad_clip / grad_norm
        } else {
            1.0
        };

        // Optimizer steps
        self.muon.step(&grads, clip_scale)?;
        self.lion.step(&grads, clip_scale)?;

        // LR schedule
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

        let loss_val = loss.to_scalar::<f32>()? as f64;
        let elapsed = step_start.elapsed().as_secs_f64();
        let n_tokens = (batch * seq_len) as f64;
        let tokens_per_sec = if elapsed > 0.0 { n_tokens / elapsed } else { 0.0 };

        Ok(StepStats {
            loss: loss_val,
            grad_norm,
            lr: self.base_lr_muon * mult,
            tokens_per_sec,
        })
    }

    /// Train for one epoch over a dataset.
    pub fn train_epoch(&mut self, dataset: &dyn Dataset, epoch: usize) -> Result<f64> {
        let loader = DataLoader::new(
            dataset,
            self.config.batch_size,
            true,
            (epoch as u64) * 1000,
            &self.device,
        );

        let mut total_loss = 0.0;
        let mut n_steps = 0;

        for batch in loader {
            let (input_ids, target_ids) = batch?;
            let stats = self.train_step(&input_ids, &target_ids)?;
            total_loss += stats.loss;
            n_steps += 1;
        }

        Ok(if n_steps > 0 { total_loss / n_steps as f64 } else { 0.0 })
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
                if checkpoint_interval > 0
                    && self.global_step.is_multiple_of(checkpoint_interval)
                {
                    if let Some(dir) = checkpoint_dir {
                        // Check disk space before saving
                        if let Ok((total, avail)) = checkpoint_manager::get_disk_space(std::path::Path::new(dir)) {
                            let usage_pct = ((total - avail) as f64 / total as f64) * 100.0;
                            if usage_pct > 90.0 {
                                println!("  WARNING: Disk {}% full ({} / {} available)",
                                    usage_pct as u32,
                                    checkpoint_manager::format_bytes(avail),
                                    checkpoint_manager::format_bytes(total)
                                );
                            }
                        }

                        // Clean old checkpoints if requested
                        if keep_last_checkpoints > 0 {
                            let dir_path = std::path::Path::new(dir);
                            if let Ok(removed) = checkpoint_manager::cleanup_old_checkpoints(dir_path, keep_last_checkpoints) {
                                if removed > 0 {
                                    println!("  Cleaned {} old checkpoint(s), keeping last {}", removed, keep_last_checkpoints);
                                }
                            }
                        }

                        let path = format!("{}/step_{}", dir, self.global_step);
                        crate::checkpoint::save_checkpoint(
                            &self.varmap,
                            &self.config,
                            self.global_step,
                            running_loss / interval_steps.max(1) as f64,
                            &path,
                        )
                        .map_err(|e| candle_core::Error::Msg(format!("checkpoint save: {}", e)))?;

                        // Report checkpoint size
                        if let Ok(size) = checkpoint_manager::dir_size(std::path::Path::new(&path)) {
                            println!("  -> checkpoint saved to {} ({})", path, checkpoint_manager::format_bytes(size));
                        } else {
                            println!("  -> checkpoint saved to {}", path);
                        }
                    }
                }
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

/// Compute total gradient norm across all variables.
pub fn compute_grad_norm(grads: &candle_core::backprop::GradStore, varmap: &VarMap) -> Result<f64> {
    let mut total = 0.0f64;
    for var in varmap.all_vars() {
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
        }
    }

    #[test]
    fn test_train_step_loss_finite() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let mut trainer = Trainer::new(cfg, device)?;

        let input_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
        let target_ids = Tensor::zeros((2, 8), DType::U32, &trainer.device)?;

        let stats = trainer.train_step(&input_ids, &target_ids)?;
        assert!(stats.loss.is_finite(), "Loss should be finite: {}", stats.loss);
        assert!(stats.grad_norm.is_finite(), "Grad norm should be finite");
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
        assert!(last < first, "Loss should decrease: first={:.4} last={:.4}", first, last);
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
        assert!(stats.lr < 0.02, "LR should be ramping during warmup: {}", stats.lr);
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
}
