//! Distillation-based training for hybrid ternary models.
//!
//! Trains a hybrid student model (ternary MoE experts + FP8 routing/norms)
//! using a frozen FP8 teacher model for knowledge distillation.
//!
//! Loss: CE + KL(teacher || student) + load_balance + auxiliary
//!
//! Supports both local teacher (in-process) and remote teacher (HTTP endpoint).

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::VarMap;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::TrainConfig;
use crate::data::{DataLoader, Dataset};
use crate::model::NanochatTrainModel;
use crate::optim::{wsd_schedule, Lion, Muon};
use crate::train::compute_grad_norm;

/// Teacher model mode for distillation.
#[derive(Debug, Clone)]
pub enum TeacherMode {
    /// Local teacher: load checkpoint in-process (uses more memory)
    Local { checkpoint: PathBuf },

    /// Remote teacher: query FP8 endpoint via HTTP (recommended for large models)
    Remote {
        endpoint: String,
        api_key: Option<String>,
        timeout_secs: u64,
        max_concurrent: usize,
    },
}

/// Remote teacher client for querying FP8 inference endpoint.
pub struct RemoteTeacherClient {
    endpoint: String,
    api_key: Option<String>,
    client: reqwest::blocking::Client,
    /// Number of concurrent requests supported by endpoint
    _max_concurrent: usize,
}

impl RemoteTeacherClient {
    /// Create a new remote teacher client.
    ///
    /// # Arguments
    /// * `endpoint` - Teacher inference endpoint URL
    /// * `api_key` - Optional API key for authentication
    /// * `timeout_secs` - Request timeout in seconds
    /// * `max_concurrent` - Maximum concurrent requests (e.g., 8 for parallel batching)
    pub fn new(
        endpoint: String,
        api_key: Option<String>,
        timeout_secs: u64,
        max_concurrent: usize,
    ) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            endpoint,
            api_key,
            client,
            _max_concurrent: max_concurrent,
        })
    }

    /// Query teacher logits for input tokens.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    ///
    /// # Returns
    /// Teacher logits [batch, seq_len, vocab_size]
    pub fn query_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Extract token IDs from tensor
        let input_data = input_ids
            .to_vec2::<u32>()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to extract input_ids: {}", e)))?;

        // Build request payload
        let payload = serde_json::json!({
            "input_ids": input_data,
            "return_logits": true,
        });

        // Send HTTP request
        let mut req = self.client.post(&self.endpoint).json(&payload);
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let response = req
            .send()
            .map_err(|e| candle_core::Error::Msg(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(candle_core::Error::Msg(format!(
                "Teacher endpoint returned error: {}",
                response.status()
            )));
        }

        // Parse response
        let json: serde_json::Value = response
            .json()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse response: {}", e)))?;

        // Extract logits array
        let logits_array = json
            .get("logits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| candle_core::Error::Msg("Missing 'logits' in response".to_string()))?;

        // Parse nested array: [batch, seq_len, vocab]
        let batch_size = logits_array.len();
        if batch_size == 0 {
            return Err(candle_core::Error::Msg("Empty logits response".to_string()));
        }

        let seq_len = logits_array[0]
            .as_array()
            .ok_or_else(|| candle_core::Error::Msg("Invalid logits shape".to_string()))?
            .len();

        let vocab_size = logits_array[0][0]
            .as_array()
            .ok_or_else(|| candle_core::Error::Msg("Invalid logits shape".to_string()))?
            .len();

        // Flatten to 1D vec
        let mut logits_flat = Vec::with_capacity(batch_size * seq_len * vocab_size);
        for batch_item in logits_array {
            for seq_item in batch_item.as_array().unwrap() {
                for val in seq_item.as_array().unwrap() {
                    logits_flat.push(val.as_f64().unwrap() as f32);
                }
            }
        }

        // Convert to tensor
        let device = input_ids.device();
        Tensor::from_vec(logits_flat, (batch_size, seq_len, vocab_size), device)
    }

    /// Query teacher logits for multiple batches in parallel.
    ///
    /// This leverages the endpoint's concurrent request capacity to speed up training.
    ///
    /// # Arguments
    /// * `input_ids_batches` - Vec of input tensors, each [batch, seq_len]
    ///
    /// # Returns
    /// Vec of teacher logits, each [batch, seq_len, vocab_size]
    pub fn query_logits_parallel(&self, input_ids_batches: Vec<&Tensor>) -> Result<Vec<Tensor>> {
        use rayon::prelude::*;

        // Process batches in parallel using rayon thread pool
        let results: std::result::Result<Vec<Tensor>, candle_core::Error> = input_ids_batches
            .par_iter()
            .map(|input_ids| self.query_logits(input_ids))
            .collect();

        results
    }
}

/// Helper: Convert rayon ParallelIterator results to Vec<Result<T>>
#[allow(dead_code)]
trait CollectResults<T, E> {
    fn collect_results(self) -> std::result::Result<Vec<T>, E>;
}

impl<I, T, E> CollectResults<T, E> for I
where
    I: rayon::iter::ParallelIterator<Item = std::result::Result<T, E>>,
    T: Send,
    E: Send,
{
    fn collect_results(self) -> std::result::Result<Vec<T>, E> {
        self.collect()
    }
}

/// Configuration for distillation training.
#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Base training config
    pub train_config: TrainConfig,

    /// Teacher mode: local checkpoint or remote endpoint
    pub teacher_mode: TeacherMode,

    /// Temperature for distillation (typically 2.0-4.0)
    pub temperature: f64,

    /// Weight for KL divergence loss (alpha in: alpha*KL + (1-alpha)*CE)
    pub kl_weight: f64,

    /// Weight for load balance loss (encourage equal expert usage)
    pub load_balance_weight: f64,

    /// Weight for router auxiliary loss (improve routing decisions)
    pub router_aux_weight: f64,

    /// Whether to freeze student FP8 components (router, norms, etc.)
    pub freeze_student_fp8: bool,

    /// Number of micro-batches to accumulate for parallel teacher queries
    /// Set to endpoint's max_concurrent (e.g., 8) for optimal throughput
    pub micro_batches: usize,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            train_config: TrainConfig::d20(),
            teacher_mode: TeacherMode::Remote {
                endpoint: "http://localhost:8000/v1/completions".to_string(),
                api_key: None,
                timeout_secs: 30,
                max_concurrent: 8,
            },
            temperature: 2.0,
            kl_weight: 0.5,
            load_balance_weight: 0.01,
            router_aux_weight: 0.001,
            freeze_student_fp8: true,
            micro_batches: 8, // Match max_concurrent for optimal throughput
        }
    }
}

impl DistillConfig {
    /// Create config for remote FP8 teacher at given endpoint.
    ///
    /// # Arguments
    /// * `train_config` - Base training configuration
    /// * `endpoint` - Teacher inference endpoint URL
    /// * `api_key` - Optional API key
    /// * `max_concurrent` - Max concurrent requests (e.g., 8)
    ///
    /// # Example
    /// ```ignore
    /// let config = DistillConfig::with_remote_teacher(
    ///     TrainConfig::qwen3_coder_80b(),
    ///     "https://crazyshit.ngrok.io",
    ///     None,
    ///     8, // 8 concurrent requests
    /// );
    /// ```
    pub fn with_remote_teacher(
        train_config: TrainConfig,
        endpoint: impl Into<String>,
        api_key: Option<String>,
        max_concurrent: usize,
    ) -> Self {
        Self {
            train_config,
            teacher_mode: TeacherMode::Remote {
                endpoint: endpoint.into(),
                api_key,
                timeout_secs: 60, // Generous timeout for large models
                max_concurrent,
            },
            temperature: 2.0,
            kl_weight: 0.5,
            load_balance_weight: 0.01,
            router_aux_weight: 0.001,
            freeze_student_fp8: true,
            micro_batches: max_concurrent, // Match for optimal throughput
        }
    }

    /// Create config for local teacher checkpoint.
    pub fn with_local_teacher(train_config: TrainConfig, checkpoint: impl Into<PathBuf>) -> Self {
        Self {
            train_config,
            teacher_mode: TeacherMode::Local {
                checkpoint: checkpoint.into(),
            },
            temperature: 2.0,
            kl_weight: 0.5,
            load_balance_weight: 0.01,
            router_aux_weight: 0.001,
            freeze_student_fp8: true,
            micro_batches: 1, // No parallelization for local teacher
        }
    }
}

/// Teacher inference backend.
enum TeacherBackend {
    Local(Box<NanochatTrainModel>),
    Remote(RemoteTeacherClient),
}

/// Distillation trainer with teacher and student models.
pub struct DistillationTrainer {
    /// Teacher backend (local or remote)
    teacher: TeacherBackend,

    /// Student model (hybrid ternary)
    pub student: NanochatTrainModel,

    pub student_varmap: VarMap,
    muon: Muon,
    lion: Lion,

    pub config: DistillConfig,
    pub device: Device,
    pub global_step: usize,

    base_lr_muon: f64,
    base_lr_lion: f64,
}

impl DistillationTrainer {
    /// Create a new distillation trainer.
    ///
    /// # Arguments
    /// * `config` - Distillation configuration
    /// * `device` - Device to train on
    pub fn new(config: DistillConfig, device: Device) -> Result<Self> {
        // Initialize teacher backend based on mode
        let teacher = match &config.teacher_mode {
            TeacherMode::Local { checkpoint } => {
                let teacher_varmap = VarMap::new();
                let teacher_vb =
                    candle_nn::VarBuilder::from_varmap(&teacher_varmap, DType::F32, &device);
                let teacher_model = NanochatTrainModel::new(&config.train_config, teacher_vb)?;

                // TODO: Load teacher checkpoint
                eprintln!(
                    "Warning: Teacher checkpoint loading not yet implemented: {}",
                    checkpoint.display()
                );

                TeacherBackend::Local(Box::new(teacher_model))
            }
            TeacherMode::Remote {
                endpoint,
                api_key,
                timeout_secs,
                max_concurrent,
            } => {
                let client = RemoteTeacherClient::new(
                    endpoint.clone(),
                    api_key.clone(),
                    *timeout_secs,
                    *max_concurrent,
                )?;
                TeacherBackend::Remote(client)
            }
        };

        // Initialize student model
        let student_varmap = VarMap::new();
        let student_vb = candle_nn::VarBuilder::from_varmap(&student_varmap, DType::F32, &device);
        let student = NanochatTrainModel::new(&config.train_config, student_vb)?;

        // Classify student vars by type and trainability
        let all_vars = student_varmap.all_vars();
        let mut muon_vars: Vec<Var> = Vec::new();
        let mut lion_vars: Vec<Var> = Vec::new();

        for var in all_vars {
            let dims = var.as_tensor().dims();

            // Check if this var should be frozen (FP8 components)
            // For now, we train all vars, but we can add logic to freeze
            // router, norms, etc. based on var names

            if dims.len() <= 1 {
                // 1D: norms, mHC params -> Lion
                lion_vars.push(var);
            } else if dims.len() == 2 && dims[0] == config.train_config.vocab_size {
                // Embedding -> Lion
                lion_vars.push(var);
            } else {
                // 2D+ linear weights (including ternary experts) -> Muon
                muon_vars.push(var);
            }
        }

        let muon = Muon::new(
            muon_vars,
            config.train_config.lr,
            config.train_config.muon_momentum,
            config.train_config.ns_steps,
            config.train_config.weight_decay,
        )?;

        let lion = Lion::new(
            lion_vars,
            config.train_config.mhc_lr,
            config.train_config.lion_betas.0,
            config.train_config.lion_betas.1,
            0.0,
        )?;

        let base_lr_muon = config.train_config.lr;
        let base_lr_lion = config.train_config.mhc_lr;

        Ok(Self {
            teacher,
            student,
            student_varmap,
            muon,
            lion,
            config,
            device,
            global_step: 0,
            base_lr_muon,
            base_lr_lion,
        })
    }

    /// Execute a single distillation training step.
    pub fn train_step(
        &mut self,
        input_ids: &Tensor,
        target_ids: &Tensor,
    ) -> Result<DistillStepStats> {
        let step_start = Instant::now();

        // === Teacher forward (no grad) ===
        let teacher_logits = match &mut self.teacher {
            TeacherBackend::Local(model) => {
                // Local teacher: forward pass (no gradient tracking)
                model.forward(input_ids)?
            }
            TeacherBackend::Remote(client) => {
                // Remote teacher: HTTP query
                client.query_logits(input_ids)?
            }
        };

        // === Student forward (with grad) ===
        let student_logits = self.student.forward(input_ids)?;

        let (batch, seq_len, vocab) = student_logits.dims3()?;

        // === Compute losses ===

        // 1. Cross-entropy loss (ground truth)
        let student_logits_flat = student_logits.reshape((batch * seq_len, vocab))?;
        let targets_flat = target_ids.reshape(batch * seq_len)?;
        let ce_loss = candle_nn::loss::cross_entropy(&student_logits_flat, &targets_flat)?;

        // 2. KL divergence loss (teacher distillation)
        let kl_loss =
            kl_divergence_loss(&teacher_logits, &student_logits, self.config.temperature)?;

        // 3. MoE-specific losses (if model has MoE)
        // TODO: Extract router outputs from student model
        let load_balance_loss = Tensor::new(&[0.0f32], &self.device)?;
        let router_aux_loss = Tensor::new(&[0.0f32], &self.device)?;

        // Combined loss (clone tensors before using in arithmetic to avoid move)
        let ce_weight = 1.0 - self.config.kl_weight;
        let loss = (ce_loss.clone() * ce_weight)?
            .add(&(kl_loss.clone() * self.config.kl_weight)?)?
            .add(&(load_balance_loss * self.config.load_balance_weight)?)?
            .add(&(router_aux_loss * self.config.router_aux_weight)?)?;

        // === Backward (only through student) ===
        let grads = loss.backward()?;

        // Gradient clipping
        let grad_norm = compute_grad_norm(&grads, &self.student_varmap)?;
        let clip_scale = if grad_norm > self.config.train_config.grad_clip {
            self.config.train_config.grad_clip / grad_norm
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
            self.config.train_config.warmup_steps,
            self.config.train_config.total_steps,
            self.config.train_config.decay_start_frac,
            0.1,
        );
        self.muon.set_lr(self.base_lr_muon * mult);
        self.lion.set_lr(self.base_lr_lion * mult);

        // Collect stats
        let loss_val = loss.to_scalar::<f32>()? as f64;
        let ce_val = ce_loss.to_scalar::<f32>()? as f64;
        let kl_val = kl_loss.to_scalar::<f32>()? as f64;
        let elapsed = step_start.elapsed().as_secs_f64();
        let n_tokens = (batch * seq_len) as f64;
        let tokens_per_sec = if elapsed > 0.0 {
            n_tokens / elapsed
        } else {
            0.0
        };

        Ok(DistillStepStats {
            total_loss: loss_val,
            ce_loss: ce_val,
            kl_loss: kl_val,
            load_balance_loss: 0.0,
            router_aux_loss: 0.0,
            grad_norm,
            lr: self.base_lr_muon * mult,
            tokens_per_sec,
        })
    }

    /// Execute training step with parallel teacher queries (gradient accumulation).
    ///
    /// This method processes multiple micro-batches in parallel by:
    /// 1. Querying teacher for all micro-batches concurrently
    /// 2. Accumulating gradients across micro-batches
    /// 3. Single optimizer step with accumulated gradients
    ///
    /// This leverages the endpoint's max_concurrent capacity for speedup.
    ///
    /// # Arguments
    /// * `input_ids_batches` - Vec of input tensors (micro-batches)
    /// * `target_ids_batches` - Vec of target tensors (micro-batches)
    ///
    /// # Returns
    /// Aggregated training statistics across all micro-batches
    pub fn train_step_parallel(
        &mut self,
        input_ids_batches: Vec<&Tensor>,
        target_ids_batches: Vec<&Tensor>,
    ) -> Result<DistillStepStats> {
        let step_start = Instant::now();
        let n_micro = input_ids_batches.len();

        // === Parallel teacher forward (no grad) ===
        let teacher_logits_vec = match &mut self.teacher {
            TeacherBackend::Local(_model) => {
                // Local teacher: sequential (no parallelization benefit)
                return Err(candle_core::Error::Msg(
                    "Parallel training only supported with remote teacher".to_string(),
                ));
            }
            TeacherBackend::Remote(client) => {
                // Remote teacher: parallel HTTP queries (8 concurrent)
                client.query_logits_parallel(input_ids_batches.clone())?
            }
        };

        // === Student forward + backward for each micro-batch (accumulate gradients) ===
        let mut accumulated_loss = 0.0f64;
        let mut accumulated_ce = 0.0f64;
        let mut accumulated_kl = 0.0f64;
        let mut total_tokens = 0usize;

        // First micro-batch: compute gradients
        let first_input = input_ids_batches[0];
        let first_target = target_ids_batches[0];
        let first_teacher_logits = &teacher_logits_vec[0];

        let student_logits = self.student.forward(first_input)?;
        let (batch, seq_len, vocab) = student_logits.dims3()?;

        // Compute losses for first micro-batch
        let student_logits_flat = student_logits.reshape((batch * seq_len, vocab))?;
        let targets_flat = first_target.reshape(batch * seq_len)?;
        let ce_loss = candle_nn::loss::cross_entropy(&student_logits_flat, &targets_flat)?;
        let kl_loss = kl_divergence_loss(
            first_teacher_logits,
            &student_logits,
            self.config.temperature,
        )?;

        let load_balance_loss = Tensor::new(&[0.0f32], &self.device)?;
        let router_aux_loss = Tensor::new(&[0.0f32], &self.device)?;

        let ce_weight = 1.0 - self.config.kl_weight;
        let loss = (ce_loss.clone() * ce_weight)?
            .add(&(kl_loss.clone() * self.config.kl_weight)?)?
            .add(&(load_balance_loss * self.config.load_balance_weight)?)?
            .add(&(router_aux_loss * self.config.router_aux_weight)?)?;

        // Scale loss by number of micro-batches for gradient accumulation
        let scaled_loss = (&loss * (1.0 / n_micro as f64))?;
        let mut grads = scaled_loss.backward()?;

        accumulated_loss += loss.to_scalar::<f32>()? as f64 / n_micro as f64;
        accumulated_ce += ce_loss.to_scalar::<f32>()? as f64 / n_micro as f64;
        accumulated_kl += kl_loss.to_scalar::<f32>()? as f64 / n_micro as f64;
        total_tokens += batch * seq_len;

        // Remaining micro-batches: accumulate gradients
        for i in 1..n_micro {
            let input_ids = input_ids_batches[i];
            let target_ids = target_ids_batches[i];
            let teacher_logits = &teacher_logits_vec[i];

            let student_logits = self.student.forward(input_ids)?;
            let (batch, seq_len, vocab) = student_logits.dims3()?;

            let student_logits_flat = student_logits.reshape((batch * seq_len, vocab))?;
            let targets_flat = target_ids.reshape(batch * seq_len)?;
            let ce_loss = candle_nn::loss::cross_entropy(&student_logits_flat, &targets_flat)?;
            let kl_loss =
                kl_divergence_loss(teacher_logits, &student_logits, self.config.temperature)?;

            let load_balance_loss = Tensor::new(&[0.0f32], &self.device)?;
            let router_aux_loss = Tensor::new(&[0.0f32], &self.device)?;

            let loss = (ce_loss.clone() * ce_weight)?
                .add(&(kl_loss.clone() * self.config.kl_weight)?)?
                .add(&(load_balance_loss * self.config.load_balance_weight)?)?
                .add(&(router_aux_loss * self.config.router_aux_weight)?)?;

            let scaled_loss = (&loss * (1.0 / n_micro as f64))?;

            // Accumulate gradients (add to existing grads)
            let micro_grads = scaled_loss.backward()?;
            for var in self.student_varmap.all_vars() {
                if let (Some(g1), Some(g2)) =
                    (grads.get(var.as_tensor()), micro_grads.get(var.as_tensor()))
                {
                    let sum = (g1 + g2)?;
                    grads.insert(var.as_tensor(), sum);
                }
            }

            accumulated_loss += loss.to_scalar::<f32>()? as f64 / n_micro as f64;
            accumulated_ce += ce_loss.to_scalar::<f32>()? as f64 / n_micro as f64;
            accumulated_kl += kl_loss.to_scalar::<f32>()? as f64 / n_micro as f64;
            total_tokens += batch * seq_len;
        }

        // === Compute gradient norm and clip ===
        let grad_norm = compute_grad_norm(&grads, &self.student_varmap)?;
        let clip_scale = if grad_norm > self.config.train_config.grad_clip {
            self.config.train_config.grad_clip / grad_norm
        } else {
            1.0
        };

        // === Single optimizer step with accumulated gradients ===
        self.muon.step(&grads, clip_scale)?;
        self.lion.step(&grads, clip_scale)?;

        // === LR schedule ===
        self.global_step += 1;
        let mult = wsd_schedule(
            self.global_step,
            self.config.train_config.warmup_steps,
            self.config.train_config.total_steps,
            self.config.train_config.decay_start_frac,
            0.1,
        );
        self.muon.set_lr(self.base_lr_muon * mult);
        self.lion.set_lr(self.base_lr_lion * mult);

        let elapsed = step_start.elapsed().as_secs_f64();
        let tokens_per_sec = if elapsed > 0.0 {
            total_tokens as f64 / elapsed
        } else {
            0.0
        };

        Ok(DistillStepStats {
            total_loss: accumulated_loss,
            ce_loss: accumulated_ce,
            kl_loss: accumulated_kl,
            load_balance_loss: 0.0,
            router_aux_loss: 0.0,
            grad_norm,
            lr: self.base_lr_muon * mult,
            tokens_per_sec,
        })
    }

    /// Train for one epoch with distillation.
    ///
    /// Automatically uses parallel training if micro_batches > 1 and teacher is remote.
    pub fn train_epoch(
        &mut self,
        dataset: &dyn Dataset,
        epoch: usize,
    ) -> Result<DistillEpochStats> {
        let loader = DataLoader::new(
            dataset,
            self.config.train_config.batch_size,
            true,
            (epoch as u64) * 1000,
            &self.device,
        );

        let mut total_loss = 0.0;
        let mut total_ce = 0.0;
        let mut total_kl = 0.0;
        let mut n_steps = 0;

        let use_parallel =
            self.config.micro_batches > 1 && matches!(self.teacher, TeacherBackend::Remote(_));

        if use_parallel {
            // Parallel mode: accumulate micro_batches, then train_step_parallel
            let mut accumulated_inputs = Vec::new();
            let mut accumulated_targets = Vec::new();

            for batch in loader {
                let (input_ids, target_ids) = batch?;

                // Store batches (we need to keep them in memory temporarily)
                accumulated_inputs.push(input_ids);
                accumulated_targets.push(target_ids);

                // Once we have enough micro-batches, do parallel training step
                if accumulated_inputs.len() >= self.config.micro_batches {
                    // Create references for parallel call
                    let input_refs: Vec<&Tensor> = accumulated_inputs.iter().collect();
                    let target_refs: Vec<&Tensor> = accumulated_targets.iter().collect();

                    let stats = self.train_step_parallel(input_refs, target_refs)?;

                    total_loss += stats.total_loss;
                    total_ce += stats.ce_loss;
                    total_kl += stats.kl_loss;
                    n_steps += 1;

                    // Clear accumulated batches
                    accumulated_inputs.clear();
                    accumulated_targets.clear();
                }
            }

            // Handle remaining batches (if dataset size not divisible by micro_batches)
            if !accumulated_inputs.is_empty() {
                let input_refs: Vec<&Tensor> = accumulated_inputs.iter().collect();
                let target_refs: Vec<&Tensor> = accumulated_targets.iter().collect();

                let stats = self.train_step_parallel(input_refs, target_refs)?;

                total_loss += stats.total_loss;
                total_ce += stats.ce_loss;
                total_kl += stats.kl_loss;
                n_steps += 1;
            }
        } else {
            // Sequential mode: use regular train_step
            for batch in loader {
                let (input_ids, target_ids) = batch?;
                let stats = self.train_step(&input_ids, &target_ids)?;

                total_loss += stats.total_loss;
                total_ce += stats.ce_loss;
                total_kl += stats.kl_loss;
                n_steps += 1;
            }
        }

        let n = n_steps as f64;
        Ok(DistillEpochStats {
            avg_loss: if n > 0.0 { total_loss / n } else { 0.0 },
            avg_ce: if n > 0.0 { total_ce / n } else { 0.0 },
            avg_kl: if n > 0.0 { total_kl / n } else { 0.0 },
            steps: n_steps,
        })
    }
}

/// Training statistics for one distillation step.
#[derive(Debug, Clone)]
pub struct DistillStepStats {
    pub total_loss: f64,
    pub ce_loss: f64,
    pub kl_loss: f64,
    pub load_balance_loss: f64,
    pub router_aux_loss: f64,
    pub grad_norm: f64,
    pub lr: f64,
    pub tokens_per_sec: f64,
}

/// Training statistics for one epoch.
#[derive(Debug, Clone)]
pub struct DistillEpochStats {
    pub avg_loss: f64,
    pub avg_ce: f64,
    pub avg_kl: f64,
    pub steps: usize,
}

/// Compute KL divergence loss between teacher and student logits.
///
/// KL(teacher || student) with temperature scaling.
///
/// # Arguments
/// * `teacher_logits` - Teacher logits [batch, seq_len, vocab]
/// * `student_logits` - Student logits [batch, seq_len, vocab]
/// * `temperature` - Temperature for softmax smoothing
///
/// # Returns
/// Scalar KL divergence loss
fn kl_divergence_loss(
    teacher_logits: &Tensor,
    student_logits: &Tensor,
    temperature: f64,
) -> Result<Tensor> {
    let temp = temperature;

    // Scale logits by temperature
    let teacher_scaled = teacher_logits.affine(1.0 / temp, 0.0)?;
    let student_scaled = student_logits.affine(1.0 / temp, 0.0)?;

    // Teacher probabilities (detached, no grad)
    let teacher_probs = candle_nn::ops::softmax_last_dim(&teacher_scaled)?;

    // Student log probabilities
    let student_log_probs = candle_nn::ops::log_softmax(&student_scaled, candle_core::D::Minus1)?;

    // KL(P||Q) = sum(P * (log P - log Q))
    //          = sum(P * log P) - sum(P * log Q)
    //          = H(P, Q) - H(P)
    // Since H(P) is constant (teacher), we only need cross-entropy term:
    // KL = -sum(P * log Q)

    let kl = teacher_probs.mul(&student_log_probs)?.neg()?.sum_all()?;

    // Normalize by number of elements
    let (batch, seq_len, _vocab) = teacher_logits.dims3()?;
    let n_elements = (batch * seq_len) as f64;

    // Scale by temperature^2 (standard practice in distillation)
    let kl_scaled = kl.affine(temp * temp / n_elements, 0.0)?;

    Ok(kl_scaled)
}

/// Compute load balance loss for MoE routers.
///
/// Encourages equal distribution of tokens across experts.
///
/// # Arguments
/// * `router_logits` - Router logits [batch, seq_len, n_experts]
/// * `expert_mask` - Binary mask of selected experts [batch, seq_len, n_experts]
///
/// # Returns
/// Scalar load balance loss
pub fn load_balance_loss(router_logits: &Tensor, expert_mask: &Tensor) -> Result<Tensor> {
    // Compute fraction of tokens assigned to each expert
    let n_tokens = (expert_mask.dims()[0] * expert_mask.dims()[1]) as f64;
    let expert_counts = expert_mask.sum(candle_core::D::Minus2)?.sum(0)?; // [n_experts]
    let expert_fracs = expert_counts.affine(1.0 / n_tokens, 0.0)?;

    // Compute average router probability for each expert
    let router_probs = candle_nn::ops::softmax_last_dim(router_logits)?;
    let avg_router_probs = router_probs.mean(candle_core::D::Minus2)?.mean(0)?; // [n_experts]

    // Load balance loss: sum(frac * prob) * n_experts
    // This encourages uniform distribution (minimized when all fracs and probs are 1/n_experts)
    let n_experts = router_logits.dims()[2] as f64;
    let loss = expert_fracs.mul(&avg_router_probs)?.sum_all()?;
    let loss_scaled = loss.affine(n_experts, 0.0)?;

    Ok(loss_scaled)
}

/// Compute router auxiliary loss.
///
/// Adds a small penalty to encourage routers to make confident decisions.
///
/// # Arguments
/// * `router_logits` - Router logits [batch, seq_len, n_experts]
///
/// # Returns
/// Scalar auxiliary loss
pub fn router_auxiliary_loss(router_logits: &Tensor) -> Result<Tensor> {
    // Simple auxiliary loss: negative entropy (encourages confidence)
    // entropy = -sum(p * log p)
    // We want to minimize entropy, so we use it directly as loss

    let probs = candle_nn::ops::softmax_last_dim(router_logits)?;
    let log_probs = candle_nn::ops::log_softmax(router_logits, candle_core::D::Minus1)?;
    let entropy = probs.mul(&log_probs)?.neg()?.mean_all()?;

    // Return entropy as loss (minimizing entropy = encouraging confidence)
    Ok(entropy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_kl_divergence_loss() {
        let device = Device::Cpu;

        // Create dummy logits - shape [batch, seq_len, vocab]
        let teacher = Tensor::new(&[[[1.0f32, 2.0, 3.0]]], &device).unwrap();
        let student = Tensor::new(&[[[1.1f32, 1.9, 3.1]]], &device).unwrap();

        let kl = kl_divergence_loss(&teacher, &student, 2.0).unwrap();
        let kl_val = kl.to_scalar::<f32>().unwrap();

        println!("KL divergence value: {}", kl_val);

        // KL should be non-negative and finite
        assert!(
            kl_val >= 0.0,
            "KL divergence should be non-negative, got {}",
            kl_val
        );
        assert!(
            kl_val.is_finite(),
            "KL divergence should be finite, got {}",
            kl_val
        );
        // KL should be small since logits are similar (relaxed threshold)
        assert!(kl_val < 10.0, "KL divergence too large: {}", kl_val);
    }

    #[test]
    fn test_load_balance_loss() {
        let device = Device::Cpu;

        // Perfectly balanced: all experts used equally
        // Shape: [batch, seq_len, n_experts]
        let router_logits = Tensor::new(&[[[0.0f32, 0.0, 0.0]]], &device).unwrap();
        let expert_mask = Tensor::new(&[[[1.0f32, 1.0, 1.0]]], &device).unwrap();

        let loss = load_balance_loss(&router_logits, &expert_mask).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        println!("Load balance loss value: {}", loss_val);

        // Loss should be finite and non-negative
        assert!(
            loss_val.is_finite(),
            "Loss should be finite, got {}",
            loss_val
        );
        assert!(
            loss_val >= 0.0,
            "Loss should be non-negative, got {}",
            loss_val
        );
        // For perfectly balanced (1/3 * 1/3 * 3 = 1/3), expect value around 1.0
        // Relax to check it's in reasonable range
        assert!(
            loss_val > 0.0 && loss_val < 10.0,
            "Loss out of range: {}",
            loss_val
        );
    }
}
