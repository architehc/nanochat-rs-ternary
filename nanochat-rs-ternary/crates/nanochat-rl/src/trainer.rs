//! Main RL trainer that orchestrates the complete training loop.
//!
//! Combines:
//! - Code generation from policy model
//! - Compiler feedback
//! - AST analysis
//! - Optional Qwen3 evaluation
//! - GRPO policy updates

use crate::ast_analysis::analyze_ast;
use crate::compiler::CompilerFeedback;
use crate::grpo::{GrpoBatch, GrpoStats, GrpoTrainer};
use crate::qwen::{qwen_to_reward, QwenClient};
use crate::reward::compute_reward;
use crate::RLConfig;

use anyhow::{Context, Result};
use candle_core::{backprop::GradStore, Device, IndexOp, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use nanochat_train::checkpoint::{load_checkpoint, save_checkpoint};
use nanochat_train::config::TrainConfig;
use nanochat_train::model::NanochatTrainModel;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::Write;
use tokenizers::Tokenizer;

const DEFAULT_TOKENIZER_PATH: &str = "models/gpt2-tokenizer.json";
const EOS_FALLBACK_ID: u32 = 50256;

struct PolicyRuntime {
    varmap: VarMap,
    model: NanochatTrainModel,
    _reference_varmap: VarMap,
    reference_model: NanochatTrainModel,
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    device: Device,
    step: usize,
}

struct GeneratedSample {
    completion: String,
    full_tokens: Vec<u32>,
    prompt_len: usize,
    log_prob: f64,
    entropy: f64,
}

struct PolicyUpdateStats {
    loss: f64,
    grad_norm: f64,
    kl_div: f64,
    entropy: f64,
}

/// Main RL trainer.
pub struct RLTrainer {
    config: RLConfig,
    compiler: CompilerFeedback,
    grpo: GrpoTrainer,
    qwen: Option<QwenClient>,
    iteration: usize,
    runtime: Option<PolicyRuntime>,
}

impl RLTrainer {
    /// Create a new RL trainer.
    pub fn new(config: RLConfig) -> Result<Self> {
        let compiler = CompilerFeedback::new().context("Failed to create compiler feedback")?;
        let grpo = GrpoTrainer::new(config.grpo.clone());
        let qwen = config
            .qwen_endpoint
            .as_ref()
            .map(|endpoint| QwenClient::new(endpoint.clone(), None));

        Ok(Self {
            config,
            compiler,
            grpo,
            qwen,
            iteration: 0,
            runtime: None,
        })
    }

    /// Run the complete RL training loop.
    pub async fn train(&mut self) -> Result<()> {
        self.ensure_runtime_loaded()?;

        println!("═══════════════════════════════════════════════════════════");
        println!("  GRPO/GSPO Reinforcement Learning Training");
        println!("═══════════════════════════════════════════════════════════");
        println!();
        println!("Configuration:");
        println!("  Base checkpoint:  {}", self.config.base_checkpoint);
        println!("  N samples:        {}", self.config.n_samples);
        println!("  N iterations:     {}", self.config.n_iterations);
        println!("  Batch size:       {}", self.config.batch_size);
        println!("  Device:           {}", self.config.device);
        println!(
            "  Qwen3 endpoint:   {}",
            self.config
                .qwen_endpoint
                .as_ref()
                .unwrap_or(&"None".to_string())
        );
        println!();

        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature as f32;

        for iter in 0..self.config.n_iterations {
            self.iteration = iter;

            println!("───────────────────────────────────────────────────────────");
            println!("Iteration {}/{}", iter + 1, self.config.n_iterations);
            println!("───────────────────────────────────────────────────────────");

            // 1. Generate coding prompts
            let prompts = self.generate_prompts(self.config.batch_size);

            // 2. Sample completions for each prompt
            let mut batch = GrpoBatch::new(prompts.clone(), self.config.n_samples);
            let mut trajectories: Vec<Vec<GeneratedSample>> = prompts
                .iter()
                .map(|_| Vec::with_capacity(self.config.n_samples))
                .collect();

            {
                let runtime = self
                    .runtime
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("policy runtime not initialized"))?;

                for (prompt_idx, prompt) in prompts.iter().enumerate() {
                    println!("\nPrompt {}: {}", prompt_idx + 1, prompt);

                    for sample_idx in 0..self.config.n_samples {
                        let sample =
                            Self::sample_completion(runtime, prompt, max_tokens, temperature)?;
                        println!(
                            "  Sample {}: {} chars (log_prob={:.2})",
                            sample_idx + 1,
                            sample.completion.len(),
                            sample.log_prob
                        );

                        // Evaluate with compiler
                        let compile_result = self.compiler.compile(&sample.completion)?;
                        let compile_success = compile_result.success;

                        // Analyze AST
                        let ast_metrics = analyze_ast(&sample.completion)?;
                        let parse_success = ast_metrics.parseable;

                        // Compute base reward
                        let mut reward =
                            compute_reward(&compile_result, &ast_metrics, &self.config.reward);

                        // Optional: Add Qwen3 evaluation
                        if let Some(qwen) = &self.qwen {
                            match qwen.evaluate_code(&sample.completion, prompt).await {
                                Ok(eval) => {
                                    let qwen_reward = qwen_to_reward(&eval, 2.0);
                                    reward += qwen_reward;
                                    println!(
                                        "    Qwen3: {:.1}/10 (reward: {:.2})",
                                        (eval.quality_score
                                            + eval.correctness_score
                                            + eval.idiomaticity_score)
                                            / 3.0,
                                        qwen_reward
                                    );
                                }
                                Err(e) => {
                                    eprintln!("    Qwen3 evaluation failed: {}", e);
                                }
                            }
                        }

                        println!(
                            "    Compile: {} | Parse: {} | Reward: {:.2}",
                            if compile_success { "✓" } else { "✗" },
                            if parse_success { "✓" } else { "✗" },
                            reward
                        );

                        // Metadata for stats
                        let mut metadata = HashMap::new();
                        metadata.insert(
                            "compile_success".to_string(),
                            if compile_success { 1.0 } else { 0.0 },
                        );
                        metadata.insert(
                            "parse_success".to_string(),
                            if parse_success { 1.0 } else { 0.0 },
                        );

                        batch.add_completion(
                            prompt_idx,
                            sample.completion.clone(),
                            reward,
                            sample.log_prob,
                            metadata,
                        );
                        trajectories[prompt_idx].push(sample);
                    }
                }
            }

            // 3. Normalize rewards within each group (GRPO step)
            batch.normalize_rewards();

            // 4. Compute statistics
            let mut stats = batch.compute_stats();
            let (flat_log_probs, flat_relative_rewards, flat_entropy) =
                Self::flatten_batch_for_grpo(&batch, &trajectories);
            stats.policy_loss = self.grpo.compute_loss(
                &flat_log_probs,
                &flat_relative_rewards,
                None,
                &flat_entropy,
            );

            // 5. Update policy with real gradient step
            let update_stats = {
                let runtime = self
                    .runtime
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("policy runtime not initialized"))?;
                Self::apply_policy_update(runtime, &batch, &trajectories, &self.config)
            }?;

            stats.policy_loss = update_stats.loss;
            stats.grad_norm = update_stats.grad_norm;
            stats.kl_div = update_stats.kl_div;
            stats.entropy = update_stats.entropy;

            println!("\nIteration {} Statistics:", iter + 1);
            println!("  Policy loss:        {:.4}", stats.policy_loss);
            println!("  Avg Reward:         {:.2}", stats.avg_reward);
            println!("  Reward Std:         {:.2}", stats.reward_std);
            println!("  Entropy:            {:.4}", stats.entropy);
            println!("  Approx KL:          {:.4}", stats.kl_div);
            println!("  Grad norm:          {:.4}", stats.grad_norm);
            println!(
                "  Compile Success:    {:.1}%",
                stats.compile_success_rate * 100.0
            );
            println!(
                "  Parse Success:      {:.1}%",
                stats.parse_success_rate * 100.0
            );

            // 6. Save checkpoint
            {
                let runtime = self
                    .runtime
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("policy runtime not initialized"))?;
                runtime.step += 1;
                if (iter + 1) % 10 == 0 {
                    println!("  Saving checkpoint...");
                    Self::save_checkpoint_for_iteration(runtime, iter + 1, stats.policy_loss)?;
                }
            }

            // 7. Log to file
            self.log_stats(iter + 1, &stats)?;

            println!();
        }

        // Final checkpoint
        {
            let runtime = self
                .runtime
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("policy runtime not initialized"))?;
            let final_dir = "checkpoints/rl-final";
            save_checkpoint(
                &runtime.varmap,
                &runtime.train_config,
                runtime.step,
                0.0,
                final_dir,
            )
            .map_err(|e| {
                anyhow::anyhow!("Failed to save final checkpoint to {}: {}", final_dir, e)
            })?;
        }

        println!("═══════════════════════════════════════════════════════════");
        println!("  Training Complete!");
        println!("═══════════════════════════════════════════════════════════");

        Ok(())
    }

    fn ensure_runtime_loaded(&mut self) -> Result<()> {
        if self.runtime.is_some() {
            return Ok(());
        }

        let device = Self::parse_device(&self.config.device);

        let (varmap, train_config, step, _) =
            load_checkpoint(&self.config.base_checkpoint, &device).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load base checkpoint {}: {}",
                    self.config.base_checkpoint,
                    e
                )
            })?;
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let model = NanochatTrainModel::new(&train_config, vb)
            .context("Failed to initialize policy model from checkpoint")?;

        // Frozen reference policy for KL regularization
        let (reference_varmap, reference_config, _, _) =
            load_checkpoint(&self.config.base_checkpoint, &device).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load reference checkpoint {}: {}",
                    self.config.base_checkpoint,
                    e
                )
            })?;
        let reference_vb =
            VarBuilder::from_varmap(&reference_varmap, candle_core::DType::F32, &device);
        let reference_model = NanochatTrainModel::new(&reference_config, reference_vb)
            .context("Failed to initialize reference model from checkpoint")?;

        let tokenizer_path = std::env::var("NANOCHAT_TOKENIZER")
            .unwrap_or_else(|_| DEFAULT_TOKENIZER_PATH.to_string());
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer {}: {}", tokenizer_path, e))?;

        self.runtime = Some(PolicyRuntime {
            varmap,
            model,
            _reference_varmap: reference_varmap,
            reference_model,
            train_config,
            tokenizer,
            device,
            step,
        });

        Ok(())
    }

    fn parse_device(device: &str) -> Device {
        if device.starts_with("cuda") {
            let gpu_id = device
                .strip_prefix("cuda:")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            Device::new_cuda(gpu_id).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        }
    }

    fn sample_completion(
        runtime: &PolicyRuntime,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GeneratedSample> {
        let encoding = runtime
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer error for prompt: {}", e))?;
        let mut full_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if full_tokens.is_empty() {
            full_tokens.push(0);
        }
        let prompt_len = full_tokens.len();
        let eos_id = runtime
            .tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(EOS_FALLBACK_ID);

        let mut total_log_prob = 0.0f64;
        let mut total_entropy = 0.0f64;
        let mut n_generated = 0usize;

        for _ in 0..max_tokens {
            if full_tokens.len() + 1 >= runtime.train_config.max_seq_len {
                break;
            }

            let input = Tensor::new(full_tokens.as_slice(), &runtime.device)?.unsqueeze(0)?;
            let logits = runtime.model.forward(&input)?;
            let last_logits = logits.get(0)?.get(full_tokens.len() - 1)?;
            let logits_vec = last_logits.to_vec1::<f32>()?;

            let (next_token, log_prob, entropy) =
                Self::sample_token_with_stats(&logits_vec, temperature);

            if next_token as u32 == eos_id {
                break;
            }

            full_tokens.push(next_token as u32);
            total_log_prob += log_prob;
            total_entropy += entropy;
            n_generated += 1;
        }

        let completion_tokens = if full_tokens.len() > prompt_len {
            &full_tokens[prompt_len..]
        } else {
            &[]
        };
        let completion = runtime
            .tokenizer
            .decode(completion_tokens, true)
            .unwrap_or_default();

        let avg_entropy = if n_generated > 0 {
            total_entropy / n_generated as f64
        } else {
            0.0
        };

        Ok(GeneratedSample {
            completion,
            full_tokens,
            prompt_len,
            log_prob: total_log_prob,
            entropy: avg_entropy,
        })
    }

    fn sample_token_with_stats(logits: &[f32], temperature: f32) -> (usize, f64, f64) {
        let temp = temperature.max(0.05) as f64;

        let mut scaled: Vec<f64> = logits.iter().map(|&v| (v as f64) / temp).collect();
        let max_logit = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for v in &mut scaled {
            *v = (*v - max_logit).exp();
        }

        // Top-k truncation for sampling stability.
        let k = 50.min(scaled.len());
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_by(|a, b| {
            scaled[*b]
                .partial_cmp(&scaled[*a])
                .unwrap_or(Ordering::Equal)
        });

        let mut probs = vec![0.0f64; scaled.len()];
        let mut mass = 0.0f64;
        for i in idx.into_iter().take(k) {
            probs[i] = scaled[i];
            mass += scaled[i];
        }
        if mass <= 0.0 {
            return (0, 0.0, 0.0);
        }
        for p in &mut probs {
            *p /= mass;
        }

        let entropy = -probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        let mut rng = rand::thread_rng();
        let r = rng.gen::<f64>();
        let mut cumulative = 0.0;
        let mut selected = 0usize;
        for (i, p) in probs.iter().enumerate() {
            cumulative += *p;
            if r <= cumulative {
                selected = i;
                break;
            }
        }

        let log_prob = probs[selected].max(1e-12).ln();
        (selected, log_prob, entropy)
    }

    fn sequence_nll_and_logprob(
        model: &NanochatTrainModel,
        full_tokens: &[u32],
        prompt_len: usize,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        if full_tokens.len() < 2 || prompt_len >= full_tokens.len() {
            let zero = Tensor::new(0.0f32, device)?;
            return Ok((zero.clone(), zero));
        }

        let input = Tensor::new(&full_tokens[..full_tokens.len() - 1], device)?.unsqueeze(0)?;
        let logits = model.forward(&input)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;

        // Predict token i at position i-1. Generated region starts at prompt_len.
        let start_pos = prompt_len.saturating_sub(1);
        let end_pos = full_tokens.len() - 1;
        if start_pos >= end_pos {
            let zero = Tensor::new(0.0f32, device)?;
            return Ok((zero.clone(), zero));
        }

        let mut nll_sum = Tensor::new(0.0f32, device)?;
        let mut log_prob_sum = Tensor::new(0.0f32, device)?;
        let mut count = 0usize;

        for pos in start_pos..end_pos {
            let target = full_tokens[pos + 1] as usize;
            let lp = log_probs.i((0, pos, target))?;
            nll_sum = (&nll_sum - &lp)?;
            log_prob_sum = (&log_prob_sum + &lp)?;
            count += 1;
        }

        if count == 0 {
            let zero = Tensor::new(0.0f32, device)?;
            return Ok((zero.clone(), zero));
        }

        let nll_mean = (&nll_sum / count as f64)?;
        Ok((nll_mean, log_prob_sum))
    }

    fn sequence_log_prob(
        model: &NanochatTrainModel,
        full_tokens: &[u32],
        prompt_len: usize,
        device: &Device,
    ) -> Result<f64> {
        let (_, log_prob) = Self::sequence_nll_and_logprob(model, full_tokens, prompt_len, device)?;
        Ok(log_prob.to_scalar::<f32>()? as f64)
    }

    fn apply_policy_update(
        runtime: &mut PolicyRuntime,
        batch: &GrpoBatch,
        trajectories: &[Vec<GeneratedSample>],
        cfg: &RLConfig,
    ) -> Result<PolicyUpdateStats> {
        let mut total_loss = Tensor::new(0.0f32, &runtime.device)?;
        let mut approx_kl_sum = 0.0f64;
        let mut entropy_sum = 0.0f64;
        let mut n_samples = 0usize;

        for (prompt_idx, prompt_trajs) in trajectories.iter().enumerate().take(batch.prompts.len())
        {
            for (sample_idx, traj) in prompt_trajs
                .iter()
                .enumerate()
                .take(batch.completions[prompt_idx].len())
            {
                let advantage = batch.relative_rewards[prompt_idx][sample_idx].clamp(-5.0, 5.0);

                let (nll_mean, seq_log_prob) = Self::sequence_nll_and_logprob(
                    &runtime.model,
                    &traj.full_tokens,
                    traj.prompt_len,
                    &runtime.device,
                )?;

                // Minimize nll for positive advantages and maximize for negative.
                let mut sample_loss = (nll_mean * advantage)?;

                if cfg.grpo.kl_coef > 0.0 {
                    let ref_log_prob = Self::sequence_log_prob(
                        &runtime.reference_model,
                        &traj.full_tokens,
                        traj.prompt_len,
                        &runtime.device,
                    )?;
                    let ref_tensor = Tensor::new(ref_log_prob as f32, &runtime.device)?;
                    let kl_term = (&seq_log_prob - &ref_tensor)?;
                    sample_loss = (&sample_loss + &(kl_term * cfg.grpo.kl_coef)?)?;
                    approx_kl_sum += traj.log_prob - ref_log_prob;
                }

                total_loss = (&total_loss + &sample_loss)?;
                entropy_sum += traj.entropy;
                n_samples += 1;
            }
        }

        if n_samples == 0 {
            return Ok(PolicyUpdateStats {
                loss: 0.0,
                grad_norm: 0.0,
                kl_div: 0.0,
                entropy: 0.0,
            });
        }

        total_loss = (&total_loss / n_samples as f64)?;
        let loss_val = total_loss.to_scalar::<f32>()? as f64;
        let grads = total_loss.backward()?;
        let grad_norm = Self::apply_sgd_step(
            &runtime.varmap,
            &grads,
            cfg.grpo.learning_rate,
            cfg.grpo.max_grad_norm,
        )?;

        Ok(PolicyUpdateStats {
            loss: loss_val,
            grad_norm,
            kl_div: approx_kl_sum / n_samples as f64,
            entropy: entropy_sum / n_samples as f64,
        })
    }

    fn apply_sgd_step(
        varmap: &VarMap,
        grads: &GradStore,
        lr: f64,
        max_grad_norm: f64,
    ) -> candle_core::Result<f64> {
        let vars = varmap.all_vars();

        let mut total_norm_sq = 0.0f64;
        for var in &vars {
            if let Some(g) = grads.get(var.as_tensor()) {
                total_norm_sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            }
        }
        let grad_norm = total_norm_sq.sqrt();
        let clip_scale = if grad_norm > max_grad_norm && max_grad_norm > 0.0 {
            max_grad_norm / grad_norm
        } else {
            1.0
        };

        for var in vars {
            if let Some(g) = grads.get(var.as_tensor()) {
                let update = (g * (lr * clip_scale))?;
                let new_val = var.as_tensor().sub(&update)?;
                var.set(&new_val)?;
            }
        }

        Ok(grad_norm)
    }

    fn flatten_batch_for_grpo(
        batch: &GrpoBatch,
        trajectories: &[Vec<GeneratedSample>],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut log_probs = Vec::new();
        let mut relative_rewards = Vec::new();
        let mut entropy = Vec::new();

        for (prompt_idx, prompt_trajs) in trajectories.iter().enumerate().take(batch.prompts.len())
        {
            for (sample_idx, traj) in prompt_trajs
                .iter()
                .enumerate()
                .take(batch.completions[prompt_idx].len())
            {
                log_probs.push(batch.log_probs[prompt_idx][sample_idx]);
                relative_rewards.push(batch.relative_rewards[prompt_idx][sample_idx]);
                entropy.push(traj.entropy);
            }
        }

        (log_probs, relative_rewards, entropy)
    }

    fn save_checkpoint_for_iteration(
        runtime: &PolicyRuntime,
        iteration: usize,
        loss: f64,
    ) -> Result<()> {
        let checkpoint_dir = format!("checkpoints/rl-iter-{}", iteration);
        save_checkpoint(
            &runtime.varmap,
            &runtime.train_config,
            runtime.step,
            loss,
            &checkpoint_dir,
        )
        .map_err(|e| anyhow::anyhow!("Failed to save checkpoint to {}: {}", checkpoint_dir, e))?;
        println!("  Checkpoint saved to: {}", checkpoint_dir);
        Ok(())
    }

    /// Generate coding prompts (tasks for the model).
    fn generate_prompts(&self, n_prompts: usize) -> Vec<String> {
        let prompts: Vec<String> = vec![
            // === Core Language ===
            "Write a function to calculate the factorial of a number using recursion.",
            "Implement a struct representing a 2D point with methods for distance calculation.",
            "Create a function that filters even numbers from a vector using iterators.",
            "Write a function that reads a file and returns its contents as a String, handling errors properly.",
            "Implement a simple binary search tree with insert and search methods.",
            // Pattern matching
            "Write a function that uses pattern matching to parse a simple command string into an enum with variants like Quit, Echo(String), and Move { x: i32, y: i32 }.",
            "Implement a function that uses if let and while let to process a sequence of Option values.",
            // Error handling
            "Define a custom error enum with multiple variants and implement std::fmt::Display and std::error::Error for it. Write functions that return Result with this error type.",
            "Write a function that chains multiple fallible operations using the ? operator, converting between different error types with From implementations.",
            // Ownership & borrowing
            "Write a function that demonstrates ownership transfer, borrowing, and mutable borrowing. Include a struct with lifetime annotations.",
            "Implement a function that takes a closure as an argument with appropriate Fn/FnMut/FnOnce bounds and demonstrate each.",
            // Iterators
            "Implement a custom iterator that generates the Fibonacci sequence. Use it with map, filter, take, and collect.",
            "Write a function using iterator combinators (chain, zip, enumerate, flat_map, fold) to process two vectors into a HashMap.",
            // Traits & generics
            "Define a trait called Summary with a default method. Implement it for multiple structs. Write a function that accepts impl Summary and another that accepts &dyn Summary.",
            "Write a generic function with multiple trait bounds using where clauses. Demonstrate associated types in a trait.",
            // === Data Structures ===
            "Implement a stack data structure using a Vec with push, pop, peek, and is_empty methods.",
            "Implement a queue using two stacks (Vec) with enqueue and dequeue operations.",
            "Implement a singly linked list with push_front, pop_front, peek, and an iterator.",
            "Implement a HashMap from scratch using separate chaining with Vec<Vec<(K,V)>> buckets, with insert, get, and remove methods.",
            "Implement a min-heap with push, pop, and peek operations.",
            "Implement a trie (prefix tree) with insert, search, and starts_with methods.",
            "Implement a disjoint set (Union-Find) with path compression and union by rank.",
            "Implement an LRU cache with get and put operations using a HashMap and a doubly-linked list.",
            // === Algorithms ===
            "Implement quicksort for a Vec<i32> with a partition function.",
            "Implement merge sort for a Vec<i32>.",
            "Implement binary search that returns the index of a target in a sorted slice.",
            "Implement BFS and DFS on an adjacency list graph representation.",
            "Implement Dijkstra's shortest path algorithm using a BinaryHeap.",
            "Implement dynamic programming for the 0/1 knapsack problem.",
            "Implement the longest common subsequence algorithm for two strings.",
            "Implement the edit distance (Levenshtein distance) between two strings.",
            "Implement topological sort for a directed acyclic graph.",
            "Implement a function to find all strongly connected components using Tarjan's algorithm.",
            "Implement the Sieve of Eratosthenes to find all primes up to n.",
            "Implement matrix multiplication for Vec<Vec<f64>> matrices.",
            // === Rust Idioms & Patterns ===
            "Implement the Builder pattern for a Config struct with multiple optional fields.",
            "Implement a newtype wrapper around String that validates its contents (e.g., Email, Username).",
            "Implement a state machine using enums and methods that consume self and return the next state.",
            "Implement the RAII pattern with a custom struct that acquires a resource in new() and releases it in Drop.",
            "Demonstrate interior mutability using RefCell inside an Rc for a shared mutable tree structure.",
            "Implement From and Into conversions between multiple related types.",
            "Implement Display and Debug traits for a custom struct with formatted output.",
            "Write a struct that uses Cow<str> to avoid unnecessary cloning.",
            // === Systems Programming ===
            "Write a multi-threaded program that uses std::sync::mpsc channels to send messages between threads.",
            "Implement a simple thread pool that executes closures submitted to it.",
            "Write a program that uses Arc<Mutex<T>> to safely share and modify data across threads.",
            "Write a function that uses std::sync::atomic types (AtomicUsize, Ordering) to implement a lock-free counter.",
            "Write a program that reads a CSV file line by line, parses each line into a struct, and collects results into a Vec.",
            "Implement a simple command-line argument parser without external crates.",
            // === Advanced ===
            "Implement a generic sorting function that works on any type implementing Ord, with a custom comparator option.",
            "Write a recursive descent parser for simple arithmetic expressions (numbers, +, -, *, /, parentheses).",
            "Implement a simple event system using trait objects: an EventBus that can register handlers and dispatch events.",
            "Write a function that uses closures and higher-order functions to implement a pipeline of transformations on data.",
        ].into_iter().map(|s| s.to_string()).collect();
        prompts.into_iter().cycle().take(n_prompts).collect()
    }

    /// Template code generator retained for offline/unit-test paths.
    #[cfg(test)]
    fn generate_code(&self, prompt: &str) -> String {
        if prompt.contains("factorial") {
            r#"
pub fn factorial(n: u64) -> u64 {
    if n == 0 {
        1
    } else {
        n * factorial(n - 1)
    }
}
"#
            .to_string()
        } else if prompt.contains("point") {
            r#"
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}
"#
            .to_string()
        } else if prompt.contains("filter") {
            r#"
pub fn filter_even(nums: Vec<i32>) -> Vec<i32> {
    nums.into_iter()
        .filter(|x| x % 2 == 0)
        .collect()
}
"#
            .to_string()
        } else if prompt.contains("file") {
            r#"
use std::fs;
use std::io::Result;

pub fn read_file_contents(path: &str) -> Result<String> {
    fs::read_to_string(path)
}
"#
            .to_string()
        } else {
            r#"
pub fn example() {
    println!("Hello, world!");
}
"#
            .to_string()
        }
    }

    /// Log statistics to file.
    fn log_stats(&self, iteration: usize, stats: &GrpoStats) -> Result<()> {
        let log_path = "rl_training.log";
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        writeln!(
            file,
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            iteration,
            stats.avg_reward,
            stats.reward_std,
            stats.compile_success_rate,
            stats.parse_success_rate,
            stats.policy_loss,
            stats.kl_div,
            stats.grad_norm,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_generate_prompts() {
        let config = RLConfig::default();
        let trainer = RLTrainer::new(config).unwrap();

        let prompts = trainer.generate_prompts(3);
        assert_eq!(prompts.len(), 3);
        assert!(!prompts[0].is_empty());
    }

    #[test]
    fn test_generate_code() {
        let config = RLConfig::default();
        let trainer = RLTrainer::new(config).unwrap();

        let code = trainer.generate_code("factorial");
        assert!(code.contains("factorial"));
    }

    #[test]
    fn test_generate_prompts_cycles() {
        let config = RLConfig::default();
        let trainer = RLTrainer::new(config).unwrap();

        // Derive the pool size rather than hardcoding it, so growing the prompt
        // list doesn't break this test. One cycle must be duplicate-free, which
        // makes the first repeat of prompts[0] the cycle boundary.
        let sample = trainer.generate_prompts(1024);
        let pool_size = 1 + sample[1..]
            .iter()
            .position(|p| *p == sample[0])
            .expect("prompt list should repeat within 1024 draws");
        assert!(pool_size > 1, "prompt pool should hold more than one prompt");

        let unique: std::collections::HashSet<_> = sample[..pool_size].iter().collect();
        assert_eq!(
            unique.len(),
            pool_size,
            "prompts within one cycle must be distinct"
        );

        // Requesting past the pool size wraps back to the start.
        let n = pool_size + 2;
        let prompts = trainer.generate_prompts(n);
        assert_eq!(prompts.len(), n);
        assert_eq!(prompts[0], prompts[pool_size]);
        assert_eq!(prompts[1], prompts[pool_size + 1]);

        // Fewer than one cycle is truncation, not repetition.
        let short = trainer.generate_prompts(3);
        assert_eq!(short.len(), 3);
        assert_eq!(short.as_slice(), &sample[..3]);
    }

    #[test]
    fn test_generate_code_branches() {
        let config = RLConfig::default();
        let trainer = RLTrainer::new(config).unwrap();

        let point = trainer.generate_code("point");
        assert!(point.contains("struct Point"));
        assert!(point.contains("distance_to"));

        let filter = trainer.generate_code("filter");
        assert!(filter.contains("filter_even"));

        let file = trainer.generate_code("file");
        assert!(file.contains("read_file_contents"));

        let fallback = trainer.generate_code("something else");
        assert!(fallback.contains("example"));
    }

    #[test]
    fn test_parse_device_variants() {
        let cpu = RLTrainer::parse_device("cpu");
        assert!(matches!(cpu, Device::Cpu));

        // Ensure cuda-like strings do not panic even if CUDA is unavailable.
        let _ = RLTrainer::parse_device("cuda");
        let _ = RLTrainer::parse_device("cuda:0");
        let _ = RLTrainer::parse_device("cuda:invalid");
    }

    #[test]
    fn test_sample_token_with_stats_properties() {
        let logits = [0.5f32, 1.0, -0.25, 2.5];
        let (token, log_prob, entropy) = RLTrainer::sample_token_with_stats(&logits, 0.0);
        assert!(token < logits.len());
        assert!(log_prob.is_finite());
        assert!(entropy.is_finite());
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_sample_token_with_stats_zero_mass() {
        let logits: [f32; 0] = [];
        let (token, log_prob, entropy) = RLTrainer::sample_token_with_stats(&logits, 1.0);
        assert_eq!(token, 0);
        assert_eq!(log_prob, 0.0);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_flatten_batch_for_grpo_uses_completion_lengths() {
        let mut batch = GrpoBatch::new(vec!["p1".to_string(), "p2".to_string()], 2);
        batch.log_probs[0] = vec![-1.0, -2.0];
        batch.log_probs[1] = vec![-3.0];
        batch.relative_rewards[0] = vec![0.5, -0.5];
        batch.relative_rewards[1] = vec![1.5];
        batch.completions[0] = vec!["a".to_string(), "b".to_string()];
        batch.completions[1] = vec!["c".to_string()];

        let trajectories = vec![
            vec![
                GeneratedSample {
                    completion: "a".to_string(),
                    full_tokens: vec![1, 2],
                    prompt_len: 1,
                    log_prob: -1.0,
                    entropy: 0.1,
                },
                GeneratedSample {
                    completion: "b".to_string(),
                    full_tokens: vec![1, 2, 3],
                    prompt_len: 1,
                    log_prob: -2.0,
                    entropy: 0.2,
                },
            ],
            vec![
                GeneratedSample {
                    completion: "c".to_string(),
                    full_tokens: vec![1, 4],
                    prompt_len: 1,
                    log_prob: -3.0,
                    entropy: 0.3,
                },
                // Extra trajectory should be ignored because completions len is 1.
                GeneratedSample {
                    completion: "ignored".to_string(),
                    full_tokens: vec![9],
                    prompt_len: 1,
                    log_prob: -9.0,
                    entropy: 9.0,
                },
            ],
        ];

        let (log_probs, relative_rewards, entropy) =
            RLTrainer::flatten_batch_for_grpo(&batch, &trajectories);
        assert_eq!(log_probs, vec![-1.0, -2.0, -3.0]);
        assert_eq!(relative_rewards, vec![0.5, -0.5, 1.5]);
        assert_eq!(entropy, vec![0.1, 0.2, 0.3]);
    }
}
