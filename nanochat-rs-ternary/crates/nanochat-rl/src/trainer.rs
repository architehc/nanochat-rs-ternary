//! Main RL trainer that orchestrates the complete training loop
//!
//! Combines:
//! - Code generation from policy model
//! - Compiler feedback
//! - AST analysis
//! - Optional Qwen3 evaluation
//! - GRPO policy updates

use crate::compiler::CompilerFeedback;
use crate::ast_analysis::analyze_ast;
use crate::reward::compute_reward;
use crate::grpo::{GrpoTrainer, GrpoBatch, GrpoStats};
use crate::qwen::{QwenClient, qwen_to_reward};
use crate::RLConfig;

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::Write;

/// Main RL trainer
pub struct RLTrainer {
    config: RLConfig,
    compiler: CompilerFeedback,
    grpo: GrpoTrainer,
    qwen: Option<QwenClient>,
    iteration: usize,
}

impl RLTrainer {
    /// Create a new RL trainer
    pub fn new(config: RLConfig) -> Result<Self> {
        let compiler = CompilerFeedback::new()
            .context("Failed to create compiler feedback")?;

        let grpo = GrpoTrainer::new(config.grpo.clone());

        let qwen = config.qwen_endpoint.as_ref().map(|endpoint| {
            QwenClient::new(endpoint.clone(), None)
        });

        Ok(Self {
            config,
            compiler,
            grpo,
            qwen,
            iteration: 0,
        })
    }

    /// Run the complete RL training loop
    pub async fn train(&mut self) -> Result<()> {
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
        println!("  Qwen3 endpoint:   {}", self.config.qwen_endpoint.as_ref().unwrap_or(&"None".to_string()));
        println!();

        // TODO: Load model from checkpoint
        // let model = load_model(&self.config.base_checkpoint, &self.config.device)?;

        for iter in 0..self.config.n_iterations {
            self.iteration = iter;

            println!("───────────────────────────────────────────────────────────");
            println!("Iteration {}/{}", iter + 1, self.config.n_iterations);
            println!("───────────────────────────────────────────────────────────");

            // 1. Generate coding prompts
            let prompts = self.generate_prompts(self.config.batch_size);

            // 2. Sample completions for each prompt
            let mut batch = GrpoBatch::new(prompts.clone(), self.config.n_samples);

            for (prompt_idx, prompt) in prompts.iter().enumerate() {
                println!("\nPrompt {}: {}", prompt_idx + 1, prompt);

                for sample_idx in 0..self.config.n_samples {
                    // Generate code (TODO: use actual model)
                    let code = self.generate_code(&prompt)?;
                    println!("  Sample {}: {} chars", sample_idx + 1, code.len());

                    // Evaluate with compiler
                    let compile_result = self.compiler.compile(&code)?;
                    let compile_success = compile_result.success;

                    // Analyze AST
                    let ast_metrics = analyze_ast(&code)?;
                    let parse_success = ast_metrics.parseable;

                    // Compute base reward
                    let mut reward = compute_reward(
                        &compile_result,
                        &ast_metrics,
                        &self.config.reward,
                    );

                    // Optional: Add Qwen3 evaluation
                    if let Some(qwen) = &self.qwen {
                        match qwen.evaluate_code(&code, prompt).await {
                            Ok(eval) => {
                                let qwen_reward = qwen_to_reward(&eval, 2.0);
                                reward += qwen_reward;
                                println!("    Qwen3: {:.1}/10 (reward: {:.2})",
                                         (eval.quality_score + eval.correctness_score + eval.idiomaticity_score) / 3.0,
                                         qwen_reward);
                            }
                            Err(e) => {
                                eprintln!("    Qwen3 evaluation failed: {}", e);
                            }
                        }
                    }

                    println!("    Compile: {} | Parse: {} | Reward: {:.2}",
                             if compile_success { "✓" } else { "✗" },
                             if parse_success { "✓" } else { "✗" },
                             reward);

                    // Metadata for stats
                    let mut metadata = HashMap::new();
                    metadata.insert("compile_success".to_string(), if compile_success { 1.0 } else { 0.0 });
                    metadata.insert("parse_success".to_string(), if parse_success { 1.0 } else { 0.0 });

                    // TODO: Get actual log_prob from model
                    let log_prob = -5.0; // Placeholder

                    batch.add_completion(prompt_idx, code, reward, log_prob, metadata);
                }
            }

            // 3. Normalize rewards within each group (GRPO step)
            batch.normalize_rewards();

            // 4. Compute statistics
            let stats = batch.compute_stats();

            println!("\nIteration {} Statistics:", iter + 1);
            println!("  Avg Reward:         {:.2}", stats.avg_reward);
            println!("  Reward Std:         {:.2}", stats.reward_std);
            println!("  Compile Success:    {:.1}%", stats.compile_success_rate * 100.0);
            println!("  Parse Success:      {:.1}%", stats.parse_success_rate * 100.0);

            // 5. Update policy (TODO: implement actual gradient update)
            // let loss = self.grpo.compute_loss(...);
            // update_model_parameters(model, loss);

            // 6. Save checkpoint
            if (iter + 1) % 10 == 0 {
                println!("  Saving checkpoint...");
                self.save_checkpoint(iter + 1)?;
            }

            // 7. Log to file
            self.log_stats(iter + 1, &stats)?;

            println!();
        }

        println!("═══════════════════════════════════════════════════════════");
        println!("  Training Complete!");
        println!("═══════════════════════════════════════════════════════════");

        Ok(())
    }

    /// Generate coding prompts (tasks for the model)
    fn generate_prompts(&self, n_prompts: usize) -> Vec<String> {
        // For now, use a diverse set of Rust coding tasks
        // In production, these would come from a dataset
        vec![
            "Write a function to calculate the factorial of a number using recursion.".to_string(),
            "Implement a struct representing a 2D point with methods for distance calculation.".to_string(),
            "Create a function that filters even numbers from a vector using iterators.".to_string(),
            "Write a function that reads a file and returns its contents as a String, handling errors properly.".to_string(),
            "Implement a simple binary search tree with insert and search methods.".to_string(),
        ]
        .into_iter()
        .cycle()
        .take(n_prompts)
        .collect()
    }

    /// Generate code for a prompt (placeholder - will use actual model)
    fn generate_code(&self, prompt: &str) -> Result<String> {
        // TODO: Use actual model for generation
        // For now, return template code based on prompt keywords

        if prompt.contains("factorial") {
            Ok(r#"
pub fn factorial(n: u64) -> u64 {
    if n == 0 {
        1
    } else {
        n * factorial(n - 1)
    }
}
"#.to_string())
        } else if prompt.contains("point") {
            Ok(r#"
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
"#.to_string())
        } else if prompt.contains("filter") {
            Ok(r#"
pub fn filter_even(nums: Vec<i32>) -> Vec<i32> {
    nums.into_iter()
        .filter(|x| x % 2 == 0)
        .collect()
}
"#.to_string())
        } else if prompt.contains("file") {
            Ok(r#"
use std::fs;
use std::io::Result;

pub fn read_file_contents(path: &str) -> Result<String> {
    fs::read_to_string(path)
}
"#.to_string())
        } else {
            Ok(r#"
pub fn example() {
    println!("Hello, world!");
}
"#.to_string())
        }
    }

    /// Save checkpoint
    fn save_checkpoint(&self, iteration: usize) -> Result<()> {
        // TODO: Implement actual checkpoint saving
        let checkpoint_dir = format!("checkpoints/rl-iter-{}", iteration);
        std::fs::create_dir_all(&checkpoint_dir)?;
        println!("  Checkpoint saved to: {}", checkpoint_dir);
        Ok(())
    }

    /// Log statistics to file
    fn log_stats(&self, iteration: usize, stats: &GrpoStats) -> Result<()> {
        let log_path = "rl_training.log";
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        writeln!(
            file,
            "{},{:.4},{:.4},{:.4},{:.4}",
            iteration,
            stats.avg_reward,
            stats.reward_std,
            stats.compile_success_rate,
            stats.parse_success_rate,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let code = trainer.generate_code("Write a factorial function").unwrap();
        assert!(code.contains("factorial"));
    }
}
