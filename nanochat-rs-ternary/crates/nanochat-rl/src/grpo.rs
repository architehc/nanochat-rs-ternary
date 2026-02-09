//! GRPO: Group Relative Policy Optimization
//!
//! Implements GRPO algorithm for RL training of code generation models.
//! GRPO uses relative rewards within groups of samples to stabilize training.
//!
//! Algorithm:
//! 1. For each prompt, sample N completions from current policy
//! 2. Evaluate each completion (compiler + AST)
//! 3. Compute relative rewards within the group: r_i = (reward_i - mean) / std
//! 4. Update policy using these relative rewards as advantage estimates
//!
//! Benefits:
//! - More stable than absolute rewards (less sensitive to reward scale)
//! - Naturally normalized (zero mean, unit variance)
//! - Encourages diversity through group comparison

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GRPO algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    /// Learning rate for policy updates
    pub learning_rate: f64,

    /// KL divergence penalty coefficient
    pub kl_coef: f64,

    /// Clip ratio for PPO-style clipping (optional)
    pub clip_ratio: Option<f64>,

    /// Entropy bonus coefficient (encourage exploration)
    pub entropy_coef: f64,

    /// Value function coefficient (if using critic)
    pub value_coef: f64,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,

    /// Number of optimization epochs per batch
    pub n_epochs: usize,

    /// Mini-batch size for optimization
    pub mini_batch_size: usize,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            kl_coef: 0.1,
            clip_ratio: Some(0.2), // PPO-style clipping
            entropy_coef: 0.01,
            value_coef: 0.5,
            max_grad_norm: 1.0,
            n_epochs: 4,
            mini_batch_size: 4,
        }
    }
}

/// Training statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoStats {
    /// Average reward
    pub avg_reward: f64,

    /// Average relative reward
    pub avg_relative_reward: f64,

    /// Reward standard deviation
    pub reward_std: f64,

    /// Policy loss
    pub policy_loss: f64,

    /// Value loss (if using critic)
    pub value_loss: f64,

    /// Entropy
    pub entropy: f64,

    /// KL divergence from reference policy
    pub kl_div: f64,

    /// Gradient norm
    pub grad_norm: f64,

    /// Compilation success rate
    pub compile_success_rate: f64,

    /// Average AST parseability rate
    pub parse_success_rate: f64,
}

/// GRPO trainer
pub struct GrpoTrainer {
    config: GrpoConfig,
}

impl GrpoTrainer {
    pub fn new(config: GrpoConfig) -> Self {
        Self { config }
    }

    /// Compute policy gradient loss for a batch of samples
    ///
    /// This is the core GRPO algorithm:
    /// loss = -mean(log_probs * relative_rewards)
    ///
    /// Where:
    /// - log_probs: log probability of generated tokens under current policy
    /// - relative_rewards: normalized rewards (zero mean, unit variance per group)
    pub fn compute_loss(
        &self,
        log_probs: &[f64],      // Log probabilities of generated sequences
        relative_rewards: &[f64], // Relative rewards (normalized)
        ref_log_probs: Option<&[f64]>, // Reference policy log probs (for KL penalty)
        entropy: &[f64],        // Entropy of policy distribution
    ) -> f64 {
        assert_eq!(log_probs.len(), relative_rewards.len());

        let n = log_probs.len() as f64;
        let mut loss = 0.0;

        // Policy gradient loss: -E[log_prob * advantage]
        for i in 0..log_probs.len() {
            loss -= log_probs[i] * relative_rewards[i];
        }
        loss /= n;

        // KL divergence penalty (if reference policy provided)
        if let Some(ref_probs) = ref_log_probs {
            let mut kl = 0.0;
            for i in 0..log_probs.len() {
                kl += ref_probs[i] - log_probs[i];
            }
            kl /= n;
            loss += self.config.kl_coef * kl;
        }

        // Entropy bonus (encourage exploration)
        let avg_entropy: f64 = entropy.iter().sum::<f64>() / n;
        loss -= self.config.entropy_coef * avg_entropy;

        loss
    }

    /// Compute PPO-style clipped loss (optional, more stable)
    pub fn compute_clipped_loss(
        &self,
        log_probs: &[f64],
        old_log_probs: &[f64],
        relative_rewards: &[f64],
    ) -> f64 {
        let clip_ratio = self.config.clip_ratio.unwrap_or(0.2);
        let n = log_probs.len() as f64;
        let mut loss = 0.0;

        for i in 0..log_probs.len() {
            // Importance sampling ratio
            let ratio = (log_probs[i] - old_log_probs[i]).exp();

            // Clipped ratio
            let clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio);

            // Take minimum of clipped and unclipped objective
            let unclipped = ratio * relative_rewards[i];
            let clipped = clipped_ratio * relative_rewards[i];
            loss -= unclipped.min(clipped);
        }

        loss / n
    }
}

/// Batch of training samples for GRPO
#[derive(Debug, Clone)]
pub struct GrpoBatch {
    /// Prompts (coding tasks)
    pub prompts: Vec<String>,

    /// Generated completions (n_samples per prompt)
    pub completions: Vec<Vec<String>>,

    /// Rewards for each completion
    pub rewards: Vec<Vec<f64>>,

    /// Relative rewards (normalized within each group)
    pub relative_rewards: Vec<Vec<f64>>,

    /// Log probabilities of completions under current policy
    pub log_probs: Vec<Vec<f64>>,

    /// Metadata (compile success, parse success, etc.)
    pub metadata: Vec<Vec<HashMap<String, f64>>>,
}

impl GrpoBatch {
    /// Create a new batch
    pub fn new(prompts: Vec<String>, n_samples: usize) -> Self {
        let n_prompts = prompts.len();
        Self {
            prompts,
            completions: vec![Vec::with_capacity(n_samples); n_prompts],
            rewards: vec![Vec::with_capacity(n_samples); n_prompts],
            relative_rewards: vec![Vec::with_capacity(n_samples); n_prompts],
            log_probs: vec![Vec::with_capacity(n_samples); n_prompts],
            metadata: vec![Vec::with_capacity(n_samples); n_prompts],
        }
    }

    /// Add a completion to the batch
    pub fn add_completion(
        &mut self,
        prompt_idx: usize,
        completion: String,
        reward: f64,
        log_prob: f64,
        metadata: HashMap<String, f64>,
    ) {
        self.completions[prompt_idx].push(completion);
        self.rewards[prompt_idx].push(reward);
        self.log_probs[prompt_idx].push(log_prob);
        self.metadata[prompt_idx].push(metadata);
    }

    /// Normalize rewards within each group (core GRPO step)
    pub fn normalize_rewards(&mut self) {
        for i in 0..self.prompts.len() {
            let rewards = &self.rewards[i];
            if rewards.is_empty() {
                continue;
            }

            // Compute mean and std
            let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
            let variance = rewards.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / rewards.len() as f64;
            let std = variance.sqrt().max(1e-6); // Avoid division by zero

            // Normalize
            self.relative_rewards[i] = rewards.iter()
                .map(|r| (r - mean) / std)
                .collect();
        }
    }

    /// Compute statistics for this batch
    pub fn compute_stats(&self) -> GrpoStats {
        let mut total_reward = 0.0;
        let mut total_relative = 0.0;
        let mut n_samples = 0;
        let mut compile_success = 0;
        let mut parse_success = 0;

        for i in 0..self.prompts.len() {
            for j in 0..self.rewards[i].len() {
                total_reward += self.rewards[i][j];
                total_relative += self.relative_rewards[i][j];
                n_samples += 1;

                if let Some(compile) = self.metadata[i][j].get("compile_success") {
                    if *compile > 0.5 {
                        compile_success += 1;
                    }
                }

                if let Some(parse) = self.metadata[i][j].get("parse_success") {
                    if *parse > 0.5 {
                        parse_success += 1;
                    }
                }
            }
        }

        let n = n_samples as f64;

        // Compute reward std
        let avg_reward = total_reward / n;
        let mut variance = 0.0;
        for rewards in &self.rewards {
            for r in rewards {
                variance += (r - avg_reward).powi(2);
            }
        }
        let reward_std = (variance / n).sqrt();

        GrpoStats {
            avg_reward,
            avg_relative_reward: total_relative / n,
            reward_std,
            policy_loss: 0.0,  // Will be filled during optimization
            value_loss: 0.0,
            entropy: 0.0,
            kl_div: 0.0,
            grad_norm: 0.0,
            compile_success_rate: compile_success as f64 / n,
            parse_success_rate: parse_success as f64 / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpo_batch_normalize() {
        let mut batch = GrpoBatch::new(vec!["prompt1".to_string()], 3);

        // Add samples with different rewards
        batch.add_completion(0, "code1".to_string(), 10.0, -5.0, HashMap::new());
        batch.add_completion(0, "code2".to_string(), 5.0, -6.0, HashMap::new());
        batch.add_completion(0, "code3".to_string(), 15.0, -4.0, HashMap::new());

        batch.normalize_rewards();

        // Check that relative rewards have mean ~0
        let mean: f64 = batch.relative_rewards[0].iter().sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-6, "Mean should be near zero: {}", mean);

        // Check that highest reward maps to highest relative reward
        let max_idx = batch.rewards[0].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        assert!(batch.relative_rewards[0][max_idx] > 0.0);
    }

    #[test]
    fn test_compute_loss() {
        let config = GrpoConfig::default();
        let trainer = GrpoTrainer::new(config);

        let log_probs = vec![-2.0, -3.0, -1.5];
        let relative_rewards = vec![1.0, -0.5, 0.5];
        let entropy = vec![0.5, 0.6, 0.4];

        let loss = trainer.compute_loss(&log_probs, &relative_rewards, None, &entropy);

        // Loss should be finite
        assert!(loss.is_finite());
    }
}
