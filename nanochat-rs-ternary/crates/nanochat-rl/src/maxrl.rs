//! MaxRL: Maximum Likelihood Reinforcement Learning
//!
//! Based on "Maximum Likelihood Reinforcement Learning" (alphaxiv.org/abs/2602.02710)
//!
//! Key improvement over GRPO: Directly maximizes likelihood of correct outputs
//! rather than just relative rewards. Achieves 20x better test-time scaling.
//!
//! Perfect for code generation where correctness is binary (compiles or not).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MaxRL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxRLConfig {
    /// Learning rate for policy updates
    pub learning_rate: f64,

    /// Correctness threshold (reward above which sample is considered "correct")
    pub correctness_threshold: f64,

    /// Temperature for advantage weighting (beta in paper)
    pub temperature: f64,

    /// KL divergence penalty coefficient
    pub kl_coef: f64,

    /// Number of samples per prompt
    pub n_samples: usize,

    /// Clip ratio for PPO-style clipping
    pub clip_ratio: Option<f64>,

    /// Maximum gradient norm
    pub max_grad_norm: f64,
}

impl Default for MaxRLConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            correctness_threshold: 20.0, // Reward > 20 = "correct"
            temperature: 1.0,            // Beta parameter
            kl_coef: 0.1,
            n_samples: 8, // More samples for better ML estimate
            clip_ratio: Some(0.2),
            max_grad_norm: 1.0,
        }
    }
}

/// MaxRL trainer
pub struct MaxRLTrainer {
    config: MaxRLConfig,
}

impl MaxRLTrainer {
    pub fn new(config: MaxRLConfig) -> Self {
        Self { config }
    }

    /// Compute MaxRL loss
    ///
    /// Key difference from GRPO:
    /// - GRPO: Uses all samples with relative rewards
    /// - MaxRL: Maximizes likelihood of only "correct" samples
    ///
    /// MaxRL objective:
    /// L = -E[log π(a|s) | reward(a,s) > threshold]
    ///
    /// With temperature-based weighting:
    /// L = -E[w(r) * log π(a|s)]  where w(r) = exp((r - threshold) / temp)
    pub fn compute_maxrl_loss(
        &self,
        log_probs: &[f64],             // Log probabilities of generated sequences
        rewards: &[f64],               // Rewards for each sequence
        ref_log_probs: Option<&[f64]>, // Reference policy (for KL)
    ) -> (f64, MaxRLStats) {
        assert_eq!(log_probs.len(), rewards.len());

        let n = log_probs.len() as f64;
        let mut loss = 0.0;
        let mut n_correct = 0;
        let mut total_weight = 0.0;

        // Compute correctness-weighted loss
        for i in 0..log_probs.len() {
            if rewards[i] > self.config.correctness_threshold {
                // This sample is "correct" - include in ML objective
                n_correct += 1;

                // Temperature-based weighting (higher reward = higher weight)
                // Clamp exponent to avoid f64 overflow (exp(709) ≈ f64::MAX)
                let excess_reward = rewards[i] - self.config.correctness_threshold;
                let exponent = (excess_reward / self.config.temperature).clamp(-500.0, 500.0);
                let weight = exponent.exp();

                loss -= weight * log_probs[i];
                total_weight += weight;
            }
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            loss /= total_weight;
        } else {
            // No correct samples - fallback to standard policy gradient
            // with penalties for all samples
            for i in 0..log_probs.len() {
                // Negative reward as penalty
                let penalty = -rewards[i].min(0.0);
                loss += penalty * log_probs[i];
            }
            loss /= n;
        }

        // KL divergence penalty (if reference policy provided)
        if let Some(ref_probs) = ref_log_probs {
            assert_eq!(
                ref_probs.len(),
                log_probs.len(),
                "ref_log_probs length ({}) must match log_probs length ({})",
                ref_probs.len(),
                log_probs.len()
            );
            let mut kl = 0.0;
            for i in 0..log_probs.len() {
                kl += log_probs[i] - ref_probs[i];
            }
            kl /= n;
            loss += self.config.kl_coef * kl;
        }

        let stats = MaxRLStats {
            loss,
            n_correct,
            n_total: log_probs.len(),
            correctness_rate: n_correct as f64 / n,
            avg_correct_reward: if n_correct > 0 {
                rewards
                    .iter()
                    .zip(std::iter::repeat(()))
                    .filter(|(r, _)| **r > self.config.correctness_threshold)
                    .map(|(r, _)| r)
                    .sum::<f64>()
                    / n_correct as f64
            } else {
                0.0
            },
        };

        (loss, stats)
    }

    /// Compute MaxRL loss with advantage normalization
    ///
    /// This variant normalizes advantages within correct samples only,
    /// providing more stable gradients while maintaining ML objective.
    pub fn compute_maxrl_loss_normalized(
        &self,
        log_probs: &[f64],
        rewards: &[f64],
    ) -> (f64, MaxRLStats) {
        // Separate correct and incorrect samples
        let correct_indices: Vec<usize> = rewards
            .iter()
            .enumerate()
            .filter(|(_, &r)| r > self.config.correctness_threshold)
            .map(|(i, _)| i)
            .collect();

        let n_correct = correct_indices.len();

        if n_correct == 0 {
            // No correct samples - return high loss
            return (
                1000.0,
                MaxRLStats {
                    loss: 1000.0,
                    n_correct: 0,
                    n_total: log_probs.len(),
                    correctness_rate: 0.0,
                    avg_correct_reward: 0.0,
                },
            );
        }

        // Compute normalized advantages for correct samples
        let correct_rewards: Vec<f64> = correct_indices.iter().map(|&i| rewards[i]).collect();

        let mean = correct_rewards.iter().sum::<f64>() / n_correct as f64;
        let variance = correct_rewards
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / n_correct as f64;
        let std_dev = variance.sqrt().max(1e-6);

        // Compute loss with normalized advantages
        let mut loss = 0.0;
        for &i in &correct_indices {
            let normalized_advantage = (rewards[i] - mean) / std_dev;
            let weight = (normalized_advantage / self.config.temperature).exp();
            loss -= weight * log_probs[i];
        }

        loss /= n_correct as f64;

        let stats = MaxRLStats {
            loss,
            n_correct,
            n_total: log_probs.len(),
            correctness_rate: n_correct as f64 / log_probs.len() as f64,
            avg_correct_reward: mean,
        };

        (loss, stats)
    }
}

/// MaxRL training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxRLStats {
    /// Loss value
    pub loss: f64,

    /// Number of correct samples (reward > threshold)
    pub n_correct: usize,

    /// Total number of samples
    pub n_total: usize,

    /// Correctness rate (n_correct / n_total)
    pub correctness_rate: f64,

    /// Average reward of correct samples
    pub avg_correct_reward: f64,
}

/// MaxRL batch with correctness filtering
#[derive(Debug, Clone)]
pub struct MaxRLBatch {
    /// Prompts
    pub prompts: Vec<String>,

    /// All generated completions
    pub completions: Vec<Vec<String>>,

    /// Rewards for each completion
    pub rewards: Vec<Vec<f64>>,

    /// Log probabilities
    pub log_probs: Vec<Vec<f64>>,

    /// Metadata
    pub metadata: Vec<Vec<HashMap<String, f64>>>,

    /// Filtered indices of correct samples
    pub correct_indices: Vec<Vec<usize>>,
}

impl MaxRLBatch {
    /// Create a new MaxRL batch
    pub fn new(prompts: Vec<String>, n_samples: usize) -> Self {
        let n_prompts = prompts.len();
        Self {
            prompts,
            completions: vec![Vec::with_capacity(n_samples); n_prompts],
            rewards: vec![Vec::with_capacity(n_samples); n_prompts],
            log_probs: vec![Vec::with_capacity(n_samples); n_prompts],
            metadata: vec![Vec::with_capacity(n_samples); n_prompts],
            correct_indices: vec![Vec::new(); n_prompts],
        }
    }

    /// Filter correct samples based on threshold
    pub fn filter_correct(&mut self, threshold: f64) {
        for i in 0..self.prompts.len() {
            self.correct_indices[i] = self.rewards[i]
                .iter()
                .enumerate()
                .filter(|(_, &r)| r > threshold)
                .map(|(idx, _)| idx)
                .collect();
        }
    }

    /// Get statistics about correctness
    pub fn correctness_stats(&self) -> (usize, usize, f64) {
        let total_correct: usize = self.correct_indices.iter().map(|v| v.len()).sum();
        let total_samples: usize = self.rewards.iter().map(|v| v.len()).sum();
        let rate = total_correct as f64 / total_samples.max(1) as f64;

        (total_correct, total_samples, rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxrl_all_correct() {
        let config = MaxRLConfig {
            correctness_threshold: 20.0,
            temperature: 1.0,
            ..Default::default()
        };
        let trainer = MaxRLTrainer::new(config);

        // All samples are correct
        let log_probs = vec![-2.0, -3.0, -2.5];
        let rewards = vec![25.0, 30.0, 22.0];

        let (loss, stats) = trainer.compute_maxrl_loss(&log_probs, &rewards, None);

        assert!(loss.is_finite());
        assert_eq!(stats.n_correct, 3);
        assert_eq!(stats.correctness_rate, 1.0);
        assert!(stats.avg_correct_reward > 20.0);
    }

    #[test]
    fn test_maxrl_mixed_correct() {
        let config = MaxRLConfig {
            correctness_threshold: 20.0,
            ..Default::default()
        };
        let trainer = MaxRLTrainer::new(config);

        // Mix of correct and incorrect
        let log_probs = vec![-2.0, -3.0, -2.5, -4.0];
        let rewards = vec![25.0, 10.0, 22.0, -5.0];

        let (loss, stats) = trainer.compute_maxrl_loss(&log_probs, &rewards, None);

        assert!(loss.is_finite());
        assert_eq!(stats.n_correct, 2); // Only first and third
        assert_eq!(stats.correctness_rate, 0.5);
    }

    #[test]
    fn test_maxrl_none_correct() {
        let config = MaxRLConfig {
            correctness_threshold: 20.0,
            ..Default::default()
        };
        let trainer = MaxRLTrainer::new(config);

        // No correct samples
        let log_probs = vec![-2.0, -3.0, -2.5];
        let rewards = vec![10.0, 5.0, -10.0];

        let (loss, stats) = trainer.compute_maxrl_loss(&log_probs, &rewards, None);

        assert!(loss.is_finite());
        assert_eq!(stats.n_correct, 0);
        assert_eq!(stats.correctness_rate, 0.0);
    }

    #[test]
    fn test_maxrl_batch_filtering() {
        let mut batch = MaxRLBatch::new(vec!["prompt1".to_string()], 4);

        batch.rewards[0] = vec![25.0, 10.0, 30.0, -5.0];
        batch.filter_correct(20.0);

        assert_eq!(batch.correct_indices[0], vec![0, 2]); // First and third are correct

        let (n_correct, n_total, rate) = batch.correctness_stats();
        assert_eq!(n_correct, 2);
        assert_eq!(n_total, 4);
        assert_eq!(rate, 0.5);
    }
}
