//! Training-Free GRPO: Group Relative Policy Optimization without Backprop
//!
//! Based on: "Training-Free Group Relative Policy Optimization" (arXiv:2024.xxxxx)
//!
//! Key innovation: Learn from experience library of successful rollouts
//! without gradient-based optimization. Uses semantic advantages to guide
//! token selection during generation.
//!
//! Benefits:
//! - Zero backprop cost (no gradient computation)
//! - Continual learning from sparse feedback
//! - Memory-efficient (stores token priors, not gradients)
//! - Compatible with any reward function

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Experience from a successful rollout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Input prompt
    pub prompt: String,

    /// Generated response
    pub response: String,

    /// Advantage score (relative to group mean)
    pub advantage: f64,

    /// Timestamp for experience aging
    pub timestamp: u64,

    /// Optional metadata (task ID, domain, etc.)
    pub metadata: HashMap<String, String>,
}

impl Experience {
    /// Create new experience
    pub fn new(prompt: String, response: String, advantage: f64) -> Self {
        Self {
            prompt,
            response,
            advantage,
            timestamp: Self::current_timestamp(),
            metadata: HashMap::new(),
        }
    }

    /// Get current Unix timestamp
    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Age of experience in seconds
    pub fn age_seconds(&self) -> u64 {
        Self::current_timestamp().saturating_sub(self.timestamp)
    }

    /// Check if experience is stale (older than threshold)
    pub fn is_stale(&self, max_age_seconds: u64) -> bool {
        self.age_seconds() > max_age_seconds
    }
}

/// Configuration for Training-Free GRPO
#[derive(Debug, Clone)]
pub struct GRPOConfig {
    /// Number of rollouts per prompt for group comparison
    pub group_size: usize,

    /// Maximum experiences to store in library
    pub max_library_size: usize,

    /// Maximum age of experiences in seconds (for pruning)
    pub max_experience_age: u64,

    /// Minimum advantage to store experience
    pub min_advantage_threshold: f64,

    /// Temperature for sampling from experience distribution
    pub sampling_temperature: f64,

    /// Whether to use advantage weighting for sampling
    pub use_advantage_weighting: bool,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            group_size: 8,                      // 8 rollouts per prompt
            max_library_size: 10000,            // 10K experiences max
            max_experience_age: 86400 * 7,      // 1 week
            min_advantage_threshold: 0.0,       // Only store positive advantage
            sampling_temperature: 1.0,          // Neutral temperature
            use_advantage_weighting: true,      // Weight by advantage
        }
    }
}

/// Training-Free GRPO experience library
pub struct TrainingFreeGRPO {
    config: GRPOConfig,

    /// Experience library (successful rollouts)
    experience_library: Vec<Experience>,

    /// Statistics
    total_rollouts: usize,
    total_experiences_stored: usize,
    total_experiences_pruned: usize,
}

impl TrainingFreeGRPO {
    /// Create new Training-Free GRPO with default config
    pub fn new() -> Self {
        Self::with_config(GRPOConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GRPOConfig) -> Self {
        Self {
            config,
            experience_library: Vec::new(),
            total_rollouts: 0,
            total_experiences_stored: 0,
            total_experiences_pruned: 0,
        }
    }

    /// Generate group rollouts and extract successful experiences
    ///
    /// # Arguments
    /// * `rollouts` - Pre-generated rollouts (prompt, response pairs)
    /// * `rewards` - Reward for each rollout
    ///
    /// # Algorithm
    /// 1. Compute mean reward across group
    /// 2. Calculate advantage for each rollout (reward - mean)
    /// 3. Store experiences with positive advantage
    /// 4. Prune library if exceeds max size
    pub fn add_rollout_group(
        &mut self,
        rollouts: Vec<(String, String)>, // (prompt, response)
        rewards: Vec<f64>,
    ) {
        assert_eq!(rollouts.len(), rewards.len());
        assert_eq!(rollouts.len(), self.config.group_size);

        self.total_rollouts += rollouts.len();

        // Compute group statistics
        let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;

        // Extract successful experiences (above mean)
        for ((prompt, response), &reward) in rollouts.iter().zip(rewards.iter()) {
            let advantage = reward - mean_reward;

            if advantage > self.config.min_advantage_threshold {
                let experience = Experience::new(
                    prompt.clone(),
                    response.clone(),
                    advantage,
                );

                self.experience_library.push(experience);
                self.total_experiences_stored += 1;
            }
        }

        // Prune if library too large
        self.prune_library();
    }

    /// Prune experience library to stay within size limits
    ///
    /// Strategy:
    /// 1. Remove stale experiences (age > max_age)
    /// 2. If still too large, keep highest advantage experiences
    fn prune_library(&mut self) {
        let initial_size = self.experience_library.len();

        // Remove stale experiences
        self.experience_library.retain(|exp| {
            !exp.is_stale(self.config.max_experience_age)
        });

        let after_age_prune = self.experience_library.len();
        self.total_experiences_pruned += initial_size - after_age_prune;

        // If still too large, keep top experiences by advantage
        if self.experience_library.len() > self.config.max_library_size {
            // Sort by advantage (descending)
            self.experience_library.sort_by(|a, b| {
                b.advantage.partial_cmp(&a.advantage).unwrap()
            });

            // Truncate to max size
            let to_remove = self.experience_library.len() - self.config.max_library_size;
            self.experience_library.truncate(self.config.max_library_size);
            self.total_experiences_pruned += to_remove;
        }
    }

    /// Sample an experience from library
    ///
    /// Uses advantage-weighted sampling if enabled
    pub fn sample_experience(&self) -> Option<&Experience> {
        if self.experience_library.is_empty() {
            return None;
        }

        if !self.config.use_advantage_weighting {
            // Uniform sampling
            use rand::seq::SliceRandom;
            return self.experience_library.choose(&mut rand::thread_rng());
        }

        // Advantage-weighted sampling with temperature
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Compute weights: exp(advantage / temperature)
        let weights: Vec<f64> = self.experience_library
            .iter()
            .map(|exp| (exp.advantage / self.config.sampling_temperature).exp())
            .collect();

        let total_weight: f64 = weights.iter().sum();
        let mut cumulative = 0.0;
        let target = rng.gen::<f64>() * total_weight;

        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight;
            if cumulative >= target {
                return Some(&self.experience_library[i]);
            }
        }

        // Fallback (shouldn't happen)
        self.experience_library.last()
    }

    /// Get experiences matching a specific prompt (or prefix)
    pub fn get_experiences_for_prompt(&self, prompt: &str) -> Vec<&Experience> {
        self.experience_library
            .iter()
            .filter(|exp| exp.prompt.starts_with(prompt))
            .collect()
    }

    /// Get statistics about the experience library
    pub fn stats(&self) -> GRPOStats {
        let avg_advantage = if !self.experience_library.is_empty() {
            self.experience_library.iter().map(|e| e.advantage).sum::<f64>()
                / self.experience_library.len() as f64
        } else {
            0.0
        };

        let max_advantage = self.experience_library
            .iter()
            .map(|e| e.advantage)
            .fold(f64::NEG_INFINITY, f64::max);

        let min_advantage = self.experience_library
            .iter()
            .map(|e| e.advantage)
            .fold(f64::INFINITY, f64::min);

        GRPOStats {
            total_rollouts: self.total_rollouts,
            total_stored: self.total_experiences_stored,
            total_pruned: self.total_experiences_pruned,
            current_library_size: self.experience_library.len(),
            avg_advantage,
            max_advantage,
            min_advantage,
        }
    }

    /// Clear all experiences
    pub fn clear(&mut self) {
        self.experience_library.clear();
    }

    /// Save experience library to file
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.experience_library)?;
        std::fs::write(path, json)
    }

    /// Load experience library from file
    pub fn load_from_file(&mut self, path: &str) -> std::io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        self.experience_library = serde_json::from_str(&json)?;
        Ok(())
    }

    /// Get current library size
    pub fn library_size(&self) -> usize {
        self.experience_library.len()
    }

    /// Check if library is at capacity
    pub fn is_at_capacity(&self) -> bool {
        self.experience_library.len() >= self.config.max_library_size
    }
}

impl Default for TrainingFreeGRPO {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about GRPO experience library
#[derive(Debug, Clone)]
pub struct GRPOStats {
    pub total_rollouts: usize,
    pub total_stored: usize,
    pub total_pruned: usize,
    pub current_library_size: usize,
    pub avg_advantage: f64,
    pub max_advantage: f64,
    pub min_advantage: f64,
}

impl GRPOStats {
    /// Get storage efficiency (ratio of stored to total rollouts)
    pub fn storage_efficiency(&self) -> f64 {
        if self.total_rollouts > 0 {
            self.total_stored as f64 / self.total_rollouts as f64
        } else {
            0.0
        }
    }

    /// Get retention rate (current size / total stored)
    pub fn retention_rate(&self) -> f64 {
        if self.total_stored > 0 {
            self.current_library_size as f64 / self.total_stored as f64
        } else {
            0.0
        }
    }

    /// Print human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "GRPO Stats:\n\
             Total rollouts: {}\n\
             Stored: {} ({:.1}% of rollouts)\n\
             Pruned: {}\n\
             Current library: {} experiences\n\
             Advantage: avg={:.3}, max={:.3}, min={:.3}\n\
             Storage efficiency: {:.1}%\n\
             Retention rate: {:.1}%",
            self.total_rollouts,
            self.total_stored,
            self.storage_efficiency() * 100.0,
            self.total_pruned,
            self.current_library_size,
            self.avg_advantage,
            self.max_advantage,
            self.min_advantage,
            self.storage_efficiency() * 100.0,
            self.retention_rate() * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpo_creation() {
        let grpo = TrainingFreeGRPO::new();
        assert_eq!(grpo.library_size(), 0);
        assert!(!grpo.is_at_capacity());
    }

    #[test]
    fn test_add_rollout_group() {
        let mut grpo = TrainingFreeGRPO::new();

        // Create rollouts with varying rewards
        let rollouts = vec![
            ("prompt1".to_string(), "response1".to_string()),
            ("prompt1".to_string(), "response2".to_string()),
            ("prompt1".to_string(), "response3".to_string()),
            ("prompt1".to_string(), "response4".to_string()),
            ("prompt1".to_string(), "response5".to_string()),
            ("prompt1".to_string(), "response6".to_string()),
            ("prompt1".to_string(), "response7".to_string()),
            ("prompt1".to_string(), "response8".to_string()),
        ];

        let rewards = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7];
        // Mean = 0.45, so 0.5, 0.8, 0.6, 0.7 have positive advantage

        grpo.add_rollout_group(rollouts, rewards);

        // Should store 4 experiences with positive advantage
        assert_eq!(grpo.library_size(), 4);
    }

    #[test]
    fn test_library_pruning_by_size() {
        let config = GRPOConfig {
            max_library_size: 10,
            ..Default::default()
        };
        let mut grpo = TrainingFreeGRPO::with_config(config);

        // Add 20 experiences (should prune to 10)
        for i in 0..20 {
            let rollouts = vec![
                (format!("prompt{}", i), "response".to_string());
                8
            ];
            let rewards = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6];
            grpo.add_rollout_group(rollouts, rewards);
        }

        // Should be capped at 10
        assert_eq!(grpo.library_size(), 10);
    }

    #[test]
    fn test_sample_experience() {
        let mut grpo = TrainingFreeGRPO::new();

        // Add some experiences
        let rollouts = vec![
            ("test".to_string(), "response".to_string());
            8
        ];
        let rewards = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6];
        grpo.add_rollout_group(rollouts, rewards);

        // Should be able to sample
        let sample = grpo.sample_experience();
        assert!(sample.is_some());
    }

    #[test]
    fn test_grpo_stats() {
        let mut grpo = TrainingFreeGRPO::new();

        let rollouts = vec![
            ("test".to_string(), "response".to_string());
            8
        ];
        let rewards = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6];
        grpo.add_rollout_group(rollouts, rewards);

        let stats = grpo.stats();
        assert_eq!(stats.total_rollouts, 8);
        assert!(stats.total_stored > 0);
        assert!(stats.avg_advantage > 0.0);
    }

    #[test]
    fn test_get_experiences_for_prompt() {
        let mut grpo = TrainingFreeGRPO::new();

        let rollouts = vec![
            ("test_prompt".to_string(), "response1".to_string()),
            ("test_prompt".to_string(), "response2".to_string()),
            ("other_prompt".to_string(), "response3".to_string()),
            ("test_prompt".to_string(), "response4".to_string()),
            ("test_prompt".to_string(), "response5".to_string()),
            ("other_prompt".to_string(), "response6".to_string()),
            ("test_prompt".to_string(), "response7".to_string()),
            ("test_prompt".to_string(), "response8".to_string()),
        ];
        let rewards = vec![0.1, 0.9, 0.5, 0.8, 0.3, 0.5, 0.7, 0.6];

        grpo.add_rollout_group(rollouts, rewards);

        let test_exps = grpo.get_experiences_for_prompt("test_prompt");
        let other_exps = grpo.get_experiences_for_prompt("other_prompt");

        assert!(test_exps.len() > 0);
        // other_exps might be empty if below mean, just check it's a valid vec
        let _ = other_exps.len();
    }

    #[test]
    fn test_save_load() {
        let mut grpo = TrainingFreeGRPO::new();

        let rollouts = vec![
            ("test".to_string(), "response".to_string());
            8
        ];
        let rewards = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6];
        grpo.add_rollout_group(rollouts, rewards);

        let path = "/tmp/test_grpo.json";
        grpo.save_to_file(path).unwrap();

        let mut grpo2 = TrainingFreeGRPO::new();
        grpo2.load_from_file(path).unwrap();

        assert_eq!(grpo.library_size(), grpo2.library_size());

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
