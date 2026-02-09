//! Compare GRPO vs MaxRL on Rust code generation
//!
//! Demonstrates the key difference:
//! - GRPO: Optimizes relative rewards (all samples contribute)
//! - MaxRL: Maximizes likelihood of correct samples only
//!
//! Expected result: MaxRL achieves better test-time scaling (20x according to paper)

use nanochat_rl::{
    CompilerFeedback, analyze_ast, compute_reward, RewardConfig,
    GrpoConfig, GrpoTrainer, MaxRLConfig, MaxRLTrainer,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  GRPO vs MaxRL Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    // Setup
    let compiler = CompilerFeedback::new()?;
    let reward_config = RewardConfig::default();

    // Test samples: mix of correct and incorrect code
    let samples = vec![
        (
            "Correct: Iterator",
            r#"
pub fn sum_squares(nums: &[i32]) -> i32 {
    nums.iter().map(|x| x * x).sum()
}
"#,
        ),
        (
            "Correct: Error handling",
            r#"
use std::fs;
use std::io;

pub fn read_file(path: &str) -> io::Result<String> {
    fs::read_to_string(path)
}
"#,
        ),
        (
            "Incorrect: Syntax error",
            r#"
pub fn broken() {
    let x = 5  // Missing semicolon
}
"#,
        ),
        (
            "Incorrect: Type mismatch",
            r#"
pub fn wrong() -> String {
    42  // Returns i32, not String
}
"#,
        ),
        (
            "Correct: Struct",
            r#"
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}
"#,
        ),
    ];

    // Evaluate all samples
    let mut log_probs = Vec::new();
    let mut rewards = Vec::new();

    println!("Evaluating {} code samples...\n", samples.len());

    for (name, code) in &samples {
        // Compile and analyze
        let compile_result = compiler.compile(code)?;
        let ast_metrics = analyze_ast(code)?;
        let reward = compute_reward(&compile_result, &ast_metrics, &reward_config);

        // Simulated log probability (in real training, from model)
        let log_prob = -5.0 + (reward / 50.0); // Higher reward = higher prob

        println!("{}", name);
        println!("  Compiles: {}", if compile_result.success { "✓" } else { "✗" });
        println!("  Reward: {:.2}", reward);
        println!("  Log prob: {:.4}", log_prob);

        log_probs.push(log_prob);
        rewards.push(reward);
        println!();
    }

    // ═══════════════════════════════════════════════════════════
    // GRPO: Uses all samples with relative rewards
    // ═══════════════════════════════════════════════════════════

    println!("═══════════════════════════════════════════════════════════");
    println!("  GRPO (Group Relative Policy Optimization)");
    println!("═══════════════════════════════════════════════════════════\n");

    let grpo_config = GrpoConfig::default();
    let grpo_trainer = GrpoTrainer::new(grpo_config);

    // Compute relative rewards (GRPO normalizes within group)
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let variance = rewards.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / rewards.len() as f64;
    let std_dev = variance.sqrt().max(1e-6);

    let relative_rewards: Vec<f64> = rewards.iter()
        .map(|r| (r - mean) / std_dev)
        .collect();

    println!("Relative rewards (normalized):");
    for (i, (name, _)) in samples.iter().enumerate() {
        println!("  {}: {:.4}", name.split(':').next().unwrap(), relative_rewards[i]);
    }
    println!();

    // GRPO loss (all samples contribute)
    let grpo_loss = grpo_trainer.compute_loss(
        &log_probs,
        &relative_rewards,
        None,
        &vec![0.5; samples.len()], // Dummy entropy
    );

    println!("GRPO Loss: {:.4}", grpo_loss);
    println!("All {} samples contribute to gradient", samples.len());
    println!();

    // ═══════════════════════════════════════════════════════════
    // MaxRL: Maximizes likelihood of correct samples only
    // ═══════════════════════════════════════════════════════════

    println!("═══════════════════════════════════════════════════════════");
    println!("  MaxRL (Maximum Likelihood RL)");
    println!("═══════════════════════════════════════════════════════════\n");

    let maxrl_config = MaxRLConfig {
        correctness_threshold: 20.0, // Reward > 20 = "correct"
        temperature: 1.0,
        ..Default::default()
    };
    let maxrl_trainer = MaxRLTrainer::new(maxrl_config);

    let (maxrl_loss, maxrl_stats) = maxrl_trainer.compute_maxrl_loss(
        &log_probs,
        &rewards,
        None,
    );

    println!("Correctness filtering (threshold = 20.0):");
    for (i, (name, _)) in samples.iter().enumerate() {
        let is_correct = rewards[i] > 20.0;
        println!("  {}: {} (reward: {:.2})",
                 name.split(':').next().unwrap(),
                 if is_correct { "✓ Correct" } else { "✗ Ignored" },
                 rewards[i]);
    }
    println!();

    println!("MaxRL Statistics:");
    println!("  Loss: {:.4}", maxrl_loss);
    println!("  Correct samples: {} / {}", maxrl_stats.n_correct, maxrl_stats.n_total);
    println!("  Correctness rate: {:.1}%", maxrl_stats.correctness_rate * 100.0);
    println!("  Avg correct reward: {:.2}", maxrl_stats.avg_correct_reward);
    println!("  Only {} samples contribute to gradient", maxrl_stats.n_correct);
    println!();

    // ═══════════════════════════════════════════════════════════
    // Comparison and Analysis
    // ═══════════════════════════════════════════════════════════

    println!("═══════════════════════════════════════════════════════════");
    println!("  Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Key Differences:");
    println!();
    println!("GRPO:");
    println!("  • Uses ALL samples (correct + incorrect)");
    println!("  • Normalizes rewards within group");
    println!("  • Incorrect samples get negative relative rewards");
    println!("  • Optimizes: -E[log π(a) * relative_reward]");
    println!();
    println!("MaxRL:");
    println!("  • Uses ONLY correct samples (reward > threshold)");
    println!("  • Ignores incorrect samples completely");
    println!("  • Weights by excess reward over threshold");
    println!("  • Optimizes: -E[log π(correct_a)]");
    println!();

    println!("Why MaxRL is Better for Code Generation:");
    println!("  1. Direct ML objective: Maximizes P(compilable code)");
    println!("  2. Efficient learning: Focuses on successful patterns");
    println!("  3. Better scaling: 20x improvement in test-time efficiency");
    println!("  4. Natural for binary tasks: Code either compiles or doesn't");
    println!();

    println!("Recommendation:");
    println!("  Use MaxRL for Rust code generation where:");
    println!("  • Correctness is measurable (rustc compilation)");
    println!("  • Task has clear success criterion");
    println!("  • Want maximum efficiency with compute");
    println!();

    Ok(())
}
