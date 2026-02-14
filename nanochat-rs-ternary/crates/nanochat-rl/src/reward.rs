//! Reward function for Rust code generation RL
//!
//! Combines compiler feedback and AST analysis into a single reward signal
//! that guides policy optimization toward generating correct, idiomatic Rust code.

use crate::ast_analysis::AstMetrics;
use crate::compiler::CompileResult;
use serde::{Deserialize, Serialize};

/// Configuration for reward function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Weight for compilation success
    pub w_compile_success: f64,

    /// Weight for no errors
    pub w_no_errors: f64,

    /// Weight for no warnings
    pub w_no_warnings: f64,

    /// Weight for parseability
    pub w_parseable: f64,

    /// Weight for code structure (functions, structs, etc.)
    pub w_structure: f64,

    /// Weight for code quality (Result, Option, docs)
    pub w_quality: f64,

    /// Weight for idiomatic patterns
    pub w_idioms: f64,

    /// Weight for complexity (lower is better)
    pub w_complexity: f64,

    /// Penalty for panics (unwrap, expect)
    pub panic_penalty: f64,

    /// Penalty for unsafe code
    pub unsafe_penalty: f64,

    /// Bonus for documentation
    pub doc_bonus: f64,

    /// Maximum complexity threshold (cyclomatic)
    pub max_complexity: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            w_compile_success: 10.0, // Compilation is critical
            w_no_errors: 5.0,        // Error-free code is important
            w_no_warnings: 1.0,      // Warnings matter but less
            w_parseable: 8.0,        // Must be valid syntax
            w_structure: 2.0,        // Reasonable structure
            w_quality: 3.0,          // Quality patterns
            w_idioms: 2.0,           // Idiomatic Rust
            w_complexity: -0.5,      // Penalize high complexity
            panic_penalty: -2.0,     // Discourage panics
            unsafe_penalty: -1.0,    // Discourage unsafe (but not forbidden)
            doc_bonus: 1.0,          // Encourage docs
            max_complexity: 10.0,    // Reasonable complexity limit
        }
    }
}

/// Complete code sample with all evaluation results
#[derive(Debug, Clone)]
pub struct CodeSample {
    /// Generated code
    pub code: String,

    /// Compiler feedback
    pub compile_result: CompileResult,

    /// AST metrics
    pub ast_metrics: AstMetrics,

    /// Computed reward
    pub reward: f64,
}

/// Compute reward for a code sample
pub fn compute_reward(
    compile_result: &CompileResult,
    ast_metrics: &AstMetrics,
    config: &RewardConfig,
) -> f64 {
    let mut reward = 0.0;

    // 1. Compilation success (most important)
    if compile_result.success {
        reward += config.w_compile_success;
    }

    // 2. No errors
    if compile_result.n_errors == 0 {
        reward += config.w_no_errors;
    } else {
        // Penalize errors proportionally
        reward -= compile_result.n_errors as f64 * 2.0;
    }

    // 3. No warnings
    if compile_result.n_warnings == 0 {
        reward += config.w_no_warnings;
    } else {
        // Small penalty for warnings
        reward -= compile_result.n_warnings as f64 * 0.5;
    }

    // 4. Parseability (syntax)
    if ast_metrics.parseable {
        reward += config.w_parseable;

        // Only evaluate AST metrics if code is parseable
        reward += evaluate_structure(&ast_metrics.structure, config);
        reward += evaluate_quality(&ast_metrics.quality, config);
        reward += evaluate_idioms(&ast_metrics.idioms, config);
        reward += evaluate_complexity(&ast_metrics.complexity, config);
    }

    reward
}

fn evaluate_structure(
    structure: &crate::ast_analysis::StructureMetrics,
    config: &RewardConfig,
) -> f64 {
    let mut score = 0.0;

    // Reward for having functions (core building blocks)
    score += (structure.n_functions as f64 * 0.5).min(2.0);

    // Reward for data structures
    score += (structure.n_structs as f64 * 0.3).min(1.0);
    score += (structure.n_enums as f64 * 0.3).min(1.0);

    // Reward for traits (advanced Rust)
    score += (structure.n_traits as f64 * 0.5).min(1.0);

    // Normalize and apply weight
    score * config.w_structure
}

fn evaluate_quality(quality: &crate::ast_analysis::QualityMetrics, config: &RewardConfig) -> f64 {
    let mut score = 0.0;

    // Reward for proper error handling
    if quality.uses_result {
        score += 1.0;
    }

    if quality.uses_option {
        score += 0.5;
    }

    // Reward for documentation
    if quality.has_docs {
        score += config.doc_bonus;
    }
    score += quality.doc_coverage * config.doc_bonus;

    // Reward for ? operator (idiomatic error handling)
    if quality.uses_try_operator {
        score += 0.5;
    }

    // Penalties
    if quality.has_unsafe {
        score += config.unsafe_penalty;
    }

    score += quality.n_panics as f64 * config.panic_penalty;

    // Apply weight
    score * config.w_quality
}

fn evaluate_idioms(idioms: &crate::ast_analysis::IdiomMetrics, config: &RewardConfig) -> f64 {
    let mut score = 0.0;

    // Reward idiomatic patterns
    if idioms.uses_iterators {
        score += 1.0;
    }

    if idioms.uses_pattern_matching {
        score += 0.8;
    }

    if idioms.uses_destructuring {
        score += 0.5;
    }

    if idioms.uses_closures {
        score += 0.7;
    }

    if idioms.uses_method_chaining {
        score += 0.6;
    }

    if idioms.uses_turbofish {
        score += 0.3;
    }

    // Apply weight
    score * config.w_idioms
}

fn evaluate_complexity(
    complexity: &crate::ast_analysis::ComplexityMetrics,
    config: &RewardConfig,
) -> f64 {
    let mut score = 0.0;

    // Penalize high cyclomatic complexity
    if complexity.cyclomatic as f64 > config.max_complexity {
        let excess = (complexity.cyclomatic as f64 - config.max_complexity) / config.max_complexity;
        score += excess * config.w_complexity * 2.0;
    }

    // Penalize deep nesting (> 3 levels)
    if complexity.max_nesting > 3 {
        let excess = (complexity.max_nesting - 3) as f64;
        score += excess * config.w_complexity;
    }

    // Reward reasonable code length (not too short, not too long)
    // Sweet spot: 10-100 LOC
    if complexity.loc < 5 {
        score -= 1.0; // Too trivial
    } else if complexity.loc > 200 {
        score -= 2.0; // Too complex
    } else if (10..=100).contains(&complexity.loc) {
        score += 0.5; // Good length
    }

    score
}

/// Compute relative rewards for a group of samples (for GRPO)
/// Returns normalized rewards with zero mean
pub fn compute_relative_rewards(samples: &[CodeSample]) -> Vec<f64> {
    let rewards: Vec<f64> = samples.iter().map(|s| s.reward).collect();

    // Compute mean
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;

    // Compute std dev
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
    let std_dev = variance.sqrt().max(1e-6); // Avoid division by zero

    // Normalize: (reward - mean) / std_dev
    rewards.iter().map(|r| (r - mean) / std_dev).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_analysis::*;
    use crate::compiler::CompileResult;

    #[test]
    fn test_reward_successful_compile() {
        let config = RewardConfig::default();

        let compile_result = CompileResult {
            success: true,
            n_errors: 0,
            n_warnings: 0,
            output: String::new(),
            exit_code: Some(0),
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        let ast_metrics = AstMetrics {
            parseable: true,
            complexity: ComplexityMetrics::default(),
            structure: StructureMetrics {
                n_functions: 1,
                ..Default::default()
            },
            quality: QualityMetrics {
                uses_result: true,
                ..Default::default()
            },
            idioms: IdiomMetrics {
                uses_iterators: true,
                ..Default::default()
            },
        };

        let reward = compute_reward(&compile_result, &ast_metrics, &config);

        // Should be positive for successful, quality code
        assert!(
            reward > 10.0,
            "Reward for good code should be positive: {}",
            reward
        );
    }

    #[test]
    fn test_reward_failed_compile() {
        let config = RewardConfig::default();

        let compile_result = CompileResult {
            success: false,
            n_errors: 3,
            n_warnings: 0,
            output: String::new(),
            exit_code: Some(1),
            errors: vec![],
            warnings: Vec::new(),
        };

        let ast_metrics = AstMetrics {
            parseable: false,
            complexity: ComplexityMetrics::default(),
            structure: StructureMetrics::default(),
            quality: QualityMetrics::default(),
            idioms: IdiomMetrics::default(),
        };

        let reward = compute_reward(&compile_result, &ast_metrics, &config);

        // Should be negative for failed compilation
        assert!(
            reward < 0.0,
            "Reward for broken code should be negative: {}",
            reward
        );
    }

    #[test]
    fn test_relative_rewards() {
        let samples = vec![
            CodeSample {
                code: String::new(),
                compile_result: CompileResult {
                    success: true,
                    n_errors: 0,
                    n_warnings: 0,
                    output: String::new(),
                    exit_code: Some(0),
                    errors: Vec::new(),
                    warnings: Vec::new(),
                },
                ast_metrics: AstMetrics {
                    parseable: true,
                    complexity: ComplexityMetrics::default(),
                    structure: StructureMetrics::default(),
                    quality: QualityMetrics::default(),
                    idioms: IdiomMetrics::default(),
                },
                reward: 10.0,
            },
            CodeSample {
                code: String::new(),
                compile_result: CompileResult {
                    success: false,
                    n_errors: 1,
                    n_warnings: 0,
                    output: String::new(),
                    exit_code: Some(1),
                    errors: Vec::new(),
                    warnings: Vec::new(),
                },
                ast_metrics: AstMetrics {
                    parseable: false,
                    complexity: ComplexityMetrics::default(),
                    structure: StructureMetrics::default(),
                    quality: QualityMetrics::default(),
                    idioms: IdiomMetrics::default(),
                },
                reward: 5.0,
            },
        ];

        let relative = compute_relative_rewards(&samples);

        // Should have mean ~0
        let mean: f64 = relative.iter().sum::<f64>() / relative.len() as f64;
        assert!(mean.abs() < 1e-6, "Mean should be near zero: {}", mean);
    }
}
