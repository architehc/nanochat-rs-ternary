//! Evaluation metrics for code generation, including pass@k calculation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pass@k metric: probability that at least one of k generated samples passes all tests.
///
/// Calculated using the unbiased estimator:
/// pass@k = E[1 - C(n-c, k) / C(n, k)]
/// where:
/// - n = total samples generated per problem
/// - c = samples that passed tests
/// - C(n, k) = n choose k
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassAtK {
    /// k value (number of samples)
    pub k: usize,

    /// pass@k score (0.0 to 1.0)
    pub score: f64,

    /// Total problems evaluated
    pub num_problems: usize,

    /// Problems with at least one passing solution
    pub num_solved: usize,
}

impl PassAtK {
    /// Calculate pass@k from evaluation results.
    ///
    /// # Arguments
    /// * `results` - Map from task_id to (num_passed, num_total)
    /// * `k` - Number of samples to consider
    pub fn calculate(results: &HashMap<String, (usize, usize)>, k: usize) -> Self {
        let num_problems = results.len();
        let mut pass_at_k_sum = 0.0;
        let mut num_solved = 0;

        for (_task_id, (c, n)) in results.iter() {
            let c = *c;
            let n = *n;

            if c > 0 {
                num_solved += 1;
            }

            if n < k {
                // Not enough samples, skip
                continue;
            }

            // Calculate pass@k for this problem using unbiased estimator
            // pass@k = 1 - C(n-c, k) / C(n, k)
            let prob = 1.0 - (binomial_coeff(n - c, k) / binomial_coeff(n, k));
            pass_at_k_sum += prob;
        }

        let score = if num_problems > 0 {
            pass_at_k_sum / num_problems as f64
        } else {
            0.0
        };

        Self {
            k,
            score,
            num_problems,
            num_solved,
        }
    }
}

/// Calculate binomial coefficient C(n, k) = n! / (k! * (n-k)!)
fn binomial_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }

    // Use multiplicative formula to avoid overflow
    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1.0;

    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }

    result
}

/// Complete evaluation metrics for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Model name/identifier
    pub model_name: String,

    /// Dataset name (HumanEval, MBPP, etc.)
    pub dataset_name: String,

    /// pass@1 score
    pub pass_at_1: PassAtK,

    /// pass@10 score (if available)
    pub pass_at_10: Option<PassAtK>,

    /// pass@100 score (if available)
    pub pass_at_100: Option<PassAtK>,

    /// Total problems attempted
    pub total_problems: usize,

    /// Problems with at least one passing solution
    pub problems_solved: usize,

    /// Average execution time per problem (ms)
    pub avg_execution_time_ms: f64,

    /// Error breakdown
    pub error_counts: HashMap<String, usize>,

    /// Per-problem results (optional, for detailed analysis)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_problem_results: Option<HashMap<String, ProblemResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemResult {
    pub task_id: String,
    pub num_passed: usize,
    pub num_total: usize,
    pub avg_time_ms: f64,
    pub error_types: Vec<String>,
}

impl EvalMetrics {
    pub fn print_summary(&self) {
        println!("═══════════════════════════════════════════════════════════");
        println!("  Code Generation Evaluation Results");
        println!("═══════════════════════════════════════════════════════════");
        println!();
        println!("Model: {}", self.model_name);
        println!("Dataset: {}", self.dataset_name);
        println!();
        println!("Pass@k Scores:");
        println!(
            "  pass@1:  {:.2}% ({}/{})",
            self.pass_at_1.score * 100.0,
            self.pass_at_1.num_solved,
            self.pass_at_1.num_problems
        );
        if let Some(p10) = &self.pass_at_10 {
            println!(
                "  pass@10: {:.2}% ({}/{})",
                p10.score * 100.0,
                p10.num_solved,
                p10.num_problems
            );
        }
        if let Some(p100) = &self.pass_at_100 {
            println!(
                "  pass@100: {:.2}% ({}/{})",
                p100.score * 100.0,
                p100.num_solved,
                p100.num_problems
            );
        }
        println!();
        println!("Performance:");
        println!(
            "  Problems solved: {}/{}",
            self.problems_solved, self.total_problems
        );
        println!("  Avg execution time: {:.1}ms", self.avg_execution_time_ms);
        println!();

        if !self.error_counts.is_empty() {
            println!("Error Breakdown:");
            let mut errors: Vec<_> = self.error_counts.iter().collect();
            errors.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
            for (error_type, count) in errors.iter().take(5) {
                let pct = (**count as f64 / self.total_problems as f64) * 100.0;
                println!("  {}: {} ({:.1}%)", error_type, count, pct);
            }
            println!();
        }

        println!("═══════════════════════════════════════════════════════════");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_coeff() {
        assert_eq!(binomial_coeff(5, 0), 1.0);
        assert_eq!(binomial_coeff(5, 5), 1.0);
        assert_eq!(binomial_coeff(5, 1), 5.0);
        assert_eq!(binomial_coeff(5, 2), 10.0);
        assert_eq!(binomial_coeff(5, 3), 10.0);
        assert_eq!(binomial_coeff(10, 3), 120.0);
    }

    #[test]
    fn test_pass_at_k_perfect() {
        // All samples pass
        let mut results = HashMap::new();
        results.insert("task1".to_string(), (10, 10)); // 10/10 passed
        results.insert("task2".to_string(), (10, 10));

        let pass_at_1 = PassAtK::calculate(&results, 1);
        assert_eq!(pass_at_1.score, 1.0);
        assert_eq!(pass_at_1.num_solved, 2);
    }

    #[test]
    fn test_pass_at_k_zero() {
        // No samples pass
        let mut results = HashMap::new();
        results.insert("task1".to_string(), (0, 10)); // 0/10 passed
        results.insert("task2".to_string(), (0, 10));

        let pass_at_1 = PassAtK::calculate(&results, 1);
        assert_eq!(pass_at_1.score, 0.0);
        assert_eq!(pass_at_1.num_solved, 0);
    }

    #[test]
    fn test_pass_at_k_partial() {
        // Mix of passing and failing
        let mut results = HashMap::new();
        results.insert("task1".to_string(), (5, 10)); // 5/10 passed
        results.insert("task2".to_string(), (1, 10)); // 1/10 passed

        let pass_at_1 = PassAtK::calculate(&results, 1);
        // pass@1 for task1: 5/10 = 0.5
        // pass@1 for task2: 1/10 = 0.1
        // Average: (0.5 + 0.1) / 2 = 0.3
        assert!((pass_at_1.score - 0.3).abs() < 0.01);
        assert_eq!(pass_at_1.num_solved, 2);

        let pass_at_10 = PassAtK::calculate(&results, 10);
        // pass@10 should be higher (probability of at least one success in 10 tries)
        assert!(pass_at_10.score > pass_at_1.score);
    }

    #[test]
    fn test_pass_at_k_skips_when_n_less_than_k() {
        let mut results = HashMap::new();
        results.insert("task1".to_string(), (1, 1));
        let pass_at_5 = PassAtK::calculate(&results, 5);
        assert_eq!(pass_at_5.k, 5);
        assert_eq!(pass_at_5.num_problems, 1);
        assert_eq!(pass_at_5.num_solved, 1);
        assert_eq!(pass_at_5.score, 0.0);
    }

    #[test]
    fn test_eval_metrics_print_summary_with_optional_fields() {
        let pass1 = PassAtK {
            k: 1,
            score: 0.5,
            num_problems: 2,
            num_solved: 1,
        };
        let pass10 = PassAtK {
            k: 10,
            score: 0.8,
            num_problems: 2,
            num_solved: 2,
        };

        let mut error_counts = HashMap::new();
        error_counts.insert("SyntaxError".to_string(), 3);
        error_counts.insert("Timeout".to_string(), 1);

        let metrics = EvalMetrics {
            model_name: "unit-model".to_string(),
            dataset_name: "HumanEval".to_string(),
            pass_at_1: pass1,
            pass_at_10: Some(pass10),
            pass_at_100: None,
            total_problems: 2,
            problems_solved: 1,
            avg_execution_time_ms: 12.3,
            error_counts,
            per_problem_results: None,
        };

        metrics.print_summary();
    }
}
