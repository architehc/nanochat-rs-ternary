//! Main evaluation harness that coordinates dataset loading, model querying, and metric calculation.

use crate::datasets::CodeProblem;
use crate::executor::CodeExecutor;
use crate::metrics::{EvalMetrics, PassAtK, ProblemResult};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for code generation evaluation.
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Model endpoint URL (e.g., "http://localhost:8080/v1/completions")
    pub model_endpoint: String,

    /// Model name/identifier
    pub model_name: String,

    /// Number of solutions to generate per problem
    pub num_samples: usize,

    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f64,

    /// Maximum tokens to generate
    pub max_tokens: usize,

    /// Timeout for code execution (seconds)
    pub execution_timeout: u64,

    /// Number of problems to evaluate (None = all)
    pub max_problems: Option<usize>,

    /// Whether to use parallel execution
    pub parallel: bool,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            model_endpoint: "http://localhost:8080/v1/completions".to_string(),
            model_name: "nanochat-ternary".to_string(),
            num_samples: 10,
            temperature: 0.8,
            max_tokens: 512,
            execution_timeout: 5,
            max_problems: None,
            parallel: true,
        }
    }
}

/// Main evaluator that runs code generation eval.
pub struct Evaluator {
    config: EvaluationConfig,
    executor: CodeExecutor,
    client: reqwest::blocking::Client,
}

impl Evaluator {
    pub fn new(config: EvaluationConfig) -> Self {
        let executor = CodeExecutor::new().with_timeout(config.execution_timeout);
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();

        Self {
            config,
            executor,
            client,
        }
    }

    /// Evaluate on a list of code problems.
    pub async fn evaluate(
        &self,
        problems: &[CodeProblem],
        dataset_name: &str,
    ) -> Result<EvalMetrics, anyhow::Error> {
        let problems = if let Some(max) = self.config.max_problems {
            &problems[..max.min(problems.len())]
        } else {
            problems
        };

        println!(
            "Evaluating {} on {} ({} problems)",
            self.config.model_name,
            dataset_name,
            problems.len()
        );
        println!("Generating {} samples per problem", self.config.num_samples);
        println!();

        // Progress bar
        let pb = ProgressBar::new(problems.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        let mut all_results: HashMap<String, (usize, usize)> = HashMap::new();
        let mut per_problem_results: HashMap<String, ProblemResult> = HashMap::new();
        let mut error_counts: HashMap<String, usize> = HashMap::new();
        let mut total_execution_time = 0.0;
        let mut problems_solved = 0;

        for problem in problems {
            pb.set_message(format!("Evaluating {}", problem.task_id));

            // Generate solutions
            let solutions = self.generate_solutions(problem).await?;

            // Execute and test each solution
            let mut num_passed = 0;
            let mut exec_times = Vec::new();
            let mut error_types = Vec::new();

            for solution in &solutions {
                match self
                    .executor
                    .execute(solution, &problem.test, &problem.entry_point)
                    .await
                {
                    Ok(result) => {
                        if result.passed {
                            num_passed += 1;
                        } else if let Some(error_type) = &result.error_type {
                            let error_name = format!("{:?}", error_type);
                            *error_counts.entry(error_name.clone()).or_insert(0) += 1;
                            error_types.push(error_name);
                        }
                        exec_times.push(result.time_ms as f64);
                    }
                    Err(e) => {
                        *error_counts
                            .entry("ExecutionError".to_string())
                            .or_insert(0) += 1;
                        error_types.push(format!("ExecutionError: {}", e));
                    }
                }
            }

            let avg_time = if !exec_times.is_empty() {
                exec_times.iter().sum::<f64>() / exec_times.len() as f64
            } else {
                0.0
            };

            total_execution_time += avg_time;

            if num_passed > 0 {
                problems_solved += 1;
            }

            all_results.insert(problem.task_id.clone(), (num_passed, solutions.len()));
            per_problem_results.insert(
                problem.task_id.clone(),
                ProblemResult {
                    task_id: problem.task_id.clone(),
                    num_passed,
                    num_total: solutions.len(),
                    avg_time_ms: avg_time,
                    error_types,
                },
            );

            pb.inc(1);
        }

        pb.finish_with_message("Evaluation complete");
        println!();

        // Calculate pass@k metrics
        let pass_at_1 = PassAtK::calculate(&all_results, 1);
        let pass_at_10 = if self.config.num_samples >= 10 {
            Some(PassAtK::calculate(&all_results, 10))
        } else {
            None
        };
        let pass_at_100 = if self.config.num_samples >= 100 {
            Some(PassAtK::calculate(&all_results, 100))
        } else {
            None
        };

        let avg_execution_time_ms = total_execution_time / problems.len() as f64;

        Ok(EvalMetrics {
            model_name: self.config.model_name.clone(),
            dataset_name: dataset_name.to_string(),
            pass_at_1,
            pass_at_10,
            pass_at_100,
            total_problems: problems.len(),
            problems_solved,
            avg_execution_time_ms,
            error_counts,
            per_problem_results: Some(per_problem_results),
        })
    }

    /// Generate solutions for a problem by querying the model.
    async fn generate_solutions(
        &self,
        problem: &CodeProblem,
    ) -> Result<Vec<String>, anyhow::Error> {
        let mut solutions = Vec::new();

        for _ in 0..self.config.num_samples {
            let solution = self.query_model(&problem.prompt).await?;
            solutions.push(solution);
        }

        Ok(solutions)
    }

    /// Query the model endpoint for code completion.
    async fn query_model(&self, prompt: &str) -> Result<String, anyhow::Error> {
        #[derive(Serialize)]
        struct CompletionRequest {
            prompt: String,
            max_tokens: usize,
            temperature: f64,
            stop: Vec<String>,
        }

        #[derive(Deserialize)]
        struct CompletionResponse {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            text: String,
        }

        let request = CompletionRequest {
            prompt: prompt.to_string(),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            stop: vec![
                "\nclass ".to_string(),
                "\ndef ".to_string(),
                "\n#".to_string(),
            ],
        };

        let response = self
            .client
            .post(&self.config.model_endpoint)
            .json(&request)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Model endpoint returned error: {} - {}",
                response.status(),
                response.text()?
            );
        }

        let completion: CompletionResponse = response.json()?;

        if completion.choices.is_empty() {
            anyhow::bail!("Model returned no completions");
        }

        Ok(completion.choices[0].text.clone())
    }
}

/// Evaluation report with comparison across models/configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub reports: Vec<EvalMetrics>,
    pub timestamp: String,
}

impl EvalReport {
    pub fn new() -> Self {
        Self {
            reports: Vec::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn add_result(&mut self, metrics: EvalMetrics) {
        self.reports.push(metrics);
    }

    pub fn print_comparison(&self) {
        println!("═══════════════════════════════════════════════════════════");
        println!("  Model Comparison");
        println!("═══════════════════════════════════════════════════════════");
        println!();
        println!(
            "{:<30} {:>10} {:>10} {:>10}",
            "Model", "pass@1", "pass@10", "Solved"
        );
        println!("{}", "-".repeat(64));

        for metrics in &self.reports {
            let pass_10_str = metrics
                .pass_at_10
                .as_ref()
                .map(|p| format!("{:.2}%", p.score * 100.0))
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "{:<30} {:>9.2}% {:>10} {:>4}/{:<3}",
                metrics.model_name,
                metrics.pass_at_1.score * 100.0,
                pass_10_str,
                metrics.problems_solved,
                metrics.total_problems
            );
        }
        println!();
    }

    pub fn save_json(&self, path: &str) -> Result<(), anyhow::Error> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl Default for EvalReport {
    fn default() -> Self {
        Self::new()
    }
}
