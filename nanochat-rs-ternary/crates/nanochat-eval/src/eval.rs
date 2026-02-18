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
            model_endpoint: "http://localhost:8080/v1/chat/completions".to_string(),
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

    /// Query the model endpoint for code completion using chat completions API.
    async fn query_model(&self, prompt: &str) -> Result<String, anyhow::Error> {
        #[derive(Serialize)]
        struct ChatMessage {
            role: String,
            content: String,
        }

        #[derive(Serialize)]
        struct ChatCompletionRequest {
            messages: Vec<ChatMessage>,
            max_tokens: usize,
            temperature: f64,
        }

        #[derive(Deserialize)]
        struct ChatCompletionResponse {
            choices: Vec<ChatChoice>,
        }

        #[derive(Deserialize)]
        struct ChatChoice {
            message: ChatChoiceMessage,
        }

        #[derive(Deserialize)]
        struct ChatChoiceMessage {
            content: String,
        }

        let request = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
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

        let completion: ChatCompletionResponse = response.json()?;

        if completion.choices.is_empty() {
            anyhow::bail!("Model returned no completions");
        }

        Ok(completion.choices[0].message.content.clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::ProblemResult;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    use std::thread;
    use tempfile::tempdir;

    fn spawn_mock_server(status: &str, body: String, max_requests: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let addr = listener.local_addr().expect("mock addr");
        let status_line = status.to_string();
        thread::spawn(move || {
            for _ in 0..max_requests {
                let (mut stream, _) = listener.accept().expect("accept");
                let mut buf = [0u8; 8192];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status_line,
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
                stream.flush().expect("flush response");
            }
        });
        format!("http://{}/v1/chat/completions", addr)
    }

    fn block_on_immediate<F: std::future::Future>(future: F) -> F::Output {
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(std::ptr::null(), &VTABLE)
        }
        fn wake(_: *const ()) {}
        fn wake_by_ref(_: *const ()) {}
        fn drop(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);

        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = Box::pin(future);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("future unexpectedly pending in synchronous test"),
        }
    }

    #[test]
    fn test_query_model_http_error() {
        let endpoint = spawn_mock_server("500 Internal Server Error", "{\"error\":\"nope\"}".into(), 1);
        let evaluator = Evaluator::new(EvaluationConfig {
            model_endpoint: endpoint,
            ..EvaluationConfig::default()
        });
        let err = block_on_immediate(evaluator.query_model("fn x() {}")).unwrap_err();
        assert!(err.to_string().contains("Model endpoint returned error"));
    }

    #[test]
    fn test_query_model_empty_choices_error() {
        let endpoint = spawn_mock_server("200 OK", "{\"choices\":[]}".into(), 1);
        let evaluator = Evaluator::new(EvaluationConfig {
            model_endpoint: endpoint,
            ..EvaluationConfig::default()
        });
        let err = block_on_immediate(evaluator.query_model("fn x() {}")).unwrap_err();
        assert!(err.to_string().contains("no completions"));
    }

    #[test]
    fn test_generate_solutions_success() -> Result<(), anyhow::Error> {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "def add(a, b):\n    return a + b\n"
                }
            }]
        })
        .to_string();
        let endpoint = spawn_mock_server("200 OK", body, 2);
        let evaluator = Evaluator::new(EvaluationConfig {
            model_endpoint: endpoint,
            model_name: "unit-model".to_string(),
            num_samples: 2,
            temperature: 0.0,
            max_tokens: 64,
            execution_timeout: 2,
            max_problems: None,
            parallel: false,
        });
        let problem = CodeProblem {
            task_id: "Unit/0".to_string(),
            prompt: "Write add".to_string(),
            entry_point: "add".to_string(),
            canonical_solution: None,
            test: "def check(candidate):\n    assert candidate(1, 2) == 3\n".to_string(),
            metadata: serde_json::json!({}),
        };
        let samples = block_on_immediate(evaluator.generate_solutions(&problem))?;
        assert_eq!(samples.len(), 2);
        assert!(samples.iter().all(|s| s.contains("def add")));
        Ok(())
    }

    #[test]
    fn test_eval_report_roundtrip_and_print() -> Result<(), anyhow::Error> {
        let pass = PassAtK {
            k: 1,
            score: 0.75,
            num_problems: 4,
            num_solved: 3,
        };
        let mut problems = HashMap::new();
        problems.insert(
            "Unit/0".to_string(),
            ProblemResult {
                task_id: "Unit/0".to_string(),
                num_passed: 1,
                num_total: 1,
                avg_time_ms: 12.0,
                error_types: vec![],
            },
        );

        let metrics = EvalMetrics {
            model_name: "m".to_string(),
            dataset_name: "d".to_string(),
            pass_at_1: pass.clone(),
            pass_at_10: Some(pass.clone()),
            pass_at_100: None,
            total_problems: 4,
            problems_solved: 3,
            avg_execution_time_ms: 12.5,
            error_counts: HashMap::new(),
            per_problem_results: Some(problems),
        };

        let mut report = EvalReport::new();
        report.add_result(metrics);
        report.print_comparison();

        let dir = tempdir()?;
        let path = dir.path().join("report.json");
        report.save_json(path.to_str().expect("utf8 path"))?;
        let text = std::fs::read_to_string(&path)?;
        assert!(text.contains("\"reports\""));
        assert!(text.contains("\"timestamp\""));
        Ok(())
    }
}
