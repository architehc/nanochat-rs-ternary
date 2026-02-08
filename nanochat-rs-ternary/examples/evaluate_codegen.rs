//! Code generation quality evaluation tool for HumanEval and MBPP.
//!
//! Evaluates ternary model code generation quality and compares with baselines.
//!
//! Usage:
//!   # Evaluate on HumanEval
//!   cargo run --release --example evaluate_codegen -- \
//!     --dataset humaneval \
//!     --data-path HumanEval.jsonl \
//!     --model-endpoint http://localhost:8080/v1/completions \
//!     --model-name qwen3-ternary \
//!     --num-samples 10 \
//!     --output results.json
//!
//!   # Evaluate on MBPP
//!   cargo run --release --example evaluate_codegen -- \
//!     --dataset mbpp \
//!     --data-path mbpp.json \
//!     --model-endpoint http://localhost:8080/v1/completions \
//!     --num-samples 10

use nanochat_eval::{
    HumanEvalDataset, MBPPDataset, EvaluationConfig, Evaluator, EvalReport,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "evaluate_codegen")]
#[command(about = "Evaluate code generation quality on HumanEval/MBPP")]
struct Args {
    /// Dataset to evaluate on (humaneval, mbpp)
    #[arg(long, default_value = "humaneval")]
    dataset: String,

    /// Path to dataset file
    #[arg(long)]
    data_path: String,

    /// Model endpoint URL
    #[arg(long, default_value = "http://localhost:8080/v1/completions")]
    model_endpoint: String,

    /// Model name/identifier
    #[arg(long, default_value = "nanochat-ternary")]
    model_name: String,

    /// Number of solutions to generate per problem
    #[arg(long, default_value = "10")]
    num_samples: usize,

    /// Temperature for sampling (0.0 = greedy, 0.8 = creative)
    #[arg(long, default_value = "0.8")]
    temperature: f64,

    /// Maximum tokens to generate
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Timeout for code execution (seconds)
    #[arg(long, default_value = "5")]
    execution_timeout: u64,

    /// Maximum number of problems to evaluate (for testing)
    #[arg(long)]
    max_problems: Option<usize>,

    /// Output file for results (JSON)
    #[arg(long)]
    output: Option<String>,

    /// Baseline model endpoint for comparison (optional)
    #[arg(long)]
    baseline_endpoint: Option<String>,

    /// Baseline model name
    #[arg(long)]
    baseline_name: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Code Generation Quality Evaluation");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Load dataset
    println!("Loading dataset: {} from {}", args.dataset, args.data_path);
    let (problems, dataset_name) = match args.dataset.as_str() {
        "humaneval" => {
            let dataset = HumanEvalDataset::load(&args.data_path)?;
            println!("  Loaded {} HumanEval problems", dataset.len());
            (dataset.problems().to_vec(), "HumanEval")
        }
        "mbpp" => {
            let dataset = MBPPDataset::load(&args.data_path)?;
            println!("  Loaded {} MBPP problems", dataset.len());
            (dataset.problems().to_vec(), "MBPP")
        }
        other => {
            eprintln!("Unknown dataset: {}. Use 'humaneval' or 'mbpp'", other);
            std::process::exit(1);
        }
    };
    println!();

    // Create evaluation report
    let mut report = EvalReport::new();

    // Evaluate main model
    println!("Evaluating {} model...", args.model_name);
    let config = EvaluationConfig {
        model_endpoint: args.model_endpoint.clone(),
        model_name: args.model_name.clone(),
        num_samples: args.num_samples,
        temperature: args.temperature,
        max_tokens: args.max_tokens,
        execution_timeout: args.execution_timeout,
        max_problems: args.max_problems,
        parallel: true,
    };

    let evaluator = Evaluator::new(config);
    let metrics = evaluator.evaluate(&problems, dataset_name).await?;
    metrics.print_summary();
    report.add_result(metrics);

    // Evaluate baseline if provided
    if let (Some(baseline_endpoint), Some(baseline_name)) =
        (args.baseline_endpoint, args.baseline_name)
    {
        println!();
        println!("Evaluating {} baseline model...", baseline_name);

        let baseline_config = EvaluationConfig {
            model_endpoint: baseline_endpoint,
            model_name: baseline_name,
            num_samples: args.num_samples,
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            execution_timeout: args.execution_timeout,
            max_problems: args.max_problems,
            parallel: true,
        };

        let baseline_evaluator = Evaluator::new(baseline_config);
        let baseline_metrics = baseline_evaluator.evaluate(&problems, dataset_name).await?;
        baseline_metrics.print_summary();
        report.add_result(baseline_metrics);
    }

    // Print comparison if multiple models
    if report.reports.len() > 1 {
        println!();
        report.print_comparison();
    }

    // Save results
    if let Some(output_path) = args.output {
        println!();
        println!("Saving results to {}...", output_path);
        report.save_json(&output_path)?;
        println!("✓ Results saved");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("Evaluation complete!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
