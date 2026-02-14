//! Comprehensive Model Benchmarking System
//!
//! Generates code samples and evaluates:
//! - Compilation success rate
//! - AST quality metrics
//! - Code complexity
//! - Error types and patterns
//!
//! Usage:
//!   cargo run --release -p nanochat-eval --example benchmark_model -- \
//!     --checkpoint checkpoints/stable-v2/step_10000 \
//!     --n-samples 100 \
//!     --output benchmark_results.json

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use nanochat_rl::CompilerFeedback;
use nanochat_train::{checkpoint::load_checkpoint, model::NanochatTrainModel};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "benchmark_model")]
#[command(about = "Comprehensive model evaluation benchmark")]
struct Args {
    /// Checkpoint to evaluate
    #[arg(long)]
    checkpoint: String,

    /// Number of samples per prompt
    #[arg(long, default_value = "100")]
    n_samples: usize,

    /// Temperature for sampling
    #[arg(long, default_value = "0.8")]
    temperature: f32,

    /// Max tokens to generate
    #[arg(long, default_value = "200")]
    max_tokens: usize,

    /// Output JSON file
    #[arg(long, default_value = "benchmark_results.json")]
    output: String,

    /// Device
    #[arg(long, default_value = "cuda:0")]
    device: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResults {
    checkpoint: String,
    timestamp: String,
    total_samples: usize,

    // Compilation metrics
    compile_success_rate: f64,
    compile_errors: Vec<String>,

    // Code quality metrics
    avg_lines: f64,
    avg_functions: f64,
    avg_complexity: f64,

    // Performance metrics
    avg_generation_time_ms: f64,
    tokens_per_second: f64,

    // Sample breakdown
    samples_by_prompt: Vec<PromptResults>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PromptResults {
    prompt: String,
    n_samples: usize,
    compile_success: usize,
    avg_lines: f64,
    common_errors: Vec<String>,
    sample_outputs: Vec<String>,
}

/// Test prompts covering different Rust patterns
fn get_test_prompts() -> Vec<String> {
    vec![
        // Basic functions
        "fn factorial(n: u64) -> u64 {".to_string(),
        "fn is_palindrome(s: &str) -> bool {".to_string(),
        "fn reverse_string(s: String) -> String {".to_string(),
        // Data structures
        "struct Point { x: f64, y: f64 }".to_string(),
        "enum Color { Red, Green, Blue }".to_string(),
        "impl Point {".to_string(),
        // Collections
        "fn find_duplicates(arr: &[i32]) -> Vec<i32> {".to_string(),
        "fn merge_sorted(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {".to_string(),
        // Error handling
        "fn read_file(path: &str) -> Result<String, std::io::Error> {".to_string(),
        "fn parse_int(s: &str) -> Option<i32> {".to_string(),
        // Iterators
        "fn sum_squares(nums: &[i32]) -> i32 {".to_string(),
        "fn filter_even(nums: Vec<i32>) -> Vec<i32> {".to_string(),
        // Async
        "async fn fetch_data(url: &str) -> Result<String, Box<dyn std::error::Error>> {"
            .to_string(),
        // Traits
        "trait Parser {".to_string(),
        "impl ToString for Point {".to_string(),
        // Macros
        "macro_rules! vec_of_strings {".to_string(),
    ]
}

/// Generate code sample, returns (code, num_tokens_generated)
fn generate_sample(
    model: &NanochatTrainModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    device: &Device,
) -> Result<(String, usize)> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let initial_len = token_ids.len();

    for _ in 0..max_tokens {
        let input = Tensor::new(token_ids.as_slice(), device)?;
        let input = input.unsqueeze(0)?;

        let logits = model.forward(&input)?;
        let last_logits = logits.get(0)?.get(token_ids.len() - 1)?;
        let logits_vec = last_logits.to_vec1::<f32>()?;

        // Sample with temperature
        let next_token = sample_token(&logits_vec, temperature);

        // Check for EOS
        if next_token == tokenizer.token_to_id("<|endoftext|>").unwrap_or(50256) as usize {
            break;
        }

        token_ids.push(next_token as u32);
    }

    let output = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
    let tokens_generated = token_ids.len() - initial_len;
    Ok((output, tokens_generated))
}

/// Sample token with temperature
fn sample_token(logits: &[f32], temperature: f32) -> usize {
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| (i, (logit / temperature).exp()))
        .collect();

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    probs.truncate(50);

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in &mut probs {
        *p /= sum;
    }

    let rng = rand::random::<f32>();
    let mut cumsum = 0.0;
    for (idx, p) in &probs {
        cumsum += p;
        if rng <= cumsum {
            return *idx;
        }
    }

    probs[0].0
}

/// Count lines in code
fn count_lines(code: &str) -> usize {
    code.lines().filter(|l| !l.trim().is_empty()).count()
}

/// Parse code and count functions using syn
fn count_functions(code: &str) -> usize {
    let parsed = syn::parse_file(code);
    if let Ok(file) = parsed {
        let mut count = 0;
        for item in file.items {
            match item {
                syn::Item::Fn(_) => count += 1,
                syn::Item::Impl(impl_item) => {
                    count += impl_item
                        .items
                        .iter()
                        .filter(|item| matches!(item, syn::ImplItem::Fn(_)))
                        .count();
                }
                syn::Item::Trait(trait_item) => {
                    count += trait_item
                        .items
                        .iter()
                        .filter(|item| matches!(item, syn::TraitItem::Fn(_)))
                        .count();
                }
                _ => {}
            }
        }
        count
    } else {
        0
    }
}

/// Calculate cyclomatic complexity (simplified)
/// Counts decision points: if, while, for, match, &&, ||
fn calculate_complexity(code: &str) -> usize {
    let keywords = ["if ", "while ", "for ", "match ", "&&", "||", "?"];
    let mut complexity = 1; // Base complexity
    for keyword in &keywords {
        complexity += code.matches(keyword).count();
    }
    complexity
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Model Benchmark Suite");
    println!("═══════════════════════════════════════════════════════════\n");

    // Setup device
    let device = if args.device.starts_with("cuda") {
        let gpu_id = args
            .device
            .strip_prefix("cuda:")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        Device::new_cuda(gpu_id).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    println!("Device: {:?}\n", device);

    // Load checkpoint
    println!("Loading checkpoint: {}", args.checkpoint);
    let (varmap, config, step, _) = load_checkpoint(&args.checkpoint, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint: {}", e))?;
    println!("✓ Loaded (step {})\n", step);

    // Create model
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)?;
    println!("✓ Model initialized\n");

    // Load tokenizer
    let tokenizer = Tokenizer::from_file("models/gpt2-tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("✓ Tokenizer loaded\n");

    // Setup compiler
    let compiler = CompilerFeedback::new()?;
    println!("✓ Compiler ready\n");

    // Get test prompts
    let prompts = get_test_prompts();
    let samples_per_prompt = args.n_samples / prompts.len();

    println!("Benchmark Configuration:");
    println!("  Prompts: {}", prompts.len());
    println!("  Samples per prompt: {}", samples_per_prompt);
    println!("  Total samples: {}", prompts.len() * samples_per_prompt);
    println!("  Temperature: {}", args.temperature);
    println!("  Max tokens: {}\n", args.max_tokens);

    println!("═══════════════════════════════════════════════════════════");
    println!("Running Benchmark...");
    println!("═══════════════════════════════════════════════════════════\n");

    let start_time = Instant::now();
    let mut total_compile_success = 0;
    let mut total_samples = 0;
    let mut total_lines = 0.0;
    let mut total_functions = 0.0;
    let mut total_complexity = 0.0;
    let mut total_tokens_generated = 0;
    let mut total_gen_time = 0.0;
    let mut prompt_results = Vec::new();
    let mut all_compile_errors = Vec::new();

    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        println!(
            "Prompt {}/{}: {}",
            prompt_idx + 1,
            prompts.len(),
            prompt.chars().take(50).collect::<String>()
        );

        let mut compile_success = 0;
        let mut prompt_lines = 0.0;
        let mut prompt_errors = Vec::new();
        let mut sample_outputs = Vec::new();

        for sample_idx in 0..samples_per_prompt {
            let gen_start = Instant::now();

            // Generate
            let (code, tokens_generated) = generate_sample(
                &model,
                &tokenizer,
                prompt,
                args.max_tokens,
                args.temperature,
                &device,
            )?;

            let gen_time = gen_start.elapsed().as_millis() as f64;
            total_gen_time += gen_time;
            total_tokens_generated += tokens_generated;

            // Compile
            let compile_result = compiler.compile(&code)?;
            let lines = count_lines(&code);
            let functions = count_functions(&code);
            let complexity = calculate_complexity(&code);

            if compile_result.success {
                compile_success += 1;
                total_compile_success += 1;
            } else {
                if let Some(err) = compile_result.errors.first() {
                    let err_msg = format!("{}: {}", err.level, err.message);
                    prompt_errors.push(err_msg.clone());
                    all_compile_errors.push(err_msg);
                }
            }

            prompt_lines += lines as f64;
            total_lines += lines as f64;
            total_functions += functions as f64;
            total_complexity += complexity as f64;
            total_samples += 1;

            // Save first 3 samples
            if sample_idx < 3 {
                sample_outputs.push(code.chars().take(200).collect());
            }

            if (sample_idx + 1) % 10 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }

        println!(" ✓ {}/{} compiled", compile_success, samples_per_prompt);

        prompt_results.push(PromptResults {
            prompt: prompt.clone(),
            n_samples: samples_per_prompt,
            compile_success,
            avg_lines: prompt_lines / samples_per_prompt as f64,
            common_errors: prompt_errors.into_iter().take(3).collect(),
            sample_outputs,
        });
    }

    let elapsed = start_time.elapsed();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Benchmark Results");
    println!("═══════════════════════════════════════════════════════════\n");

    let success_rate = total_compile_success as f64 / total_samples as f64 * 100.0;
    println!("Compilation:");
    println!(
        "  Success: {}/{} ({:.1}%)",
        total_compile_success, total_samples, success_rate
    );
    println!("  Failures: {}", total_samples - total_compile_success);
    println!();

    println!("Code Quality:");
    println!(
        "  Avg lines per sample: {:.1}",
        total_lines / total_samples as f64
    );
    println!(
        "  Avg functions per sample: {:.1}",
        total_functions / total_samples as f64
    );
    println!(
        "  Avg complexity per sample: {:.1}",
        total_complexity / total_samples as f64
    );
    println!();

    println!("Performance:");
    println!("  Total time: {:.1}s", elapsed.as_secs_f64());
    println!(
        "  Avg generation time: {:.0}ms",
        total_gen_time / total_samples as f64
    );
    println!(
        "  Samples per second: {:.2}",
        total_samples as f64 / elapsed.as_secs_f64()
    );
    println!(
        "  Tokens per second: {:.1}",
        total_tokens_generated as f64 / (total_gen_time / 1000.0)
    );
    println!();

    // Save results
    let results = BenchmarkResults {
        checkpoint: args.checkpoint.clone(),
        timestamp: chrono::Local::now().to_rfc3339(),
        total_samples,
        compile_success_rate: success_rate,
        compile_errors: all_compile_errors.into_iter().take(10).collect(),
        avg_lines: total_lines / total_samples as f64,
        avg_functions: total_functions / total_samples as f64,
        avg_complexity: total_complexity / total_samples as f64,
        avg_generation_time_ms: total_gen_time / total_samples as f64,
        tokens_per_second: total_tokens_generated as f64 / (total_gen_time / 1000.0),
        samples_by_prompt: prompt_results,
    };

    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&args.output, json)?;

    println!("Results saved to: {}", args.output);
    println!("\n═══════════════════════════════════════════════════════════");

    Ok(())
}
