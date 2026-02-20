//! MaxRL Training Loop for Rust Code Generation
//!
//! Maximum Likelihood RL - Only learns from compilable code
//! Paper: https://www.alphaxiv.org/abs/2602.02710
//!
//! Key difference from GRPO:
//! - GRPO: Uses all samples with relative rewards
//! - MaxRL: Uses only correct samples (reward > threshold)
//! - Result: 20x better test-time scaling

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder};
use clap::Parser;
use nanochat_rl::{
    analyze_ast, compute_reward, CompilerFeedback, MaxRLConfig, MaxRLTrainer, RewardConfig,
};
use nanochat_train::{
    checkpoint::{load_checkpoint, save_checkpoint},
    model::NanochatTrainModel,
};
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "train_maxrl")]
#[command(about = "MaxRL training for Rust code generation")]
struct Args {
    /// Base checkpoint to start from
    #[arg(long)]
    checkpoint: String,

    /// Output directory for trained model
    #[arg(long, default_value = "checkpoints/maxrl-rust")]
    output: String,

    /// Number of RL iterations
    #[arg(long, default_value = "200")]
    iterations: usize,

    /// Samples per prompt
    #[arg(long, default_value = "8")]
    n_samples: usize,

    /// Batch size
    #[arg(long, default_value = "3")]
    batch_size: usize,

    /// Correctness threshold (reward above = correct)
    #[arg(long, default_value = "20.0")]
    correctness_threshold: f64,

    /// Temperature for weighting correct samples
    #[arg(long, default_value = "1.0")]
    temperature: f64,

    /// Learning rate
    #[arg(long, default_value = "1e-5")]
    lr: f64,

    /// Device
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Checkpoint save interval
    #[arg(long, default_value = "50")]
    save_interval: usize,
}

/// Rust code generation prompts
fn get_prompts() -> Vec<String> {
    vec![
        "Write a function to calculate the factorial of a number using recursion.".to_string(),
        "Implement a struct representing a 2D point with methods for distance calculation."
            .to_string(),
        "Create a function that reverses a string in place.".to_string(),
        "Write a binary search function for a sorted array.".to_string(),
        "Implement a simple stack data structure with push, pop, and peek.".to_string(),
        "Create a function to check if a string is a palindrome.".to_string(),
        "Write a function to find the maximum element in a slice.".to_string(),
        "Implement a function that merges two sorted arrays.".to_string(),
    ]
}

/// Generate code sample using the model
fn generate_sample(
    model: &NanochatTrainModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    device: &Device,
) -> Result<String> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

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
    Ok(output)
}

/// Sample token with temperature
fn sample_token(logits: &[f32], temperature: f32) -> usize {
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| (i, (logit / temperature).exp()))
        .collect();

    // Top-k sampling (k=50)
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    probs.truncate(50);

    // Normalize
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in &mut probs {
        *p /= sum;
    }

    // Sample
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

/// Compute log probabilities for a generated sequence
fn compute_log_probs(
    model: &NanochatTrainModel,
    tokenizer: &Tokenizer,
    text: &str,
    device: &Device,
) -> Result<f64> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    let token_ids = encoding.get_ids();

    let mut total_log_prob = 0.0;

    for i in 1..token_ids.len() {
        let context = &token_ids[..i];
        let input = Tensor::new(context, device)?;
        let input = input.unsqueeze(0)?;

        let logits = model.forward(&input)?;
        let last_logits = logits.get(0)?.get(i - 1)?;
        let logits_vec = last_logits.to_vec1::<f32>()?;

        // Compute log softmax
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits_vec.iter().map(|&l| (l - max_logit).exp()).sum();
        let log_sum = max_logit + exp_sum.ln();

        let target_token = token_ids[i] as usize;
        let log_prob = logits_vec[target_token] - log_sum;
        total_log_prob += log_prob as f64;
    }

    Ok(total_log_prob)
}

/// Build weighted supervised loss from correct MaxRL samples.
///
/// Weights follow the same temperature scaling used by MaxRL:
/// exp((reward - threshold) / temperature).
fn compute_weighted_ml_loss(
    model: &NanochatTrainModel,
    tokenizer: &Tokenizer,
    samples_with_rewards: &[(String, f64)],
    correctness_threshold: f64,
    temperature: f64,
    device: &Device,
) -> Result<Option<Tensor>> {
    let mut weighted_loss: Option<Tensor> = None;
    let mut total_weight = 0.0f64;
    let temp = temperature.max(1e-6);

    for (code, reward) in samples_with_rewards {
        if *reward <= correctness_threshold {
            continue;
        }

        let encoding = tokenizer
            .encode(code, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let token_ids = encoding.get_ids();
        if token_ids.len() < 2 {
            continue;
        }

        let input = Tensor::new(&token_ids[..token_ids.len() - 1], device)?.unsqueeze(0)?;
        let target = Tensor::new(&token_ids[1..], device)?.unsqueeze(0)?;
        let sample_loss = model.forward_loss(&input, &target)?;

        let exponent = ((*reward - correctness_threshold) / temp).clamp(-20.0, 20.0);
        let weight = exponent.exp();
        let scaled = sample_loss.affine(weight, 0.0)?;

        weighted_loss = Some(match weighted_loss {
            Some(acc) => acc.add(&scaled)?,
            None => scaled,
        });
        total_weight += weight;
    }

    if total_weight == 0.0 {
        return Ok(None);
    }

    let loss = weighted_loss
        .ok_or_else(|| anyhow::anyhow!("Missing weighted loss despite non-zero total weight"))?
        .affine(1.0 / total_weight, 0.0)?;
    Ok(Some(loss))
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  MaxRL Training for Rust Code Generation");
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

    // Load base checkpoint
    println!("Loading base checkpoint: {}", args.checkpoint);
    let (varmap, config, step, _) = load_checkpoint(&args.checkpoint, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint: {}", e))?;
    println!("✓ Loaded (step {})\n", step);

    // Create model
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)?;
    println!("✓ Model initialized\n");

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr,
            ..Default::default()
        },
    )?;
    println!("✓ Optimizer initialized (AdamW, lr={})\n", args.lr);

    // Load tokenizer
    let tokenizer = Tokenizer::from_file("models/gpt2-tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("✓ Tokenizer loaded\n");

    // Setup compiler feedback
    let compiler = CompilerFeedback::new()?;
    let reward_config = RewardConfig::default();
    println!("✓ Compiler feedback ready\n");

    // Setup MaxRL trainer
    let maxrl_config = MaxRLConfig {
        correctness_threshold: args.correctness_threshold,
        temperature: args.temperature,
        ..Default::default()
    };
    let maxrl_trainer = MaxRLTrainer::new(maxrl_config);
    println!("MaxRL Configuration:");
    println!("  Correctness threshold: {}", args.correctness_threshold);
    println!("  Temperature: {}", args.temperature);
    println!("  Samples per prompt: {}", args.n_samples);
    println!("  Iterations: {}\n", args.iterations);

    // Get prompts
    let prompts = get_prompts();

    println!("═══════════════════════════════════════════════════════════");
    println!("Starting MaxRL Training Loop");
    println!("═══════════════════════════════════════════════════════════\n");

    let start_time = Instant::now();
    let mut last_loss = 0.0f64;

    for iter in 1..=args.iterations {
        println!("───────────────────────────────────────────────────────────");
        println!("Iteration {}/{}", iter, args.iterations);
        println!("───────────────────────────────────────────────────────────\n");

        let mut all_log_probs = Vec::new();
        let mut all_rewards = Vec::new();
        let mut samples_with_rewards = Vec::new();

        // Generate samples for each prompt
        for (prompt_idx, prompt) in prompts.iter().take(args.batch_size).enumerate() {
            println!(
                "Prompt {}: {}",
                prompt_idx + 1,
                prompt.chars().take(60).collect::<String>()
            );

            for sample_idx in 1..=args.n_samples {
                // Generate code
                let code = generate_sample(&model, &tokenizer, prompt, 200, 0.8, &device)?;

                // Evaluate
                let compile_result = compiler.compile(&code)?;
                let ast_metrics = analyze_ast(&code)?;
                let reward = compute_reward(&compile_result, &ast_metrics, &reward_config);

                // Compute log probs
                let log_prob = compute_log_probs(&model, &tokenizer, &code, &device)?;

                all_log_probs.push(log_prob);
                all_rewards.push(reward);
                samples_with_rewards.push((code, reward));

                let status = if compile_result.success { "✓" } else { "✗" };
                println!(
                    "  Sample {}: {} | Reward: {:.2} | LogProb: {:.2}",
                    sample_idx, status, reward, log_prob
                );
            }
            println!();
        }

        // Compute MaxRL loss
        let (loss, stats) = maxrl_trainer.compute_maxrl_loss(&all_log_probs, &all_rewards, None);

        println!("Iteration {} Statistics:", iter);
        println!("  Loss: {:.4}", loss);
        println!(
            "  Correct samples: {}/{} ({:.1}%)",
            stats.n_correct,
            stats.n_total,
            stats.correctness_rate * 100.0
        );
        println!("  Avg correct reward: {:.2}", stats.avg_correct_reward);
        println!(
            "  Compile success: {:.1}%",
            all_rewards
                .iter()
                .filter(|&&r| r > args.correctness_threshold)
                .count() as f64
                / all_rewards.len() as f64
                * 100.0
        );
        println!();
        last_loss = loss;

        if let Some(update_loss) = compute_weighted_ml_loss(
            &model,
            &tokenizer,
            &samples_with_rewards,
            args.correctness_threshold,
            args.temperature,
            &device,
        )? {
            let update_loss_value = update_loss.to_scalar::<f32>()? as f64;
            let grads = update_loss.backward()?;
            optimizer.step(&grads)?;
            println!(
                "  Optimizer step: update_loss={:.4} using {} correct samples",
                update_loss_value, stats.n_correct
            );
        } else {
            println!("  Optimizer step: skipped (no correct samples)");
        }
        println!();

        // Save checkpoint
        if iter % args.save_interval == 0 {
            let checkpoint_path = format!("{}/iter_{}", args.output, iter);
            println!("  Saving checkpoint: {}", checkpoint_path);
            save_checkpoint(&varmap, &config, iter, loss, &checkpoint_path)?;
        }

        let elapsed = start_time.elapsed().as_secs();
        println!("  Elapsed: {}m {}s\n", elapsed / 60, elapsed % 60);
    }

    // Save final checkpoint
    let final_path = format!("{}/final", args.output);
    println!("Saving final checkpoint: {}", final_path);
    save_checkpoint(&varmap, &config, args.iterations, last_loss, &final_path)?;

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  MaxRL Training Complete!");
    println!("═══════════════════════════════════════════════════════════\n");

    let total_time = start_time.elapsed().as_secs();
    println!(
        "Total time: {}h {}m",
        total_time / 3600,
        (total_time % 3600) / 60
    );
    println!("Final checkpoint: {}", final_path);

    Ok(())
}
