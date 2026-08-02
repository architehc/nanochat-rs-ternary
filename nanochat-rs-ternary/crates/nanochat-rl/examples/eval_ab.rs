//! A/B evaluation: RL-tuned checkpoint vs base checkpoint.
//!
//! Samples completions from both checkpoints over the GRPO training prompts
//! plus a held-out set the RL run never saw, compiles prompt + completion
//! with rustc, and reports compile rate, parse rate, and mean reward per
//! checkpoint and per prompt set.
//!
//! ```bash
//! NANOCHAT_TOKENIZER=data/rust_v4_4k/tokenizer.json \
//! cargo run --release -p nanochat-rl --example eval_ab \
//!   --features nanochat-train/cuda -- \
//!   --base checkpoints/qwen35_hybrid_seq512/step_82000 \
//!   --rl checkpoints/rl-final
//! ```

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use nanochat_rl::{analyze_ast, compute_reward, CompilerFeedback, RewardConfig};
use nanochat_train::checkpoint::load_checkpoint;
use nanochat_train::model::NanochatTrainModel;
use rand::Rng;
use std::cmp::Ordering;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "Compare compile/parse rates of two checkpoints")]
struct Args {
    /// Base checkpoint (pre-RL)
    #[arg(long)]
    base: String,

    /// RL-tuned checkpoint
    #[arg(long)]
    rl: String,

    /// Samples per prompt
    #[arg(long, default_value = "8")]
    n_samples: usize,

    /// Max tokens per completion
    #[arg(long, default_value = "200")]
    max_tokens: usize,

    /// Sampling temperature
    #[arg(long, default_value = "0.8")]
    temperature: f32,
}

/// Subset of the GRPO training prompts (seen by the RL run).
fn train_prompts() -> Vec<&'static str> {
    vec![
        "/// Compute the factorial of n recursively.\nfn factorial(n: u64) -> u64 {\n",
        "/// Return only the even numbers from the input.\nfn filter_even(nums: &[i32]) -> Vec<i32> {\n",
        "/// Binary search over a sorted slice; returns the index of target.\nfn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n",
        "/// Sieve of Eratosthenes: all primes up to n.\nfn primes_up_to(n: usize) -> Vec<usize> {\n",
        "/// Check whether a string is a palindrome.\nfn is_palindrome(s: &str) -> bool {\n",
        "/// Greatest common divisor via Euclid's algorithm.\nfn gcd(a: u64, b: u64) -> u64 {\n",
        "/// Merge two sorted vectors into one sorted vector.\nfn merge(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {\n",
        "/// Maximum element of a slice, if any.\nfn max_element(xs: &[i32]) -> Option<i32> {\n",
        "/// Count vowels in a string.\nfn count_vowels(s: &str) -> usize {\n",
        "/// Sum all elements of a slice.\nfn sum(xs: &[i64]) -> i64 {\n",
    ]
}

/// Prompts the RL run never trained on.
fn heldout_prompts() -> Vec<&'static str> {
    vec![
        "/// Compute the n-th power of base without overflow checks.\nfn power(base: u64, n: u32) -> u64 {\n",
        "/// Return the indices of the two numbers that add up to target.\nfn two_sum(nums: &[i32], target: i32) -> Option<(usize, usize)> {\n",
        "/// Convert a temperature from Celsius to Kelvin.\nfn celsius_to_kelvin(c: f64) -> f64 {\n",
        "/// Count how many times needle occurs in haystack.\nfn count_occurrences(haystack: &str, needle: char) -> usize {\n",
        "/// Return the running (prefix) sums of the input.\nfn prefix_sums(xs: &[i64]) -> Vec<i64> {\n",
        "/// A simple counter.\npub struct Counter {\n    count: u64,\n}\n\nimpl Counter {\n    /// Increment and return the new value.\n    pub fn increment(&mut self) -> u64 {\n",
        "/// Remove consecutive duplicate elements.\nfn dedup_consecutive(xs: Vec<i32>) -> Vec<i32> {\n",
        "/// Interleave two slices into one vector.\nfn interleave(a: &[i32], b: &[i32]) -> Vec<i32> {\n",
    ]
}

struct SetStats {
    n: usize,
    compiled: usize,
    parsed: usize,
    reward_sum: f64,
}

impl SetStats {
    fn new() -> Self {
        Self { n: 0, compiled: 0, parsed: 0, reward_sum: 0.0 }
    }
    fn line(&self) -> String {
        format!(
            "compile {:>5.1}% ({}/{}) | parse {:>5.1}% ({}/{}) | mean reward {:+.2}",
            100.0 * self.compiled as f64 / self.n.max(1) as f64,
            self.compiled,
            self.n,
            100.0 * self.parsed as f64 / self.n.max(1) as f64,
            self.parsed,
            self.n,
            self.reward_sum / self.n.max(1) as f64,
        )
    }
}

fn sample_token(logits: &[f32], temperature: f32) -> usize {
    let temp = temperature.max(0.05) as f64;
    let mut scaled: Vec<f64> = logits.iter().map(|&v| (v as f64) / temp).collect();
    let max_logit = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for v in &mut scaled {
        *v = (*v - max_logit).exp();
    }
    let k = 50.min(scaled.len());
    let mut idx: Vec<usize> = (0..scaled.len()).collect();
    idx.sort_by(|a, b| scaled[*b].partial_cmp(&scaled[*a]).unwrap_or(Ordering::Equal));
    let mut probs = vec![0.0f64; scaled.len()];
    let mut mass = 0.0f64;
    for i in idx.into_iter().take(k) {
        probs[i] = scaled[i];
        mass += scaled[i];
    }
    if mass <= 0.0 {
        return 0;
    }
    let r = rand::thread_rng().gen::<f64>() * mass;
    let mut cumulative = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumulative += *p;
        if r <= cumulative {
            return i;
        }
    }
    0
}

/// Batched lockstep decoding: all rows share the prompt, so the batch stays
/// rectangular. Finished rows get EOS filler and are trimmed afterwards.
fn sample_batch(
    model: &NanochatTrainModel,
    tokenizer: &Tokenizer,
    device: &Device,
    max_seq_len: usize,
    prompt: &str,
    n_samples: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<Vec<String>> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_tokens.is_empty() {
        prompt_tokens.push(0);
    }
    let prompt_len = prompt_tokens.len();
    let eos_id = tokenizer
        .token_to_id("<|endoftext|>")
        .or_else(|| tokenizer.token_to_id("<eos>"))
        .unwrap_or(u32::MAX);

    let mut rows: Vec<Vec<u32>> = vec![prompt_tokens; n_samples];
    let mut finished = vec![false; n_samples];
    let mut n_generated = vec![0usize; n_samples];

    for _ in 0..max_tokens {
        let seq_len = rows[0].len();
        if seq_len + 1 >= max_seq_len || finished.iter().all(|&f| f) {
            break;
        }
        let flat: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let input = Tensor::from_vec(flat, (n_samples, seq_len), device)?;
        let logits = model.forward(&input)?;
        let last = logits.i((.., seq_len - 1, ..))?.to_vec2::<f32>()?;
        for (i, row_logits) in last.iter().enumerate() {
            if finished[i] {
                rows[i].push(eos_id);
                continue;
            }
            let next = sample_token(row_logits, temperature) as u32;
            rows[i].push(next);
            if next == eos_id {
                finished[i] = true;
            } else {
                n_generated[i] += 1;
            }
        }
    }

    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut toks = std::mem::take(&mut rows[i]);
        toks.truncate(prompt_len + n_generated[i]);
        out.push(tokenizer.decode(&toks[prompt_len..], true).unwrap_or_default());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn eval_checkpoint(
    label: &str,
    ckpt: &str,
    device: &Device,
    tokenizer: &Tokenizer,
    compiler: &CompilerFeedback,
    reward_cfg: &RewardConfig,
    args: &Args,
) -> Result<(SetStats, SetStats)> {
    println!("── {} ({})", label, ckpt);
    let (varmap, train_config, step, _) = load_checkpoint(ckpt, device)
        .map_err(|e| anyhow::anyhow!("loading {}: {}", ckpt, e))?;
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let model = NanochatTrainModel::new(&train_config, vb)?;
    println!("   loaded (step {})", step);

    let mut result = Vec::new();
    for (set_name, prompts) in [("train", train_prompts()), ("heldout", heldout_prompts())] {
        let mut stats = SetStats::new();
        for prompt in prompts {
            let completions = sample_batch(
                &model,
                tokenizer,
                device,
                train_config.max_seq_len,
                prompt,
                args.n_samples,
                args.max_tokens,
                args.temperature,
            )?;
            for completion in completions {
                let full_code = format!("{}{}", prompt, completion);
                let compile_result = compiler.compile(&full_code)?;
                let ast = analyze_ast(&full_code)?;
                stats.n += 1;
                stats.compiled += compile_result.success as usize;
                stats.parsed += ast.parseable as usize;
                stats.reward_sum += compute_reward(&compile_result, &ast, reward_cfg);
            }
        }
        println!("   {:>7}: {}", set_name, stats.line());
        result.push(stats);
    }
    let heldout = result.pop().unwrap();
    let train = result.pop().unwrap();
    Ok((train, heldout))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_cuda(0)?;
    let tokenizer_path = std::env::var("NANOCHAT_TOKENIZER")
        .unwrap_or_else(|_| "data/rust_v4_4k/tokenizer.json".to_string());
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer {}: {}", tokenizer_path, e))?;
    let compiler = CompilerFeedback::new()?;
    let reward_cfg = RewardConfig::default();

    println!(
        "A/B eval: {} samples/prompt, max_tokens {}, temperature {}\n",
        args.n_samples, args.max_tokens, args.temperature
    );

    let (base_train, base_held) = eval_checkpoint(
        "BASE", &args.base, &device, &tokenizer, &compiler, &reward_cfg, &args,
    )?;
    let (rl_train, rl_held) = eval_checkpoint(
        "RL", &args.rl, &device, &tokenizer, &compiler, &reward_cfg, &args,
    )?;

    println!("\n══ Summary (RL vs BASE) ══");
    for (name, base, rl) in [
        ("train prompts", &base_train, &rl_train),
        ("heldout prompts", &base_held, &rl_held),
    ] {
        println!(
            "{:>15}: compile {:+.1}pp | parse {:+.1}pp | reward {:+.2}",
            name,
            100.0 * (rl.compiled as f64 / rl.n.max(1) as f64
                - base.compiled as f64 / base.n.max(1) as f64),
            100.0 * (rl.parsed as f64 / rl.n.max(1) as f64
                - base.parsed as f64 / base.n.max(1) as f64),
            rl.reward_sum / rl.n.max(1) as f64 - base.reward_sum / base.n.max(1) as f64,
        );
    }
    Ok(())
}
