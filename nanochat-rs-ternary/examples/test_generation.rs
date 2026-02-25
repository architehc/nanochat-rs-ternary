//! Test code generation with trained checkpoint
//!
//! Usage:
//!   cargo run --release --example test_generation --features nanochat-train/cuda

use nanochat_train::{
    checkpoint::load_checkpoint,
    model::NanochatTrainModel,
};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use anyhow::Result;

fn sample_token(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| (i, (logit / temperature).exp()))
        .collect();

    // Sort by probability descending
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top-k
    probs.truncate(top_k);

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

fn generate(
    model: &NanochatTrainModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    device: &Device,
) -> Result<String> {
    // Encode prompt
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {}", e))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

    print!("{}", prompt);
    io::stdout().flush()?;

    // Generate tokens
    for _ in 0..max_tokens {
        // Forward pass
        let input_tensor = Tensor::new(token_ids.as_slice(), device)?;
        let input_tensor = input_tensor.unsqueeze(0)?; // Add batch dimension

        let logits = model.forward(&input_tensor)?;

        // Get logits for last token
        let last_logits = logits.get(0)?.get(token_ids.len() - 1)?;
        let logits_vec = last_logits.to_vec1::<f32>()?;

        // Sample next token
        let next_token = sample_token(&logits_vec, temperature, top_k);

        // Check for EOS
        if next_token == tokenizer.token_to_id("<|endoftext|>").unwrap_or(50256) as usize {
            break;
        }

        // Decode and print
        if let Some(text) = tokenizer.id_to_token(next_token as u32) {
            print!("{}", text.replace("Ġ", " "));
            io::stdout().flush()?;
        }

        token_ids.push(next_token as u32);
    }

    println!("\n");

    // Decode full sequence
    let output = tokenizer.decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;
    Ok(output)
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Testing Trained Rust Code Generation Model");
    println!("═══════════════════════════════════════════════════════════\n");

    // Setup device
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}\n", device);

    // Load checkpoint
    let ckpt_dir = std::env::args().nth(1)
        .unwrap_or_else(|| "checkpoints/nano-500m-haar/step_10000".to_string());
    let tokenizer_path = std::env::args().nth(2)
        .unwrap_or_else(|| "data/rust_prepared/tokenizer.json".to_string());

    println!("Loading checkpoint from: {}", ckpt_dir);
    let (varmap, config, step, loss) = load_checkpoint(&ckpt_dir, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint: {}", e))?;
    println!("✓ Checkpoint loaded (step {}, loss {:.4})\n", step, loss);

    // Create model
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)?;
    println!("✓ Model initialized ({:.1}M params)\n", model.param_count() as f32 / 1_000_000.0);

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("✓ Tokenizer loaded ({})\n", tokenizer_path);

    println!("═══════════════════════════════════════════════════════════");
    println!("  Generating Rust Code");
    println!("═══════════════════════════════════════════════════════════\n");

    // Test prompts
    let prompts = vec![
        "fn fibonacci(n: usize) -> usize {",
        "struct Point { x: f64, y: f64 }\n\nimpl Point {",
        "use std::collections::HashMap;\n\nfn count_words(text: &str) -> HashMap<String, usize> {",
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {",
        "async fn fetch_data(url: &str) -> Result<String, Box<dyn std::error::Error>> {",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        println!("─────────────────────────────────────────────────────────");
        println!("Test {}/{}:", i + 1, prompts.len());
        println!("─────────────────────────────────────────────────────────\n");

        match generate(&model, &tokenizer, prompt, 150, 0.3, 10, &device) {
            Ok(_) => {},
            Err(e) => eprintln!("Error generating: {}", e),
        }

        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  Testing Complete!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
