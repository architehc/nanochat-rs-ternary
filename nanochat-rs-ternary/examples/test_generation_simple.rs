//! Simple greedy generation test
//! Usage: cargo run --release --example test_generation_simple --features nanochat-train/cuda

use nanochat_train::{
    checkpoint::load_checkpoint,
    model::NanochatTrainModel,
};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Simple Greedy Generation Test");
    println!("═══════════════════════════════════════════════════════════\n");

    // Setup
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}\n", device);

    // Load checkpoint
    println!("Loading checkpoint...");
    let (varmap, config, step, _) = load_checkpoint("checkpoints/production-supervised/step_15500", &device)
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint: {}", e))?;
    println!("✓ Loaded (step {})\n", step);

    // Create model
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)?;
    println!("✓ Model ready\n");

    // Load tokenizer
    let tokenizer = Tokenizer::from_file("models/gpt2-tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("✓ Tokenizer loaded\n");

    println!("═══════════════════════════════════════════════════════════");
    println!("Test prompt: \"fn main() {{\"");
    println!("═══════════════════════════════════════════════════════════\n");

    let prompt = "fn main() {";
    print!("{}", prompt);
    io::stdout().flush()?;

    // Encode
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!(" (encoded to {} tokens)", token_ids.len());
    println!("\nGenerating (greedy decoding, max 50 tokens):\n");

    // Generate with greedy decoding
    for i in 0..50 {
        let input = Tensor::new(token_ids.as_slice(), &device)?;
        let input = input.unsqueeze(0)?; // Batch dimension

        let logits = model.forward(&input)?;

        // Get last token logits
        let last_logits = logits.get(0)?.get(token_ids.len() - 1)?;
        let logits_vec = last_logits.to_vec1::<f32>()?;

        // Greedy: take argmax
        let (next_token, max_logit) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if i < 5 {
            println!("  Token {}: {} (logit: {:.2})", i + 1, next_token, max_logit);
        }

        // Check for EOS
        if next_token == tokenizer.token_to_id("<|endoftext|>").unwrap_or(50256) as usize {
            println!("\n[EOS]");
            break;
        }

        // Decode and print
        if let Some(text) = tokenizer.id_to_token(next_token as u32) {
            print!("{}", text.replace("Ġ", " "));
            io::stdout().flush()?;
        }

        token_ids.push(next_token as u32);
    }

    println!("\n\n═══════════════════════════════════════════════════════════");
    println!("Done!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
