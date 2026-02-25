//! Simple generation test for the small Rust coder model.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use nanochat_train::{checkpoint::load_checkpoint, model::NanochatTrainModel};
use std::io::{self, Write};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let checkpoint_path = "checkpoints/rust-small/final";
    
    println!("Loading checkpoint: {}", checkpoint_path);
    let (varmap, config, step, _) = load_checkpoint(checkpoint_path, &device)
        .map_err(|e| format!("Load checkpoint error: {}", e))?;
    println!("✓ Loaded (step {})\n", step);

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)
        .map_err(|e| format!("Create model error: {}", e))?;
    
    let tokenizer = Tokenizer::from_file("models/gpt2-tokenizer.json")
        .map_err(|e| format!("Load tokenizer error: {}", e))?;
    println!("✓ Model and Tokenizer ready\n");

    let prompts = [
        "fn fibonacci(n: usize) -> usize {",
        "struct Point {",
        "impl Iterator for",
        "#[derive(Debug)]",
    ];

    for prompt in prompts {
        println!("Prompt: \"{}\"", prompt);
        print!("Output: ");
        io::stdout().flush()?;

        let encoding = tokenizer.encode(prompt, false).map_err(|e| format!("Encode error: {}", e))?;
        let mut gen_tokens: Vec<u32> = encoding.get_ids().to_vec();

        for _ in 0..20 {
            let input = Tensor::new(gen_tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.forward(&input)?;
            let last_logits = logits.get(0)?.get(gen_tokens.len() - 1)?;
            let logits_vec = last_logits.to_vec1::<f32>()?;

            let (next_token, _) = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let token_str = tokenizer
                .id_to_token(next_token as u32)
                .unwrap_or_else(|| format!("{}", next_token));
            print!("{}", token_str.replace("Ġ", " "));
            io::stdout().flush()?;

            gen_tokens.push(next_token as u32);
            if next_token == 50256 { break; } // stop at EOS
        }
        println!("\n");
    }

    Ok(())
}
