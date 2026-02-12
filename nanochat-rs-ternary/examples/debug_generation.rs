//! Debug generation - instrument model internals to find the bug
//!
//! This script adds extensive logging to understand why the model
//! produces repeated tokens despite reasonable training loss.

use nanochat_train::{
    checkpoint::load_checkpoint,
    model::NanochatTrainModel,
};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use anyhow::Result;

fn tensor_stats(tensor: &Tensor, name: &str) -> Result<()> {
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let std = {
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    };
    println!("  {} stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
             name, mean, std, min, max);
    Ok(())
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  DEBUG GENERATION - Finding the Architecture Bug");
    println!("═══════════════════════════════════════════════════════════\n");

    // Setup
    let device = Device::Cpu;
    println!("Device: {:?}\n", device);

    // Load checkpoint
    let checkpoint_path = "checkpoints/stable-v2/step_20000";
    println!("Loading checkpoint: {}", checkpoint_path);
    let (varmap, config, step, _) = load_checkpoint(checkpoint_path, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint: {}", e))?;
    println!("✓ Loaded (step {})\n", step);

    println!("Model config:");
    println!("  dim: {}", config.dim);
    println!("  n_layers: {}", config.n_layers);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  weight_tied: {}\n", config.weight_tied);

    // Create model
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)?;
    println!("✓ Model ready\n");

    // Load tokenizer
    let tokenizer = Tokenizer::from_file("models/gpt2-tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("✓ Tokenizer loaded\n");

    // Test 1: Check embedding layer
    println!("═══════════════════════════════════════════════════════════");
    println!("TEST 1: Embedding Layer Sanity");
    println!("═══════════════════════════════════════════════════════════\n");

    let embed_weights = model.tok_embed.embeddings();
    tensor_stats(embed_weights, "embedding weights")?;

    // Check a few specific embeddings
    let token_0 = embed_weights.get(0)?;
    let token_100 = embed_weights.get(100)?;
    let token_1391 = embed_weights.get(1391)?; // The problematic token

    println!("  Token 0 embedding (first 5): {:?}",
             token_0.narrow(0, 0, 5)?.to_vec1::<f32>()?);
    println!("  Token 100 embedding (first 5): {:?}",
             token_100.narrow(0, 0, 5)?.to_vec1::<f32>()?);
    println!("  Token 1391 embedding (first 5): {:?}",
             token_1391.narrow(0, 0, 5)?.to_vec1::<f32>()?);

    // Check if embeddings are diverse
    let emb_0_vec = token_0.to_vec1::<f32>()?;
    let emb_100_vec = token_100.to_vec1::<f32>()?;
    let cosine_sim = emb_0_vec.iter().zip(emb_100_vec.iter())
        .map(|(a, b)| a * b).sum::<f32>() /
        (emb_0_vec.iter().map(|x| x.powi(2)).sum::<f32>().sqrt() *
         emb_100_vec.iter().map(|x| x.powi(2)).sum::<f32>().sqrt());
    println!("  Cosine similarity (token 0 vs 100): {:.4}", cosine_sim);
    println!();

    // Test 2: Check LM head weights
    println!("═══════════════════════════════════════════════════════════");
    println!("TEST 2: LM Head Weights");
    println!("═══════════════════════════════════════════════════════════\n");

    // Get LM head weights (transposed embedding if weight-tied)
    let lm_weights = if config.weight_tied {
        println!("  Weight-tied: using transposed embeddings");
        embed_weights.t()?
    } else {
        println!("  Separate LM head");
        model.lm_head_weight.as_ref().unwrap().t()?
    };

    tensor_stats(&lm_weights, "LM head weights")?;

    // Check specific output weights for token 1391
    // lm_weights is [vocab, dim] after transpose, so get(1391) gets the weight vector for token 1391
    let lm_weights_shape = lm_weights.dims();
    println!("  LM weights shape: {:?}", lm_weights_shape);

    // Get the weight vector that produces logit for token 1391
    // This is row 1391 if shape is [vocab, dim]
    if lm_weights_shape[0] == config.vocab_size {
        let output_weight_1391 = lm_weights.get(1391)?;
        println!("  Output weight for token 1391 (first 5): {:?}",
                 output_weight_1391.narrow(0, 0, 5)?.to_vec1::<f32>()?);
    } else {
        println!("  Unexpected LM weight shape, skipping token 1391 check");
    }
    println!();

    // Test 3: Forward pass with simple prompt
    println!("═══════════════════════════════════════════════════════════");
    println!("TEST 3: Forward Pass Analysis");
    println!("═══════════════════════════════════════════════════════════\n");

    let prompt = "fn main() {";
    println!("Prompt: \"{}\"", prompt);

    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Encoded to {} tokens: {:?}\n", token_ids.len(), token_ids);

    // Forward pass
    let input = Tensor::new(token_ids.as_slice(), &device)?;
    let input = input.unsqueeze(0)?; // [1, seq_len]
    println!("Input shape: {:?}", input.shape());

    let logits = model.forward(&input)?;
    println!("Output shape: {:?}", logits.shape());

    // Get last token logits
    let last_logits = logits.get(0)?.get(token_ids.len() - 1)?;
    tensor_stats(&last_logits, "last token logits")?;

    let logits_vec = last_logits.to_vec1::<f32>()?;

    // Find top-5 tokens
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 predicted tokens:");
    for (i, &(token_id, logit)) in indexed.iter().take(5).enumerate() {
        let token_str = tokenizer.id_to_token(token_id as u32)
            .unwrap_or_else(|| format!("<unk:{}>", token_id));
        println!("  {}. Token {} (\"{}\"): logit = {:.4}",
                 i + 1, token_id, token_str, logit);
    }

    // Check logit distribution
    let logit_mean = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
    let logit_std = {
        let variance = logits_vec.iter().map(|x| (x - logit_mean).powi(2)).sum::<f32>()
            / logits_vec.len() as f32;
        variance.sqrt()
    };
    println!("\nLogit distribution:");
    println!("  Mean: {:.4}", logit_mean);
    println!("  Std: {:.4}", logit_std);
    println!("  Max: {:.4}", indexed[0].1);
    println!("  Gap (max - 2nd): {:.4}", indexed[0].1 - indexed[1].1);

    // Calculate entropy before softmax
    let logit_exp_sum: f32 = logits_vec.iter().map(|&x| (x - indexed[0].1).exp()).sum();
    let prob_max = 1.0 / logit_exp_sum;
    println!("  Probability of max token (after softmax): {:.6}", prob_max);

    // Test 4: Check if all positions produce the same prediction
    println!("\n═══════════════════════════════════════════════════════════");
    println!("TEST 4: Position-wise Predictions");
    println!("═══════════════════════════════════════════════════════════\n");

    for pos in 0..token_ids.len() {
        let pos_logits = logits.get(0)?.get(pos)?;
        let pos_logits_vec = pos_logits.to_vec1::<f32>()?;

        let (top_token, top_logit) = pos_logits_vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let token_str = tokenizer.id_to_token(top_token as u32)
            .unwrap_or_else(|| format!("<unk:{}>", top_token));

        println!("  Position {}: top token {} (\"{}\"), logit = {:.2}",
                 pos, top_token, token_str, top_logit);
    }

    // Test 5: Generate with temperature
    println!("\n═══════════════════════════════════════════════════════════");
    println!("TEST 5: Generation with Temperature Scaling");
    println!("═══════════════════════════════════════════════════════════\n");

    for temp in &[0.1, 0.5, 1.0, 2.0] {
        print!("Temperature {}: ", temp);
        io::stdout().flush()?;

        let mut gen_tokens = token_ids.clone();
        for _ in 0..10 {
            let input = Tensor::new(gen_tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.forward(&input)?;
            let last_logits = logits.get(0)?.get(gen_tokens.len() - 1)?;
            let scaled = (last_logits / *temp as f64)?;
            let logits_vec = scaled.to_vec1::<f32>()?;

            let (next_token, _) = logits_vec.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let token_str = tokenizer.id_to_token(next_token as u32)
                .unwrap_or_else(|| format!("{}", next_token));
            print!("{}", token_str.replace("Ġ", " "));
            io::stdout().flush()?;

            gen_tokens.push(next_token as u32);
        }
        println!();
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Debug complete! Check output above for anomalies.");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
