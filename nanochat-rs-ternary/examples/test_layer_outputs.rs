//! Test if transformer layers are producing non-zero outputs

use nanochat_train::{
    checkpoint::load_checkpoint,
    model::NanochatTrainModel,
};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use anyhow::Result;

fn tensor_stats(tensor: &Tensor, name: &str) -> Result<()> {
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let absmax = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let std = {
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    };
    println!("  {}: mean={:.6}, std={:.6}, absmax={:.6}", name, mean, std, absmax);
    Ok(())
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Testing Layer Outputs");
    println!("═══════════════════════════════════════════════════════════\n");

    // Load model
    let device = Device::Cpu;
    let checkpoint_path = "checkpoints/stable-v2/step_20000";
    let (varmap, config, step, _) = load_checkpoint(checkpoint_path, &device)
        .map_err(|e| anyhow::anyhow!("Load checkpoint error: {}", e))?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)?;
    println!("✓ Loaded checkpoint step {}\n", step);

    // Load tokenizer
    let tokenizer = Tokenizer::from_file("models/gpt2-tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Load tokenizer error: {}", e))?;

    // Test with a simple prompt
    let prompt = "fn main() {";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Prompt: \"{}\"", prompt);
    println!("Tokens: {:?}\n", token_ids);

    // Run full forward pass
    let input = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;

    // Check what the model outputs BEFORE the LM head
    // We can't access internals easily, so let's check the logits
    let logits = model.forward(&input)?; // [1, seq_len, vocab]

    // For each position, check if it predicts its own input token
    println!("Checking if model predicts input tokens:\n");
    for (pos, &token_id) in token_ids.iter().enumerate() {
        let pos_logits = logits.get(0)?.get(pos)?;
        let pos_logits_vec = pos_logits.to_vec1::<f32>()?;

        let (top_token, top_logit) = pos_logits_vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let token_str = tokenizer.id_to_token(token_id)
            .unwrap_or_else(|| format!("{}", token_id));
        let top_str = tokenizer.id_to_token(top_token as u32)
            .unwrap_or_else(|| format!("{}", top_token));

        let matches = top_token as u32 == token_id;
        println!("Position {}: input={} (\"{}\"), predicts={} (\"{}\"), logit={:.2} {}",
                 pos, token_id, token_str, top_token, top_str, top_logit,
                 if matches { "✗ SAME!" } else { "✓ different" });
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("If block delta is near-zero, the transformer isn't learning!");
    println!("If collapsed output ≈ original embedding, model is identity!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
