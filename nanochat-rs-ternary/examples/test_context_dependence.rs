//! Test if model uses previous context or just copies current token

use nanochat_train::{
    checkpoint::load_checkpoint,
    model::NanochatTrainModel,
};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Testing Context Dependence");
    println!("═══════════════════════════════════════════════════════════\n");

    // Load model
    let device = Device::Cpu;
    let checkpoint_path = "checkpoints/stable-v2/step_20000";
    let (varmap, config, step, _) = load_checkpoint(checkpoint_path, &device)
        .map_err(|e| anyhow::anyhow!("Load checkpoint error: {}", e))?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)?;
    let tokenizer = Tokenizer::from_file("models/gpt2-tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Load tokenizer error: {}", e))?;
    println!("✓ Loaded checkpoint step {}\n", step);

    // Test: Same ending token, different prefix
    let test_cases = vec![
        ("fn main() {", "After 'fn main()'"),
        ("struct Foo {", "After 'struct Foo'"),
        ("let x = {", "After 'let x ='"),
        ("match foo {", "After 'match foo'"),
    ];

    println!("Testing: Does the model use previous context?");
    println!("If it just copies the current '{{' token, all predictions will be identical.\n");

    for (prompt, desc) in test_cases {
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let input = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input)?;

        // Get logits for predicting AFTER the '{' token
        let last_logits = logits.get(0)?.get(token_ids.len() - 1)?;
        let logits_vec = last_logits.to_vec1::<f32>()?;

        // Top 3 predictions
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("{}: \"{}\"", desc, prompt);
        for (i, &(tok_id, logit)) in indexed.iter().take(3).enumerate() {
            let tok_str = tokenizer.id_to_token(tok_id as u32)
                .unwrap_or_else(|| format!("{}", tok_id));
            println!("  {}. Token {} (\"{}\"): logit = {:.2}",
                     i + 1, tok_id, tok_str, logit);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("Result:");
    println!("  - If all top predictions are '{{', model ignores context");
    println!("  - If predictions differ, model uses context");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
