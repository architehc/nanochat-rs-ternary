//! Test if loss calculation properly aligns logits with targets

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use nanochat_train::config::TrainConfig;
use nanochat_train::model::NanochatTrainModel;

fn main() -> Result<()> {
    println!("Testing loss alignment...\n");

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mut config = TrainConfig::d20();
    config.vocab_size = 10; // Small vocab for testing

    let model = NanochatTrainModel::new(&config, vb)?;

    // Create test input: [0, 1, 2, 3]
    let input_ids = Tensor::new(&[0u32, 1, 2, 3], &device)?.unsqueeze(0)?;
    // Target should be: [1, 2, 3, 4] (next tokens)
    let target_ids = Tensor::new(&[1u32, 2, 3, 4], &device)?.unsqueeze(0)?;

    println!("Input:  [0, 1, 2, 3]");
    println!("Target: [1, 2, 3, 4]");
    println!();

    // Forward pass
    let logits = model.forward(&input_ids)?; // [1, 4, 10]
    println!("Logits shape: {:?}", logits.dims());

    // Check what each position predicts
    for pos in 0..4 {
        let pos_logits = logits.get(0)?.get(pos)?; // [10]
        let pos_logits_vec = pos_logits.to_vec1::<f32>()?;

        let (predicted_token, max_logit) = pos_logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let target = target_ids.get(0)?.get(pos)?.to_vec0::<u32>()?;

        println!(
            "Position {}: logits predicts token {}, target is token {} {}",
            pos,
            predicted_token,
            target,
            if predicted_token as u32 == target {
                "✓"
            } else {
                "✗"
            }
        );
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Expected behavior:");
    println!("  logits[0] should learn to predict target[0] = 1");
    println!("  logits[1] should learn to predict target[1] = 2");
    println!("  logits[2] should learn to predict target[2] = 3");
    println!("  logits[3] should learn to predict target[3] = 4");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
