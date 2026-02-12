//! Inspect mHC parameters to find identity mapping

use nanochat_train::checkpoint::load_checkpoint;
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use nanochat_train::model::NanochatTrainModel;
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Inspecting mHC Parameters");
    println!("═══════════════════════════════════════════════════════════\n");

    let device = Device::Cpu;
    let checkpoint_path = "checkpoints/stable-v2/step_20000";
    let (varmap, config, _, _) = load_checkpoint(checkpoint_path, &device)
        .map_err(|e| anyhow::anyhow!("Load error: {}", e))?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)?;

    println!("Checking first block's mHC parameters:\n");

    // We need to check the mHC parameters directly
    // Let's look at the varmap to see what parameters are there
    println!("Looking for mHC-related parameters in varmap...\n");

    // Try to access mHC parameters through the model
    // The blocks have mhc_attn and mhc_ffn
    // But they're private, so we need to inspect the varmap directly

    // List all parameter names
    let mut param_names: Vec<String> = Vec::new();
    for (name, _) in varmap.all_vars().iter() {
        param_names.push(name.clone());
    }
    param_names.sort();

    println!("All parameters (filtering for mHC in block 0):");
    for name in param_names.iter() {
        if name.contains("block.0") && (name.contains("mhc") || name.contains("alpha")) {
            println!("  {}", name);
            if let Some(var) = varmap.get(&[&name]).ok() {
                let tensor = var.as_tensor();
                let vals = tensor.to_vec1::<f32>()?;
                println!("    Values: {:?}", vals);

                // Check specific properties
                if name.contains("pre_logits") {
                    let sum: f32 = vals.iter().sum();
                    println!("    Sum of pre_logits: {:.6}", sum);
                    // After softmax, check if they sum to 1
                    let exp_sum: f32 = vals.iter().map(|x| x.exp()).sum();
                    let softmax: Vec<f32> = vals.iter().map(|x| x.exp() / exp_sum).collect();
                    let softmax_sum: f32 = softmax.iter().sum();
                    println!("    Softmax values: {:?}", softmax);
                    println!("    Softmax sum: {:.6}", softmax_sum);
                }
            }
        }
    }

    Ok(())
}
