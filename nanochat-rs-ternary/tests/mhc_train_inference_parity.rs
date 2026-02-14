//! Test that training and inference mHC implementations produce identical outputs.
//!
//! This is critical for exported model correctness - if train and inference diverge,
//! exported checkpoints will behave differently at inference time.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use mhc_lite::MhcLiteN2;
use nanochat_train::mhc::MhcLiteN2Train;

#[test]
fn test_mhc_apply_parity() -> Result<()> {
    let device = Device::Cpu;

    // Create training version with known parameters
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mhc_train = MhcLiteN2Train::new(vb.pp("test"))?;

    // Extract parameters for inference version
    let alpha_logit = mhc_train.alpha_logit.to_vec1::<f32>()?[0];
    let pre_logits = mhc_train.pre_logits.to_vec1::<f32>()?;
    let pre_bias = mhc_train.pre_bias.to_vec1::<f32>()?;
    let post_logits = mhc_train.post_logits.to_vec1::<f32>()?;
    let post_bias = mhc_train.post_bias.to_vec1::<f32>()?;

    let mhc_inf = MhcLiteN2 {
        alpha_logit,
        pre_logits: [pre_logits[0], pre_logits[1]],
        pre_bias: [pre_bias[0], pre_bias[1]],
        post_logits: [post_logits[0], post_logits[1]],
        post_bias: [post_bias[0], post_bias[1]],
    };

    // Test data
    let dim = 64;
    let batch = 2;
    let seq = 4;

    // Create test inputs
    let x_exp_data: Vec<f32> = (0..batch * seq * 2 * dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let layer_out_data: Vec<f32> = (0..batch * seq * dim)
        .map(|i| (i as f32 * 0.02).cos())
        .collect();

    // Training forward (candle tensors)
    let x_exp_train = Tensor::from_vec(x_exp_data.clone(), (batch, seq, 2 * dim), &device)?;
    let layer_out_train = Tensor::from_vec(layer_out_data.clone(), (batch, seq, dim), &device)?;
    let result_train = mhc_train.apply(&x_exp_train, &layer_out_train, dim)?;
    let result_train_flat = result_train.flatten_all()?.to_vec1::<f32>()?;

    // Inference forward (flat arrays, reshaped to [batch*seq, 2*dim] and [batch*seq, dim])
    let x_exp_inf = x_exp_data; // Already flat
    let layer_out_inf = layer_out_data; // Already flat

    // Reshape for inference: flatten batch and seq dimensions
    let x_exp_inf_flat: Vec<f32> = (0..batch * seq)
        .flat_map(|bs| {
            let offset = bs * 2 * dim;
            x_exp_inf[offset..offset + 2 * dim].to_vec()
        })
        .collect();
    let layer_out_inf_flat: Vec<f32> = (0..batch * seq)
        .flat_map(|bs| {
            let offset = bs * dim;
            layer_out_inf[offset..offset + dim].to_vec()
        })
        .collect();

    let result_inf = mhc_inf.apply(&x_exp_inf_flat, &layer_out_inf_flat, dim);

    // Compare results
    assert_eq!(
        result_train_flat.len(),
        result_inf.len(),
        "Output lengths must match"
    );

    let max_diff = result_train_flat
        .iter()
        .zip(result_inf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "Max difference between train and inference: {:.10}",
        max_diff
    );

    assert!(
        max_diff < 1e-5,
        "Train and inference outputs differ by {:.10} (threshold: 1e-5)",
        max_diff
    );

    Ok(())
}

#[test]
fn test_mhc_with_different_params() -> Result<()> {
    // Test with non-default parameters to ensure parity holds generally
    let device = Device::Cpu;

    // Create mHC with custom initialization to test parity with non-default params
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Manually create with custom init
    let alpha_logit = vb.get_with_hints(1, "alpha_logit", candle_nn::Init::Const(1.5))?;
    let pre_logits = vb.get_with_hints(
        2,
        "pre_logits",
        candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.5,
        },
    )?;
    let pre_bias = vb.get_with_hints(2, "pre_bias", candle_nn::Init::Const(0.5))?;
    let post_logits = vb.get_with_hints(
        2,
        "post_logits",
        candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.3,
        },
    )?;
    let post_bias = vb.get_with_hints(2, "post_bias", candle_nn::Init::Const(0.5))?;

    let mhc_train = MhcLiteN2Train {
        alpha_logit,
        pre_logits,
        pre_bias,
        post_logits,
        post_bias,
    };

    // Extract for inference
    let alpha_logit = mhc_train.alpha_logit.to_vec1::<f32>()?[0];
    let pre_logits = mhc_train.pre_logits.to_vec1::<f32>()?;
    let pre_bias = mhc_train.pre_bias.to_vec1::<f32>()?;
    let post_logits = mhc_train.post_logits.to_vec1::<f32>()?;
    let post_bias = mhc_train.post_bias.to_vec1::<f32>()?;

    let mhc_inf = MhcLiteN2 {
        alpha_logit,
        pre_logits: [pre_logits[0], pre_logits[1]],
        pre_bias: [pre_bias[0], pre_bias[1]],
        post_logits: [post_logits[0], post_logits[1]],
        post_bias: [post_bias[0], post_bias[1]],
    };

    // Simple test case
    let dim = 32;
    let x_exp_data: Vec<f32> = (0..2 * dim).map(|i| i as f32 * 0.1).collect();
    let layer_out_data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin()).collect();

    let x_exp_train = Tensor::from_vec(x_exp_data.clone(), (1, 1, 2 * dim), &device)?;
    let layer_out_train = Tensor::from_vec(layer_out_data.clone(), (1, 1, dim), &device)?;
    let result_train = mhc_train.apply(&x_exp_train, &layer_out_train, dim)?;
    let result_train_flat = result_train.flatten_all()?.to_vec1::<f32>()?;

    let result_inf = mhc_inf.apply(&x_exp_data, &layer_out_data, dim);

    let max_diff = result_train_flat
        .iter()
        .zip(result_inf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max diff with custom params: {:.10}", max_diff);

    assert!(
        max_diff < 1e-5,
        "Parity broken with custom params: max_diff={:.10}",
        max_diff
    );

    Ok(())
}
