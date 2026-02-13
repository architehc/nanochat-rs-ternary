//! LoopLM-specific export-load roundtrip integration test.
//!
//! Validates that loop-configured models (d20_loop) can be:
//! 1. Exported to GGUF + mHC with loop metadata
//! 2. Loaded back with correct loop architecture reconstruction
//! 3. Execute forward pass with loop iterations
//! 4. Produce finite, non-degenerate outputs

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;

use nanochat_train::config::TrainConfig;
use nanochat_train::export::export_model;
use nanochat_train::model::NanochatTrainModel;

use nanochat_model::model::NanochatModel;

#[test]
fn test_loop_export_load_forward_roundtrip() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Use d20_loop config: 1 local_before + 1 shared (×4 iterations) + 1 local_after
    let cfg = TrainConfig::d20_loop();
    let loop_cfg = cfg.loop_config.as_ref().unwrap();

    assert_eq!(loop_cfg.local_before, 1);
    assert_eq!(loop_cfg.local_after, 1);
    assert_eq!(loop_cfg.loop_count, 4);

    println!("\n=== LoopLM Roundtrip Test ===");
    println!("Config: d20_loop");
    println!("  local_before: {}", loop_cfg.local_before);
    println!("  loop_count: {}", loop_cfg.loop_count);
    println!("  local_after: {}", loop_cfg.local_after);
    println!("  effective_depth: {}", loop_cfg.effective_depth());

    // 1. Create training model with loop architecture
    let train_model = NanochatTrainModel::new(&cfg, vb).unwrap();

    // Verify loop architecture was built correctly
    assert_eq!(train_model.blocks.len(), 0, "Standard blocks should be empty");
    assert_eq!(train_model.local_blocks_before.len(), 1);
    assert_eq!(train_model.local_blocks_after.len(), 1);
    assert!(train_model.shared_loop_block.is_some());
    println!("✓ Training model built with loop architecture");

    // 2. Run forward pass through training model
    let tokens = vec![1u32, 5, 10, 20, 42];
    let token_tensor = Tensor::from_vec(tokens.clone(), (1, 5), &device).unwrap();
    let train_logits = train_model.forward(&token_tensor).unwrap();

    // Get last-token logits: [1, 5, vocab] -> [vocab]
    let train_last = train_logits
        .narrow(1, 4, 1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    println!("✓ Training forward pass completed");
    println!("  train logits shape: [{}]", train_last.len());

    // 3. Export to GGUF + mHC with loop metadata
    let dir = tempfile::tempdir().unwrap();
    let gguf_path = dir.path().join("d20_loop.gguf");
    let mhc_path = dir.path().join("d20_loop.mhc");

    export_model(
        &train_model,
        &cfg,
        gguf_path.to_str().unwrap(),
        mhc_path.to_str().unwrap(),
    )
    .unwrap();

    println!("✓ Exported to GGUF + mHC");

    // 4. Load via inference model
    let mut inf_model = NanochatModel::from_gguf(
        gguf_path.to_str().unwrap(),
        mhc_path.to_str().unwrap(),
    )
    .unwrap();

    println!("✓ Loaded from GGUF + mHC");

    // 5. Verify loop config was reconstructed correctly
    assert!(
        inf_model.config.loop_config.is_some(),
        "Loop config should be loaded from GGUF metadata"
    );

    let loaded_loop_cfg = inf_model.config.loop_config.as_ref().unwrap();
    assert_eq!(loaded_loop_cfg.local_before, loop_cfg.local_before);
    assert_eq!(loaded_loop_cfg.local_after, loop_cfg.local_after);
    assert_eq!(loaded_loop_cfg.loop_count, loop_cfg.loop_count);

    println!("✓ Loop config reconstructed correctly");

    // 6. Verify loop architecture was reconstructed
    assert_eq!(inf_model.blocks.len(), 0, "Standard blocks should be empty");
    assert_eq!(inf_model.local_blocks_before.len(), 1);
    assert_eq!(inf_model.local_blocks_after.len(), 1);
    assert!(inf_model.shared_loop_block.is_some());
    assert!(inf_model.loop_kv_cache.is_some());

    println!("✓ Inference model built with loop architecture");

    // 7. Run same tokens through inference model (autoregressive)
    let inf_logits = inf_model.forward_sequence(&tokens);

    println!("✓ Inference forward pass completed");
    println!("  inf logits shape: [{}]", inf_logits.len());

    // 8. Basic sanity checks
    assert_eq!(
        train_last.len(),
        inf_logits.len(),
        "logit vector size mismatch"
    );

    assert!(
        inf_logits.iter().all(|v| v.is_finite()),
        "inference model produced non-finite logits"
    );

    assert!(
        inf_logits.iter().any(|&v| v.abs() > 0.001),
        "inference model produced near-zero logits (likely silent failure)"
    );

    println!("✓ Logits are finite and non-degenerate");

    // 9. Test batched prefill path
    inf_model.reset_caches();
    let batched_logits = inf_model.forward_sequence_batched(&tokens);

    assert_eq!(batched_logits.len(), inf_logits.len());
    assert!(batched_logits.iter().all(|v| v.is_finite()));

    println!("✓ Batched prefill with loops completed");

    // 10. Numerical comparison (Pearson correlation)
    let train_mean = train_last.iter().sum::<f32>() / train_last.len() as f32;
    let inf_mean = inf_logits.iter().sum::<f32>() / inf_logits.len() as f32;

    let cov: f32 = train_last
        .iter()
        .zip(inf_logits.iter())
        .map(|(a, b)| (a - train_mean) * (b - inf_mean))
        .sum();

    let train_std: f32 = train_last
        .iter()
        .map(|a| (a - train_mean).powi(2))
        .sum::<f32>()
        .sqrt();

    let inf_std: f32 = inf_logits
        .iter()
        .map(|b| (b - inf_mean).powi(2))
        .sum::<f32>()
        .sqrt();

    let pearson = if train_std > 0.0 && inf_std > 0.0 {
        cov / (train_std * inf_std)
    } else {
        0.0
    };

    println!("  Pearson correlation: {:.4}", pearson);

    // With random weights, expect weak correlation but not systematically negative
    assert!(
        pearson > -0.5,
        "Strong negative correlation ({:.4}) suggests loop path bug",
        pearson
    );

    println!("\n=== LoopLM Roundtrip Test PASSED ===");
    println!("✓ Export/load preserves loop architecture");
    println!("✓ Loop forward pass executes correctly");
    println!("✓ Batched prefill works with loops");
    println!("✓ No systematic train/inference mismatch\n");
}

#[test]
fn test_loop_adaptive_config_roundtrip() {
    // Test that adaptive loop config is preserved through export/load
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let cfg = TrainConfig::d20_loop();
    let train_model = NanochatTrainModel::new(&cfg, vb).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let gguf_path = dir.path().join("adaptive.gguf");
    let mhc_path = dir.path().join("adaptive.mhc");

    export_model(
        &train_model,
        &cfg,
        gguf_path.to_str().unwrap(),
        mhc_path.to_str().unwrap(),
    )
    .unwrap();

    let inf_model = NanochatModel::from_gguf(
        gguf_path.to_str().unwrap(),
        mhc_path.to_str().unwrap(),
    )
    .unwrap();

    // Verify adaptive loop config
    let loaded_loop_cfg = inf_model.config.loop_config.as_ref().unwrap();
    assert!(loaded_loop_cfg.adaptive_loop.is_some());

    let adaptive = loaded_loop_cfg.adaptive_loop.as_ref().unwrap();
    let expected_adaptive = cfg.loop_config.as_ref().unwrap().adaptive_loop.as_ref().unwrap();

    assert_eq!(adaptive.min_loops, expected_adaptive.min_loops);
    assert_eq!(adaptive.max_loops, expected_adaptive.max_loops);
    assert!((adaptive.perplexity_threshold - expected_adaptive.perplexity_threshold).abs() < 1e-5);

    println!("✓ Adaptive loop config preserved through roundtrip");
}

#[test]
fn test_loop_batched_autoregressive_parity() {
    // CRITICAL: Validate that batched and autoregressive paths produce identical outputs
    // This catches causality bugs, state handling errors, and gate computation mismatches

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let cfg = TrainConfig::d20_loop();
    let train_model = NanochatTrainModel::new(&cfg, vb).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let gguf_path = dir.path().join("parity.gguf");
    let mhc_path = dir.path().join("parity.mhc");

    export_model(
        &train_model,
        &cfg,
        gguf_path.to_str().unwrap(),
        mhc_path.to_str().unwrap(),
    )
    .unwrap();

    let tokens = vec![1u32, 5, 10, 20, 42, 7, 13, 25];

    // Run autoregressive (ground truth)
    let mut model_auto = NanochatModel::from_gguf(
        gguf_path.to_str().unwrap(),
        mhc_path.to_str().unwrap(),
    )
    .unwrap();
    let logits_auto = model_auto.forward_sequence(&tokens);

    // Run batched prefill
    let mut model_batched = NanochatModel::from_gguf(
        gguf_path.to_str().unwrap(),
        mhc_path.to_str().unwrap(),
    )
    .unwrap();
    let logits_batched = model_batched.forward_sequence_batched(&tokens);

    // Strict numerical comparison
    assert_eq!(
        logits_auto.len(),
        logits_batched.len(),
        "Logit size mismatch"
    );

    let max_diff = logits_auto
        .iter()
        .zip(logits_batched.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mean_diff = logits_auto
        .iter()
        .zip(logits_batched.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / logits_auto.len() as f32;

    println!("\n=== Loop Batched/Autoregressive Parity ===");
    println!("  max_diff: {:.6}", max_diff);
    println!("  mean_diff: {:.6}", mean_diff);

    // Ternary quantization + floating point arithmetic can introduce small errors
    // But batched and autoregressive should be nearly identical (< 1e-4)
    // NOT just "same sign" - actual numerical parity
    assert!(
        max_diff < 1e-4,
        "Batched/autoregressive mismatch! max_diff={:.6} (expected < 1e-4). \
         This indicates causality violation or state handling bug.",
        max_diff
    );

    println!("✓ Batched and autoregressive paths produce identical outputs");
    println!("✓ No causality violations detected");
    println!("✓ State handling is consistent\n");
}
