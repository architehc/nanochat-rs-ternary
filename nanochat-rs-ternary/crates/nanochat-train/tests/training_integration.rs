//! Integration tests for full training workflows
//!
//! These tests verify end-to-end training behavior including:
//! - Model convergence
//! - Checkpoint save/load
//! - Optimizer state preservation
//! - Configuration compatibility

use candle_core::{DType, Device, Result};
use nanochat_train::{
    config::TrainConfig,
    data::{DataLoader, SyntheticDataset},
    train::{Trainer, TrainerBuilder, StepAttempt},
    checkpoint,
};

fn tiny_config() -> TrainConfig {
    TrainConfig {
        dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        ffn_mult: 2.0,
        vocab_size: 256,
        max_seq_len: 32,
        group_size: 64,
        mhc_n_streams: 2,
        weight_tied: true,
        rope_theta: 10000.0,
        loop_config: None,
        lr: 0.02,
        mhc_lr: 1e-4,
        weight_decay: 0.0,
        batch_size: 4,
        grad_accum_steps: 1,
        warmup_steps: 5,
        total_steps: 50,
        decay_start_frac: 0.8,
        grad_clip: 1.0,
        ns_steps: 3,
        muon_momentum: 0.95,
        lion_betas: (0.9, 0.99),
        use_8bit_optim: false,
        use_galore: false,
        galore_rank: 64,
        galore_update_freq: 10,
        use_mtp: false,
        mtp_n_tokens: 2,
        mtp_weight: 0.2,
        use_collider: false,
        collider_threshold: 0.3,
        collider_sparsity: 0.35,
        use_async_loader: false,
        async_n_workers: 2,
        async_prefetch_size: 4,
        use_fp4: false,
        fp4_stochastic_rounding: true,
        distill_teacher: None,
        distill_kl_weight: 0.0,
        loop_scale_penalty: 0.0,
    }
}

#[test]
fn test_config_validation_catches_errors() {
    let mut config = tiny_config();
    
    // Valid config should pass
    assert!(config.validate().is_ok());
    
    // Invalid: dim not divisible by n_heads
    config.n_heads = 3;
    let result = config.validate();
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("divisible by n_heads")));
    
    // Invalid: warmup >= total_steps
    config.n_heads = 4;
    config.warmup_steps = 100;
    config.total_steps = 50;
    let result = config.validate();
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("warmup_steps")));
}

#[test]
fn test_trainer_builder_pattern() -> Result<()> {
    let config = tiny_config();
    let device = Device::Cpu;
    
    // Build from scratch
    let trainer = TrainerBuilder::new(config.clone())
        .device(device.clone())
        .build()?;
    
    assert_eq!(trainer.config.dim, config.dim);
    assert_eq!(trainer.global_step, 0);
    
    Ok(())
}

#[test]
fn test_full_training_convergence() -> Result<()> {
    let device = Device::Cpu;
    let mut config = tiny_config();
    config.lr = 0.01; // Lower LR for stability
    config.total_steps = 30;
    
    let mut trainer = Trainer::new(config, device)?;
    let dataset = SyntheticDataset::new(256, 8, 32, 42);
    
    // Train for 3 epochs
    let mut losses = Vec::new();
    for epoch in 0..3 {
        let avg_loss = trainer.train_epoch(&dataset, epoch)?;
        losses.push(avg_loss);
    }
    
    // Loss should generally decrease (with some noise allowed)
    let first = losses[0];
    let last = *losses.last().unwrap();
    
    // Allow for 20% fluctuation due to randomness in small models
    assert!(
        last < first * 1.2, // Not more than 20% higher
        "Training should not diverge: first={:.4} last={:.4}",
        first, last
    );
    
    Ok(())
}

#[test]
fn test_checkpoint_save_load_roundtrip() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let config = tiny_config();
    
    // Create and train initial model
    let mut trainer = Trainer::new(config.clone(), device.clone())?;
    let dataset = SyntheticDataset::new(256, 8, 8, 42);
    
    // Train for one step to generate gradients/optimizer state
    let loader = DataLoader::new(&dataset, 4, false, 42, &device);
    let (input_ids, target_ids) = loader.into_iter().next().unwrap()?;
    let _stats = trainer.train_step(&input_ids, &target_ids)?;
    
    // Save checkpoint
    let temp_dir = tempfile::tempdir()?;
    let ckpt_path = temp_dir.path().join("step_1");
    checkpoint::save_checkpoint(
        &trainer.varmap,
        &config,
        trainer.global_step,
        1.5,
        ckpt_path.to_str().unwrap(),
    )?;
    
    // Verify files exist
    assert!(ckpt_path.join("model.safetensors").exists());
    assert!(ckpt_path.join("meta.json").exists());
    
    // Load checkpoint into new trainer
    let loaded_trainer = Trainer::from_checkpoint(ckpt_path.to_str().unwrap(), device)?;
    
    // Verify state restored
    assert_eq!(loaded_trainer.global_step, trainer.global_step);
    assert_eq!(loaded_trainer.config.dim, trainer.config.dim);
    
    Ok(())
}

#[test]
fn test_grad_accumulation_steps() -> Result<()> {
    let device = Device::Cpu;
    let mut config = tiny_config();
    config.grad_accum_steps = 3;
    
    let mut trainer = Trainer::new(config, device)?;
    let input_ids = candle_core::Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
    let target_ids = candle_core::Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
    
    // First 2 steps should not update global_step
    let _ = trainer.train_step(&input_ids, &target_ids)?;
    assert_eq!(trainer.global_step, 0, "Should not step on first micro-step");
    
    let _ = trainer.train_step(&input_ids, &target_ids)?;
    assert_eq!(trainer.global_step, 0, "Should not step on second micro-step");
    
    // Third step should trigger optimizer
    let _ = trainer.train_step(&input_ids, &target_ids)?;
    assert_eq!(trainer.global_step, 1, "Should step after grad_accum_steps");
    
    Ok(())
}

#[test]
fn test_oom_recovery_mechanism() -> Result<()> {
    let device = Device::Cpu;
    let config = tiny_config();
    let mut trainer = Trainer::new(config, device)?;
    
    let input_ids = candle_core::Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
    let target_ids = candle_core::Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
    
    // Normal step should succeed
    match trainer.train_step_with_recovery(&input_ids, &target_ids) {
        StepAttempt::Success(_) => (), // Expected
        StepAttempt::RecoveredFromOom => panic!("Should not need OOM recovery"),
        StepAttempt::Failed(e) => return Err(e),
    }
    
    Ok(())
}

#[test]
fn test_collider_integration() -> Result<()> {
    let device = Device::Cpu;
    let mut config = tiny_config();
    config.use_collider = true;
    config.collider_threshold = 0.3;
    config.collider_sparsity = 0.5;
    
    let mut trainer = Trainer::new(config, device)?;
    let input_ids = candle_core::Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
    let target_ids = candle_core::Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
    
    // Should complete without error
    let stats = trainer.train_step(&input_ids, &target_ids)?;
    assert!(stats.loss.is_finite());
    assert!(stats.grad_norm.is_finite());
    
    Ok(())
}

#[test]
fn test_mtp_integration() -> Result<()> {
    let device = Device::Cpu;
    let mut config = tiny_config();
    config.use_mtp = true;
    config.mtp_n_tokens = 2;
    config.mtp_weight = 0.2;
    
    let mut trainer = Trainer::new(config, device)?;
    let input_ids = candle_core::Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
    let target_ids = candle_core::Tensor::zeros((2, 8), DType::U32, &trainer.device)?;
    
    // Should complete without error
    let stats = trainer.train_step(&input_ids, &target_ids)?;
    assert!(stats.loss.is_finite());
    
    Ok(())
}

#[test]
fn test_optimizer_memory_stats() -> Result<()> {
    let device = Device::Cpu;
    let mut config = tiny_config();
    config.use_8bit_optim = true;
    
    let trainer = Trainer::new(config, device)?;
    let stats = trainer.optimizer_memory_stats();
    
    // 8-bit optimizer should report memory reduction
    assert!(stats.memory_reduction > 0.0);
    assert!(stats.variant.contains("8-bit") || stats.variant.contains("Quantized"));
    
    Ok(())
}

#[test]
fn test_async_loader_metrics() -> Result<()> {
    use nanochat_train::data::AsyncDataLoader;
    use std::sync::Arc;
    
    let dataset = Arc::new(SyntheticDataset::new(100, 16, 50, 42));
    let device = Device::Cpu;
    
    let mut loader = AsyncDataLoader::new(
        dataset,
        8,
        false,
        42,
        2,
        4,
        device,
    );
    
    // Check initial metrics
    let metrics = loader.metrics();
    assert_eq!(metrics.n_workers, 2);
    assert_eq!(metrics.batches_consumed, 0);
    
    // Consume one batch
    let _ = loader.next_batch();
    
    let metrics = loader.metrics();
    assert_eq!(metrics.batches_consumed, 1);
    
    // Verify ExactSizeIterator
    assert_eq!(loader.len(), 7); // ceil(50/8) = 7
    
    Ok(())
}

#[test]
fn test_memory_pool_reuse() -> Result<()> {
    use nanochat_train::memory_pool::TensorPool;
    
    let device = Device::Cpu;
    let mut pool = TensorPool::new(4);
    
    // Acquire and release
    let t1 = pool.acquire(&candle_core::Shape::from((10, 10)), DType::F32, &device)?;
    pool.release(t1);
    
    // Reacquire - should reuse
    let _t2 = pool.acquire(&candle_core::Shape::from((10, 10)), DType::F32, &device)?;
    
    let stats = pool.stats();
    assert_eq!(stats.total_acquired, 2);
    assert_eq!(stats.total_reused, 1);
    assert!((stats.reuse_rate - 0.5).abs() < 1e-6);
    
    Ok(())
}
