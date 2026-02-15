//! Integration test for E2 optimizer enhancements

use candle_core::{DType, Device};
use candle_nn::VarMap;
use nanochat_train::config::TrainConfig;
use nanochat_train::model::NanochatTrainModel;
use nanochat_train::optim::MuonOptimizer;

#[test]
fn test_8bit_optimizer_initialization() {
    let device = Device::Cpu;
    let mut config = TrainConfig::tiny_cpu();

    // Enable 8-bit optimizer
    config.use_8bit_optim = true;
    config.use_galore = false;

    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = NanochatTrainModel::new(&config, vb).expect("Failed to create model");

    // Get vars and create optimizer
    let all_vars = varmap.all_vars();
    let mut muon_vars = Vec::new();

    for var in all_vars {
        let dims = var.as_tensor().dims();
        if dims.len() > 1 && dims[0] != config.vocab_size {
            muon_vars.push(var);
        }
    }

    let opt = MuonOptimizer::from_config(
        muon_vars,
        config.lr,
        config.muon_momentum,
        config.ns_steps,
        config.weight_decay,
        config.use_8bit_optim,
        config.use_galore,
        config.galore_rank,
        config.galore_update_freq,
    )
    .expect("Failed to create optimizer");

    // Check that we got the quantized variant
    let stats = opt.memory_stats();
    assert_eq!(stats.variant, "8-bit Quantized Muon");
    assert!(
        stats.memory_reduction > 0.7,
        "Expected >70% memory reduction"
    );

    println!("✅ 8-bit optimizer initialized successfully");
    println!("   Variant: {}", stats.variant);
    println!(
        "   Memory reduction: {:.1}%",
        stats.memory_reduction * 100.0
    );
}

#[test]
fn test_galore_optimizer_initialization() {
    let device = Device::Cpu;
    let mut config = TrainConfig::d20(); // Larger model for GaLore (dim=256)

    // Enable GaLore
    config.use_8bit_optim = false;
    config.use_galore = true;
    config.galore_rank = 64; // Smaller rank for test

    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = NanochatTrainModel::new(&config, vb).expect("Failed to create model");

    // Get vars and create optimizer
    let all_vars = varmap.all_vars();
    let mut muon_vars = Vec::new();

    for var in all_vars {
        let dims = var.as_tensor().dims();
        if dims.len() > 1 && dims[0] != config.vocab_size {
            muon_vars.push(var);
        }
    }

    let opt = MuonOptimizer::from_config(
        muon_vars,
        config.lr,
        config.muon_momentum,
        config.ns_steps,
        config.weight_decay,
        config.use_8bit_optim,
        config.use_galore,
        config.galore_rank,
        config.galore_update_freq,
    )
    .expect("Failed to create optimizer");

    // Check that we got the GaLore variant
    let stats = opt.memory_stats();
    assert_eq!(stats.variant, "GaLore2 Muon");
    // Note: Memory reduction may be 0 if no matrices are large enough (min_dim=256)
    // With d20 config (dim=256), we should get some projection

    println!("✅ GaLore optimizer initialized successfully");
    println!("   Variant: {}", stats.variant);
    println!(
        "   Memory reduction: {:.1}%",
        stats.memory_reduction * 100.0
    );
    println!("   Details: {}", stats.details);
}

#[test]
fn test_baseline_optimizer_initialization() {
    let device = Device::Cpu;
    let config = TrainConfig::tiny_cpu();
    // Both flags should be false by default
    assert!(!config.use_8bit_optim);
    assert!(!config.use_galore);

    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = NanochatTrainModel::new(&config, vb).expect("Failed to create model");

    // Get vars and create optimizer
    let all_vars = varmap.all_vars();
    let mut muon_vars = Vec::new();

    for var in all_vars {
        let dims = var.as_tensor().dims();
        if dims.len() > 1 && dims[0] != config.vocab_size {
            muon_vars.push(var);
        }
    }

    let opt = MuonOptimizer::from_config(
        muon_vars,
        config.lr,
        config.muon_momentum,
        config.ns_steps,
        config.weight_decay,
        config.use_8bit_optim,
        config.use_galore,
        config.galore_rank,
        config.galore_update_freq,
    )
    .expect("Failed to create optimizer");

    // Check that we got the standard variant
    let stats = opt.memory_stats();
    assert_eq!(stats.variant, "Standard Muon");
    assert_eq!(
        stats.memory_reduction, 0.0,
        "Baseline should have no reduction"
    );

    println!("✅ Baseline optimizer initialized successfully");
    println!("   Variant: {}", stats.variant);
}
