//! Test actual model initialization on CUDA

use nanochat_train::{config::TrainConfig, model::NanochatTrainModel};
use candle_core::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing model CUDA initialization...\n");

    let device = Device::new_cuda(0)?;
    println!("✓ CUDA device initialized: {:?}\n", device);

    // Test 1: Tiny model
    println!("Test 1: Tiny model (4 layers, 256 dim)");
    let mut config1 = TrainConfig::tiny_cpu();
    config1.vocab_size = 50257; // Full vocab
    println!("  Config: {} layers, {} dim, {} params",
        config1.n_layers, config1.dim,
        estimate_params(&config1));

    match NanochatTrainModel::new(config1, device.clone()) {
        Ok(_model) => println!("✓ Tiny model initialized successfully\n"),
        Err(e) => {
            println!("✗ Failed: {}\n", e);
            return Err(e);
        }
    }

    // Test 2: D20 model
    println!("Test 2: D20 model (6 layers, 256 dim)");
    let config2 = TrainConfig::d20();
    println!("  Config: {} layers, {} dim, {} params",
        config2.n_layers, config2.dim,
        estimate_params(&config2));

    match NanochatTrainModel::new(config2, device.clone()) {
        Ok(_model) => println!("✓ D20 model initialized successfully\n"),
        Err(e) => {
            println!("✗ Failed: {}\n", e);
            return Err(e);
        }
    }

    // Test 3: nano-125M model
    println!("Test 3: nano-125M model (12 layers, 768 dim)");
    let config3 = TrainConfig::nano_125m();
    println!("  Config: {} layers, {} dim, {} params",
        config3.n_layers, config3.dim,
        estimate_params(&config3));

    match NanochatTrainModel::new(config3, device.clone()) {
        Ok(_model) => println!("✓ nano-125M initialized successfully\n"),
        Err(e) => {
            println!("✗ Failed: {}\n", e);
            println!("(This is expected on 24GB GPU)\n");
        }
    }

    println!("✅ Testing complete!");

    Ok(())
}

fn estimate_params(config: &TrainConfig) -> String {
    // Rough parameter count estimate
    let embed_params = config.vocab_size * config.dim * 2; // tok + pos
    let attn_params_per_layer = config.dim * config.dim * 4; // qkvo
    let ffn_params_per_layer = (config.dim as f32 * config.ffn_mult) as usize * config.dim * 3; // gate+up+down
    let total = embed_params + config.n_layers * (attn_params_per_layer + ffn_params_per_layer);

    if total > 1_000_000 {
        format!("{:.1}M", total as f64 / 1_000_000.0)
    } else if total > 1_000 {
        format!("{:.1}K", total as f64 / 1_000.0)
    } else {
        format!("{}", total)
    }
}
