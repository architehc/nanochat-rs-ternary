//! Test Trainer initialization on CUDA with different model sizes

use nanochat_train::{config::TrainConfig, train::Trainer};
use candle_core::Device;

fn main() {
    println!("Testing Trainer CUDA initialization...\n");

    let device = match Device::new_cuda(0) {
        Ok(d) => {
            println!("✓ CUDA device initialized: {:?}\n", d);
            d
        }
        Err(e) => {
            println!("✗ CUDA initialization failed: {}\n", e);
            return;
        }
    };

    // Test 1: Tiny model
    println!("Test 1: Tiny model (4 layers, 256 dim, vocab=50257)");
    let mut config1 = TrainConfig::tiny_cpu();
    config1.vocab_size = 50257; // Full vocab
    println!("  Estimated params: ~{}M", estimate_params(&config1));

    match Trainer::new(config1, device.clone()) {
        Ok(_trainer) => println!("✓ Tiny model trainer initialized successfully\n"),
        Err(e) => {
            println!("✗ Failed: {}\n", e);
            return;
        }
    }

    // Test 2: D20 model
    println!("Test 2: D20 model (6 layers, 256 dim, vocab=50257)");
    let config2 = TrainConfig::d20();
    println!("  Estimated params: ~{}M", estimate_params(&config2));

    match Trainer::new(config2, device.clone()) {
        Ok(_trainer) => println!("✓ D20 model trainer initialized successfully\n"),
        Err(e) => {
            println!("✗ Failed: {}\n", e);
            return;
        }
    }

    // Test 3: nano-125M model
    println!("Test 3: nano-125M model (12 layers, 768 dim, vocab=50257)");
    let config3 = TrainConfig::nano_125m();
    println!("  Estimated params: ~{}M", estimate_params(&config3));

    match Trainer::new(config3, device.clone()) {
        Ok(_trainer) => println!("✓ nano-125M trainer initialized successfully\n"),
        Err(e) => {
            println!("✗ Failed: {}\n", e);
            println!("(This might be expected on 24GB GPU)\n");
        }
    }

    println!("✅ Testing complete!");
}

fn estimate_params(config: &TrainConfig) -> String {
    // Rough parameter count estimate
    let embed_params = config.vocab_size * config.dim * 2; // tok + pos
    let attn_params_per_layer = config.dim * config.dim * 4; // qkvo
    let ffn_params_per_layer = (config.dim as f32 * config.ffn_mult) as usize * config.dim * 3; // gate+up+down
    let total = embed_params + config.n_layers * (attn_params_per_layer + ffn_params_per_layer);

    format!("{:.1}", total as f64 / 1_000_000.0)
}
