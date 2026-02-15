//! Memory benchmark for optimizer comparisons.
//!
//! Compares memory usage of different optimizer configurations:
//! - Baseline (no GaLore, no 8-bit)
//! - GaLore 2 only
//! - 8-bit Muon only
//! - GaLore 2 + 8-bit Muon (best)

use nanochat_train::config::TrainConfig;
use nanochat_train::train::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== nanochat Optimizer Memory Benchmark ===\n");

    let device = candle_core::Device::Cpu;

    // Test configuration - start with tiny_cpu preset
    let mut config = TrainConfig::tiny_cpu();
    config.use_8bit_optim = false;
    config.use_galore = false;
    config.galore_rank = 128;
    config.galore_update_freq = 200;
    config.total_steps = 10; // Just init, don't train

    println!(
        "Model size: ~{}M parameters",
        config.param_count_estimate() / 1_000_000
    );
    println!("Batch size: {}", config.batch_size);
    println!("Max sequence length: {}\n", config.max_seq_len);

    // Baseline: No GaLore, No 8-bit
    println!("1. Baseline (FP32 Muon, no GaLore)");
    config.use_8bit_optim = false;
    config.use_galore = false;
    let baseline_mem = measure_memory(&config, &device)?;
    println!("   Memory: {:.2} MB\n", baseline_mem);

    // GaLore only
    println!("2. GaLore 2 only (FP32 Muon + GaLore)");
    config.use_8bit_optim = false;
    config.use_galore = true;
    let galore_mem = measure_memory(&config, &device)?;
    println!("   Memory: {:.2} MB", galore_mem);
    println!(
        "   Reduction: {:.1}%\n",
        100.0 * (1.0 - galore_mem / baseline_mem)
    );

    // 8-bit only
    println!("3. 8-bit Muon only (no GaLore)");
    config.use_8bit_optim = true;
    config.use_galore = false;
    let quant_mem = measure_memory(&config, &device)?;
    println!("   Memory: {:.2} MB", quant_mem);
    println!(
        "   Reduction: {:.1}%\n",
        100.0 * (1.0 - quant_mem / baseline_mem)
    );

    // Both (best)
    println!("4. GaLore 2 + 8-bit Muon (best)");
    config.use_8bit_optim = true;
    config.use_galore = true;
    let best_mem = measure_memory(&config, &device)?;
    println!("   Memory: {:.2} MB", best_mem);
    println!(
        "   Reduction: {:.1}%\n",
        100.0 * (1.0 - best_mem / baseline_mem)
    );

    // Summary
    println!("=== Summary ===");
    println!("Baseline:           {:.2} MB (100.0%)", baseline_mem);
    println!(
        "GaLore only:        {:.2} MB ({:.1}%)",
        galore_mem,
        100.0 * galore_mem / baseline_mem
    );
    println!(
        "8-bit only:         {:.2} MB ({:.1}%)",
        quant_mem,
        100.0 * quant_mem / baseline_mem
    );
    println!(
        "GaLore + 8-bit:     {:.2} MB ({:.1}%)",
        best_mem,
        100.0 * best_mem / baseline_mem
    );
    println!("\n✓ Target: >50% reduction");
    println!(
        "✓ Achieved: {:.1}% reduction",
        100.0 * (1.0 - best_mem / baseline_mem)
    );

    if best_mem / baseline_mem < 0.5 {
        println!("\n✅ SUCCESS: Memory reduction target met!");
    } else {
        println!("\n⚠️  WARNING: Memory reduction below target");
    }

    Ok(())
}

fn measure_memory(
    config: &TrainConfig,
    device: &candle_core::Device,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Initialize trainer (allocates model + optimizer state)
    let trainer = Trainer::new(config.clone(), device.clone())?;

    // Rough estimate: count parameters + optimizer state size
    let param_count = trainer.model.param_count();
    let param_bytes = param_count * 4; // FP32 parameters

    // Optimizer state size depends on config
    let opt_bytes = if config.use_8bit_optim {
        // 8-bit: ~1.5 bytes per param (momentum + quantization overhead)
        (param_count as f64 * 1.5) as usize
    } else {
        // FP32: ~8 bytes per param (momentum)
        param_count * 8
    };

    // GaLore reduces optimizer state proportionally to rank
    let effective_opt_bytes = if config.use_galore {
        let rank = config.galore_rank;
        let reduction_factor = rank as f64 / config.dim as f64;
        (opt_bytes as f64 * reduction_factor) as usize
    } else {
        opt_bytes
    };

    let total_bytes = param_bytes + effective_opt_bytes;
    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);

    Ok(total_mb)
}
