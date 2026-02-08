//! Sensitivity analysis tool for layer-wise quantization.
//!
//! This tool analyzes which layers in a model are most sensitive to
//! ternarization, helping optimize hybrid quantization strategies.
//!
//! Usage:
//!   cargo run --release --example sensitivity_analysis -- \
//!     --config d20 \
//!     --checkpoint path/to/checkpoint \
//!     --max-batches 100 \
//!     --threshold 0.05
//!
//! Output: Ranked list of layers by sensitivity with recommendations.

use candle_core::Device;
use nanochat_train::{
    config::TrainConfig,
    data::{Dataset, SyntheticDataset},
    sensitivity::SensitivityAnalyzer,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "sensitivity_analysis")]
#[command(about = "Analyze layer-wise quantization sensitivity")]
struct Args {
    /// Model configuration preset
    #[arg(long, default_value = "d20")]
    config: String,

    /// Path to model checkpoint (optional - will use random init if not provided)
    #[arg(long)]
    checkpoint: Option<String>,

    /// Maximum batches to evaluate per layer (trade speed vs accuracy)
    #[arg(long, default_value = "50")]
    max_batches: usize,

    /// Sensitivity threshold for recommendations (e.g., 0.05 = 5% loss increase)
    #[arg(long, default_value = "0.05")]
    threshold: f64,

    /// Output file for detailed report (JSON)
    #[arg(long)]
    output: Option<String>,

    /// Device (cpu, cuda:0, etc.)
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Layer-wise Quantization Sensitivity Analysis");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Parse device
    let device = if args.device.starts_with("cuda") {
        #[cfg(feature = "cuda")]
        {
            let gpu_id = args.device.strip_prefix("cuda:")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            Device::new_cuda(gpu_id)
                .map_err(|e| format!("Failed to create CUDA device: {}", e))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("CUDA not available, using CPU");
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    // Load config
    let config = match args.config.as_str() {
        "d20" => TrainConfig::d20(),
        "nano_125m" => TrainConfig::nano_125m(),
        "nano_1b" => TrainConfig::nano_1b(),
        "tiny_cpu" => TrainConfig::tiny_cpu(),
        other => {
            eprintln!("Unknown config: {}. Using d20.", other);
            TrainConfig::d20()
        }
    };

    println!("Configuration:");
    println!("  Model: {} ({} params)", args.config, format_count(config.param_count_estimate()));
    println!("  Device: {:?}", device);
    println!("  Max batches/layer: {}", args.max_batches);
    println!("  Sensitivity threshold: {:.1}%", args.threshold * 100.0);
    if let Some(ckpt) = &args.checkpoint {
        println!("  Checkpoint: {}", ckpt);
    } else {
        println!("  Checkpoint: None (using random initialization)");
    }
    println!();

    // Create analyzer
    println!("Initializing analyzer...");
    let mut analyzer = SensitivityAnalyzer::new(config.clone(), device)?;

    // Load checkpoint if provided
    if let Some(checkpoint_path) = &args.checkpoint {
        println!("Loading checkpoint from {}...", checkpoint_path);
        analyzer.load_checkpoint(checkpoint_path)?;
    } else {
        println!("Using random initialization (for demonstration)");
    }
    println!();

    // Create validation dataset
    println!("Creating validation dataset...");
    let dataset = SyntheticDataset::new(
        config.vocab_size as u32,
        config.max_seq_len,
        args.max_batches * config.batch_size,
        42,
    );
    println!("  Dataset: {} samples", dataset.len());
    println!();

    // Run analysis
    println!("═══════════════════════════════════════════════════════════");
    println!();
    let report = analyzer.analyze(&dataset, args.max_batches)?;
    println!();

    // Print summary
    report.print_summary(args.threshold);

    // Save detailed report if requested
    if let Some(output_path) = args.output {
        println!("Saving detailed report to {}...", output_path);
        save_report_json(&report, &output_path)?;
        println!("✓ Report saved");
        println!();
    }

    // Print actionable recommendations
    println!("Actionable Recommendations:");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let (ternarize, keep_fp8) = report.recommend_ternarization(args.threshold);
    let (savings_bytes, savings_pct) = report.compute_savings(args.threshold);

    println!("Hybrid Quantization Strategy:");
    println!("  1. Ternarize {} layers ({}% of params)", ternarize.len(), savings_pct as usize);
    println!("     → Memory savings: {}", format_bytes(savings_bytes));
    println!("     → Expected quality impact: < {:.1}%", args.threshold * 100.0);
    println!();
    println!("  2. Keep {} layers in FP8/BF16", keep_fp8.len());
    println!("     → These are most sensitive to quantization");
    println!();

    println!("Next Steps:");
    println!("  1. Use qwen3_converter with selective ternarization:");
    println!("     cargo run --bin qwen3_converter -- \\");
    println!("       --input checkpoint.safetensors \\");
    println!("       --output hybrid.gguf \\");
    println!("       --ternarize-list ternarize_layers.txt");
    println!();
    println!("  2. Train with distillation on hybrid model:");
    println!("     cargo run --example distill_qwen3 -- \\");
    println!("       --teacher-endpoint <fp8-endpoint> \\");
    println!("       --checkpoint-dir checkpoints/hybrid");
    println!();
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

/// Save report as JSON.
fn save_report_json(report: &nanochat_train::sensitivity::SensitivityReport, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let json = serde_json::json!({
        "baseline_loss": report.baseline_loss,
        "total_params": report.total_params,
        "layers": report.layers.iter().map(|l| {
            serde_json::json!({
                "name": l.layer_name,
                "type": format!("{:?}", l.layer_type),
                "baseline_loss": l.baseline_loss,
                "ternary_loss": l.ternary_loss,
                "sensitivity": l.sensitivity,
                "num_params": l.num_params,
                "memory_savings": l.memory_savings,
            })
        }).collect::<Vec<_>>(),
        "type_stats": report.type_stats.iter().map(|(k, v)| {
            (format!("{:?}", k), serde_json::json!({
                "count": v.count,
                "avg_sensitivity": v.avg_sensitivity,
                "max_sensitivity": v.max_sensitivity,
                "min_sensitivity": v.min_sensitivity,
                "total_params": v.total_params,
                "total_memory_savings": v.total_memory_savings,
            }))
        }).collect::<serde_json::Map<String, serde_json::Value>>(),
    });

    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json)?.as_bytes())?;

    Ok(())
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2}GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2}MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.2}KB", bytes as f64 / 1_024.0)
    } else {
        format!("{}B", bytes)
    }
}
