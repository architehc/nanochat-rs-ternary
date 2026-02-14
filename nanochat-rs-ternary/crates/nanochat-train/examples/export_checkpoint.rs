//! Export trained checkpoint to GGUF + mHC binary format.
//!
//! Usage:
//!   cargo run --release --example export_checkpoint -- \
//!     --checkpoint checkpoints/tiny-cpu-demo/final \
//!     --output models/tiny-cpu.gguf

use candle_core::{DType, Device};
use clap::Parser;
use nanochat_train::{checkpoint, export, model::NanochatTrainModel};

#[derive(Parser)]
#[command(name = "export_checkpoint")]
#[command(about = "Export trained checkpoint to GGUF + mHC format")]
struct Args {
    /// Checkpoint directory to load
    #[arg(long)]
    checkpoint: String,

    /// Output GGUF file path
    #[arg(long)]
    output: String,

    /// Optional separate mHC file path (default: <output>.mhc)
    #[arg(long)]
    mhc_output: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Exporting Checkpoint to GGUF Format");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Checkpoint: {}", args.checkpoint);
    println!("Output: {}", args.output);
    println!();

    // Load checkpoint
    println!("Loading checkpoint...");
    let device = Device::Cpu;
    let (varmap, config, step, loss) = checkpoint::load_checkpoint(&args.checkpoint, &device)
        .map_err(|e| format!("Failed to load checkpoint: {}", e))?;

    println!("✓ Checkpoint loaded");
    println!("  Step: {}", step);
    println!("  Loss: {:.4}", loss);
    println!(
        "  Model: {}D, {} layers, {} heads",
        config.dim, config.n_layers, config.n_heads
    );
    println!();

    // Create model from checkpoint
    println!("Creating model...");
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = NanochatTrainModel::new(&config, vb)
        .map_err(|e| format!("Failed to create model: {}", e))?;

    println!("✓ Model created");
    println!(
        "  Parameters: {}",
        format_count(config.param_count_estimate())
    );
    println!();

    // Determine mHC output path
    let mhc_path = args
        .mhc_output
        .unwrap_or_else(|| format!("{}.mhc", args.output.trim_end_matches(".gguf")));

    // Export to GGUF + mHC
    println!("Exporting to GGUF...");
    export::export_gguf(&model, &config, &args.output)
        .map_err(|e| format!("Failed to export GGUF: {}", e))?;

    println!("✓ GGUF exported to: {}", args.output);

    println!("Exporting mHC parameters...");
    export::export_mhc(&model, &config, &mhc_path)
        .map_err(|e| format!("Failed to export mHC: {}", e))?;

    println!("✓ mHC exported to: {}", mhc_path);
    println!();

    // Show file sizes
    if let Ok(gguf_size) = std::fs::metadata(&args.output).map(|m| m.len()) {
        println!("GGUF file size: {}", format_bytes(gguf_size));
    }
    if let Ok(mhc_size) = std::fs::metadata(&mhc_path).map(|m| m.len()) {
        println!("mHC file size: {}", format_bytes(mhc_size));
    }
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("Export complete!");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Next steps:");
    println!("1. Start inference server:");
    println!("   cargo run --release --bin nanochat-serve -- \\");
    println!("     --model {} \\", args.output);
    println!("     --mhc {} \\", mhc_path);
    println!("     --port 8080");
    println!();
    println!("2. Test generation:");
    println!("   curl http://localhost:8080/v1/completions \\");
    println!("     -H 'Content-Type: application/json' \\");
    println!("     -d '{{\"prompt\": \"def hello():\", \"max_tokens\": 50}}'");

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

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}
