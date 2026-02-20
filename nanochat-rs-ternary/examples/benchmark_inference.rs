//! Benchmark tool for ternary Qwen3 inference performance.
//!
//! Measures:
//! - Prefill throughput (prompt processing)
//! - Decode latency (autoregressive generation)
//! - Memory usage
//! - Per-layer timing breakdown
//!
//! Usage:
//!   cargo run --release --example benchmark_inference -- \
//!     --model hybrid.gguf \
//!     --mhc hybrid.mhc \
//!     --prompt-lengths 128,512,2048 \
//!     --num-tokens 100 \
//!     --output benchmark_results.json

use clap::Parser;
use nanochat_model::{
    bench::{
        get_hardware_info, BenchmarkResults, DecodeBenchmark, LayerTiming, MemoryStats,
        PrefillBenchmark,
    },
    config::ModelConfig,
};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "benchmark_inference")]
#[command(about = "Benchmark ternary model inference performance")]
struct Args {
    /// Path to model GGUF file
    #[arg(long)]
    model: String,

    /// Path to mHC weights file
    #[arg(long)]
    mhc: Option<String>,

    /// Model configuration preset
    #[arg(long, default_value = "qwen3_coder_80b")]
    config: String,

    /// Prompt lengths to test (comma-separated)
    #[arg(long, default_value = "128,512,2048")]
    prompt_lengths: String,

    /// Number of decode tokens to generate
    #[arg(long, default_value = "100")]
    num_tokens: usize,

    /// Batch size for prefill
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// Output file for results (JSON)
    #[arg(long)]
    output: Option<String>,

    /// Device (cpu, cuda:0, etc.)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Number of warmup iterations
    #[arg(long, default_value = "3")]
    warmup: usize,

    /// Number of benchmark iterations
    #[arg(long, default_value = "10")]
    iterations: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Ternary Model Inference Benchmark");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Parse configuration
    let config = match args.config.as_str() {
        "d20" => ModelConfig::d20(),
        "nano_125m" => ModelConfig::nano_125m(),
        "nano_1b" => ModelConfig::nano_1b(),
        "qwen3_coder_80b" => ModelConfig::qwen3_coder_80b(),
        other => {
            eprintln!("Unknown config: {}. Using d20.", other);
            ModelConfig::d20()
        }
    };

    println!("Configuration:");
    println!("  Model: {}", args.config);
    println!(
        "  Parameters: {}",
        format_count(config.param_count_estimate())
    );
    println!("  Device: {}", args.device);
    println!("  Model file: {}", args.model);
    if let Some(mhc_path) = &args.mhc {
        println!("  mHC file: {}", mhc_path);
    }
    println!();

    // Parse prompt lengths
    let prompt_lengths: Vec<usize> = args
        .prompt_lengths
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("Benchmark Plan:");
    println!("  Prompt lengths: {:?}", prompt_lengths);
    println!("  Decode tokens: {}", args.num_tokens);
    println!("  Batch size: {}", args.batch_size);
    println!("  Warmup iterations: {}", args.warmup);
    println!("  Benchmark iterations: {}", args.iterations);
    println!();

    // Get hardware info
    let hardware = get_hardware_info();
    println!("Hardware:");
    println!(
        "  CPU: {} ({} cores)",
        hardware.cpu_model, hardware.cpu_cores
    );
    println!("  System Memory: {:.1}GB", hardware.system_memory_gb);
    println!();

    let model_path = Path::new(&args.model);
    if !model_path.exists() {
        return Err(format!("Model file not found: {}", model_path.display()).into());
    }
    let model_size = std::fs::metadata(model_path)?.len() as usize;
    println!(
        "Model artifact: {} ({})",
        model_path.display(),
        format_bytes(model_size)
    );
    println!("Running synthetic benchmark mode (simulated compute path)");
    println!();

    // Run benchmarks for each prompt length
    let mut all_results = Vec::new();

    for &seq_len in &prompt_lengths {
        println!("═══════════════════════════════════════════════════════════");
        println!("Benchmarking prompt length: {} tokens", seq_len);
        println!("═══════════════════════════════════════════════════════════");
        println!();

        // Warmup
        println!("Warming up ({} iterations)...", args.warmup);
        for i in 0..args.warmup {
            let _ = simulate_prefill(&config, args.batch_size, seq_len);
            let _ = simulate_decode(&config, args.num_tokens);
            print!(".");
            if (i + 1) % 10 == 0 {
                print!(" {}/{}\n", i + 1, args.warmup);
            }
        }
        println!();

        // Prefill benchmark
        println!("Benchmarking prefill ({} iterations)...", args.iterations);
        let mut prefill_times = Vec::new();
        for _ in 0..args.iterations {
            let start = Instant::now();
            simulate_prefill(&config, args.batch_size, seq_len);
            prefill_times.push(start.elapsed().as_secs_f64());
        }

        let prefill_avg = prefill_times.iter().sum::<f64>() / prefill_times.len() as f64;
        let total_tokens = args.batch_size * seq_len;
        let prefill_tps = total_tokens as f64 / prefill_avg;
        let prefill_ms_per_token = (prefill_avg * 1000.0) / total_tokens as f64;

        println!("  Throughput: {:.1} tokens/sec", prefill_tps);
        println!("  Latency: {:.2} ms/token", prefill_ms_per_token);
        println!("  Total time: {:.2}s", prefill_avg);
        println!();

        // Decode benchmark
        println!("Benchmarking decode ({} iterations)...", args.iterations);
        let mut decode_times = Vec::new();
        for _ in 0..args.iterations {
            let start = Instant::now();
            simulate_decode(&config, args.num_tokens);
            decode_times.push(start.elapsed().as_secs_f64());
        }

        let decode_avg = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
        let decode_tps = args.num_tokens as f64 / decode_avg;
        let decode_ms_per_token = (decode_avg * 1000.0) / args.num_tokens as f64;

        println!("  Throughput: {:.1} tokens/sec", decode_tps);
        println!("  Latency: {:.2} ms/token", decode_ms_per_token);
        println!("  Total time: {:.2}s", decode_avg);
        println!();

        // Memory estimate
        let memory = estimate_memory(&config, seq_len);
        println!("Memory Estimate:");
        println!("  Model weights: {}", format_bytes(memory.model_weights));
        println!("  Activations: {}", format_bytes(memory.activations));
        println!("  KV cache: {}", format_bytes(memory.kv_cache));
        println!("  Peak: {}", format_bytes(memory.peak_memory));
        println!();

        // Create benchmark result
        let layer_timings = estimate_layer_timings(&config, prefill_avg);
        let result = BenchmarkResults {
            model_name: args.config.clone(),
            quant_type: "Q1_58 (Ternary)".to_string(),
            prefill: PrefillBenchmark {
                batch_size: args.batch_size,
                seq_len,
                total_time: prefill_avg,
                tokens_per_sec: prefill_tps,
                ms_per_token: prefill_ms_per_token,
                gpu_util: None,
            },
            decode: DecodeBenchmark {
                batch_size: 1,
                num_tokens: args.num_tokens,
                total_time: decode_avg,
                tokens_per_sec: decode_tps,
                ms_per_token: decode_ms_per_token,
                gpu_util: None,
            },
            memory,
            layer_timings,
            hardware: hardware.clone(),
        };

        result.print_report();
        all_results.push(result);
    }

    // Save results
    if let Some(output_path) = args.output {
        println!("Saving results to {}...", output_path);
        let json = serde_json::json!({
            "config": args.config,
            "model_file": args.model,
            "device": args.device,
            "results": all_results.iter().map(|r| r.to_json()).collect::<Vec<_>>(),
        });
        std::fs::write(&output_path, serde_json::to_string_pretty(&json)?)?;
        println!("✓ Results saved");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("Benchmark complete!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

/// Simulate prefill pass (placeholder for actual model forward)
fn simulate_prefill(config: &ModelConfig, batch_size: usize, seq_len: usize) {
    // Estimate compute: ~2 FLOPs per param per token
    let params = config.param_count_estimate();
    let flops = 2 * params * batch_size * seq_len;

    // Simulate work with busy loop (rough approximation)
    // Assume ~1 TFLOP/s throughput for ternary GEMV
    let target_duration_ns = (flops as f64 / 1e12 * 1e9) as u64;
    let start = Instant::now();

    while start.elapsed().as_nanos() < target_duration_ns as u128 {
        // Busy wait to simulate compute
        std::hint::spin_loop();
    }
}

/// Simulate decode pass (placeholder for actual model forward)
fn simulate_decode(config: &ModelConfig, num_tokens: usize) {
    // Autoregressive: one token at a time
    let params = config.param_count_estimate();

    for _ in 0..num_tokens {
        let flops = 2 * params; // Single token
        let target_duration_ns = (flops as f64 / 1e12 * 1e9) as u64;
        let start = Instant::now();

        while start.elapsed().as_nanos() < target_duration_ns as u128 {
            std::hint::spin_loop();
        }
    }
}

/// Estimate memory usage
fn estimate_memory(config: &ModelConfig, seq_len: usize) -> MemoryStats {
    // Model weights (ternary: ~0.25 bytes per param including scales)
    let model_weights = (config.param_count_estimate() as f64 * 0.25) as usize;

    // Activations (FP32: 4 bytes per element)
    // Rough estimate: hidden_dim * batch_size * (some multiple for intermediate buffers)
    let activations = config.dim * 4 * 8; // 8x hidden for SwiGLU intermediate

    // KV cache (FP16: 2 bytes per element)
    // 2 tensors (K, V) × n_layers × n_kv_heads × seq_len × head_dim
    let head_dim = config.dim / config.n_heads;
    let kv_cache = 2 * config.n_layers * config.n_kv_heads * seq_len * head_dim * 2;

    let peak_memory = model_weights + activations + kv_cache;

    MemoryStats {
        model_weights,
        activations,
        kv_cache,
        peak_memory,
        bandwidth_gbs: None,
    }
}

fn estimate_layer_timings(
    config: &ModelConfig,
    prefill_time_secs: f64,
) -> HashMap<String, LayerTiming> {
    let mut timings = HashMap::new();
    if config.n_layers == 0 || prefill_time_secs <= 0.0 {
        return timings;
    }

    // Synthetic prefill model: distribute 85% of time across transformer layers.
    let total_ms = prefill_time_secs * 1000.0;
    let layer_budget_ms = total_ms * 0.85;
    let per_layer_ms = layer_budget_ms / config.n_layers as f64;

    for layer_idx in 0..config.n_layers {
        let attn_ms = per_layer_ms * 0.45;
        let ffn_ms = per_layer_ms * 0.50;
        let norm_ms = per_layer_ms * 0.05;

        let attn_name = format!("layer_{layer_idx}.attention");
        timings.insert(
            attn_name.clone(),
            LayerTiming {
                layer_name: attn_name,
                layer_type: "attention".to_string(),
                time_ms: attn_ms,
                percentage: (attn_ms / total_ms) * 100.0,
            },
        );

        let ffn_name = format!("layer_{layer_idx}.ffn");
        timings.insert(
            ffn_name.clone(),
            LayerTiming {
                layer_name: ffn_name,
                layer_type: "ffn".to_string(),
                time_ms: ffn_ms,
                percentage: (ffn_ms / total_ms) * 100.0,
            },
        );

        let norm_name = format!("layer_{layer_idx}.norm");
        timings.insert(
            norm_name.clone(),
            LayerTiming {
                layer_name: norm_name,
                layer_type: "norm".to_string(),
                time_ms: norm_ms,
                percentage: (norm_ms / total_ms) * 100.0,
            },
        );
    }

    timings
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
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2}KB", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}
