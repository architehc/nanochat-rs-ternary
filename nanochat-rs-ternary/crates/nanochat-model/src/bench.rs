//! Benchmarking utilities for model inference.
//!
//! Measures:
//! - Throughput (tokens/sec)
//! - Latency (ms/token)
//! - Memory usage
//! - Layer-wise breakdown
//! - GEMV kernel performance

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark results for model inference.
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Model name/identifier
    pub model_name: String,

    /// Quantization type (FP32, FP16, FP8, Ternary, etc.)
    pub quant_type: String,

    /// Prefill benchmarks (prompt processing)
    pub prefill: PrefillBenchmark,

    /// Decode benchmarks (autoregressive generation)
    pub decode: DecodeBenchmark,

    /// Memory usage statistics
    pub memory: MemoryStats,

    /// Per-layer timing breakdown
    pub layer_timings: HashMap<String, LayerTiming>,

    /// Hardware info
    pub hardware: HardwareInfo,
}

/// Prefill (prompt processing) benchmark results.
#[derive(Debug, Clone)]
pub struct PrefillBenchmark {
    /// Batch size tested
    pub batch_size: usize,

    /// Prompt length (tokens)
    pub seq_len: usize,

    /// Total time (seconds)
    pub total_time: f64,

    /// Tokens per second
    pub tokens_per_sec: f64,

    /// Time per token (milliseconds)
    pub ms_per_token: f64,

    /// GPU utilization (0-100%)
    pub gpu_util: Option<f64>,
}

/// Decode (autoregressive) benchmark results.
#[derive(Debug, Clone)]
pub struct DecodeBenchmark {
    /// Batch size tested
    pub batch_size: usize,

    /// Number of tokens generated
    pub num_tokens: usize,

    /// Total time (seconds)
    pub total_time: f64,

    /// Tokens per second
    pub tokens_per_sec: f64,

    /// Time per token (milliseconds)
    pub ms_per_token: f64,

    /// GPU utilization (0-100%)
    pub gpu_util: Option<f64>,
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Model weights (bytes)
    pub model_weights: usize,

    /// Activations (bytes, single batch)
    pub activations: usize,

    /// KV cache (bytes, full context)
    pub kv_cache: usize,

    /// Peak memory usage (bytes)
    pub peak_memory: usize,

    /// Memory bandwidth utilization (GB/s)
    pub bandwidth_gbs: Option<f64>,
}

/// Per-layer timing information.
#[derive(Debug, Clone)]
pub struct LayerTiming {
    pub layer_name: String,
    pub layer_type: String,
    pub time_ms: f64,
    pub percentage: f64,
}

/// Hardware information.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub gpu_model: Option<String>,
    pub gpu_memory_gb: Option<f64>,
    pub system_memory_gb: f64,
}

impl BenchmarkResults {
    /// Print formatted benchmark report.
    pub fn print_report(&self) {
        println!("═══════════════════════════════════════════════════════════");
        println!("  Inference Benchmark Results");
        println!("═══════════════════════════════════════════════════════════");
        println!();
        println!("Model: {}", self.model_name);
        println!("Quantization: {}", self.quant_type);
        println!();

        // Hardware
        println!("Hardware:");
        println!(
            "  CPU: {} ({} cores)",
            self.hardware.cpu_model, self.hardware.cpu_cores
        );
        if let Some(gpu) = &self.hardware.gpu_model {
            println!("  GPU: {}", gpu);
            if let Some(mem) = self.hardware.gpu_memory_gb {
                println!("  GPU Memory: {:.1}GB", mem);
            }
        }
        println!("  System Memory: {:.1}GB", self.hardware.system_memory_gb);
        println!();

        // Memory usage
        println!("Memory Usage:");
        println!(
            "  Model weights: {}",
            format_bytes(self.memory.model_weights)
        );
        println!("  Activations: {}", format_bytes(self.memory.activations));
        println!("  KV cache: {}", format_bytes(self.memory.kv_cache));
        println!("  Peak usage: {}", format_bytes(self.memory.peak_memory));
        if let Some(bw) = self.memory.bandwidth_gbs {
            println!("  Bandwidth: {:.1} GB/s", bw);
        }
        println!();

        // Prefill performance
        println!("Prefill (Prompt Processing):");
        println!("  Batch size: {}", self.prefill.batch_size);
        println!("  Prompt length: {} tokens", self.prefill.seq_len);
        println!(
            "  Throughput: {:.1} tokens/sec",
            self.prefill.tokens_per_sec
        );
        println!("  Latency: {:.2} ms/token", self.prefill.ms_per_token);
        println!("  Total time: {:.2}s", self.prefill.total_time);
        if let Some(util) = self.prefill.gpu_util {
            println!("  GPU utilization: {:.1}%", util);
        }
        println!();

        // Decode performance
        println!("Decode (Autoregressive Generation):");
        println!("  Batch size: {}", self.decode.batch_size);
        println!("  Tokens generated: {}", self.decode.num_tokens);
        println!("  Throughput: {:.1} tokens/sec", self.decode.tokens_per_sec);
        println!("  Latency: {:.2} ms/token", self.decode.ms_per_token);
        println!("  Total time: {:.2}s", self.decode.total_time);
        if let Some(util) = self.decode.gpu_util {
            println!("  GPU utilization: {:.1}%", util);
        }
        println!();

        // Layer timings (top 10 slowest)
        if !self.layer_timings.is_empty() {
            println!("Top 10 Slowest Layers:");
            let mut timings: Vec<_> = self.layer_timings.values().collect();
            timings.sort_by(|a, b| b.time_ms.partial_cmp(&a.time_ms).unwrap());
            for (i, timing) in timings.iter().take(10).enumerate() {
                println!(
                    "  {}. {} ({}) - {:.2}ms ({:.1}%)",
                    i + 1,
                    timing.layer_name,
                    timing.layer_type,
                    timing.time_ms,
                    timing.percentage
                );
            }
            println!();
        }

        println!("═══════════════════════════════════════════════════════════");
    }

    /// Export results as JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "model_name": self.model_name,
            "quant_type": self.quant_type,
            "prefill": {
                "batch_size": self.prefill.batch_size,
                "seq_len": self.prefill.seq_len,
                "total_time": self.prefill.total_time,
                "tokens_per_sec": self.prefill.tokens_per_sec,
                "ms_per_token": self.prefill.ms_per_token,
                "gpu_util": self.prefill.gpu_util,
            },
            "decode": {
                "batch_size": self.decode.batch_size,
                "num_tokens": self.decode.num_tokens,
                "total_time": self.decode.total_time,
                "tokens_per_sec": self.decode.tokens_per_sec,
                "ms_per_token": self.decode.ms_per_token,
                "gpu_util": self.decode.gpu_util,
            },
            "memory": {
                "model_weights": self.memory.model_weights,
                "activations": self.memory.activations,
                "kv_cache": self.memory.kv_cache,
                "peak_memory": self.memory.peak_memory,
                "bandwidth_gbs": self.memory.bandwidth_gbs,
            },
            "layer_timings": self.layer_timings.values().map(|t| {
                serde_json::json!({
                    "layer_name": t.layer_name,
                    "layer_type": t.layer_type,
                    "time_ms": t.time_ms,
                    "percentage": t.percentage,
                })
            }).collect::<Vec<_>>(),
            "hardware": {
                "cpu_model": self.hardware.cpu_model,
                "cpu_cores": self.hardware.cpu_cores,
                "gpu_model": self.hardware.gpu_model,
                "gpu_memory_gb": self.hardware.gpu_memory_gb,
                "system_memory_gb": self.hardware.system_memory_gb,
            },
        })
    }
}

/// Simple timer for measuring operations.
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn stop(&self) -> f64 {
        let elapsed = self.elapsed_secs();
        println!("[{}] {:.2}ms", self.name, elapsed * 1000.0);
        elapsed
    }
}

/// Format bytes as human-readable.
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

/// Get hardware information.
pub fn get_hardware_info() -> HardwareInfo {
    use std::fs;

    // CPU info
    let cpu_model = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|content| {
            content
                .lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let cpu_cores = num_cpus::get();

    // Memory info
    let system_memory_gb = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|content| {
            content
                .lines()
                .find(|line| line.starts_with("MemTotal"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
                .map(|kb| kb / 1024.0 / 1024.0)
        })
        .unwrap_or(0.0);

    // GPU info (would need CUDA calls for real implementation)
    let gpu_model = None;
    let gpu_memory_gb = None;

    HardwareInfo {
        cpu_model,
        cpu_cores,
        gpu_model,
        gpu_memory_gb,
        system_memory_gb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10.0);
        assert!(elapsed < 20.0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500B");
        assert_eq!(format_bytes(1024), "1.00KB");
        assert_eq!(format_bytes(1_048_576), "1.00MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00GB");
    }
}
