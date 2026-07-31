//! What GEMM rate each dtype actually reaches on this GPU, at this model's shapes.
//!
//! This is the evidence for choosing a compute precision. It answers two things:
//!
//!   1. How much faster is bf16 than f32 here (the tensor-core payoff)?
//!   2. What is the *ceiling* for any faster format — FP8, NVFP4 — given how much
//!      of a training step is GEMM at all?
//!
//! Deliberately tiny in memory so it can run alongside a training job.
//!
//! ```bash
//! cargo run --release --features cuda --example bench_gemm
//! ```

// CUDA-only: candle has no bf16 GEMM on the CPU backend.
#[cfg(feature = "cuda")]
mod imp {

use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

/// Sustained TFLOPS for `[m,k] @ [k,n]` at `dtype`.
fn rate(dev: &Device, m: usize, k: usize, n: usize, dtype: DType, reps: usize) -> Result<f64> {
    let a = Tensor::randn(0f32, 1.0, (m, k), dev)?.to_dtype(dtype)?;
    let b = Tensor::randn(0f32, 1.0, (k, n), dev)?.to_dtype(dtype)?;
    let _ = a.matmul(&b)?; // warm up cuBLAS handles / algo selection
    dev.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = a.matmul(&b)?;
    }
    dev.synchronize()?;
    let secs = t0.elapsed().as_secs_f64();
    Ok(2.0 * m as f64 * k as f64 * n as f64 * reps as f64 / secs / 1e12)
}

pub fn run() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;

    // Shapes the qwen35-hybrid model actually issues (batch 4 x seq 512 = 2048
    // tokens), plus a large square one to show the asymptotic rate.
    let shapes = [
        ("ffn up/gate  [2048,1024]x[1024,3584]", 2048, 1024, 3584),
        ("ffn down     [2048,3584]x[3584,1024]", 2048, 3584, 1024),
        ("attn/dn proj [2048,1024]x[1024,1024]", 2048, 1024, 1024),
        ("lm head      [2048,1024]x[1024,4096]", 2048, 1024, 4096),
        ("large square [4096,4096]x[4096,4096]", 4096, 4096, 4096),
    ];

    println!("{:<40} {:>10} {:>10} {:>8}", "shape", "f32", "bf16", "speedup");
    println!("{}", "-".repeat(72));
    let (mut sum_f32, mut sum_bf16) = (0.0, 0.0);
    for (label, m, k, n) in shapes {
        let f = rate(&dev, m, k, n, DType::F32, 30)?;
        let b = rate(&dev, m, k, n, DType::BF16, 30)?;
        sum_f32 += f;
        sum_bf16 += b;
        println!("{:<40} {:>7.1} TF {:>7.1} TF {:>7.2}x", label, f, b, b / f);
    }
    println!("{}", "-".repeat(72));
    println!(
        "{:<40} {:>7.1} TF {:>7.1} TF {:>7.2}x",
        "mean",
        sum_f32 / shapes.len() as f64,
        sum_bf16 / shapes.len() as f64,
        sum_bf16 / sum_f32
    );

    println!("\nWhat a lower-precision format could buy:");
    println!("  A training step for this model is ~3.2 TFLOP of GEMM. At the bf16");
    println!("  rate above that is a small slice of the measured step time — run");
    println!("  `profile_step` for the exact percentage on this machine. Any format");
    println!("  faster than bf16 (FP8, NVFP4) can only shrink that slice, so the");
    println!("  step-level speedup is capped by Amdahl regardless of the format's");
    println!("  peak TOPS. candle has no FP4 dtype in any case; see");
    println!("  docs/BLACKWELL_LOW_PRECISION.md.");
    Ok(())
}

}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    imp::run()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example requires --features cuda");
}
