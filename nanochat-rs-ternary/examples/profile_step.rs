//! Per-phase profiler for a single training step.
//!
//! Answers "where does the wall-clock of a step actually go?" by timing each
//! phase with a device synchronize around it. Without the sync, candle's CUDA
//! backend only returns launch time and every phase looks free except the one
//! that happens to touch the host.
//!
//! ```bash
//! cargo run --release --features cuda --example profile_step -- --config qwen35-hybrid --seq-len 512 --batch-size 2
//! ```

// The whole example is CUDA-only; without the feature candle has no
// `cuda_backend` module to read VRAM from, so it degrades to a message.
#[cfg(feature = "cuda")]
mod imp {


use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use nanochat_train::config::TrainConfig;
use nanochat_train::train::{compute_grad_norm, trim_all_cuda_memory_pools, Trainer};

/// (free, total) VRAM in bytes, straight from the driver.
fn vram() -> (usize, usize) {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let mut free = 0usize;
    let mut total = 0usize;
    unsafe {
        let _ = sys::cuMemGetInfo_v2(&mut free, &mut total);
    }
    (free, total)
}

fn vram_used_gb() -> f64 {
    let (free, total) = vram();
    (total - free) as f64 / 1e9
}

/// Time `f` with a device sync on both sides, so the number is GPU time.
fn timed<T>(dev: &Device, label: &str, f: impl FnOnce() -> candle_core::Result<T>) -> candle_core::Result<(T, f64)> {
    dev.synchronize()?;
    let t0 = Instant::now();
    let out = f()?;
    dev.synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    println!("  {:<38} {:>9.1} ms", label, dt * 1e3);
    Ok((out, dt))
}

pub fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let config_name = get("--config").unwrap_or_else(|| "qwen35-hybrid".to_string());
    let batch_size: usize = get("--batch-size").and_then(|s| s.parse().ok()).unwrap_or(2);
    let seq_len: usize = get("--seq-len").and_then(|s| s.parse().ok()).unwrap_or(512);
    let iters: usize = get("--iters").and_then(|s| s.parse().ok()).unwrap_or(5);

    let mut config = match config_name.as_str() {
        "qwen35-hybrid" | "qwen35_hybrid" => TrainConfig::qwen35_hybrid(),
        other => anyhow::bail!("unknown config {other}; add it to profile_step.rs"),
    };
    config.batch_size = batch_size;
    config.use_bf16_compute = args.iter().any(|a| a == "--bf16");
    config.grad_accum_steps = 1;

    let device = Device::new_cuda(0)?;
    println!("config={config_name} batch={batch_size} seq_len={seq_len} iters={iters}");

    let mut trainer = Trainer::new(config.clone(), device.clone())?;
    let n_params: usize = trainer
        .varmap
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    println!("params={:.1}M vars={}\n", n_params as f64 / 1e6, trainer.varmap.all_vars().len());

    let vocab = config.vocab_size as u32;
    let mk = || -> candle_core::Result<Tensor> {
        let data: Vec<u32> = (0..batch_size * seq_len)
            .map(|i| ((i * 7919) % vocab as usize) as u32)
            .collect();
        Tensor::from_vec(data, (batch_size, seq_len), &device)
    };
    let input_ids = mk()?;
    let target_ids = mk()?;

    println!("VRAM after model init:      {:.2} GB", vram_used_gb());

    // Warmup: first step pays for cuBLAS handles, kernel JIT, and pool growth.
    println!("warmup...");
    trainer.train_step(&input_ids, &target_ids)?;
    device.synchronize()?;
    println!("VRAM after first step:      {:.2} GB", vram_used_gb());

    // Peak during a forward: everything the backward will need is live here.
    {
        let hidden = trainer.model.forward_hidden_only(&input_ids)?;
        device.synchronize()?;
        println!("VRAM at end of forward:     {:.2} GB", vram_used_gb());
        drop(hidden);
    }
    trim_all_cuda_memory_pools();
    println!("VRAM after trim:            {:.2} GB", vram_used_gb());

    // ---- whole-step baseline ----
    println!("\n=== full train_step ===");
    let mut step_times = Vec::new();
    for _ in 0..iters {
        device.synchronize()?;
        let t0 = Instant::now();
        trainer.train_step(&input_ids, &target_ids)?;
        device.synchronize()?;
        step_times.push(t0.elapsed().as_secs_f64());
    }
    step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = step_times[step_times.len() / 2];
    let tokens = (batch_size * seq_len) as f64;
    println!(
        "  median {:.1} ms  ->  {:.0} tok/s",
        median * 1e3,
        tokens / median
    );

    // ---- phase breakdown ----
    // Re-runs the same phases train_step does, in order, so the numbers add up
    // to roughly the whole-step time above.
    println!("\n=== phase breakdown (median of {iters}) ===");
    let mut acc: Vec<(String, Vec<f64>)> = Vec::new();
    let record = |acc: &mut Vec<(String, Vec<f64>)>, k: &str, v: f64| {
        match acc.iter_mut().find(|(n, _)| n == k) {
            Some((_, xs)) => xs.push(v),
            None => acc.push((k.to_string(), vec![v])),
        }
    };

    for _ in 0..iters {
        let (hidden, t) = timed(&device, "forward_hidden_only", || {
            trainer.model.forward_hidden_only(&input_ids)
        })?;
        record(&mut acc, "forward_hidden_only", t);

        let (loss, t) = timed(&device, "lm_head + cross-entropy", || {
            let logits = trainer.model.project_hidden_to_logits(&hidden)?;
            let logits_flat = logits.reshape((batch_size * seq_len, config.vocab_size))?;
            let targets_flat = target_ids.reshape(batch_size * seq_len)?;
            let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?;
            candle_nn::loss::nll(&log_probs, &targets_flat)
        })?;
        record(&mut acc, "lm_head + cross-entropy", t);

        let (grads, t) = timed(&device, "backward", || loss.backward())?;
        record(&mut acc, "backward", t);

        let (_, t) = timed(&device, "compute_grad_norm", || {
            compute_grad_norm(&grads, &trainer.varmap)
        })?;
        record(&mut acc, "compute_grad_norm", t);

        drop(grads);
        drop(loss);
        drop(hidden);

        let (_, t) = timed(&device, "trim_all_cuda_memory_pools", || {
            trim_all_cuda_memory_pools();
            Ok(())
        })?;
        record(&mut acc, "trim_all_cuda_memory_pools", t);
        println!("  ---");
    }

    println!("\n=== summary (median) ===");
    let mut total = 0.0;
    for (name, mut xs) in acc {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = xs[xs.len() / 2];
        total += m;
        println!(
            "  {:<38} {:>9.1} ms  ({:>5.1}% of step)",
            name,
            m * 1e3,
            100.0 * m / median
        );
    }
    println!(
        "  {:<38} {:>9.1} ms  ({:>5.1}% of step)",
        "[accounted]",
        total * 1e3,
        100.0 * total / median
    );
    println!(
        "  {:<38} {:>9.1} ms  ({:>5.1}% of step)  <- optimizer + bookkeeping",
        "[unaccounted]",
        (median - total) * 1e3,
        100.0 * (median - total) / median
    );

    // ---- roofline: how much of the step is actually GEMM? ----
    //
    // This is the number that decides whether a faster matmul format (bf16
    // tensor cores, FP8, NVFP4) can help at all. If GEMMs are a small slice of
    // the step, Amdahl caps the win no matter how fast the format is.
    println!("\n=== roofline ===");
    {
        // Linear-layer FLOPs dominate; 2 per multiply-accumulate, ~3x for
        // fwd+bwd (one forward GEMM, two backward GEMMs per weight).
        let tokens_per_step = (batch_size * seq_len) as f64;
        let fwd_flops = 2.0 * n_params as f64 * tokens_per_step;
        let step_flops = 3.0 * fwd_flops;

        // Measured achievable GEMM rate at this model's shapes.
        let a = Tensor::randn(0f32, 1.0, (batch_size * seq_len, 1024), &device)?;
        let b = Tensor::randn(0f32, 0.02, (1024, 3584), &device)?;
        let gemm_flops = 2.0 * (batch_size * seq_len) as f64 * 1024.0 * 3584.0;
        let reps = 20;

        let mut rate = |dtype: DType| -> candle_core::Result<f64> {
            let (a, b) = (a.to_dtype(dtype)?, b.to_dtype(dtype)?);
            let _ = a.matmul(&b)?; // warm up
            device.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..reps {
                let _ = a.matmul(&b)?;
            }
            device.synchronize()?;
            Ok(gemm_flops * reps as f64 / t0.elapsed().as_secs_f64())
        };

        let f32_rate = rate(DType::F32)?;
        let bf16_rate = rate(DType::BF16)?;

        println!("  model GEMM per step        {:>9.2} TFLOP", step_flops / 1e12);
        println!("  measured f32  GEMM rate    {:>9.1} TFLOPS", f32_rate / 1e12);
        println!("  measured bf16 GEMM rate    {:>9.1} TFLOPS", bf16_rate / 1e12);

        let gemm_s = step_flops / f32_rate;
        println!(
            "  => GEMM time in step       {:>9.1} ms  ({:.1}% of {:.0} ms step)",
            gemm_s * 1e3,
            100.0 * gemm_s / median,
            median * 1e3
        );
        println!(
            "  => everything else         {:>9.1} ms  ({:.1}%)  <- elementwise + launch overhead",
            (median - gemm_s) * 1e3,
            100.0 * (median - gemm_s) / median
        );
        // Ceiling for any faster GEMM format, including NVFP4: it can only ever
        // remove the GEMM slice, and only down to zero.
        println!(
            "  => ceiling if GEMMs were FREE: {:.0} ms ({:.2}x) — the cap on any format change",
            (median - gemm_s) * 1e3,
            median / (median - gemm_s)
        );
    }

    // ---- isolated micro-benchmarks ----
    println!("\n=== isolated costs ===");
    let w = Tensor::randn(0f32, 0.02, (3584, 1024), &device)?;
    let x = Tensor::randn(0f32, 1.0, (batch_size * seq_len, 1024), &device)?;
    let (_, _) = timed(&device, "raw matmul [B*S,1024]x[1024,3584]", || {
        let wt = w.t()?.contiguous()?;
        x.matmul(&wt)
    })?;
    let (_, _) = timed(&device, "absmean_quantize on [3584,1024]", || {
        nanochat_train::quantize::absmean_quantize(&w, 128).map(|_| ())
    })?;
    // Is candle's reduction the reason compute_grad_norm is still slow?
    let (_, _) = timed(&device, "sqr+sum_all on [3584,1024]", || {
        w.sqr()?.sum_all().map(|_| ())
    })?;
    let (_, _) = timed(&device, "sum_all alone on [3584,1024]", || {
        w.sum_all().map(|_| ())
    })?;
    let (_, _) = timed(&device, "sqr alone on [3584,1024]", || w.sqr().map(|_| ()))?;
    let big = Tensor::randn(0f32, 1.0, (4096, 4096), &device)?;
    let (_, _) = timed(&device, "sum_all on [4096,4096] (16.8M)", || {
        big.sum_all().map(|_| ())
    })?;
    // sum(x^2) is a dot product, which cuBLAS does at memory bandwidth.
    let (_, _) = timed(&device, "dot(w,w) via matmul [3584,1024]", || {
        let n = w.elem_count();
        let flat = w.reshape((1, n))?;
        flat.matmul(&flat.t()?.contiguous()?).map(|_| ())
    })?;
    let (_, _) = timed(&device, "sum(1)+sum(0) on [3584,1024]", || {
        w.sqr()?.sum(1)?.sum(0).map(|_| ())
    })?;

    // Muon's orthogonalization, the suspected bulk of the optimizer phase.
    use nanochat_train::optim::muon::newton_schulz_orthogonalize;
    let ffn_w = Tensor::randn(0f32, 0.02, (3584, 1024), &device)?;
    let sq_w = Tensor::randn(0f32, 0.02, (1024, 1024), &device)?;
    let (_, _) = timed(&device, "newton_schulz [3584,1024] ns=5", || {
        newton_schulz_orthogonalize(&ffn_w, 5).map(|_| ())
    })?;
    let (_, _) = timed(&device, "newton_schulz [1024,1024] ns=5", || {
        newton_schulz_orthogonalize(&sq_w, 5).map(|_| ())
    })?;

    let dummy = Tensor::zeros((1024, 1024), DType::F32, &device)?;
    let (_, _) = timed(&device, "single to_scalar (sync cost)", || {
        dummy.sum_all()?.to_scalar::<f32>().map(|_| ())
    })?;

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
