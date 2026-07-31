//! Per-layer-type cost breakdown: time and activation memory.
//!
//! `profile_step` says the step is slow; this says *which layer* is slow. Each
//! layer is built standalone at the qwen35_hybrid dimensions and run forward and
//! backward with a device sync around it, plus the VRAM it leaves live at the
//! end of its forward (which is what the backward has to hold).
//!
//! ```bash
//! cargo run --release --features cuda --example profile_layers
//! ```

// The whole example is CUDA-only; without the feature candle has no
// `cuda_backend` module to read VRAM from, so it degrades to a message.
#[cfg(feature = "cuda")]
mod imp {


use std::time::Instant;

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use nanochat_train::attention::{precompute_rope_freqs, AttentionTrain};
use nanochat_train::ffn::FeedForwardTrain;
use nanochat_train::gated_deltanet::GatedDeltaNetTrain;
use nanochat_train::layers::BitLinearSTE;

fn vram_used() -> f64 {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let (mut free, mut total) = (0usize, 0usize);
    unsafe {
        let _ = sys::cuMemGetInfo_v2(&mut free, &mut total);
    }
    (total - free) as f64 / 1e9
}

/// Run `f` forward and backward, reporting time for each and the VRAM held live
/// at the end of the forward.
fn bench(
    dev: &Device,
    label: &str,
    reps: usize,
    seed: &Var,
    f: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<()> {
    let x = seed.as_tensor();

    // Warmup.
    let y = f(x)?;
    let _ = y.sum_all()?.backward()?;
    dev.synchronize()?;

    let before = vram_used();

    // Forward.
    dev.synchronize()?;
    let t0 = Instant::now();
    let mut out = None;
    for _ in 0..reps {
        out = Some(f(x)?);
    }
    dev.synchronize()?;
    let fwd = t0.elapsed().as_secs_f64() / reps as f64;

    let held = vram_used() - before;
    let out = out.unwrap();

    // Backward.
    let loss = out.sum_all()?;
    dev.synchronize()?;
    let t1 = Instant::now();
    let _grads = loss.backward()?;
    dev.synchronize()?;
    let bwd = t1.elapsed().as_secs_f64();

    println!(
        "  {:<28} fwd {:>7.1} ms   bwd {:>7.1} ms   held {:>6.2} GB (x{} reps)",
        label,
        fwd * 1e3,
        bwd * 1e3,
        held,
        reps
    );
    Ok(())
}

pub fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let batch: usize = get("--batch-size").and_then(|s| s.parse().ok()).unwrap_or(2);
    let seq: usize = get("--seq-len").and_then(|s| s.parse().ok()).unwrap_or(512);

    // qwen35_hybrid dimensions.
    let (dim, n_heads, n_kv_heads, ffn_dim, group_size) = (1024usize, 8usize, 2usize, 3584usize, 128usize);

    let device = Device::new_cuda(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    println!("batch={batch} seq={seq} dim={dim} heads={n_heads} ffn={ffn_dim}\n");

    let x = Var::from_tensor(&Tensor::randn(0f32, 1.0, (batch, seq, dim), &device)?)?;

    let dn = GatedDeltaNetTrain::new(dim, n_heads, group_size, vb.pp("dn"))?;
    let attn = AttentionTrain::new(dim, n_heads, n_kv_heads, group_size, vb.pp("attn"))?;
    let ffn = FeedForwardTrain::new(dim, ffn_dim, group_size, vb.pp("ffn"))?;
    let lin = BitLinearSTE::new(dim, ffn_dim, group_size, vb.pp("lin"))?;

    println!("=== per-layer (one instance) ===");
    bench(&device, "GatedDeltaNet", 1, &x, |t| dn.forward(t))?;
    let (cos, sin) = precompute_rope_freqs(dim / n_heads, seq, 10000.0, &device)?;
    bench(&device, "Attention", 1, &x, |t| attn.forward(t, &cos, &sin))?;
    bench(&device, "FeedForward (SwiGLU)", 1, &x, |t| ffn.forward(t))?;
    bench(&device, "one BitLinearSTE", 1, &x, |t| lin.forward(t))?;

    // Same shape, but a plain matmul with no quantization: the floor.
    let w = Var::from_tensor(&Tensor::randn(0f32, 0.02f32, (dim, ffn_dim), &device)?)?;
    bench(&device, "plain matmul (no quant)", 1, &x, |t| {
        let (b, s, k) = t.dims3()?;
        t.reshape((b * s, k))?.matmul(w.as_tensor())?.reshape((b, s, ()))
    })?;

    // Chunk size trades sequential chunk-steps against intra-chunk work. If the
    // recurrence is launch-bound rather than FLOP-bound, bigger chunks win.
    println!("\n=== GatedDeltaNet vs chunk size ===");
    for chunk in [16usize, 32, 64, 128, 256, 512] {
        bench(&device, &format!("chunk={chunk}"), 1, &x, |t| {
            dn.forward_with_chunk(t, chunk)
        })?;
    }

    println!("\n=== model total implied (12 DeltaNet + 4 Attention + 16 FFN) ===");
    println!("  (multiply the per-layer numbers above by those counts)");

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
