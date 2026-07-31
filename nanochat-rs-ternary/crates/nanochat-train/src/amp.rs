//! Mixed-precision policy: bf16 compute with f32 master weights.
//!
//! # Status: measured, and it loses. Off by default.
//!
//! On `qwen35-hybrid` / RTX 5090 / seq 512, against the f32 baseline:
//!
//! ```text
//!   batch 2:  f32 1144 tok/s 15.58 GB   ->   bf16 1131 tok/s 16.86 GB
//!   batch 4:  f32 1631 tok/s 25.01 GB   ->   bf16 1606 tok/s 27.84 GB
//! ```
//!
//! Slower *and* larger. Two reasons, both properties of casting at the matmul
//! rather than of bf16:
//!
//! - The step is launch-bound and GEMM is only ~2% of it, so the 3.2x GEMM rate
//!   is worth ~0.6% — less than the three `to_dtype` kernels this adds per
//!   matmul, across ~160 linear layers, in forward *and* backward.
//! - Casting cannot reduce memory. Autograd retains the original f32 operands
//!   for the backward pass, so the bf16 copies are additive.
//!
//! Kept, defaulted off, because the implementation is correct and tested and the
//! negative result is worth being able to reproduce. The approach that would
//! actually pay is building the model in bf16 at the `VarBuilder` with a
//! separate f32 master copy — see `docs/BLACKWELL_LOW_PRECISION.md`.
//!
//! # What runs where
//!
//! Master weights, optimizer state, reductions (RMSNorm variance, softmax, the
//! loss, gradient norms) and the DeltaNet carried state stay in **f32**. Only
//! the *matmul operands* and the elementwise work feeding them drop to **bf16**.
//!
//! bf16 rather than f16 because it has f32's exponent range: no loss scaling, no
//! overflow bookkeeping, and a gradient that underflows in f16 stays finite here.
//! The cost is mantissa — 8 bits against f32's 24.
//!
//! # Why this is nearly free for a ternary QAT model
//!
//! [`crate::layers::BitLinearSTE`] feeds the matmul with *already quantized*
//! values: weights are `{-1, 0, +1}` times a per-group scale, activations are
//! integers in `[-127, 127]` times a per-token scale. bf16 represents every
//! integer up to 256 exactly, so the quantized factors survive the cast intact —
//! only the scales lose mantissa. The quantization error the model is already
//! trained to absorb is far larger than the rounding this adds.
//!
//! candle issues bf16 GEMMs through cuBLAS with `CUBLAS_COMPUTE_32F` (see
//! `gemm_reduced_precision_bf16`, which we leave off), so products accumulate in
//! f32 on the tensor cores. Inputs are bf16; the running sum is not.
//!
//! # Where this is deliberately *not* applied
//!
//! The Gated DeltaNet recurrence stays in f32. Three reasons, and they compound:
//!
//! - Its matmuls are small (`[128,128] x [128,128]` batched at the default chunk
//!   size). Small GEMMs do not reach the tensor-core rates that make bf16 worth
//!   it — the measured f32/bf16 gap shrinks from 4.8x to 1.6x as shapes get
//!   smaller (`examples/bench_gemm.rs`). Each cast is also two extra kernels, and
//!   this step is launch-bound, so casting small operands can cost more than it
//!   saves.
//! - The carried state `S` accumulates across the whole sequence. An 8-bit
//!   significand there compounds over every chunk, unlike a single GEMM.
//! - `kkt` feeds the triangular solve, whose accuracy depends on the
//!   conditioning of `I + N`; perturbing its entries is amplified by the solve.
//!
//! The linear layers carry the overwhelming majority of the FLOPs *and* have
//! exactly-representable operands, so that is where the win is.
//!
//! # CPU
//!
//! The policy is CUDA-only, and that is a **correctness** requirement rather than
//! a tuning choice: candle's CPU backend has no bf16 GEMM at all and fails with
//! `unsupported dtype BF16 for op matmul`. [`compute_dtype`] therefore refuses to
//! select bf16 off CUDA, which also keeps the portable tests exact.

use candle_core::{DType, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};

/// Process-wide switch. Off unless a trainer turns it on.
///
/// Global rather than threaded through every layer constructor because it is a
/// property of the *run*, not of any one layer — and `BitLinearSTE::new` is
/// called from ~30 sites that would otherwise all need a new parameter. Tests
/// that need to pin a precision call [`matmul_in`] directly instead of touching
/// this, so they stay independent of it under parallel execution.
static BF16_COMPUTE: AtomicBool = AtomicBool::new(false);

/// Enable or disable bf16 compute for the whole process.
pub fn set_bf16_compute(enabled: bool) {
    BF16_COMPUTE.store(enabled, Ordering::Relaxed);
}

/// Whether bf16 compute is currently enabled.
pub fn bf16_compute() -> bool {
    BF16_COMPUTE.load(Ordering::Relaxed)
}

/// The dtype matmul operands should be cast to for a tensor on `t`'s device.
///
/// f32 when the policy is off, or on CPU where bf16 is emulated.
pub fn compute_dtype(t: &Tensor) -> DType {
    if bf16_compute() && t.device().is_cuda() && t.dtype() == DType::F32 {
        DType::BF16
    } else {
        t.dtype()
    }
}

/// `a @ b` under the current precision policy, returning the dtype of `a`.
///
/// Casting is skipped entirely when the policy resolves to `a`'s own dtype, so
/// the f32 path adds no ops.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    matmul_in(a, b, compute_dtype(a))
}

/// `a @ b` with the operand dtype named explicitly.
///
/// The result is cast back to `a`'s dtype, so this is drop-in for `a.matmul(b)`.
/// Gradients flow through the casts: candle's backward for `ToDType` casts the
/// incoming gradient to the source dtype, so the *backward* matmuls run at
/// `dtype` too — which is where roughly two thirds of the matmul work is.
pub fn matmul_in(a: &Tensor, b: &Tensor, dtype: DType) -> Result<Tensor> {
    let out_dtype = a.dtype();
    if dtype == out_dtype {
        return a.matmul(b);
    }
    let out = a.to_dtype(dtype)?.matmul(&b.to_dtype(dtype)?)?;
    out.to_dtype(out_dtype)
}

/// [`matmul`] for operands whose leading dims broadcast.
pub fn broadcast_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dtype = compute_dtype(a);
    let out_dtype = a.dtype();
    if dtype == out_dtype {
        return a.broadcast_matmul(b);
    }
    let out = a.to_dtype(dtype)?.broadcast_matmul(&b.to_dtype(dtype)?)?;
    out.to_dtype(out_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[cfg(feature = "cuda")]
    use candle_core::Var;

    /// bf16 GEMM exists only on CUDA, so the numerical tests need a device.
    #[cfg(feature = "cuda")]
    fn dev() -> Result<Device> {
        Device::new_cuda(0)
    }

    /// bf16 operands must not move a matmul far from the f32 answer.
    ///
    /// The bar is relative error, since bf16 carries 8 mantissa bits: ~4e-3 per
    /// element. Errors across the K-dimension are independent, so the sum's
    /// relative error does not grow with K.
    #[cfg(feature = "cuda")]
    #[test]
    fn bf16_matmul_tracks_f32() -> Result<()> {
        let dev = dev()?;
        let a = Tensor::randn(0f32, 1.0, (64, 256), &dev)?;
        let b = Tensor::randn(0f32, 1.0, (256, 128), &dev)?;

        let want = matmul_in(&a, &b, DType::F32)?;
        let got = matmul_in(&a, &b, DType::BF16)?;

        let denom = want.abs()?.mean_all()?.to_scalar::<f32>()?;
        let err = (&got - &want)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(
            err / denom < 0.01,
            "bf16 matmul relative error {} too large",
            err / denom
        );
        Ok(())
    }

    /// The quantized operands a ternary model actually feeds the matmul are
    /// *exactly* representable in bf16, so this path must be bit-exact.
    ///
    /// Integers up to 256 fit bf16's 8-bit significand without rounding: ternary
    /// weights `{-1,0,1}` and int8 activations in `[-127,127]` both qualify.
    #[cfg(feature = "cuda")]
    #[test]
    fn bf16_is_exact_for_quantized_operands() -> Result<()> {
        let dev = dev()?;
        // Ternary weights and int8-range activations, unscaled.
        let acts: Vec<f32> = (0..64 * 128).map(|i| ((i % 255) as f32) - 127.0).collect();
        let w: Vec<f32> = (0..128 * 32).map(|i| (i % 3) as f32 - 1.0).collect();
        let a = Tensor::from_vec(acts, (64, 128), &dev)?;
        let b = Tensor::from_vec(w, (128, 32), &dev)?;

        let want = matmul_in(&a, &b, DType::F32)?;
        let got = matmul_in(&a, &b, DType::BF16)?;
        let err = (&got - &want)?.abs()?.max_keepdim(1)?.max_keepdim(0)?;
        assert_eq!(
            err.reshape(())?.to_scalar::<f32>()?,
            0.0,
            "bf16 must be exact for ternary x int8 operands"
        );
        Ok(())
    }

    /// Gradients must flow through the casts back to the f32 master weight.
    #[cfg(feature = "cuda")]
    #[test]
    fn gradients_flow_through_cast() -> Result<()> {
        let dev = dev()?;
        let w = Var::from_tensor(&Tensor::randn(0f32, 1.0, (16, 8), &dev)?)?;
        let x = Tensor::randn(0f32, 1.0, (4, 16), &dev)?;

        let y = matmul_in(&x, w.as_tensor(), DType::BF16)?;
        assert_eq!(y.dtype(), DType::F32, "result must come back as f32");

        let grads = y.sum_all()?.backward()?;
        let g = grads
            .get(&w)
            .expect("gradient must reach the f32 master weight");
        assert_eq!(g.dtype(), DType::F32);
        assert!(g.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0);
        Ok(())
    }

    /// The policy must be inert on CPU, so portable tests stay exact.
    #[test]
    fn policy_is_cuda_only() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::zeros((2, 2), DType::F32, &dev)?;
        set_bf16_compute(true);
        let dt = compute_dtype(&t);
        set_bf16_compute(false);
        assert_eq!(dt, DType::F32, "bf16 must not engage on CPU");
        Ok(())
    }
}
