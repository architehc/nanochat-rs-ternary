//! Candle-compatible XOR convolution (FWHT) with autograd support — **training path**.
//!
//! XOR convolution: `c[k] = (1/N) Σ_i a[i] · b[i ⊕ k]`. FWHT is
//! self-adjoint (H^T = H), so the backward pass uses the same FWHT
//! transform as the forward pass. No complex numbers needed.
//!
//! **Implementation:** Pure Candle tensor ops (reshape + narrow + cat butterflies).
//! Stays on whatever device the input tensor lives on (CPU or CUDA) — no device
//! transfers needed. Autograd gradients come for free from Candle's standard ops.

use candle_core::{Result, Tensor};

/// In-place-style FWHT via Candle tensor ops.
///
/// Applies the Walsh-Hadamard butterfly `a' = a + b, b' = a - b` at
/// exponentially increasing strides, expressed as reshape→narrow→add/sub→cat.
///
/// Input shape: `(..., N)` where N is power of 2.
/// Output shape: same as input.
///
/// O(N log N) operations, log2(N) butterfly stages.
fn fwht_tensor(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let ndim = dims.len();
    let n = dims[ndim - 1];
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-2 length, got {}", n);

    // Flatten all leading dims into a single batch dimension
    let batch: usize = dims[..ndim - 1].iter().product::<usize>().max(1);
    let orig_shape: Vec<usize> = dims.to_vec();

    let mut result = x.reshape((batch, n))?;

    let mut half = 1usize;
    while half < n {
        let groups = n / (2 * half);
        // Reshape to (batch, groups, 2, half):
        //   dim 0: batch
        //   dim 1: groups of 2*half elements
        //   dim 2: pair (element vs partner at distance `half`)
        //   dim 3: position within group
        result = result.reshape((batch, groups, 2, half))?;
        let a = result.narrow(2, 0, 1)?; // (batch, groups, 1, half)
        let b = result.narrow(2, 1, 1)?; // (batch, groups, 1, half)
        let sum = (&a + &b)?;
        let diff = (&a - &b)?;
        result = Tensor::cat(&[&sum, &diff], 2)?; // (batch, groups, 2, half)
        result = result.reshape((batch, n))?;
        half *= 2;
    }

    result.reshape(orig_shape)
}

/// FWHT convolution (XOR convolution) via pure tensor ops.
///
/// Computes `c[k] = (1/N) Σ_i signal[i] · kernel[i ⊕ k]` by:
/// 1. FWHT(signal)
/// 2. FWHT(kernel)
/// 3. Pointwise multiply
/// 4. FWHT(product) / N
///
/// FWHT is self-inverse up to 1/N: FWHT(FWHT(x)) = N·x.
///
/// Signal shape: `(batch, field_size)` or `(field_size,)`.
/// Kernel shape: `(field_size,)` or `(batch, field_size)`.
fn fwht_convolve_impl(signal: &Tensor, kernel: &Tensor, field_size: usize) -> Result<Tensor> {
    let s_fwht = fwht_tensor(signal)?;
    let k_fwht = fwht_tensor(kernel)?;
    let product = s_fwht.broadcast_mul(&k_fwht)?.contiguous()?;
    let result = fwht_tensor(&product)?;
    result / (field_size as f64)
}

/// High-level API: FWHT convolution without gradient tracking.
///
/// Gradients still flow if inputs are Vars — use this when you don't
/// need to distinguish from the with_grad variant.
pub fn fwht_convolve(signal: &Tensor, kernel: &Tensor, field_size: usize) -> Result<Tensor> {
    fwht_convolve_impl(signal, kernel, field_size)
}

/// FWHT convolution preserving autograd graph for backward pass.
///
/// With pure tensor ops, this is identical to `fwht_convolve` — autograd
/// is handled automatically by Candle's standard operations.
pub fn fwht_convolve_with_grad(
    signal: &Tensor,
    kernel: &Tensor,
    field_size: usize,
) -> Result<Tensor> {
    fwht_convolve_impl(signal, kernel, field_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_fwht_tensor_self_inverse() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;

        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let x = Tensor::from_vec(data.clone(), (n,), &device)?;

        // FWHT(FWHT(x)) = N * x
        let twice = fwht_tensor(&fwht_tensor(&x)?)?;
        let scaled = (twice / n as f64)?;
        let result = scaled.to_vec1::<f32>()?;

        for i in 0..n {
            assert!(
                (result[i] - data[i]).abs() < 1e-4,
                "FWHT self-inverse failed at {}: {} vs {}",
                i, result[i], data[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_candle_fwht_basic() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let mut kernel_data = vec![0.0f32; n];
        kernel_data[0] = 1.0;

        let signal = Tensor::from_vec(signal_data.clone(), (n,), &device)?;
        let kernel = Tensor::from_vec(kernel_data, (n,), &device)?;

        let result = fwht_convolve(&signal, &kernel, n)?;
        let result_data = result.to_vec1::<f32>()?;

        for i in 0..n {
            assert!(
                (result_data[i] - signal_data[i]).abs() < 1e-4,
                "candle FWHT delta conv failed at {}: {} vs {}",
                i, result_data[i], signal_data[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_candle_fwht_batched() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;
        let batch = 3;

        let signal_data: Vec<f32> = (0..batch * n)
            .map(|i| (i as f32 * 0.1).cos())
            .collect();
        let kernel_data: Vec<f32> = (0..batch * n)
            .map(|i| (-0.1 * (i % n) as f32).exp())
            .collect();

        let signal = Tensor::from_vec(signal_data, (batch, n), &device)?;
        let kernel = Tensor::from_vec(kernel_data, (batch, n), &device)?;

        let result = fwht_convolve(&signal, &kernel, n)?;
        assert_eq!(result.dims(), &[batch, n]);

        let result_data = result.to_vec2::<f32>()?;
        for b in 0..batch {
            assert!(
                result_data[b].iter().all(|v| v.is_finite()),
                "non-finite in batch {}",
                b
            );
        }
        Ok(())
    }

    #[test]
    fn test_candle_fwht_gradient() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let kernel_data: Vec<f32> = (0..n)
            .map(|i| (-0.1 * i as f32).exp())
            .collect();

        let signal = candle_core::Var::from_vec(signal_data, (n,), &device)?;
        let kernel = candle_core::Var::from_vec(kernel_data, (n,), &device)?;

        let result = fwht_convolve_with_grad(signal.as_tensor(), kernel.as_tensor(), n)?;
        let loss = result.sqr()?.sum_all()?;
        let grads = loss.backward()?;

        let grad_signal = grads.get(signal.as_tensor());
        let grad_kernel = grads.get(kernel.as_tensor());

        assert!(grad_signal.is_some(), "signal should have gradient");
        assert!(grad_kernel.is_some(), "kernel should have gradient");

        let gs = grad_signal.unwrap().to_vec1::<f32>()?;
        let gk = grad_kernel.unwrap().to_vec1::<f32>()?;

        assert!(gs.iter().all(|v| v.is_finite()), "non-finite signal grad");
        assert!(gk.iter().all(|v| v.is_finite()), "non-finite kernel grad");
        assert!(gs.iter().any(|v| v.abs() > 1e-8), "signal grad all zeros");
        assert!(gk.iter().any(|v| v.abs() > 1e-8), "kernel grad all zeros");

        Ok(())
    }

    #[test]
    fn test_fwht_gradient_finite_differences() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let eps = 1e-3f32;

        let signal_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.2, 0.0, -0.1, 0.4, -0.2];
        let kernel_data: Vec<f32> = vec![0.8, 0.3, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0];

        let signal = candle_core::Var::from_vec(signal_data.clone(), (n,), &device)?;
        let kernel = candle_core::Var::from_vec(kernel_data.clone(), (n,), &device)?;

        let result = fwht_convolve_with_grad(signal.as_tensor(), kernel.as_tensor(), n)?;
        let loss = result.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        let analytic_grad = grads
            .get(signal.as_tensor())
            .unwrap()
            .to_vec1::<f32>()?;

        for i in 0..n.min(4) {
            let mut sig_plus = signal_data.clone();
            sig_plus[i] += eps;
            let mut sig_minus = signal_data.clone();
            sig_minus[i] -= eps;

            let sp = Tensor::from_vec(sig_plus, (n,), &device)?;
            let sm = Tensor::from_vec(sig_minus, (n,), &device)?;
            let k = Tensor::from_vec(kernel_data.clone(), (n,), &device)?;

            let loss_plus = fwht_convolve(&sp, &k, n)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            let loss_minus = fwht_convolve(&sm, &k, n)?.sqr()?.sum_all()?.to_scalar::<f32>()?;

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytic = analytic_grad[i];

            assert!(
                (numerical - analytic).abs() < 0.05,
                "FWHT gradient mismatch at signal[{}]: numerical={} analytic={}",
                i, numerical, analytic
            );
        }

        Ok(())
    }

    /// Verify tensor-op FWHT matches C FFI FWHT output.
    #[test]
    fn test_fwht_tensor_matches_c_ffi() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17 + 0.3).sin()).collect();
        let kernel_data: Vec<f32> = (0..n).map(|i| (-0.15 * i as f32).exp()).collect();

        // Tensor-op path
        let signal = Tensor::from_vec(signal_data.clone(), (n,), &device)?;
        let kernel = Tensor::from_vec(kernel_data.clone(), (n,), &device)?;
        let tensor_result = fwht_convolve(&signal, &kernel, n)?.to_vec1::<f32>()?;

        // C FFI path
        let mut c_output = vec![0.0f32; n];
        ternary_kernels::fwht::fwht_convolve_f32_buf(
            &signal_data, &kernel_data, &mut c_output,
        );

        for i in 0..n {
            assert!(
                (tensor_result[i] - c_output[i]).abs() < 1e-4,
                "tensor vs C mismatch at [{}]: {} vs {}",
                i, tensor_result[i], c_output[i]
            );
        }
        Ok(())
    }
}
