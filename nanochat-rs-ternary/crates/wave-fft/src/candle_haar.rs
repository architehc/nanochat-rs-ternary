//! Candle-compatible Haar wavelet-basis scaling with autograd support — **training path**.
//!
//! Diagonal operator in the Haar basis (scale-selective filtering, NOT
//! shift-invariant convolution). Haar DWT is orthogonal: inverse = transpose
//! (adjoint). The backward pass has the same structure as the forward pass.
//!
//! **Implementation:** Pure Candle tensor ops (narrow + reshape + cat for
//! forward/inverse DWT levels). Stays on whatever device the input tensor
//! lives on (CPU or CUDA) — no device transfers needed. Autograd gradients
//! come for free from Candle's standard ops.

use candle_core::{Result, Tensor};

/// Forward Haar DWT via Candle tensor ops (multi-level).
///
/// At each level, pairs adjacent elements:
///   low[i]  = data[2*i] + data[2*i+1]   (approximation)
///   high[i] = data[2*i] - data[2*i+1]   (detail)
/// Output: [low_coeffs..., high_coeffs...]
///
/// Multi-level: repeat on the low (first half) coefficients.
///
/// Input shape: `(batch, N)` where N is power of 2.
fn haar_forward_tensor(x: &Tensor, n: usize, levels: usize) -> Result<Tensor> {
    assert!(
        n.is_power_of_two(),
        "Haar DWT requires power-of-2 length, got {}",
        n
    );
    debug_assert!(
        levels <= (n as f64).log2() as usize,
        "Haar levels ({}) exceeds log2(n={}) = {}",
        levels,
        n,
        (n as f64).log2() as usize
    );
    let dims = x.dims();
    let ndim = dims.len();
    let batch: usize = dims[..ndim - 1].iter().product::<usize>().max(1);
    let orig_shape: Vec<usize> = dims.to_vec();

    let mut result = x.reshape((batch, n))?;
    let mut current_len = n;

    for _ in 0..levels {
        if current_len < 2 {
            break;
        }
        let half = current_len / 2;

        // Extract prefix [0..current_len]
        let prefix = result.narrow(1, 0, current_len)?;

        // Reshape prefix into pairs: (batch, half, 2)
        let pairs = prefix.reshape((batch, half, 2))?;
        let even = pairs.narrow(2, 0, 1)?.squeeze(2)?; // data[2*i]   → (batch, half)
        let odd = pairs.narrow(2, 1, 1)?.squeeze(2)?;  // data[2*i+1] → (batch, half)

        let low = (&even + &odd)?;  // approximation
        let high = (&even - &odd)?; // detail
        let transformed = Tensor::cat(&[&low, &high], 1)?; // (batch, current_len)

        if current_len < n {
            let suffix = result.narrow(1, current_len, n - current_len)?;
            result = Tensor::cat(&[&transformed, &suffix], 1)?;
        } else {
            result = transformed;
        }

        current_len = half;
    }

    result.reshape(orig_shape)
}

/// Inverse Haar DWT via Candle tensor ops (multi-level).
///
/// At each level, reconstructs from [low, high]:
///   data[2*i]   = (low[i] + high[i]) * 0.5
///   data[2*i+1] = (low[i] - high[i]) * 0.5
///
/// Multi-level inverse: starts from deepest (smallest) level and works up.
fn haar_inverse_tensor(x: &Tensor, n: usize, levels: usize) -> Result<Tensor> {
    let dims = x.dims();
    let ndim = dims.len();
    let batch: usize = dims[..ndim - 1].iter().product::<usize>().max(1);
    let orig_shape: Vec<usize> = dims.to_vec();

    let mut result = x.reshape((batch, n))?;

    // Forward processed levels: n, n/2, n/4, ..., n/2^(levels-1)
    // Inverse processes in reverse: start from smallest current_len
    for l in (0..levels).rev() {
        let current_len = n >> l;
        if current_len < 2 {
            continue;
        }
        let half = current_len / 2;

        let prefix = result.narrow(1, 0, current_len)?;
        let low = prefix.narrow(1, 0, half)?;       // (batch, half)
        let high = prefix.narrow(1, half, half)?;    // (batch, half)

        let even = ((&low + &high)? * 0.5)?; // data[2*i]
        let odd = ((&low - &high)? * 0.5)?;  // data[2*i+1]

        // Interleave: stack along last dim then reshape
        // stack → (batch, half, 2), reshape → (batch, current_len)
        let interleaved = Tensor::stack(&[&even, &odd], 2)?.reshape((batch, current_len))?;

        if current_len < n {
            let suffix = result.narrow(1, current_len, n - current_len)?;
            result = Tensor::cat(&[&interleaved, &suffix], 1)?;
        } else {
            result = interleaved;
        }
    }

    result.reshape(orig_shape)
}

/// Haar wavelet-domain convolution via pure tensor ops.
///
/// Computes: IHaar(Haar(signal) ⊙ Haar(kernel))
/// Each Haar coefficient is scaled independently — this is a diagonal
/// operator in the wavelet basis (scale-selective filtering).
fn haar_convolve_impl(
    signal: &Tensor,
    kernel: &Tensor,
    field_size: usize,
    levels: usize,
) -> Result<Tensor> {
    let s_haar = haar_forward_tensor(signal, field_size, levels)?;
    let k_haar = haar_forward_tensor(kernel, field_size, levels)?;
    let product = s_haar.broadcast_mul(&k_haar)?.contiguous()?;
    haar_inverse_tensor(&product, field_size, levels)
}

/// High-level API: Haar convolution without gradient tracking.
pub fn haar_convolve(
    signal: &Tensor,
    kernel: &Tensor,
    field_size: usize,
    levels: usize,
) -> Result<Tensor> {
    haar_convolve_impl(signal, kernel, field_size, levels)
}

/// Haar convolution preserving autograd graph for backward pass.
///
/// With pure tensor ops, this is identical to `haar_convolve` — autograd
/// is handled automatically by Candle's standard operations.
pub fn haar_convolve_with_grad(
    signal: &Tensor,
    kernel: &Tensor,
    field_size: usize,
    levels: usize,
) -> Result<Tensor> {
    haar_convolve_impl(signal, kernel, field_size, levels)
}

/// Haar domain scaling: `IHaar(Haar(signal) * haar_coeffs)`.
///
/// Unlike `haar_convolve_with_grad`, the `haar_coeffs` tensor is already in the
/// Haar (wavelet) domain — no forward transform is applied to it. This allows
/// learning kernel coefficients directly in the Haar basis, giving the model
/// independent control over every wavelet scale without going through the
/// time-domain parameterization (omega/alpha/phi).
///
/// Input shapes:
/// - `signal`: `(batch, field_size)` — time-domain signal
/// - `haar_coeffs`: `(batch, field_size)` — already in Haar domain (learnable)
///
/// Output: `(batch, field_size)` — time-domain result
pub fn haar_scale_with_grad(
    signal: &Tensor,
    haar_coeffs: &Tensor,
    field_size: usize,
    levels: usize,
) -> Result<Tensor> {
    let s_haar = haar_forward_tensor(signal, field_size, levels)?;
    let product = s_haar.broadcast_mul(haar_coeffs)?.contiguous()?;
    haar_inverse_tensor(&product, field_size, levels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_haar_forward_inverse_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;
        let levels = (n as f64).log2() as usize;

        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let x = Tensor::from_vec(data.clone(), (n,), &device)?;

        let forward = haar_forward_tensor(&x, n, levels)?;
        let roundtrip = haar_inverse_tensor(&forward, n, levels)?;
        let result = roundtrip.to_vec1::<f32>()?;

        for i in 0..n {
            assert!(
                (result[i] - data[i]).abs() < 1e-4,
                "Haar roundtrip failed at {}: {} vs {}",
                i, result[i], data[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_candle_haar_basic() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;
        let levels = (n as f64).log2() as usize;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let kernel_data: Vec<f32> = (0..n)
            .map(|i| (-0.1 * i as f32).exp())
            .collect();

        let signal = Tensor::from_vec(signal_data, (n,), &device)?;
        let kernel = Tensor::from_vec(kernel_data, (n,), &device)?;

        let result = haar_convolve(&signal, &kernel, n, levels)?;
        let result_data = result.to_vec1::<f32>()?;

        assert!(
            result_data.iter().all(|v| v.is_finite()),
            "non-finite in Haar output"
        );
        assert!(
            result_data.iter().any(|v| v.abs() > 1e-8),
            "Haar output all zeros"
        );
        Ok(())
    }

    #[test]
    fn test_candle_haar_batched() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;
        let batch = 3;
        let levels = (n as f64).log2() as usize;

        let signal_data: Vec<f32> = (0..batch * n)
            .map(|i| (i as f32 * 0.1).cos())
            .collect();
        let kernel_data: Vec<f32> = (0..batch * n)
            .map(|i| (-0.1 * (i % n) as f32).exp())
            .collect();

        let signal = Tensor::from_vec(signal_data, (batch, n), &device)?;
        let kernel = Tensor::from_vec(kernel_data, (batch, n), &device)?;

        let result = haar_convolve(&signal, &kernel, n, levels)?;
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
    fn test_candle_haar_gradient() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;
        let levels = (n as f64).log2() as usize;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let kernel_data: Vec<f32> = (0..n)
            .map(|i| (-0.1 * i as f32).exp())
            .collect();

        let signal = candle_core::Var::from_vec(signal_data, (n,), &device)?;
        let kernel = candle_core::Var::from_vec(kernel_data, (n,), &device)?;

        let result = haar_convolve_with_grad(signal.as_tensor(), kernel.as_tensor(), n, levels)?;
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
    fn test_haar_gradient_finite_differences() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let levels = (n as f64).log2() as usize;
        let eps = 1e-3f32;

        let signal_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.2, 0.0, -0.1, 0.4, -0.2];
        let kernel_data: Vec<f32> = vec![0.8, 0.3, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0];

        let signal = candle_core::Var::from_vec(signal_data.clone(), (n,), &device)?;
        let kernel = candle_core::Var::from_vec(kernel_data.clone(), (n,), &device)?;

        let result =
            haar_convolve_with_grad(signal.as_tensor(), kernel.as_tensor(), n, levels)?;
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

            let loss_plus =
                haar_convolve(&sp, &k, n, levels)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            let loss_minus =
                haar_convolve(&sm, &k, n, levels)?.sqr()?.sum_all()?.to_scalar::<f32>()?;

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytic = analytic_grad[i];

            assert!(
                (numerical - analytic).abs() < 0.05,
                "Haar gradient mismatch at signal[{}]: numerical={} analytic={}",
                i, numerical, analytic
            );
        }

        Ok(())
    }

    /// Verify haar_scale_with_grad matches manual Haar(signal) * coeffs pipeline
    /// and that it differs from haar_convolve_with_grad (which also transforms the kernel).
    #[test]
    fn test_haar_scale_with_grad_basic() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;
        let levels = (n as f64).log2() as usize;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        // Coefficients already in Haar domain (not a time-domain kernel)
        let haar_coeffs_data: Vec<f32> = (0..n).map(|i| 1.0 - 0.05 * i as f32).collect();

        let signal = Tensor::from_vec(signal_data.clone(), (n,), &device)?;
        let coeffs = Tensor::from_vec(haar_coeffs_data.clone(), (n,), &device)?;

        // haar_scale_with_grad: IHaar(Haar(signal) * coeffs) — coeffs NOT transformed
        let result = haar_scale_with_grad(&signal, &coeffs, n, levels)?;
        let result_data = result.to_vec1::<f32>()?;

        assert!(result_data.iter().all(|v| v.is_finite()), "non-finite in haar_scale output");
        assert!(result_data.iter().any(|v| v.abs() > 1e-8), "haar_scale output all zeros");

        // Manual computation should match: forward transform signal, multiply, inverse
        let s_haar = haar_forward_tensor(&signal, n, levels)?;
        let product = s_haar.broadcast_mul(&coeffs)?.contiguous()?;
        let manual = haar_inverse_tensor(&product, n, levels)?;
        let manual_data = manual.to_vec1::<f32>()?;

        for i in 0..n {
            assert!(
                (result_data[i] - manual_data[i]).abs() < 1e-5,
                "haar_scale vs manual at [{}]: {} vs {}",
                i, result_data[i], manual_data[i]
            );
        }

        // Should differ from haar_convolve_with_grad which transforms both signal AND kernel
        let convolve_result = haar_convolve_with_grad(&signal, &coeffs, n, levels)?;
        let convolve_data = convolve_result.to_vec1::<f32>()?;
        let diff: f32 = result_data.iter().zip(convolve_data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "haar_scale should differ from haar_convolve (diff={})", diff);

        Ok(())
    }

    /// Verify haar_scale_with_grad supports autograd.
    #[test]
    fn test_haar_scale_with_grad_gradient() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let levels = (n as f64).log2() as usize;

        let signal_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.2, 0.0, -0.1, 0.4, -0.2];
        let coeffs_data: Vec<f32> = vec![0.8, 0.5, 0.3, 0.1, 0.9, 0.4, 0.2, 0.0];

        let signal = candle_core::Var::from_vec(signal_data, (n,), &device)?;
        let coeffs = candle_core::Var::from_vec(coeffs_data, (n,), &device)?;

        let result = haar_scale_with_grad(signal.as_tensor(), coeffs.as_tensor(), n, levels)?;
        let loss = result.sqr()?.sum_all()?;
        let grads = loss.backward()?;

        let grad_signal = grads.get(signal.as_tensor());
        let grad_coeffs = grads.get(coeffs.as_tensor());

        assert!(grad_signal.is_some(), "signal should have gradient");
        assert!(grad_coeffs.is_some(), "coeffs should have gradient");

        let gs = grad_signal.unwrap().to_vec1::<f32>()?;
        let gc = grad_coeffs.unwrap().to_vec1::<f32>()?;

        assert!(gs.iter().all(|v| v.is_finite()), "non-finite signal grad");
        assert!(gc.iter().all(|v| v.is_finite()), "non-finite coeffs grad");
        assert!(gs.iter().any(|v| v.abs() > 1e-8), "signal grad all zeros");
        assert!(gc.iter().any(|v| v.abs() > 1e-8), "coeffs grad all zeros");

        Ok(())
    }

    /// Verify tensor-op Haar matches C FFI Haar output.
    #[test]
    fn test_haar_tensor_matches_c_ffi() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;
        let levels = (n as f64).log2() as usize;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17 + 0.3).sin()).collect();
        let kernel_data: Vec<f32> = (0..n).map(|i| (-0.15 * i as f32).exp()).collect();

        // Tensor-op path
        let signal = Tensor::from_vec(signal_data.clone(), (n,), &device)?;
        let kernel = Tensor::from_vec(kernel_data.clone(), (n,), &device)?;
        let tensor_result = haar_convolve(&signal, &kernel, n, levels)?.to_vec1::<f32>()?;

        // C FFI path
        let mut c_output = vec![0.0f32; n];
        ternary_kernels::haar::haar_convolve_f32_buf(
            &signal_data, &kernel_data, &mut c_output, levels,
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
