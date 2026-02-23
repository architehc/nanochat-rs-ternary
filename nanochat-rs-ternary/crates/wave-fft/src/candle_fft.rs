//! Candle-compatible FFT convolution with autograd support.
//!
//! Implements `CustomOp2` so gradients flow through FFT convolution during training.

use candle_core::{CpuStorage, Layout, Result, Shape, Tensor};
use rustfft::{num_complex::Complex as RustFftComplex, FftPlanner};
use std::cell::RefCell;

thread_local! {
    static PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

fn fft_forward_buf(data: &mut [RustFftComplex<f32>]) {
    let n = data.len();
    PLANNER.with(|p| {
        let fft = p.borrow_mut().plan_fft_forward(n);
        fft.process(data);
    });
}

fn fft_inverse_buf(data: &mut [RustFftComplex<f32>]) {
    let n = data.len();
    PLANNER.with(|p| {
        let fft = p.borrow_mut().plan_fft_inverse(n);
        fft.process(data);
    });
    let scale = 1.0 / n as f32;
    for c in data.iter_mut() {
        c.re *= scale;
        c.im *= scale;
    }
}

/// Perform real-signal FFT convolution on raw f32 slices.
///
/// Returns the first `n` elements of the linear convolution.
fn convolve_raw(signal: &[f32], kernel: &[f32], n: usize) -> Vec<f32> {
    let fft_size = next_power_of_two(2 * n);

    let mut sig_c: Vec<RustFftComplex<f32>> = signal
        .iter()
        .map(|&v| RustFftComplex::new(v, 0.0))
        .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    let mut ker_c: Vec<RustFftComplex<f32>> = kernel
        .iter()
        .map(|&v| RustFftComplex::new(v, 0.0))
        .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    fft_forward_buf(&mut sig_c);
    fft_forward_buf(&mut ker_c);

    for i in 0..fft_size {
        sig_c[i] = sig_c[i] * ker_c[i];
    }

    fft_inverse_buf(&mut sig_c);

    sig_c[..n].iter().map(|c| c.re).collect()
}

/// Correlation (not convolution): conj(FFT(a)) * FFT(b).
/// Used for gradient computation.
fn correlate_raw(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let fft_size = next_power_of_two(2 * n);

    let mut a_c: Vec<RustFftComplex<f32>> = a
        .iter()
        .map(|&v| RustFftComplex::new(v, 0.0))
        .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    let mut b_c: Vec<RustFftComplex<f32>> = b
        .iter()
        .map(|&v| RustFftComplex::new(v, 0.0))
        .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    fft_forward_buf(&mut a_c);
    fft_forward_buf(&mut b_c);

    // conj(a) * b
    for i in 0..fft_size {
        let conj_a = RustFftComplex::new(a_c[i].re, -a_c[i].im);
        a_c[i] = conj_a * b_c[i];
    }

    fft_inverse_buf(&mut a_c);

    a_c[..n].iter().map(|c| c.re).collect()
}

/// Candle CustomOp2 for differentiable FFT convolution.
///
/// Forward: output = IFFT(FFT(signal) * FFT(kernel))[:n]
/// Backward w.r.t. signal: IFFT(FFT(grad_output) * conj(FFT(kernel)))[:n]
/// Backward w.r.t. kernel: IFFT(conj(FFT(signal)) * FFT(grad_output))[:n]
struct FftConvolveOp {
    field_size: usize,
}

impl candle_core::CustomOp2 for FftConvolveOp {
    fn name(&self) -> &'static str {
        "fft_convolve"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let signal = s1.as_slice::<f32>()?;
        let kernel = s2.as_slice::<f32>()?;

        let n = self.field_size;
        let signal_total = l1.shape().elem_count();
        let kernel_total = l2.shape().elem_count();

        // signal: [batch_dims..., field_size]
        // kernel: [batch_dims..., field_size]
        let batch = signal_total / n;
        let kernel_batch = kernel_total / n;

        assert_eq!(signal_total, batch * n, "signal not divisible by field_size");

        let mut output = vec![0.0f32; batch * n];

        if kernel_batch == batch {
            // Per-element batched convolution
            for b in 0..batch {
                let sig_start = l1.start_offset() + b * n;
                let ker_start = l2.start_offset() + b * n;
                let sig = &signal[sig_start..sig_start + n];
                let ker = &kernel[ker_start..ker_start + n];
                let result = convolve_raw(sig, ker, n);
                output[b * n..(b + 1) * n].copy_from_slice(&result);
            }
        } else if kernel_batch == 1 {
            // Broadcast kernel across batch
            let ker_start = l2.start_offset();
            let ker = &kernel[ker_start..ker_start + n];
            for b in 0..batch {
                let sig_start = l1.start_offset() + b * n;
                let sig = &signal[sig_start..sig_start + n];
                let result = convolve_raw(sig, ker, n);
                output[b * n..(b + 1) * n].copy_from_slice(&result);
            }
        } else {
            return Err(candle_core::Error::Msg(format!(
                "fft_convolve: incompatible batch sizes signal={} kernel={}",
                batch, kernel_batch
            )));
        }

        let output_shape = l1.shape().clone();
        Ok((CpuStorage::F32(output).into(), output_shape))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _ = res; // unused but required by trait

        let n = self.field_size;

        // d/d_signal = IFFT(conj(FFT(kernel)) * FFT(grad_output))[:n]
        // = correlate(kernel, grad_output)
        let mut grad_signal = fft_correlate_tensor(arg2, grad_res, n)?;

        // d/d_kernel = IFFT(conj(FFT(signal)) * FFT(grad_output))[:n]
        // = correlate(signal, grad_output)
        let mut grad_kernel = fft_correlate_tensor(arg1, grad_res, n)?;

        // When inputs were broadcast (e.g., kernel [n] with signal [batch, n]),
        // the gradient must be summed over broadcast dimensions to match input shape.
        while grad_signal.dims().len() > arg1.dims().len() {
            grad_signal = grad_signal.sum(0)?;
        }
        while grad_kernel.dims().len() > arg2.dims().len() {
            grad_kernel = grad_kernel.sum(0)?;
        }

        Ok((Some(grad_signal), Some(grad_kernel)))
    }
}

/// Compute correlation: IFFT(conj(FFT(a)) * FFT(b))[:n] element-wise over batches.
fn fft_correlate_tensor(a: &Tensor, b: &Tensor, n: usize) -> Result<Tensor> {
    let a_data = a.flatten_all()?.to_vec1::<f32>()?;
    let b_data = b.flatten_all()?.to_vec1::<f32>()?;

    let a_total = a_data.len();
    let b_total = b_data.len();
    let a_batch = a_total / n;
    let b_batch = b_total / n;

    let batch = a_batch.max(b_batch);
    let mut output = vec![0.0f32; batch * n];

    for bidx in 0..batch {
        let a_idx = if a_batch == 1 { 0 } else { bidx };
        let b_idx = if b_batch == 1 { 0 } else { bidx };
        let a_slice = &a_data[a_idx * n..(a_idx + 1) * n];
        let b_slice = &b_data[b_idx * n..(b_idx + 1) * n];
        let result = correlate_raw(a_slice, b_slice, n);
        output[bidx * n..(bidx + 1) * n].copy_from_slice(&result);
    }

    // Reshape to match the larger tensor's shape
    let target_shape = if a_batch >= b_batch {
        a.shape().clone()
    } else {
        b.shape().clone()
    };
    Tensor::from_vec(output, target_shape, a.device())
}

/// High-level API: differentiable FFT convolution for Candle tensors.
///
/// # Arguments
/// * `signal` - Tensor with last dim = `field_size`, shape `[..., field_size]`
/// * `kernel` - Tensor with last dim = `field_size`, shape `[..., field_size]` or `[field_size]`
/// * `field_size` - Length of signal/kernel along last dimension
///
/// # Returns
/// Convolution result with same shape as `signal`
pub fn fft_convolve(signal: &Tensor, kernel: &Tensor, field_size: usize) -> Result<Tensor> {
    signal.apply_op2_no_bwd(kernel, &FftConvolveOp { field_size })
}

/// Same as `fft_convolve` but preserves autograd graph for backward pass.
pub fn fft_convolve_with_grad(
    signal: &Tensor,
    kernel: &Tensor,
    field_size: usize,
) -> Result<Tensor> {
    signal.apply_op2(kernel, FftConvolveOp { field_size })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_candle_fft_basic() -> Result<()> {
        let device = Device::Cpu;
        let n = 32;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let mut kernel_data = vec![0.0f32; n];
        kernel_data[0] = 1.0; // delta

        let signal = Tensor::from_vec(signal_data.clone(), (n,), &device)?;
        let kernel = Tensor::from_vec(kernel_data, (n,), &device)?;

        let result = fft_convolve(&signal, &kernel, n)?;
        let result_data = result.to_vec1::<f32>()?;

        for i in 0..n {
            assert!(
                (result_data[i] - signal_data[i]).abs() < 1e-4,
                "candle FFT delta conv failed at {}: {} vs {}",
                i,
                result_data[i],
                signal_data[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_candle_fft_batched() -> Result<()> {
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

        let result = fft_convolve(&signal, &kernel, n)?;
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
    fn test_candle_fft_gradient() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;

        let signal_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let kernel_data: Vec<f32> = (0..n)
            .map(|i| (-0.1 * i as f32).exp())
            .collect();

        let signal = candle_core::Var::from_vec(signal_data, (n,), &device)?;
        let kernel = candle_core::Var::from_vec(kernel_data, (n,), &device)?;

        let result = fft_convolve_with_grad(signal.as_tensor(), kernel.as_tensor(), n)?;
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
        assert!(
            gs.iter().any(|v| v.abs() > 1e-8),
            "signal grad is all zeros"
        );
        assert!(
            gk.iter().any(|v| v.abs() > 1e-8),
            "kernel grad is all zeros"
        );

        Ok(())
    }

    #[test]
    fn test_gradient_finite_differences() -> Result<()> {
        // Verify autograd is numerically correct via finite differences
        let device = Device::Cpu;
        let n = 8;
        let eps = 1e-3f32;

        let signal_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.2, 0.0, -0.1, 0.4, -0.2];
        let kernel_data: Vec<f32> = vec![0.8, 0.3, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0];

        let signal = candle_core::Var::from_vec(signal_data.clone(), (n,), &device)?;
        let kernel = candle_core::Var::from_vec(kernel_data.clone(), (n,), &device)?;

        // Compute analytic gradient
        let result = fft_convolve_with_grad(signal.as_tensor(), kernel.as_tensor(), n)?;
        let loss = result.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        let analytic_grad_signal = grads
            .get(signal.as_tensor())
            .unwrap()
            .to_vec1::<f32>()?;

        // Compute numerical gradient for signal via finite differences
        for i in 0..n.min(4) {
            // Test first 4 elements for speed
            let mut sig_plus = signal_data.clone();
            sig_plus[i] += eps;
            let mut sig_minus = signal_data.clone();
            sig_minus[i] -= eps;

            let sp = Tensor::from_vec(sig_plus, (n,), &device)?;
            let sm = Tensor::from_vec(sig_minus, (n,), &device)?;
            let k = Tensor::from_vec(kernel_data.clone(), (n,), &device)?;

            let loss_plus = fft_convolve(&sp, &k, n)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            let loss_minus = fft_convolve(&sm, &k, n)?.sqr()?.sum_all()?.to_scalar::<f32>()?;

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytic = analytic_grad_signal[i];

            assert!(
                (numerical - analytic).abs() < 0.05,
                "gradient mismatch at signal[{}]: numerical={} analytic={}",
                i,
                numerical,
                analytic
            );
        }

        Ok(())
    }
}
