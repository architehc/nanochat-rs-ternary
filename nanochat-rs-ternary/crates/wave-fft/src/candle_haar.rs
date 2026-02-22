//! Candle-compatible Haar DWT convolution with autograd support.
//!
//! Haar DWT is orthogonal: inverse = transpose (adjoint).
//! The backward pass has the same structure as the forward pass.

use candle_core::{CpuStorage, Layout, Result, Shape, Tensor};
use ternary_kernels::haar;

/// Raw Haar convolution on f32 slices.
fn convolve_raw(signal: &[f32], kernel: &[f32], n: usize, levels: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    haar::haar_convolve_f32_buf(&signal[..n], &kernel[..n], &mut output, levels);
    output
}

/// Haar correlation for gradient computation.
///
/// Haar is orthogonal, so correlation == convolution with the adjoint.
/// For pointwise-multiply-in-wavelet-domain, gradients flow through
/// the same Haar convolution structure.
fn correlate_raw(a: &[f32], b: &[f32], n: usize, levels: usize) -> Vec<f32> {
    convolve_raw(a, b, n, levels)
}

/// Candle CustomOp2 for differentiable Haar convolution.
struct HaarConvolveOp {
    field_size: usize,
    levels: usize,
}

impl candle_core::CustomOp2 for HaarConvolveOp {
    fn name(&self) -> &'static str {
        "haar_convolve"
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
        let levels = self.levels;
        let signal_total = l1.shape().elem_count();
        let kernel_total = l2.shape().elem_count();

        let batch = signal_total / n;
        let kernel_batch = kernel_total / n;

        assert_eq!(signal_total, batch * n, "signal not divisible by field_size");

        let mut output = vec![0.0f32; batch * n];

        if kernel_batch == batch {
            for b in 0..batch {
                let sig_start = l1.start_offset() + b * n;
                let ker_start = l2.start_offset() + b * n;
                let sig = &signal[sig_start..sig_start + n];
                let ker = &kernel[ker_start..ker_start + n];
                let result = convolve_raw(sig, ker, n, levels);
                output[b * n..(b + 1) * n].copy_from_slice(&result);
            }
        } else if kernel_batch == 1 {
            let ker_start = l2.start_offset();
            let ker = &kernel[ker_start..ker_start + n];
            for b in 0..batch {
                let sig_start = l1.start_offset() + b * n;
                let sig = &signal[sig_start..sig_start + n];
                let result = convolve_raw(sig, ker, n, levels);
                output[b * n..(b + 1) * n].copy_from_slice(&result);
            }
        } else {
            return Err(candle_core::Error::Msg(format!(
                "haar_convolve: incompatible batch sizes signal={} kernel={}",
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
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let n = self.field_size;
        let levels = self.levels;

        let mut grad_signal = haar_correlate_tensor(arg2, grad_res, n, levels)?;
        let mut grad_kernel = haar_correlate_tensor(arg1, grad_res, n, levels)?;

        while grad_signal.dims().len() > arg1.dims().len() {
            grad_signal = grad_signal.sum(0)?;
        }
        while grad_kernel.dims().len() > arg2.dims().len() {
            grad_kernel = grad_kernel.sum(0)?;
        }

        Ok((Some(grad_signal), Some(grad_kernel)))
    }
}

/// Compute Haar correlation tensor.
fn haar_correlate_tensor(a: &Tensor, b: &Tensor, n: usize, levels: usize) -> Result<Tensor> {
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
        let result = correlate_raw(a_slice, b_slice, n, levels);
        output[bidx * n..(bidx + 1) * n].copy_from_slice(&result);
    }

    let target_shape = if a_batch >= b_batch {
        a.shape().clone()
    } else {
        b.shape().clone()
    };
    Tensor::from_vec(output, target_shape, a.device())
}

/// High-level API: Haar convolution without gradient tracking.
pub fn haar_convolve(
    signal: &Tensor,
    kernel: &Tensor,
    field_size: usize,
    levels: usize,
) -> Result<Tensor> {
    signal.apply_op2_no_bwd(kernel, &HaarConvolveOp { field_size, levels })
}

/// Haar convolution preserving autograd graph for backward pass.
pub fn haar_convolve_with_grad(
    signal: &Tensor,
    kernel: &Tensor,
    field_size: usize,
    levels: usize,
) -> Result<Tensor> {
    signal.apply_op2(kernel, HaarConvolveOp { field_size, levels })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
                (numerical - analytic).abs() < 0.5,
                "Haar gradient mismatch at signal[{}]: numerical={} analytic={}",
                i, numerical, analytic
            );
        }

        Ok(())
    }
}
