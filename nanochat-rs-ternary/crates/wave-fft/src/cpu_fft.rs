//! Pure-Rust FFT convolution using rustfft.
//!
//! Provides O(n log n) convolution for wave field attention kernels.

use num_complex::Complex;
use rustfft::{FftPlanner, num_complex::Complex as RustFftComplex};
use std::cell::RefCell;

// Thread-local FFT planner cache to avoid re-planning for the same sizes.
thread_local! {
    static PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

/// Convert num_complex to rustfft complex (they're the same layout but different types).
#[inline]
fn to_rustfft(c: Complex<f32>) -> RustFftComplex<f32> {
    RustFftComplex::new(c.re, c.im)
}

#[inline]
fn from_rustfft(c: RustFftComplex<f32>) -> Complex<f32> {
    Complex::new(c.re, c.im)
}

/// Compute next power of 2 >= n.
fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

/// In-place forward FFT.
fn fft_forward(data: &mut [RustFftComplex<f32>]) {
    let n = data.len();
    PLANNER.with(|p| {
        let fft = p.borrow_mut().plan_fft_forward(n);
        fft.process(data);
    });
}

/// In-place inverse FFT (includes 1/N normalization).
fn fft_inverse(data: &mut [RustFftComplex<f32>]) {
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

/// FFT convolution of two real signals.
///
/// Zero-pads both to `2 * n`, FFTs both, pointwise multiplies, IFFTs,
/// and returns the first `n` elements (linear convolution, not circular).
///
/// # Arguments
/// * `signal` - Input signal of length `n`
/// * `kernel` - Convolution kernel of length <= `n`
/// * `n` - Length of signal (output will also be length `n`)
pub fn fft_convolve(signal: &[f32], kernel: &[f32], n: usize) -> Vec<f32> {
    assert!(signal.len() >= n, "signal length {} < n={}", signal.len(), n);
    assert!(kernel.len() <= n, "kernel length {} > n={}", kernel.len(), n);

    let fft_size = next_power_of_two(2 * n);

    // Zero-pad signal
    let mut sig_c: Vec<RustFftComplex<f32>> = signal[..n]
        .iter()
        .map(|&v| RustFftComplex::new(v, 0.0))
        .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    // Zero-pad kernel
    let mut ker_c: Vec<RustFftComplex<f32>> = kernel
        .iter()
        .map(|&v| RustFftComplex::new(v, 0.0))
        .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    // Forward FFT both
    fft_forward(&mut sig_c);
    fft_forward(&mut ker_c);

    // Pointwise multiply
    for i in 0..fft_size {
        sig_c[i] = sig_c[i] * ker_c[i];
    }

    // Inverse FFT
    fft_inverse(&mut sig_c);

    // Return first n real elements
    sig_c[..n].iter().map(|c| c.re).collect()
}

/// FFT convolution with a pre-computed kernel in frequency domain.
///
/// For inference, the wave field kernel can be FFT'd once at model load time.
/// This avoids re-FFT-ing the kernel for every token step.
///
/// # Arguments
/// * `signal` - Input signal of length `n`
/// * `kernel_freq` - Pre-FFT'd kernel (length = fft_size)
/// * `n` - Length of signal
pub fn fft_convolve_precomputed(
    signal: &[f32],
    kernel_freq: &[Complex<f32>],
    n: usize,
) -> Vec<f32> {
    let fft_size = kernel_freq.len();
    assert!(fft_size >= 2 * n, "kernel_freq too short");

    // Zero-pad signal
    let mut sig_c: Vec<RustFftComplex<f32>> = signal[..n]
        .iter()
        .map(|&v| RustFftComplex::new(v, 0.0))
        .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    // Forward FFT signal
    fft_forward(&mut sig_c);

    // Pointwise multiply with precomputed kernel spectrum
    for i in 0..fft_size {
        let k = to_rustfft(kernel_freq[i]);
        sig_c[i] = sig_c[i] * k;
    }

    // Inverse FFT
    fft_inverse(&mut sig_c);

    sig_c[..n].iter().map(|c| c.re).collect()
}

/// Pre-compute the FFT of a kernel for use with `fft_convolve_precomputed`.
///
/// # Arguments
/// * `kernel` - Time-domain kernel of length <= `n`
/// * `n` - Signal length this kernel will be used with
///
/// # Returns
/// Frequency-domain kernel of length `fft_size = next_pow2(2*n)`
pub fn precompute_kernel_fft(kernel: &[f32], n: usize) -> Vec<Complex<f32>> {
    let fft_size = next_power_of_two(2 * n);

    let mut ker_c: Vec<RustFftComplex<f32>> = kernel
        .iter()
        .map(|&v| RustFftComplex::new(v, 0.0))
        .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    fft_forward(&mut ker_c);

    ker_c.iter().map(|c| from_rustfft(*c)).collect()
}

/// Batched FFT convolution: process `batch` independent signal/kernel pairs.
///
/// Each signal and kernel is length `n`. Signals and kernels are packed contiguously.
///
/// # Arguments
/// * `signals` - `[batch * n]` packed signals
/// * `kernels` - `[batch * n]` packed kernels
/// * `n` - Length of each signal/kernel
/// * `batch` - Number of pairs
///
/// # Returns
/// `[batch * n]` packed convolution results
pub fn fft_convolve_batch(
    signals: &[f32],
    kernels: &[f32],
    n: usize,
    batch: usize,
) -> Vec<f32> {
    assert_eq!(signals.len(), batch * n);
    assert_eq!(kernels.len(), batch * n);

    let mut output = vec![0.0f32; batch * n];

    for b in 0..batch {
        let sig = &signals[b * n..(b + 1) * n];
        let ker = &kernels[b * n..(b + 1) * n];
        let result = fft_convolve(sig, ker, n);
        output[b * n..(b + 1) * n].copy_from_slice(&result);
    }

    output
}

/// Batched FFT convolution with precomputed frequency-domain kernels.
///
/// # Arguments
/// * `signals` - `[batch * n]` packed signals
/// * `kernels_freq` - `[batch * fft_size]` packed precomputed kernel FFTs
/// * `n` - Length of each signal
/// * `batch` - Number of pairs
/// * `fft_size` - Size of each precomputed kernel FFT
pub fn fft_convolve_batch_precomputed(
    signals: &[f32],
    kernels_freq: &[Complex<f32>],
    n: usize,
    batch: usize,
    fft_size: usize,
) -> Vec<f32> {
    assert_eq!(signals.len(), batch * n);
    assert_eq!(kernels_freq.len(), batch * fft_size);

    let mut output = vec![0.0f32; batch * n];

    for b in 0..batch {
        let sig = &signals[b * n..(b + 1) * n];
        let kf = &kernels_freq[b * fft_size..(b + 1) * fft_size];
        let result = fft_convolve_precomputed(sig, kf, n);
        output[b * n..(b + 1) * n].copy_from_slice(&result);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_roundtrip() {
        // IFFT(FFT(x)) should approximately equal x
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();

        let fft_size = next_power_of_two(n);
        let mut buf: Vec<RustFftComplex<f32>> = x
            .iter()
            .map(|&v| RustFftComplex::new(v, 0.0))
            .chain(std::iter::repeat(RustFftComplex::new(0.0, 0.0)))
            .take(fft_size)
            .collect();

        fft_forward(&mut buf);
        fft_inverse(&mut buf);

        for i in 0..n {
            assert!(
                (buf[i].re - x[i]).abs() < 1e-5,
                "FFT roundtrip failed at {}: {} vs {}",
                i,
                buf[i].re,
                x[i]
            );
            assert!(
                buf[i].im.abs() < 1e-5,
                "imaginary part should be ~0, got {}",
                buf[i].im
            );
        }
    }

    #[test]
    fn test_convolve_with_delta() {
        // Convolving with delta function [1, 0, 0, ...] should return the signal
        let n = 32;
        let signal: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut kernel = vec![0.0f32; n];
        kernel[0] = 1.0;

        let result = fft_convolve(&signal, &kernel, n);

        for i in 0..n {
            assert!(
                (result[i] - signal[i]).abs() < 1e-4,
                "delta convolution failed at {}: {} vs {}",
                i,
                result[i],
                signal[i]
            );
        }
    }

    #[test]
    fn test_convolve_with_shifted_delta() {
        // Convolving with delta at position 1 should shift the signal by 1
        let n = 32;
        let signal: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut kernel = vec![0.0f32; n];
        kernel[1] = 1.0;

        let result = fft_convolve(&signal, &kernel, n);

        // result[i] should be signal[i-1] (with wraparound handled by zero-padding)
        assert!((result[0] - 0.0).abs() < 1e-4, "shifted result[0] should be ~0");
        for i in 1..n {
            assert!(
                (result[i] - signal[i - 1]).abs() < 1e-4,
                "shifted delta at {}: {} vs {}",
                i,
                result[i],
                signal[i - 1]
            );
        }
    }

    #[test]
    fn test_precomputed_matches_direct() {
        let n = 64;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
        let kernel: Vec<f32> = (0..n)
            .map(|i| (-0.1 * i as f32).exp() * (0.5 * i as f32).cos())
            .collect();

        let direct = fft_convolve(&signal, &kernel, n);
        let kernel_freq = precompute_kernel_fft(&kernel, n);
        let precomputed = fft_convolve_precomputed(&signal, &kernel_freq, n);

        for i in 0..n {
            assert!(
                (direct[i] - precomputed[i]).abs() < 1e-4,
                "precomputed mismatch at {}: {} vs {}",
                i,
                direct[i],
                precomputed[i]
            );
        }
    }

    #[test]
    fn test_batch_matches_sequential() {
        let n = 32;
        let batch = 4;

        let signals: Vec<f32> = (0..batch * n)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let kernels: Vec<f32> = (0..batch * n)
            .map(|i| (-0.05 * i as f32).exp())
            .collect();

        let batch_result = fft_convolve_batch(&signals, &kernels, n, batch);

        for b in 0..batch {
            let sig = &signals[b * n..(b + 1) * n];
            let ker = &kernels[b * n..(b + 1) * n];
            let sequential = fft_convolve(sig, ker, n);

            for i in 0..n {
                assert!(
                    (batch_result[b * n + i] - sequential[i]).abs() < 1e-4,
                    "batch {} element {} mismatch: {} vs {}",
                    b,
                    i,
                    batch_result[b * n + i],
                    sequential[i]
                );
            }
        }
    }

    #[test]
    fn test_wave_kernel_convolution() {
        // Test with a damped cosine kernel (the actual wave field kernel shape)
        let n = 128;
        let alpha = 0.1f32;
        let omega = 2.0f32;
        let phi = 0.0f32;

        let kernel: Vec<f32> = (0..n)
            .map(|t| {
                let t = t as f32;
                (-alpha * t).exp() * (omega * t + phi).cos()
            })
            .collect();

        // Simple pulse signal
        let mut signal = vec![0.0f32; n];
        signal[0] = 1.0;

        let result = fft_convolve(&signal, &kernel, n);

        // Convolving a pulse with the kernel should return the kernel
        for i in 0..n {
            assert!(
                (result[i] - kernel[i]).abs() < 1e-4,
                "wave kernel convolution failed at {}: {} vs {}",
                i,
                result[i],
                kernel[i]
            );
        }

        // Verify kernel decays (energy should decrease)
        let first_quarter_energy: f32 = kernel[..n / 4].iter().map(|v| v * v).sum();
        let last_quarter_energy: f32 = kernel[3 * n / 4..].iter().map(|v| v * v).sum();
        assert!(
            first_quarter_energy > last_quarter_energy,
            "wave kernel should decay: first_quarter={} last_quarter={}",
            first_quarter_energy,
            last_quarter_energy
        );
    }
}
