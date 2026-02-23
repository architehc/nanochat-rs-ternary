//! XOR convolution (dyadic convolution) for wave field attention — **inference path**.
//!
//! Computes `c[k] = (1/N) Σ_i a[i] · b[i ⊕ k]` where `⊕` is bitwise XOR.
//! This is NOT circular (shift) convolution. WHT diagonalizes XOR convolution
//! the same way DFT diagonalizes shift convolution — the mixing pattern is
//! pair-swaps at power-of-2 distances, not cyclic shifts.
//!
//! **Inference:** Kernel transforms are precomputed at model load (`precompute_kernel_fwht`).
//! Per-token cost is O(N log N) for the signal transform + O(N) pointwise multiply.
//! No heap allocation in the precomputed path beyond the output `Vec`.
//!
//! Uses only additions and subtractions in transform stages — integer-only compatible.
//! FWHT is self-inverse (up to 1/N scaling).

use ternary_kernels::fwht;

/// FWHT convolution of two f32 signals.
///
/// Both signal and kernel must be length `n` (power of 2).
/// Returns convolution result of length `n`.
pub fn fwht_convolve(signal: &[f32], kernel: &[f32], n: usize) -> Vec<f32> {
    assert!(n.is_power_of_two(), "FWHT requires power-of-2, got {}", n);
    assert!(signal.len() >= n);
    assert!(kernel.len() >= n);

    let mut output = vec![0.0f32; n];
    fwht::fwht_convolve_f32_buf(&signal[..n], &kernel[..n], &mut output);
    output
}

/// Pre-compute FWHT of a kernel for use with `fwht_convolve_precomputed`.
///
/// For FWHT, the pre-transformed kernel is just FWHT(kernel).
/// Returns f32 coefficients (not Complex like FFT).
pub fn precompute_kernel_fwht(kernel: &[f32], n: usize) -> Vec<f32> {
    assert!(n.is_power_of_two(), "FWHT requires power-of-2, got {}", n);

    let mut result = vec![0.0f32; n];
    result[..kernel.len().min(n)].copy_from_slice(&kernel[..kernel.len().min(n)]);
    fwht::fwht_inplace_f32(&mut result);
    result
}

/// FWHT convolution with a pre-transformed kernel.
///
/// Avoids re-transforming the kernel for every call.
pub fn fwht_convolve_precomputed(
    signal: &[f32],
    kernel_fwht: &[f32],
    n: usize,
) -> Vec<f32> {
    assert!(n.is_power_of_two(), "FWHT requires power-of-2, got {}", n);
    assert!(signal.len() >= n);
    assert_eq!(kernel_fwht.len(), n);

    // Transform signal
    let mut sig_t = vec![0.0f32; n];
    sig_t.copy_from_slice(&signal[..n]);
    fwht::fwht_inplace_f32(&mut sig_t);

    // Pointwise multiply in FWHT domain
    let mut output = vec![0.0f32; n];
    for i in 0..n {
        output[i] = sig_t[i] * kernel_fwht[i];
    }

    // Inverse FWHT (same as forward, then scale by 1/N)
    fwht::fwht_inplace_f32(&mut output);
    let inv_n = 1.0 / n as f32;
    for v in output.iter_mut() {
        *v *= inv_n;
    }

    output
}

/// Batched FWHT convolution with precomputed kernels.
///
/// signals: [batch * n] packed signals
/// kernels_fwht: [batch * n] packed pre-FWHT'd kernels
pub fn fwht_convolve_batch_precomputed(
    signals: &[f32],
    kernels_fwht: &[f32],
    n: usize,
    batch: usize,
) -> Vec<f32> {
    assert_eq!(signals.len(), batch * n);
    assert_eq!(kernels_fwht.len(), batch * n);

    let mut output = vec![0.0f32; batch * n];

    for b in 0..batch {
        let sig = &signals[b * n..(b + 1) * n];
        let kf = &kernels_fwht[b * n..(b + 1) * n];
        let result = fwht_convolve_precomputed(sig, kf, n);
        output[b * n..(b + 1) * n].copy_from_slice(&result);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_convolve_delta() {
        let n = 32;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let mut kernel = vec![0.0f32; n];
        kernel[0] = 1.0;

        let result = fwht_convolve(&signal, &kernel, n);

        for i in 0..n {
            assert!(
                (result[i] - signal[i]).abs() < 1e-4,
                "FWHT delta conv failed at {}: {} vs {}",
                i, result[i], signal[i]
            );
        }
    }

    #[test]
    fn test_precomputed_matches_direct() {
        let n = 32;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
        let kernel: Vec<f32> = (0..n)
            .map(|i| (-0.1 * i as f32).exp() * (0.5 * i as f32).cos())
            .collect();

        let direct = fwht_convolve(&signal, &kernel, n);
        let kernel_fwht = precompute_kernel_fwht(&kernel, n);
        let precomputed = fwht_convolve_precomputed(&signal, &kernel_fwht, n);

        for i in 0..n {
            assert!(
                (direct[i] - precomputed[i]).abs() < 1e-4,
                "precomputed mismatch at {}: {} vs {}",
                i, direct[i], precomputed[i]
            );
        }
    }

    #[test]
    fn test_batch_matches_sequential() {
        let n = 16;
        let batch = 4;

        let signals: Vec<f32> = (0..batch * n)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let kernels: Vec<f32> = (0..batch * n)
            .map(|i| (-0.05 * (i % n) as f32).exp())
            .collect();

        // Precompute all kernels
        let mut kernels_fwht = vec![0.0f32; batch * n];
        for b in 0..batch {
            let kf = precompute_kernel_fwht(&kernels[b * n..(b + 1) * n], n);
            kernels_fwht[b * n..(b + 1) * n].copy_from_slice(&kf);
        }

        let batch_result = fwht_convolve_batch_precomputed(&signals, &kernels_fwht, n, batch);

        for b in 0..batch {
            let sig = &signals[b * n..(b + 1) * n];
            let kf = &kernels_fwht[b * n..(b + 1) * n];
            let sequential = fwht_convolve_precomputed(sig, kf, n);

            for i in 0..n {
                assert!(
                    (batch_result[b * n + i] - sequential[i]).abs() < 1e-4,
                    "batch {} element {} mismatch",
                    b, i
                );
            }
        }
    }

    #[test]
    fn test_wave_kernel_fwht() {
        // Test with a damped cosine kernel (actual wave field shape)
        let n = 64;
        let alpha = 0.1f32;
        let omega = 2.0f32;

        let kernel: Vec<f32> = (0..n)
            .map(|t| {
                let t = t as f32;
                (-alpha * t).exp() * (omega * t).cos()
            })
            .collect();

        let mut signal = vec![0.0f32; n];
        signal[0] = 1.0;

        let result = fwht_convolve(&signal, &kernel, n);

        // Output should be finite and non-trivial
        assert!(
            result.iter().all(|v| v.is_finite()),
            "FWHT wave kernel produced non-finite output"
        );
        assert!(
            result.iter().any(|v| v.abs() > 1e-6),
            "FWHT wave kernel produced all-zero output"
        );
    }
}
