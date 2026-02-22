//! Haar wavelet-basis scaling for wave field attention — **inference path**.
//!
//! Diagonal operator in the Haar basis: each wavelet coefficient is scaled
//! independently via `IHaar(Haar(signal) ⊙ Haar(kernel))`. This is
//! scale-selective filtering (different frequency bands attenuated
//! independently), NOT shift-invariant convolution and NOT XOR convolution.
//!
//! **Inference:** Kernel transforms are precomputed at model load (`precompute_kernel_haar`).
//! Per-token cost is O(N log N) for forward + inverse Haar + O(N) pointwise multiply.
//! No heap allocation in the precomputed path beyond the output `Vec`.
//!
//! Multi-scale, localized. Uses only additions and subtractions in transform stages.

use ternary_kernels::haar;

/// Haar convolution: forward DWT, pointwise multiply, inverse DWT.
///
/// Both signal and kernel must be length `n` (power of 2).
/// `levels` is the number of decomposition levels (max: log2(n)).
pub fn haar_convolve(signal: &[f32], kernel: &[f32], n: usize, levels: usize) -> Vec<f32> {
    assert!(n.is_power_of_two(), "Haar requires power-of-2, got {}", n);
    assert!(signal.len() >= n);
    assert!(kernel.len() >= n);

    let mut output = vec![0.0f32; n];
    haar::haar_convolve_f32_buf(&signal[..n], &kernel[..n], &mut output, levels);
    output
}

/// Pre-compute Haar transform of a kernel for use with `haar_convolve_precomputed`.
pub fn precompute_kernel_haar(kernel: &[f32], n: usize, levels: usize) -> Vec<f32> {
    assert!(n.is_power_of_two(), "Haar requires power-of-2, got {}", n);

    let mut result = vec![0.0f32; n];
    result[..kernel.len().min(n)].copy_from_slice(&kernel[..kernel.len().min(n)]);
    haar::haar_forward_f32_inplace(&mut result, levels);
    result
}

/// Haar convolution with a pre-transformed kernel.
///
/// Avoids re-transforming the kernel for every call.
pub fn haar_convolve_precomputed(
    signal: &[f32],
    kernel_haar: &[f32],
    n: usize,
    levels: usize,
) -> Vec<f32> {
    assert!(n.is_power_of_two(), "Haar requires power-of-2, got {}", n);
    assert!(signal.len() >= n);
    assert_eq!(kernel_haar.len(), n);

    // Transform signal
    let mut sig_t = vec![0.0f32; n];
    sig_t.copy_from_slice(&signal[..n]);
    haar::haar_forward_f32_inplace(&mut sig_t, levels);

    // Pointwise multiply in Haar domain
    let mut output = vec![0.0f32; n];
    for i in 0..n {
        output[i] = sig_t[i] * kernel_haar[i];
    }

    // Inverse Haar
    haar::haar_inverse_f32_inplace(&mut output, levels);

    output
}

/// Batched Haar convolution with precomputed kernels.
///
/// signals: [batch * n] packed signals
/// kernels_haar: [batch * n] packed pre-Haar'd kernels
pub fn haar_convolve_batch_precomputed(
    signals: &[f32],
    kernels_haar: &[f32],
    n: usize,
    batch: usize,
    levels: usize,
) -> Vec<f32> {
    assert_eq!(signals.len(), batch * n);
    assert_eq!(kernels_haar.len(), batch * n);

    let mut output = vec![0.0f32; batch * n];

    for b in 0..batch {
        let sig = &signals[b * n..(b + 1) * n];
        let kh = &kernels_haar[b * n..(b + 1) * n];
        let result = haar_convolve_precomputed(sig, kh, n, levels);
        output[b * n..(b + 1) * n].copy_from_slice(&result);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_convolve_finite() {
        let n = 32;
        let levels = (n as f64).log2() as usize;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let kernel: Vec<f32> = (0..n)
            .map(|i| (-0.1 * i as f32).exp())
            .collect();

        let result = haar_convolve(&signal, &kernel, n, levels);

        assert!(
            result.iter().all(|v| v.is_finite()),
            "Haar convolve produced non-finite output"
        );
        assert!(
            result.iter().any(|v| v.abs() > 1e-8),
            "Haar convolve produced all-zero output"
        );
    }

    #[test]
    fn test_precomputed_matches_direct() {
        let n = 32;
        let levels = (n as f64).log2() as usize;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
        let kernel: Vec<f32> = (0..n)
            .map(|i| (-0.1 * i as f32).exp() * (0.5 * i as f32).cos())
            .collect();

        let direct = haar_convolve(&signal, &kernel, n, levels);
        let kernel_haar = precompute_kernel_haar(&kernel, n, levels);
        let precomputed = haar_convolve_precomputed(&signal, &kernel_haar, n, levels);

        for i in 0..n {
            assert!(
                (direct[i] - precomputed[i]).abs() < 1e-3,
                "Haar precomputed mismatch at {}: {} vs {}",
                i, direct[i], precomputed[i]
            );
        }
    }

    #[test]
    fn test_batch_matches_sequential() {
        let n = 16;
        let batch = 3;
        let levels = (n as f64).log2() as usize;

        let signals: Vec<f32> = (0..batch * n)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let kernels: Vec<f32> = (0..batch * n)
            .map(|i| (-0.05 * (i % n) as f32).exp())
            .collect();

        let mut kernels_haar = vec![0.0f32; batch * n];
        for b in 0..batch {
            let kh = precompute_kernel_haar(&kernels[b * n..(b + 1) * n], n, levels);
            kernels_haar[b * n..(b + 1) * n].copy_from_slice(&kh);
        }

        let batch_result =
            haar_convolve_batch_precomputed(&signals, &kernels_haar, n, batch, levels);

        for b in 0..batch {
            let sig = &signals[b * n..(b + 1) * n];
            let kh = &kernels_haar[b * n..(b + 1) * n];
            let sequential = haar_convolve_precomputed(sig, kh, n, levels);

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
    fn test_wave_kernel_haar() {
        let n = 64;
        let levels = (n as f64).log2() as usize;
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

        let result = haar_convolve(&signal, &kernel, n, levels);

        assert!(
            result.iter().all(|v| v.is_finite()),
            "Haar wave kernel produced non-finite output"
        );
        assert!(
            result.iter().any(|v| v.abs() > 1e-6),
            "Haar wave kernel produced all-zero output"
        );
    }
}
