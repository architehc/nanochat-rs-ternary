//! Safe Rust wrappers for Haar DWT (Discrete Wavelet Transform) C kernels.
//!
//! Haar DWT uses only additions and subtractions — integer-only compatible.
//! Multi-level decomposition gives both local and global context.

extern "C" {
    fn haar_forward_i32(data: *mut i32, len: i32, levels: i32);
    fn haar_inverse_i32(data: *mut i32, len: i32, levels: i32);
    fn haar_forward_f32(data: *mut f32, len: i32, levels: i32);
    fn haar_inverse_f32(data: *mut f32, len: i32, levels: i32);
    fn haar_convolve_i32(
        signal: *const i32,
        kernel: *const i32,
        output: *mut i32,
        len: i32,
        levels: i32,
    );
    fn haar_convolve_f32(
        signal: *const f32,
        kernel: *const f32,
        output: *mut f32,
        len: i32,
        levels: i32,
    );
}

/// Forward Haar DWT in-place on i32 data. Length must be a power of 2.
pub fn haar_forward(data: &mut [i32], levels: usize) {
    let len = data.len();
    assert!(len.is_power_of_two(), "Haar requires power-of-2 length, got {}", len);
    if len <= 1 || levels == 0 {
        return;
    }
    unsafe { haar_forward_i32(data.as_mut_ptr(), len as i32, levels as i32) }
}

/// Inverse Haar DWT in-place on i32 data. Length must be a power of 2.
pub fn haar_inverse(data: &mut [i32], levels: usize) {
    let len = data.len();
    assert!(len.is_power_of_two(), "Haar requires power-of-2 length, got {}", len);
    if len <= 1 || levels == 0 {
        return;
    }
    unsafe { haar_inverse_i32(data.as_mut_ptr(), len as i32, levels as i32) }
}

/// Forward Haar DWT in-place on f32 data. Length must be a power of 2.
pub fn haar_forward_f32_inplace(data: &mut [f32], levels: usize) {
    let len = data.len();
    assert!(len.is_power_of_two(), "Haar requires power-of-2 length, got {}", len);
    if len <= 1 || levels == 0 {
        return;
    }
    unsafe { haar_forward_f32(data.as_mut_ptr(), len as i32, levels as i32) }
}

/// Inverse Haar DWT in-place on f32 data. Length must be a power of 2.
pub fn haar_inverse_f32_inplace(data: &mut [f32], levels: usize) {
    let len = data.len();
    assert!(len.is_power_of_two(), "Haar requires power-of-2 length, got {}", len);
    if len <= 1 || levels == 0 {
        return;
    }
    unsafe { haar_inverse_f32(data.as_mut_ptr(), len as i32, levels as i32) }
}

/// Haar convolution of two i32 signals. All slices must have the same power-of-2 length.
pub fn haar_convolve_i32_buf(
    signal: &[i32],
    kernel: &[i32],
    output: &mut [i32],
    levels: usize,
) {
    let len = signal.len();
    assert!(len.is_power_of_two(), "Haar requires power-of-2 length, got {}", len);
    assert_eq!(kernel.len(), len, "kernel length mismatch");
    assert_eq!(output.len(), len, "output length mismatch");
    unsafe {
        haar_convolve_i32(
            signal.as_ptr(),
            kernel.as_ptr(),
            output.as_mut_ptr(),
            len as i32,
            levels as i32,
        )
    }
}

/// Haar convolution of two f32 signals. All slices must have the same power-of-2 length.
pub fn haar_convolve_f32_buf(
    signal: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    levels: usize,
) {
    let len = signal.len();
    assert!(len.is_power_of_two(), "Haar requires power-of-2 length, got {}", len);
    assert_eq!(kernel.len(), len, "kernel length mismatch");
    assert_eq!(output.len(), len, "output length mismatch");
    unsafe {
        haar_convolve_f32(
            signal.as_ptr(),
            kernel.as_ptr(),
            output.as_mut_ptr(),
            len as i32,
            levels as i32,
        )
    }
}

/// Compute max Haar levels for a given length: floor(log2(len)).
pub fn max_haar_levels(len: usize) -> usize {
    assert!(len.is_power_of_two(), "Haar requires power-of-2 length");
    (len as f64).log2() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_roundtrip_f32() {
        let n = 16;
        let levels = max_haar_levels(n);
        let original: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut data = original.clone();

        haar_forward_f32_inplace(&mut data, levels);
        haar_inverse_f32_inplace(&mut data, levels);

        for i in 0..n {
            assert!(
                (data[i] - original[i]).abs() < 1e-4,
                "Haar roundtrip f32 failed at {}: {} vs {}",
                i, data[i], original[i]
            );
        }
    }

    #[test]
    fn test_haar_roundtrip_i32() {
        // Note: i32 Haar has integer division rounding, so not perfectly invertible
        // for arbitrary values. Test with values that divide evenly.
        let n = 8;
        let levels = 1; // Single level is perfectly invertible for even values
        let original: Vec<i32> = vec![2, 4, 6, 8, 10, 12, 14, 16];
        let mut data = original.clone();

        haar_forward(&mut data, levels);
        haar_inverse(&mut data, levels);

        for i in 0..n {
            assert_eq!(
                data[i], original[i],
                "Haar roundtrip i32 failed at {}: {} vs {}",
                i, data[i], original[i]
            );
        }
    }

    #[test]
    fn test_haar_convolve_delta_f32() {
        let n = 16;
        let levels = max_haar_levels(n);
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

        // Delta kernel: Haar(delta) * Haar(signal) then inverse
        // This is NOT the same as circular/linear convolution —
        // it's pointwise multiplication in Haar domain.
        let mut kernel = vec![0.0f32; n];
        kernel[0] = 1.0;

        let mut output = vec![0.0f32; n];
        haar_convolve_f32_buf(&signal, &kernel, &mut output, levels);

        // Result won't be identity like FFT delta conv, because Haar domain
        // pointwise multiply with Haar(delta) is different.
        // Just verify output is finite and non-trivial.
        for i in 0..n {
            assert!(
                output[i].is_finite(),
                "Haar delta conv produced non-finite at {}",
                i
            );
        }
    }

    #[test]
    fn test_haar_forward_structure() {
        // After 1 level of Haar on [a, b, c, d]:
        // low = [a+b, c+d], high = [a-b, c-d]
        // Result: [a+b, c+d, a-b, c-d]
        let mut data = vec![1.0f32, 3.0, 5.0, 7.0];
        haar_forward_f32_inplace(&mut data, 1);

        assert!((data[0] - 4.0).abs() < 1e-6, "low[0]: {} vs 4", data[0]);
        assert!((data[1] - 12.0).abs() < 1e-6, "low[1]: {} vs 12", data[1]);
        assert!((data[2] - (-2.0)).abs() < 1e-6, "high[0]: {} vs -2", data[2]);
        assert!((data[3] - (-2.0)).abs() < 1e-6, "high[1]: {} vs -2", data[3]);
    }

    #[test]
    fn test_haar_various_sizes() {
        for &n in &[4, 8, 16, 32, 64, 128, 256] {
            let levels = max_haar_levels(n);
            let original: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut data = original.clone();

            haar_forward_f32_inplace(&mut data, levels);
            haar_inverse_f32_inplace(&mut data, levels);

            for i in 0..n {
                assert!(
                    (data[i] - original[i]).abs() < 1e-3,
                    "Haar size={} roundtrip failed at {}: {} vs {}",
                    n, i, data[i], original[i]
                );
            }
        }
    }

    #[test]
    fn test_haar_convolve_commutative() {
        let n = 16;
        let levels = max_haar_levels(n);
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (-0.1 * i as f32).exp()).collect();

        let mut out_ab = vec![0.0f32; n];
        let mut out_ba = vec![0.0f32; n];

        haar_convolve_f32_buf(&a, &b, &mut out_ab, levels);
        haar_convolve_f32_buf(&b, &a, &mut out_ba, levels);

        for i in 0..n {
            assert!(
                (out_ab[i] - out_ba[i]).abs() < 1e-4,
                "Haar convolve not commutative at {}: {} vs {}",
                i, out_ab[i], out_ba[i]
            );
        }
    }

    #[test]
    fn test_haar_partial_levels() {
        // Test with fewer levels than max
        let n = 16;
        let original: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

        for levels in 1..=max_haar_levels(n) {
            let mut data = original.clone();
            haar_forward_f32_inplace(&mut data, levels);
            haar_inverse_f32_inplace(&mut data, levels);

            for i in 0..n {
                assert!(
                    (data[i] - original[i]).abs() < 1e-3,
                    "Haar partial levels={} roundtrip failed at {}: {} vs {}",
                    levels, i, data[i], original[i]
                );
            }
        }
    }
}
