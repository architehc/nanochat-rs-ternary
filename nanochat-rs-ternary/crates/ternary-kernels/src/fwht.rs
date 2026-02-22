//! Safe Rust wrappers for FWHT (Fast Walsh-Hadamard Transform) C kernels.
//!
//! FWHT uses only additions and subtractions — integer-only compatible.
//! Self-inverse (up to 1/N scaling).

extern "C" {
    fn fwht_i32(data: *mut i32, len: i32);
    fn fwht_f32(data: *mut f32, len: i32);
    fn fwht_convolve_i32(signal: *const i32, kernel: *const i32, output: *mut i32, len: i32);
    fn fwht_convolve_f32(signal: *const f32, kernel: *const f32, output: *mut f32, len: i32);
}

/// In-place FWHT on i32 data. Length must be a power of 2.
pub fn fwht_inplace_i32(data: &mut [i32]) {
    let len = data.len();
    assert!(len.is_power_of_two(), "FWHT requires power-of-2 length, got {}", len);
    if len <= 1 {
        return;
    }
    unsafe { fwht_i32(data.as_mut_ptr(), len as i32) }
}

/// In-place FWHT on f32 data. Length must be a power of 2.
pub fn fwht_inplace_f32(data: &mut [f32]) {
    let len = data.len();
    assert!(len.is_power_of_two(), "FWHT requires power-of-2 length, got {}", len);
    if len <= 1 {
        return;
    }
    unsafe { fwht_f32(data.as_mut_ptr(), len as i32) }
}

/// FWHT convolution of two i32 signals. All slices must have the same power-of-2 length.
/// Result: IFWHT(FWHT(signal) * FWHT(kernel)) / N
pub fn fwht_convolve_i32_buf(signal: &[i32], kernel: &[i32], output: &mut [i32]) {
    let len = signal.len();
    assert!(len.is_power_of_two(), "FWHT requires power-of-2 length, got {}", len);
    assert_eq!(kernel.len(), len, "kernel length mismatch");
    assert_eq!(output.len(), len, "output length mismatch");
    unsafe { fwht_convolve_i32(signal.as_ptr(), kernel.as_ptr(), output.as_mut_ptr(), len as i32) }
}

/// FWHT convolution of two f32 signals. All slices must have the same power-of-2 length.
/// Result: IFWHT(FWHT(signal) * FWHT(kernel)) / N
pub fn fwht_convolve_f32_buf(signal: &[f32], kernel: &[f32], output: &mut [f32]) {
    let len = signal.len();
    assert!(len.is_power_of_two(), "FWHT requires power-of-2 length, got {}", len);
    assert_eq!(kernel.len(), len, "kernel length mismatch");
    assert_eq!(output.len(), len, "output length mismatch");
    unsafe { fwht_convolve_f32(signal.as_ptr(), kernel.as_ptr(), output.as_mut_ptr(), len as i32) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_self_inverse_i32() {
        // FWHT(FWHT(x)) == N * x
        let n = 8;
        let original: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut data = original.clone();

        fwht_inplace_i32(&mut data);
        fwht_inplace_i32(&mut data);

        for i in 0..n {
            assert_eq!(
                data[i],
                original[i] * n as i32,
                "FWHT self-inverse failed at {}: {} vs {}",
                i,
                data[i],
                original[i] * n as i32
            );
        }
    }

    #[test]
    fn test_fwht_self_inverse_f32() {
        let n = 16;
        let original: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut data = original.clone();

        fwht_inplace_f32(&mut data);
        fwht_inplace_f32(&mut data);

        for i in 0..n {
            let expected = original[i] * n as f32;
            assert!(
                (data[i] - expected).abs() < 1e-4,
                "FWHT self-inverse f32 failed at {}: {} vs {}",
                i,
                data[i],
                expected
            );
        }
    }

    #[test]
    fn test_fwht_convolve_delta_i32() {
        // Convolving with delta [1, 0, 0, ...] should return the signal
        let n = 8;
        let signal: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let mut kernel = vec![0i32; n];
        kernel[0] = 1;
        let mut output = vec![0i32; n];

        fwht_convolve_i32_buf(&signal, &kernel, &mut output);

        for i in 0..n {
            assert_eq!(
                output[i], signal[i],
                "FWHT delta conv failed at {}: {} vs {}",
                i, output[i], signal[i]
            );
        }
    }

    #[test]
    fn test_fwht_convolve_delta_f32() {
        let n = 16;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let mut kernel = vec![0.0f32; n];
        kernel[0] = 1.0;
        let mut output = vec![0.0f32; n];

        fwht_convolve_f32_buf(&signal, &kernel, &mut output);

        for i in 0..n {
            assert!(
                (output[i] - signal[i]).abs() < 1e-4,
                "FWHT delta conv f32 failed at {}: {} vs {}",
                i,
                output[i],
                signal[i]
            );
        }
    }

    #[test]
    fn test_fwht_various_sizes() {
        for &n in &[2, 4, 8, 16, 32, 64, 128, 256] {
            let original: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut data = original.clone();

            fwht_inplace_f32(&mut data);
            fwht_inplace_f32(&mut data);

            for i in 0..n {
                let expected = original[i] * n as f32;
                assert!(
                    (data[i] - expected).abs() < 1e-2,
                    "FWHT size={} failed at {}: {} vs {}",
                    n, i, data[i], expected
                );
            }
        }
    }

    #[test]
    fn test_fwht_convolve_commutative() {
        let n = 16;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (-0.1 * i as f32).exp()).collect();

        let mut out_ab = vec![0.0f32; n];
        let mut out_ba = vec![0.0f32; n];

        fwht_convolve_f32_buf(&a, &b, &mut out_ab);
        fwht_convolve_f32_buf(&b, &a, &mut out_ba);

        for i in 0..n {
            assert!(
                (out_ab[i] - out_ba[i]).abs() < 1e-4,
                "FWHT convolve not commutative at {}: {} vs {}",
                i, out_ab[i], out_ba[i]
            );
        }
    }
}
