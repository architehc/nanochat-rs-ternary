//! Triangle of Truth verification harness.
//!
//! Scalar reference GEMV implementation for cross-validating all kernel paths.

use crate::encode::decode_trit;
use crate::planar::PlanarWeights;

/// Scalar reference GEMV — bit-exact, no SIMD, used for verification.
///
/// Matches the C `gemv_dp4a_ref` exactly.
pub fn gemv_scalar_ref(pw: &PlanarWeights, x: &[i8], act_scale: f32, y: &mut [f32]) {
    let m = pw.rows;
    let k = pw.cols;
    let gs = pw.group_size;
    let kp = k / 4;
    let gprow = k / gs;
    let gpp = gs / 4;

    for r in 0..m {
        let mut racc = 0.0f32;
        for g in 0..gprow {
            let mut gacc = 0i32;
            let bs = r * kp + g * gpp;
            for j in 0..gpp {
                let p = pw.data[bs + j];
                let base = (g * gpp + j) * 4;
                gacc += decode_trit(p & 3) as i32 * x[base] as i32
                    + decode_trit((p >> 2) & 3) as i32 * x[base + 1] as i32
                    + decode_trit((p >> 4) & 3) as i32 * x[base + 2] as i32
                    + decode_trit((p >> 6) & 3) as i32 * x[base + 3] as i32;
            }
            racc += gacc as f32 * pw.scales_rm[r * gprow + g] * act_scale;
        }
        y[r] = racc;
    }
}

/// Result of a verification test.
#[derive(Debug)]
pub struct TestResult {
    pub pass: usize,
    pub fail: usize,
    pub max_diff: f32,
}

/// Verify GEMV output against scalar reference.
pub fn verify_gemv(
    pw: &PlanarWeights,
    x: &[i8],
    act_scale: f32,
    expected_y: &[f32],
    tol: f32,
) -> TestResult {
    let mut y = vec![0.0f32; pw.rows];
    gemv_scalar_ref(pw, x, act_scale, &mut y);

    let mut max_diff = 0.0f32;
    let mut fail = 0;

    for i in 0..pw.rows {
        let diff = (y[i] - expected_y[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tol {
            fail += 1;
        }
    }

    TestResult {
        pass: pw.rows - fail,
        fail,
        max_diff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_weights(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let v = (i as u32).wrapping_mul(2654435761) >> 16;
                (v % 200) as f32 / 100.0 - 1.0
            })
            .collect()
    }

    fn gen_activations(cols: usize) -> Vec<i8> {
        (0..cols)
            .map(|i| (((i * 37 + 13) % 200) as i32 - 100) as i8)
            .collect()
    }

    #[test]
    fn test_scalar_ref_uniform_positive() {
        let rows = 4;
        let cols = 128;
        let gs = 128;
        let weights = vec![1.0f32; rows * cols];
        let pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

        let x = vec![1i8; cols];
        let mut y = vec![0.0f32; rows];
        let act_scale = 0.5;

        gemv_scalar_ref(&pw, &x, act_scale, &mut y);

        // All weights are +1, quantized to +1
        // Each row: sum = 128 * 1 * 1 = 128, times scale * act_scale
        let gprow = cols / gs;
        for r in 0..rows {
            let mut expected = 0.0f32;
            for g in 0..gprow {
                expected += 128.0 * pw.scales_rm[r * gprow + g] * act_scale;
            }
            assert!(
                (y[r] - expected).abs() < 1e-3,
                "row {}: expected {}, got {}",
                r,
                expected,
                y[r]
            );
        }
    }

    #[test]
    fn test_scalar_ref_uniform_negative() {
        let rows = 4;
        let cols = 128;
        let gs = 128;
        let weights = vec![-1.0f32; rows * cols];
        let pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

        let x = vec![1i8; cols];
        let mut y = vec![0.0f32; rows];
        let act_scale = 0.5;

        gemv_scalar_ref(&pw, &x, act_scale, &mut y);

        let gprow = cols / gs;
        for r in 0..rows {
            let mut expected = 0.0f32;
            for g in 0..gprow {
                expected += -128.0 * pw.scales_rm[r * gprow + g] * act_scale;
            }
            assert!(
                (y[r] - expected).abs() < 1e-3,
                "row {}: expected {}, got {}",
                r,
                expected,
                y[r]
            );
        }
    }

    #[test]
    fn test_scalar_ref_cancellation() {
        let rows = 4;
        let cols = 128;
        let gs = 128;
        // Alternating +1, -1 weights
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

        let x = vec![1i8; cols];
        let mut y = vec![0.0f32; rows];

        gemv_scalar_ref(&pw, &x, 0.5, &mut y);

        // +1 and -1 should cancel
        for r in 0..rows {
            assert!(
                y[r].abs() < 1e-3,
                "row {}: expected ~0, got {}",
                r,
                y[r]
            );
        }
    }

    #[test]
    fn test_scalar_ref_various_shapes() {
        for &(m, k) in &[
            (128, 128),
            (256, 512),
            (1, 128),
            (17, 128),
            (33, 256),
            (63, 384),
            (65, 256),
            (127, 640),
        ] {
            let w = gen_weights(m, k);
            let pw = PlanarWeights::from_row_major(&w, m, k, 128);
            let x = gen_activations(k);
            let mut y = vec![0.0f32; m];

            gemv_scalar_ref(&pw, &x, 1.0 / 127.0, &mut y);

            // Just verify finite output
            for r in 0..m {
                assert!(y[r].is_finite(), "[{}x{}] row {} not finite", m, k, r);
            }
        }
    }

    #[test]
    fn test_invalid_codepoint_in_data() {
        // Create PlanarWeights with manually injected 0b10 codepoints
        let rows = 4;
        let cols = 128;
        let gs = 128;

        // Start with all zeros
        let weights = vec![0.0f32; rows * cols];
        let mut pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

        // Inject 0b10 (0xAA = 10_10_10_10 in binary) into packed data
        let kp = cols / 4;
        for r in 0..rows {
            for c in 0..kp {
                pw.data[r * kp + c] = 0xAA; // all 0b10 patterns
            }
        }

        let x = vec![100i8; cols];
        let mut y = vec![0.0f32; rows];

        gemv_scalar_ref(&pw, &x, 1.0, &mut y);

        // 0b10 decodes to 0, so output should be 0 (ignoring scale)
        for r in 0..rows {
            assert_eq!(y[r], 0.0, "row {} should be 0 for all-invalid data", r);
        }
    }

    #[test]
    fn test_verify_gemv_matching() {
        let m = 64;
        let k = 256;
        let w = gen_weights(m, k);
        let pw = PlanarWeights::from_row_major(&w, m, k, 128);
        let x = gen_activations(k);

        let mut expected = vec![0.0f32; m];
        gemv_scalar_ref(&pw, &x, 1.0 / 127.0, &mut expected);

        let result = verify_gemv(&pw, &x, 1.0 / 127.0, &expected, 1e-6);
        assert_eq!(result.fail, 0);
        assert!(result.max_diff < 1e-6);
    }

    #[test]
    fn test_verify_gemv_with_mismatches() {
        let m = 64;
        let k = 128;
        let w = gen_weights(m, k);
        let pw = PlanarWeights::from_row_major(&w, m, k, 128);
        let x = gen_activations(k);

        // Compute correct output
        let mut expected = vec![0.0f32; m];
        gemv_scalar_ref(&pw, &x, 1.0 / 127.0, &mut expected);

        // Modify expected to create mismatches
        expected[0] += 100.0;
        expected[1] += 200.0;

        let result = verify_gemv(&pw, &x, 1.0 / 127.0, &expected, 1e-3);
        assert!(result.fail > 0, "should have failures");
        assert!(result.max_diff > 50.0, "max_diff should be large");
        assert_eq!(result.pass, m - result.fail);
    }

    #[test]
    fn test_test_result_debug() {
        let result = TestResult {
            pass: 10,
            fail: 2,
            max_diff: 0.5,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("pass"));
        assert!(debug.contains("fail"));
    }
}
