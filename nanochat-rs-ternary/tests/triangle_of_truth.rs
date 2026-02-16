//! Triangle of Truth — cross-validate ALL kernel paths produce identical output.
//!
//! The scalar reference GEMV (pure Rust) is the ground truth. Every other kernel
//! path (C FFI dispatch, future GPU) must produce bit-identical f32 results.

use ternary_core::planar::PlanarWeights;
use ternary_core::verify::{gemv_scalar_ref, verify_gemv};
use ternary_kernels::cpu;

/// Generate deterministic pseudo-random weights.
fn gen_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let v = (i as u32).wrapping_mul(2654435761) >> 16;
            (v % 200) as f32 / 100.0 - 1.0
        })
        .collect()
}

/// Generate deterministic pseudo-random activations.
fn gen_activations(cols: usize) -> Vec<i8> {
    (0..cols)
        .map(|i| (((i * 37 + 13) % 200) as i32 - 100) as i8)
        .collect()
}

/// Core triangle-of-truth check: Rust scalar ref vs C FFI dispatch.
fn check_triangle(m: usize, k: usize) {
    let gs = 128;
    let w = gen_weights(m, k);
    let pw = PlanarWeights::from_row_major(&w, m, k, gs);
    let x = gen_activations(k);
    let act_scale = 1.0 / 127.0;

    // Path 1: Rust scalar reference
    let mut y_scalar = vec![0.0f32; m];
    gemv_scalar_ref(&pw, &x, act_scale, &mut y_scalar);

    // Path 2: C FFI dispatch (selects best kernel at runtime)
    let mut y_ffi = vec![0.0f32; m];
    cpu::gemv(&pw, &x, act_scale, &mut y_ffi);

    // Cross-validate
    let result = verify_gemv(&pw, &x, act_scale, &y_ffi, 1e-5);
    assert_eq!(
        result.fail, 0,
        "[{}x{}] Triangle of Truth FAILED: {} mismatches, max_diff={}",
        m, k, result.fail, result.max_diff
    );
}

#[test]
fn triangle_128x128() {
    check_triangle(128, 128);
}

#[test]
fn triangle_256x256() {
    check_triangle(256, 256);
}

#[test]
fn triangle_512x512() {
    check_triangle(512, 512);
}

#[test]
fn triangle_2048x2048() {
    check_triangle(2048, 2048);
}

#[test]
fn triangle_4096x11008() {
    check_triangle(4096, 11008);
}

#[test]
fn triangle_11008x4096() {
    check_triangle(11008, 4096);
}

/// Non-aligned M (M%32 != 0) — exercises padding/tail logic.
#[test]
fn triangle_127x128() {
    check_triangle(127, 128);
}

#[test]
fn triangle_1x4096() {
    check_triangle(1, 4096);
}

#[test]
fn triangle_17x128() {
    check_triangle(17, 128);
}

#[test]
fn triangle_33x256() {
    check_triangle(33, 256);
}

#[test]
fn triangle_63x384() {
    check_triangle(63, 384);
}

#[test]
fn triangle_65x256() {
    check_triangle(65, 256);
}

/// Invalid codepoint injection: plant 0b10 patterns, assert decode=0.
#[test]
fn triangle_invalid_codepoints() {
    let rows = 4;
    let cols = 128;
    let gs = 128;
    let weights = vec![0.0f32; rows * cols];
    let mut pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

    // Inject 0xAA = 0b10_10_10_10 (all invalid codepoints)
    let kp = cols / 4;
    for r in 0..rows {
        for c in 0..kp {
            pw.data[r * kp + c] = 0xAA;
        }
    }

    let x = vec![100i8; cols];
    let act_scale = 1.0;

    // Scalar reference
    let mut y_scalar = vec![0.0f32; rows];
    gemv_scalar_ref(&pw, &x, act_scale, &mut y_scalar);

    // C FFI
    let mut y_ffi = vec![0.0f32; rows];
    cpu::gemv(&pw, &x, act_scale, &mut y_ffi);

    // Both should be zero (invalid codepoints decode as 0)
    for r in 0..rows {
        assert_eq!(y_scalar[r], 0.0, "scalar row {} not zero", r);
        assert_eq!(y_ffi[r], 0.0, "ffi row {} not zero", r);
    }
}

/// Scale-only test: verify per-group scaling is correct.
/// Tolerance is 1e-4 to allow for FP rounding differences between kernels
/// that use different data layouts (row-major scalar vs col-major AVX2/AVX-512)
/// and different operation order (scalar multiply vs FMA).
#[test]
fn triangle_scale_only() {
    let rows = 64;
    let cols = 256;
    let gs = 128;
    let w = gen_weights(rows, cols);
    let pw = PlanarWeights::from_row_major(&w, rows, cols, gs);
    let x = gen_activations(cols);

    // Use various act_scale values
    for &act_scale in &[1.0 / 127.0, 0.5, 1.0, 0.001] {
        let mut y_scalar = vec![0.0f32; rows];
        gemv_scalar_ref(&pw, &x, act_scale, &mut y_scalar);

        let mut y_ffi = vec![0.0f32; rows];
        cpu::gemv(&pw, &x, act_scale, &mut y_ffi);

        for r in 0..rows {
            let diff = (y_scalar[r] - y_ffi[r]).abs();
            assert!(
                diff < 1e-4,
                "scale={} row {}: scalar={}, ffi={}, diff={}",
                act_scale,
                r,
                y_scalar[r],
                y_ffi[r],
                diff
            );
        }
    }
}

/// Cross-validate gemv_parallel against scalar reference.
#[test]
fn triangle_parallel_matches_scalar() {
    for &(m, k) in &[(256, 256), (4096, 4096), (4096, 11008)] {
        let gs = 128;
        let w = gen_weights(m, k);
        let pw = PlanarWeights::from_row_major(&w, m, k, gs);
        let x = gen_activations(k);
        let act_scale = 1.0 / 127.0;

        let mut y_scalar = vec![0.0f32; m];
        let mut y_parallel = vec![0.0f32; m];
        gemv_scalar_ref(&pw, &x, act_scale, &mut y_scalar);
        cpu::gemv_parallel(&pw, &x, act_scale, &mut y_parallel);

        for i in 0..m {
            let diff = (y_scalar[i] - y_parallel[i]).abs();
            assert!(
                diff < 1e-4,
                "parallel mismatch at [{}x{}] idx {}: scalar={}, parallel={}, diff={}",
                m, k, i, y_scalar[i], y_parallel[i], diff
            );
        }
    }
}

/// Uniform positive weights: every weight = +1.
#[test]
fn triangle_uniform_positive() {
    let rows = 64;
    let cols = 128;
    let gs = 128;
    let weights = vec![1.0f32; rows * cols];
    let pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);
    let x = vec![1i8; cols];
    let act_scale = 0.5;

    let mut y_scalar = vec![0.0f32; rows];
    gemv_scalar_ref(&pw, &x, act_scale, &mut y_scalar);

    let mut y_ffi = vec![0.0f32; rows];
    cpu::gemv(&pw, &x, act_scale, &mut y_ffi);

    for r in 0..rows {
        assert!(
            (y_scalar[r] - y_ffi[r]).abs() < 1e-5,
            "row {}: scalar={}, ffi={}",
            r,
            y_scalar[r],
            y_ffi[r]
        );
        // Output should be positive (sum of +1 * +1 * scale * act_scale)
        assert!(y_scalar[r] > 0.0, "row {} should be positive", r);
    }
}
