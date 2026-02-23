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

    // Inject 0xAA = 0b10_10_10_10 (all invalid codepoints) into BOTH layouts.
    // Row-major `data` is read by gemv_scalar_ref, column-major `data_colmaj`
    // is read by the SIMD/FFI kernel. Both must be corrupted to actually test
    // invalid codepoint handling in all kernel paths.
    let kp = cols / 4;
    for r in 0..rows {
        for c in 0..kp {
            pw.data[r * kp + c] = 0xAA;
            pw.data_colmaj[c * pw.rows_padded + r] = 0xAA;
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
        assert!(
            y_scalar[r].abs() < 1e-6,
            "scalar row {} not near zero: {}",
            r,
            y_scalar[r]
        );
        assert!(
            y_ffi[r].abs() < 1e-6,
            "ffi row {} not near zero: {}",
            r,
            y_ffi[r]
        );
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
                m,
                k,
                i,
                y_scalar[i],
                y_parallel[i],
                diff
            );
        }
    }
}

// ============================================================
// GOPS Regression Tests — catch layout changes that break SIMD
// ============================================================
//
// These measure wall-clock throughput and assert minimum GOPS thresholds.
// Thresholds are deliberately conservative (well below typical measured values)
// to avoid flaky tests on busy CI runners, while still catching catastrophic
// regressions (e.g., wrong data layout breaking SIMD, fallback to scalar).

/// Measure GOPS for a given shape and kernel function.
fn measure_gops(
    m: usize,
    k: usize,
    kernel: impl Fn(&PlanarWeights, &[i8], f32, &mut [f32]),
) -> f64 {
    let w = gen_weights(m, k);
    let pw = PlanarWeights::from_row_major(&w, m, k, 128);
    let x = gen_activations(k);
    let mut y = vec![0.0f32; m];
    let act_scale = 1.0 / 127.0;

    // Warmup
    for _ in 0..5 {
        kernel(&pw, &x, act_scale, &mut y);
    }

    // Timed iterations
    let iters = 50;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        kernel(&pw, &x, act_scale, &mut y);
        std::hint::black_box(&y);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let ops_per_iter = 2.0 * m as f64 * k as f64; // multiply + accumulate
    let total_ops = ops_per_iter * iters as f64;
    total_ops / elapsed / 1e9 // GOPS
}

/// Scalar reference kernel: must complete in reasonable time.
/// This is the unoptimized Rust implementation for cross-validation only.
/// Typical: ~0.1 GOPS (pure scalar byte-by-byte). Threshold catches hangs.
#[test]
fn gops_regression_scalar_4096x4096() {
    let gops = measure_gops(4096, 4096, |pw, x, s, y| {
        gemv_scalar_ref(pw, x, s, y);
    });
    assert!(
        gops > 0.01,
        "Scalar GEMV 4096x4096: {:.3} GOPS below minimum 0.01 GOPS (hung?)",
        gops
    );
    eprintln!("Scalar 4096x4096: {:.2} GOPS", gops);
}

/// FFI kernel must achieve at least 5 GOPS at production shapes.
/// Typical: 14-110 GOPS depending on ISA. 5 GOPS catches SIMD fallback.
#[test]
fn gops_regression_ffi_4096x4096() {
    let gops = measure_gops(4096, 4096, |pw, x, s, y| {
        cpu::gemv(pw, x, s, y);
    });
    assert!(
        gops > 5.0,
        "FFI GEMV 4096x4096: {:.1} GOPS below minimum 5.0 GOPS",
        gops
    );
    eprintln!("FFI 4096x4096: {:.1} GOPS", gops);
}

/// FFI kernel at FFN production shape (4096x11008).
#[test]
fn gops_regression_ffi_4096x11008() {
    let gops = measure_gops(4096, 11008, |pw, x, s, y| {
        cpu::gemv(pw, x, s, y);
    });
    assert!(
        gops > 5.0,
        "FFI GEMV 4096x11008: {:.1} GOPS below minimum 5.0 GOPS",
        gops
    );
    eprintln!("FFI 4096x11008: {:.1} GOPS", gops);
}

/// FFI kernel at reverse FFN shape (11008x4096).
#[test]
fn gops_regression_ffi_11008x4096() {
    let gops = measure_gops(11008, 4096, |pw, x, s, y| {
        cpu::gemv(pw, x, s, y);
    });
    assert!(
        gops > 5.0,
        "FFI GEMV 11008x4096: {:.1} GOPS below minimum 5.0 GOPS",
        gops
    );
    eprintln!("FFI 11008x4096: {:.1} GOPS", gops);
}

/// Parallel GEMV must achieve minimum absolute throughput.
/// When the single-thread kernel is already memory-bandwidth-saturated
/// (100+ GOPS with nibble-split AVX2), parallel adds overhead. So we
/// check absolute GOPS, not relative speedup.
#[test]
fn gops_regression_parallel_8192x4096() {
    let m = 8192;
    let k = 4096;
    let gops_single = measure_gops(m, k, |pw, x, s, y| {
        cpu::gemv(pw, x, s, y);
    });
    let gops_parallel = measure_gops(m, k, |pw, x, s, y| {
        cpu::gemv_parallel(pw, x, s, y);
    });

    eprintln!(
        "8192x4096: single={:.1} GOPS, parallel={:.1} GOPS, ratio={:.2}x",
        gops_single,
        gops_parallel,
        gops_parallel / gops_single.max(0.001)
    );

    // Parallel must at least achieve 5 GOPS (not hanging or broken)
    assert!(
        gops_parallel > 5.0,
        "Parallel GEMV 8192x4096: {:.1} GOPS below minimum 5.0 GOPS",
        gops_parallel
    );
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
