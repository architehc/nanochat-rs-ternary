//! Safe Rust wrappers over the C FFI ternary GEMV kernels.

use rayon::prelude::*;
use ternary_core::planar::PlanarWeights;

/// C struct matching the PlanarWeights layout expected by ternary_gemv.c
#[repr(C)]
struct PlanarWeightsC {
    data: *const u8,
    data_colmaj: *const u8,
    scales_rm: *const f32,
    scales_gm: *const f32,
    rows: i32,
    cols: i32,
    group_size: i32,
    rows_padded: i32,
}

extern "C" {
    // C function does not mutate through const pointers — verified in ternary_gemv.c
    fn ternary_gemv(pw: *const PlanarWeightsC, x: *const i8, act_scale: f32, y: *mut f32);

    fn gemv_dp4a_ref(
        data: *const u8,
        scales_rm: *const f32,
        x: *const i8,
        act_scale: f32,
        y: *mut f32,
        m: i32,
        k: i32,
        gs: i32,
    );

    fn ternary_kernel_name() -> *const std::ffi::c_char;
}

/// Convert PlanarWeights to the C struct for FFI.
fn to_c_struct(pw: &PlanarWeights) -> PlanarWeightsC {
    PlanarWeightsC {
        data: pw.data.as_ptr(),
        data_colmaj: pw.data_colmaj.as_ptr(),
        scales_rm: pw.scales_rm.as_ptr(),
        scales_gm: pw.scales_gm.as_ptr(),
        rows: pw.rows as i32,
        cols: pw.cols as i32,
        group_size: pw.group_size as i32,
        rows_padded: pw.rows_padded as i32,
    }
}

fn validate_planar_for_ffi(pw: &PlanarWeights, x_len: usize, y_len: usize) -> Result<(), String> {
    if x_len != pw.cols {
        return Err(format!("x.len() ({}) != cols ({})", x_len, pw.cols));
    }
    if y_len != pw.rows {
        return Err(format!("y.len() ({}) != rows ({})", y_len, pw.rows));
    }
    if pw.rows == 0 {
        return Err("rows must be > 0".to_string());
    }
    if pw.cols == 0 {
        return Err("cols must be > 0".to_string());
    }
    if pw.group_size == 0 {
        return Err("group_size must be > 0".to_string());
    }
    if !pw.cols.is_multiple_of(4) {
        return Err("cols must be divisible by 4".to_string());
    }
    if !pw.group_size.is_multiple_of(4) {
        return Err("group_size must be divisible by 4".to_string());
    }
    if !pw.cols.is_multiple_of(pw.group_size) {
        return Err("cols must be divisible by group_size".to_string());
    }
    if (pw.group_size / 4) > 32 {
        return Err(format!(
            "group_size={} exceeds kernel stack LUT limit (max 128)",
            pw.group_size
        ));
    }
    if pw.rows_padded < pw.rows {
        return Err(format!(
            "rows_padded must be >= rows ({} < {})",
            pw.rows_padded, pw.rows
        ));
    }
    if pw.rows > i32::MAX as usize {
        return Err(format!(
            "rows ({}) exceed i32::MAX for C FFI",
            pw.rows
        ));
    }
    if pw.cols > i32::MAX as usize {
        return Err(format!(
            "cols ({}) exceed i32::MAX for C FFI",
            pw.cols
        ));
    }
    if pw.group_size > i32::MAX as usize {
        return Err(format!(
            "group_size ({}) exceed i32::MAX for C FFI",
            pw.group_size
        ));
    }
    if pw.rows_padded > i32::MAX as usize {
        return Err(format!(
            "rows_padded ({}) exceed i32::MAX for C FFI",
            pw.rows_padded
        ));
    }

    let kp = pw.cols / 4;
    let gprow = pw.cols / pw.group_size;
    let rows_kp = pw
        .rows
        .checked_mul(kp)
        .ok_or_else(|| format!("rows*kp overflow: {} * {}", pw.rows, kp))?;
    let rows_gprow = pw
        .rows
        .checked_mul(gprow)
        .ok_or_else(|| format!("rows*gprow overflow: {} * {}", pw.rows, gprow))?;
    let rows_padded_kp = pw
        .rows_padded
        .checked_mul(kp)
        .ok_or_else(|| format!("rows_padded*kp overflow: {} * {}", pw.rows_padded, kp))?;
    let rows_padded_gprow = pw
        .rows_padded
        .checked_mul(gprow)
        .ok_or_else(|| {
            format!(
                "rows_padded*gprow overflow: {} * {}",
                pw.rows_padded, gprow
            )
        })?;
    if pw.data.len() < rows_kp {
        return Err(format!(
            "row-major packed data too short: {} < {}",
            pw.data.len(),
            rows_kp
        ));
    }
    if pw.scales_rm.len() < rows_gprow {
        return Err(format!(
            "row-major scales too short: {} < {}",
            pw.scales_rm.len(),
            rows_gprow
        ));
    }
    if pw.data_colmaj.len() < rows_padded_kp {
        return Err(format!(
            "col-major packed data too short: {} < {}",
            pw.data_colmaj.len(),
            rows_padded_kp
        ));
    }
    if pw.scales_gm.len() < rows_padded_gprow {
        return Err(format!(
            "group-major scales too short: {} < {}",
            pw.scales_gm.len(),
            rows_padded_gprow
        ));
    }
    Ok(())
}

/// Check if AVX2 is available on this CPU.
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

/// Check if AVX-512 is available on this CPU.
#[cfg(target_arch = "x86_64")]
pub fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512() -> bool {
    false
}

/// Check if SSSE3 is available (x86_64 only).
#[cfg(target_arch = "x86_64")]
pub fn has_ssse3() -> bool {
    std::arch::is_x86_feature_detected!("ssse3")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_ssse3() -> bool {
    false
}

/// Check if NEON is available (always true on AArch64).
#[cfg(target_arch = "aarch64")]
pub fn has_neon() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(target_arch = "aarch64"))]
pub fn has_neon() -> bool {
    false
}

/// Returns the name of the kernel selected by the C dispatch chain.
pub fn selected_kernel_name() -> &'static str {
    // SAFETY: ternary_kernel_name() returns a pointer to a static string literal
    // in the C code. The pointer is valid for the lifetime of the program.
    unsafe {
        let ptr = ternary_kernel_name();
        std::ffi::CStr::from_ptr(ptr)
            .to_str()
            .unwrap_or("unknown")
    }
}

/// Safe GEMV dispatch — calls the best available kernel automatically.
///
/// # Arguments
/// * `pw` - Planar ternary weights
/// * `x` - Quantized INT8 activations [cols]
/// * `act_scale` - Activation scale factor
/// * `y` - Output buffer [rows], caller-allocated
///
/// # Panics
/// Panics if dimensions don't match or alignment is wrong.
pub fn gemv(pw: &PlanarWeights, x: &[i8], act_scale: f32, y: &mut [f32]) {
    validate_planar_for_ffi(pw, x.len(), y.len()).expect("FFI validation failed");
    assert!(
        (pw.data.as_ptr() as usize).is_multiple_of(128),
        "data not 128B aligned"
    );
    assert!(
        (pw.scales_rm.as_ptr() as usize).is_multiple_of(128),
        "scales_rm not 128B aligned"
    );
    assert!(
        (pw.data_colmaj.as_ptr() as usize).is_multiple_of(128),
        "data_colmaj not 128B aligned"
    );
    assert!(
        (pw.scales_gm.as_ptr() as usize).is_multiple_of(128),
        "scales_gm not 128B aligned"
    );

    let c_pw = to_c_struct(pw);
    // SAFETY: All pointers in c_pw come from AlignedVec (128-byte aligned, valid for pw lifetime).
    // x and y are valid slices whose lengths are validated above against pw dimensions.
    // The C function reads exactly pw.cols bytes from x and writes exactly pw.rows floats to y.
    unsafe {
        ternary_gemv(&c_pw, x.as_ptr(), act_scale, y.as_mut_ptr());
    }
}

/// Scalar reference GEMV for verification.
pub fn gemv_scalar_ref(pw: &PlanarWeights, x: &[i8], act_scale: f32, y: &mut [f32]) {
    validate_planar_for_ffi(pw, x.len(), y.len()).expect("FFI validation failed");

    // SAFETY: All pointers come from validated PlanarWeights (AlignedVec, correct lengths).
    // x and y lengths are checked by validate_planar_for_ffi above.
    // The C scalar reference reads data[rows * kp], scales_rm[rows * gprow], x[cols],
    // and writes y[rows].
    unsafe {
        gemv_dp4a_ref(
            pw.data.as_ptr(),
            pw.scales_rm.as_ptr(),
            x.as_ptr(),
            act_scale,
            y.as_mut_ptr(),
            pw.rows as i32,
            pw.cols as i32,
            pw.group_size as i32,
        );
    }
}

/// Parallel GEMV: split M (rows) dimension across threads using rayon.
///
/// Each chunk dispatches the best available SIMD kernel (AVX2/AVX-512)
/// via `ternary_gemv()` on its row slice. Falls back to single-threaded
/// `gemv()` for small M.
///
/// Row-slicing is safe for all SIMD kernels because they access column-major
/// data as `data_colmaj[cg * rows_padded + row_offset]`, where `rows_padded`
/// is the stride. Sub-structs point into the parent's data at the row offset
/// while keeping the original `rows_padded` as stride.
///
/// # Arguments
/// * `pw` - Planar ternary weights
/// * `x` - Quantized INT8 activations [cols]
/// * `act_scale` - Activation scale factor
/// * `y` - Output buffer [rows], caller-allocated
pub fn gemv_parallel(pw: &PlanarWeights, x: &[i8], act_scale: f32, y: &mut [f32]) {
    validate_planar_for_ffi(pw, x.len(), y.len()).expect("FFI validation failed");

    const PARALLEL_THRESHOLD: usize = 4096;
    const ROWS_PER_CHUNK: usize = 256;

    if pw.rows < PARALLEL_THRESHOLD {
        return gemv(pw, x, act_scale, y);
    }

    let kp = pw.cols / 4; // packed bytes per row (row-major)
    let gprow = pw.cols / pw.group_size; // scale groups per row (row-major)

    // Validate all bounds upfront (once) instead of redundantly in each parallel chunk.
    // The worst-case access is at the end of the last chunk (row pw.rows - 1).
    {
        let data_end = pw.rows * kp;
        assert!(
            data_end <= pw.data.len(),
            "row-major packed data out of bounds: {} > {}",
            data_end,
            pw.data.len()
        );

        let scales_rm_end = pw.rows * gprow;
        assert!(
            scales_rm_end <= pw.scales_rm.len(),
            "row-major scales out of bounds: {} > {}",
            scales_rm_end,
            pw.scales_rm.len()
        );

        // Col-major last byte: column (kp-1) at stride rows_padded, plus last row
        let data_colmaj_end = (pw.rows - 1) + (kp.saturating_sub(1)) * pw.rows_padded + 1;
        assert!(
            data_colmaj_end <= pw.data_colmaj.len(),
            "col-major packed data out of bounds: {} > {}",
            data_colmaj_end,
            pw.data_colmaj.len()
        );

        // Group-major last byte: group (gprow-1) at stride rows_padded, plus last row
        let scales_gm_end = (pw.rows - 1) + (gprow.saturating_sub(1)) * pw.rows_padded + 1;
        assert!(
            scales_gm_end <= pw.scales_gm.len(),
            "group-major scales out of bounds: {} > {}",
            scales_gm_end,
            pw.scales_gm.len()
        );
    }

    y.par_chunks_mut(ROWS_PER_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, y_chunk)| {
            let row_start = chunk_idx * ROWS_PER_CHUNK;
            let chunk_rows = y_chunk.len();

            // Construct sub-PlanarWeightsC pointing into the parent's arrays.
            // Column-major and group-major arrays use rows_padded as stride,
            // so we offset by row_start within each column/group.
            // Row-major arrays are contiguous per row, so we offset by row_start * items_per_row.
            // SAFETY: Pointer arithmetic is bounds-checked by the asserts above.
            // Row-major arrays: offset by row_start * items_per_row.
            // Column-major/group-major arrays: offset by row_start (stride is rows_padded).
            let sub_pw = PlanarWeightsC {
                data: unsafe { pw.data.as_ptr().add(row_start * kp) },
                data_colmaj: unsafe { pw.data_colmaj.as_ptr().add(row_start) },
                scales_rm: unsafe { pw.scales_rm.as_ptr().add(row_start * gprow) },
                scales_gm: unsafe { pw.scales_gm.as_ptr().add(row_start) },
                rows: i32::try_from(chunk_rows).expect("chunk rows exceed i32::MAX"),
                cols: pw.cols as i32,
                group_size: pw.group_size as i32,
                rows_padded: pw.rows_padded as i32, // keep original — it's the column stride
            };

            // SAFETY: sub_pw points into the parent PlanarWeights arrays at validated offsets.
            // x is the full activation vector (shared, read-only). y_chunk is a disjoint
            // mutable slice from par_chunks_mut, so no aliasing. The C kernel reads
            // chunk_rows from the sub-struct and writes chunk_rows floats to y_chunk.
            unsafe {
                ternary_gemv(&sub_pw, x.as_ptr(), act_scale, y_chunk.as_mut_ptr());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::needless_range_loop)] // Index needed for deterministic pseudo-random generation
    fn make_test_weights(rows: usize, cols: usize) -> (PlanarWeights, Vec<i8>) {
        let gs = 128;
        // Generate deterministic test weights
        let mut w = vec![0.0f32; rows * cols];
        for i in 0..w.len() {
            let v = ((i as u32).wrapping_mul(2654435761) >> 16) % 200;
            w[i] = v as f32 / 100.0 - 1.0;
        }
        let pw = PlanarWeights::from_row_major(&w, rows, cols, gs);

        // Generate activations
        let mut x = vec![0i8; cols];
        for i in 0..cols {
            x[i] = (((i * 37 + 13) % 200) as i32 - 100) as i8;
        }
        (pw, x)
    }

    #[test]
    fn test_ffi_compiles_and_links() {
        // Just verify FFI works at all
        let (pw, x) = make_test_weights(128, 128);
        let mut y = vec![0.0f32; 128];
        gemv(&pw, &x, 1.0 / 127.0, &mut y);
        // Output should be finite
        for v in &y {
            assert!(v.is_finite(), "Output not finite: {}", v);
        }
    }

    #[test]
    fn test_gemv_matches_scalar_ref() {
        for &(m, k) in &[(128, 128), (256, 512), (64, 256)] {
            let (pw, x) = make_test_weights(m, k);
            let act_scale = 1.0 / 127.0;

            let mut y_ref = vec![0.0f32; m];
            let mut y_dispatch = vec![0.0f32; m];

            gemv_scalar_ref(&pw, &x, act_scale, &mut y_ref);
            gemv(&pw, &x, act_scale, &mut y_dispatch);

            let max_diff: f32 = y_ref
                .iter()
                .zip(y_dispatch.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-4,
                "[{}x{}] dispatch vs scalar ref max diff: {}",
                m,
                k,
                max_diff
            );
        }
    }

    #[test]
    fn test_gemv_parallel_matches_single_thread() {
        for &(m, k) in &[(4096, 512), (8192, 256), (4096, 128)] {
            let (pw, x) = make_test_weights(m, k);
            let act_scale = 1.0 / 127.0;

            // Compare parallel (SIMD per chunk) against single-thread SIMD
            let mut y_single = vec![0.0f32; m];
            let mut y_parallel = vec![0.0f32; m];

            gemv(&pw, &x, act_scale, &mut y_single);
            gemv_parallel(&pw, &x, act_scale, &mut y_parallel);

            let max_diff: f32 = y_single
                .iter()
                .zip(y_parallel.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            // Both use the same SIMD kernel, so should be bit-identical
            assert!(
                max_diff < 1e-6,
                "[{}x{}] parallel vs single-thread SIMD max diff: {}",
                m,
                k,
                max_diff
            );

            // Also verify both are close to scalar reference
            let mut y_scalar = vec![0.0f32; m];
            gemv_scalar_ref(&pw, &x, act_scale, &mut y_scalar);

            let max_diff_vs_scalar: f32 = y_parallel
                .iter()
                .zip(y_scalar.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff_vs_scalar < 1e-4,
                "[{}x{}] parallel SIMD vs scalar max diff: {}",
                m,
                k,
                max_diff_vs_scalar
            );
        }
    }

    #[test]
    fn test_gemv_parallel_small_m_fallback() {
        // M < 256 should fallback to single-threaded gemv
        let (pw, x) = make_test_weights(64, 128);
        let mut y_ref = vec![0.0f32; 64];
        let mut y_par = vec![0.0f32; 64];

        gemv_scalar_ref(&pw, &x, 1.0 / 127.0, &mut y_ref);
        gemv_parallel(&pw, &x, 1.0 / 127.0, &mut y_par);

        let max_diff: f32 = y_ref
            .iter()
            .zip(y_par.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-4, "fallback max diff: {}", max_diff);
    }

    #[test]
    fn test_shape_torture() {
        // Non-aligned M values
        for &(m, k) in &[
            (1, 128),
            (17, 128),
            (33, 256),
            (63, 384),
            (65, 256),
            (127, 640),
        ] {
            let (pw, x) = make_test_weights(m, k);
            let mut y_ref = vec![0.0f32; m];
            let mut y_dispatch = vec![0.0f32; m];

            gemv_scalar_ref(&pw, &x, 1.0 / 127.0, &mut y_ref);
            gemv(&pw, &x, 1.0 / 127.0, &mut y_dispatch);

            let max_diff: f32 = y_ref
                .iter()
                .zip(y_dispatch.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-4,
                "[{}x{}] shape torture max diff: {}",
                m,
                k,
                max_diff
            );
        }
    }
}
