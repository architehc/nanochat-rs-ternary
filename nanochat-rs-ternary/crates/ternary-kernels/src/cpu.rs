//! Safe Rust wrappers over the C FFI ternary GEMV kernels.

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
    assert_eq!(x.len(), pw.cols, "x.len() != cols");
    assert_eq!(y.len(), pw.rows, "y.len() != rows");
    assert!(pw.data.as_ptr() as usize % 128 == 0, "data not 128B aligned");
    assert!(pw.scales_rm.as_ptr() as usize % 128 == 0, "scales not 128B aligned");

    let c_pw = to_c_struct(pw);
    unsafe {
        ternary_gemv(&c_pw, x.as_ptr(), act_scale, y.as_mut_ptr());
    }
}

/// Scalar reference GEMV for verification.
pub fn gemv_scalar_ref(pw: &PlanarWeights, x: &[i8], act_scale: f32, y: &mut [f32]) {
    assert_eq!(x.len(), pw.cols);
    assert_eq!(y.len(), pw.rows);

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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_shape_torture() {
        // Non-aligned M values
        for &(m, k) in &[(1, 128), (17, 128), (33, 256), (63, 384), (65, 256), (127, 640)] {
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
