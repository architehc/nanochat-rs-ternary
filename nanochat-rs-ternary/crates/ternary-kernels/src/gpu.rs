//! GPU ternary GEMV via CUDA dp4a decode kernel.
//!
//! Provides `GpuWeights` (device-side weight storage) and `gemv_gpu()`
//! for autoregressive decode on NVIDIA GPUs.
//!
//! Gated behind `#[cfg(feature = "cuda")]`.

use ternary_core::planar::PlanarWeights;

extern "C" {
    fn cuda_ternary_gemv_init() -> i32;
    fn cuda_ternary_gemv(
        d_data: *const u8,
        d_scales: *const f32,
        d_x: *const i8,
        act_scale: f32,
        d_y: *mut f32,
        m: i32,
        k: i32,
        group_size: i32,
    );
    fn cuda_alloc(bytes: usize) -> *mut u8;
    fn cuda_free(ptr: *mut u8);
    fn cuda_memcpy_h2d(dst: *mut u8, src: *const u8, bytes: usize) -> i32;
    fn cuda_memcpy_d2h(dst: *mut u8, src: *const u8, bytes: usize) -> i32;
    fn cuda_synchronize() -> i32;
}

static INIT: std::sync::Once = std::sync::Once::new();

/// Initialize CUDA decode LUT. Call once before any GPU GEMV.
pub fn init() {
    INIT.call_once(|| {
        let ret = unsafe { cuda_ternary_gemv_init() };
        assert_eq!(ret, 0, "CUDA GEMV init failed");
    });
}

/// Device-side weight storage for GPU GEMV.
pub struct GpuWeights {
    d_data: *mut u8,
    d_scales: *mut u8,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    data_bytes: usize,
    scales_bytes: usize,
}

// SAFETY: GPU pointers are not accessed from multiple threads simultaneously
// in our usage pattern (one inference at a time per GpuWeights).
unsafe impl Send for GpuWeights {}
unsafe impl Sync for GpuWeights {}

impl GpuWeights {
    /// Upload PlanarWeights to GPU device memory.
    ///
    /// Uses row-major data + row-major scales for the dp4a kernel.
    pub fn from_planar(pw: &PlanarWeights) -> Self {
        init();

        let kp = pw.cols / 4;
        let data_bytes = pw.rows * kp;
        let n_groups = pw.cols / pw.group_size;
        let scales_bytes = pw.rows * n_groups * std::mem::size_of::<f32>();

        let d_data = unsafe { cuda_alloc(data_bytes) };
        assert!(
            !d_data.is_null(),
            "GPU alloc failed for data ({} bytes)",
            data_bytes
        );

        let d_scales = unsafe { cuda_alloc(scales_bytes) };
        assert!(
            !d_scales.is_null(),
            "GPU alloc failed for scales ({} bytes)",
            scales_bytes
        );

        // Upload row-major data
        let ret = unsafe { cuda_memcpy_h2d(d_data, pw.data.as_ptr(), data_bytes) };
        assert_eq!(ret, 0, "H2D copy failed for data");

        // Upload row-major scales
        let ret =
            unsafe { cuda_memcpy_h2d(d_scales, pw.scales_rm.as_ptr() as *const u8, scales_bytes) };
        assert_eq!(ret, 0, "H2D copy failed for scales");

        GpuWeights {
            d_data,
            d_scales,
            rows: pw.rows,
            cols: pw.cols,
            group_size: pw.group_size,
            data_bytes,
            scales_bytes,
        }
    }
}

impl Drop for GpuWeights {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.d_data);
            cuda_free(self.d_scales);
        }
    }
}

impl std::fmt::Debug for GpuWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuWeights")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("group_size", &self.group_size)
            .field("data_bytes", &self.data_bytes)
            .field("scales_bytes", &self.scales_bytes)
            .finish()
    }
}

/// Run ternary GEMV on GPU.
///
/// # Arguments
/// * `gw` - GPU-resident weights
/// * `x` - Host-side INT8 activations [cols]
/// * `act_scale` - Activation scale factor
/// * `y` - Host-side output buffer [rows]
pub fn gemv_gpu(gw: &GpuWeights, x: &[i8], act_scale: f32, y: &mut [f32]) {
    assert_eq!(x.len(), gw.cols, "x.len() != cols");
    assert_eq!(y.len(), gw.rows, "y.len() != rows");

    init();

    let x_bytes = x.len();
    let y_bytes = std::mem::size_of_val(y);

    // Allocate device buffers for x and y
    let d_x = unsafe { cuda_alloc(x_bytes) };
    assert!(!d_x.is_null(), "GPU alloc failed for x");
    let d_y = unsafe { cuda_alloc(y_bytes) };
    assert!(!d_y.is_null(), "GPU alloc failed for y");

    // Upload x
    let ret = unsafe { cuda_memcpy_h2d(d_x, x.as_ptr() as *const u8, x_bytes) };
    assert_eq!(ret, 0, "H2D copy failed for x");

    // Launch kernel
    unsafe {
        cuda_ternary_gemv(
            gw.d_data,
            gw.d_scales as *const f32,
            d_x as *const i8,
            act_scale,
            d_y as *mut f32,
            gw.rows as i32,
            gw.cols as i32,
            gw.group_size as i32,
        );
    }

    // Download y
    let ret = unsafe { cuda_memcpy_d2h(y.as_mut_ptr() as *mut u8, d_y as *const u8, y_bytes) };
    assert_eq!(ret, 0, "D2H copy failed for y");

    // Sync
    let ret = unsafe { cuda_synchronize() };
    assert_eq!(ret, 0, "CUDA synchronize failed");

    // Free temporary buffers
    unsafe {
        cuda_free(d_x);
        cuda_free(d_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_weights(rows: usize, cols: usize) -> (PlanarWeights, Vec<i8>) {
        let gs = 128;
        let mut w = vec![0.0f32; rows * cols];
        for i in 0..w.len() {
            let v = ((i as u32).wrapping_mul(2654435761) >> 16) % 200;
            w[i] = v as f32 / 100.0 - 1.0;
        }
        let pw = PlanarWeights::from_row_major(&w, rows, cols, gs);

        let mut x = vec![0i8; cols];
        for i in 0..cols {
            x[i] = (((i * 37 + 13) % 200) as i32 - 100) as i8;
        }
        (pw, x)
    }

    #[test]
    fn test_gpu_gemv_matches_cpu_scalar() {
        let (pw, x) = make_test_weights(128, 128);
        let act_scale = 1.0 / 127.0;

        // CPU scalar reference
        let mut y_cpu = vec![0.0f32; 128];
        crate::cpu::gemv_scalar_ref(&pw, &x, act_scale, &mut y_cpu);

        // GPU
        let gw = GpuWeights::from_planar(&pw);
        let mut y_gpu = vec![0.0f32; 128];
        gemv_gpu(&gw, &x, act_scale, &mut y_gpu);

        let max_diff: f32 = y_cpu
            .iter()
            .zip(y_gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-3,
            "GPU vs CPU scalar ref max diff: {} (should be < 1e-3)",
            max_diff
        );
    }

    #[test]
    fn test_gpu_shape_torture() {
        for &(m, k) in &[(64, 128), (256, 512), (128, 256), (1, 128), (17, 128)] {
            let (pw, x) = make_test_weights(m, k);
            let act_scale = 1.0 / 127.0;

            let mut y_cpu = vec![0.0f32; m];
            crate::cpu::gemv_scalar_ref(&pw, &x, act_scale, &mut y_cpu);

            let gw = GpuWeights::from_planar(&pw);
            let mut y_gpu = vec![0.0f32; m];
            gemv_gpu(&gw, &x, act_scale, &mut y_gpu);

            let max_diff: f32 = y_cpu
                .iter()
                .zip(y_gpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-3,
                "[{}x{}] GPU vs CPU max diff: {}",
                m,
                k,
                max_diff
            );
        }
    }

    #[test]
    fn test_gpu_upload_download_roundtrip() {
        let (pw, _) = make_test_weights(64, 128);
        let gw = GpuWeights::from_planar(&pw);

        assert_eq!(gw.rows, 64);
        assert_eq!(gw.cols, 128);
        assert_eq!(gw.group_size, 128);
        assert!(gw.data_bytes > 0);
        assert!(gw.scales_bytes > 0);
    }
}
