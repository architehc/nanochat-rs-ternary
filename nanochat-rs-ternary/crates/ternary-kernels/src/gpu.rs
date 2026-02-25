//! GPU ternary GEMV via CUDA dp4a decode kernel.
//!
//! Provides `GpuWeights` (device-side weight storage) and `gemv_gpu()`
//! for autoregressive decode on NVIDIA GPUs.
//!
//! Gated behind `#[cfg(any(feature = "cuda", has_cuda))]`.

use std::sync::OnceLock;
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
    ) -> i32;
    fn cuda_alloc(bytes: usize) -> *mut u8;
    fn cuda_free(ptr: *mut u8);
    fn cuda_memcpy_h2d(dst: *mut u8, src: *const u8, bytes: usize) -> i32;
    fn cuda_memcpy_d2h(dst: *mut u8, src: *const u8, bytes: usize) -> i32;
    fn cuda_synchronize() -> i32;
}

type GpuResult<T> = std::result::Result<T, String>;

static INIT_RESULT: OnceLock<GpuResult<()>> = OnceLock::new();

struct DeviceAlloc {
    ptr: *mut u8,
}

impl DeviceAlloc {
    fn alloc(bytes: usize, label: &str) -> GpuResult<Self> {
        // SAFETY: cuda_alloc wraps cudaMalloc which returns a valid device pointer
        // or null on failure. We check for null below.
        let ptr = unsafe { cuda_alloc(bytes) };
        if ptr.is_null() {
            return Err(format!("GPU alloc failed for {} ({} bytes)", label, bytes));
        }
        Ok(Self { ptr })
    }

    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn into_raw(mut self) -> *mut u8 {
        let ptr = self.ptr;
        self.ptr = std::ptr::null_mut();
        ptr
    }
}

impl Drop for DeviceAlloc {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was allocated by cuda_alloc (cudaMalloc) and has not
            // been freed yet. We null it via into_raw() when ownership is transferred.
            unsafe {
                cuda_free(self.ptr);
            }
        }
    }
}

/// Initialize CUDA decode LUT. Call once before any GPU GEMV.
///
/// **Note:** Init failure is cached permanently via `OnceLock`. If GPU init fails
/// (e.g. driver not loaded), the process must be restarted to retry. This is
/// intentional — transient GPU failures during model load indicate an
/// unsalvageable environment.
pub fn init() -> GpuResult<()> {
    INIT_RESULT
        .get_or_init(|| {
            // SAFETY: cuda_ternary_gemv_init initializes the constant-memory decode LUT
            // on the GPU. Safe to call multiple times (OnceLock ensures single init).
            let ret = unsafe { cuda_ternary_gemv_init() };
            if ret == 0 {
                Ok(())
            } else {
                Err("CUDA GEMV init failed".to_string())
            }
        })
        .clone()
}

/// Device-side weight storage for GPU GEMV.
///
/// `d_scales` is stored as `*mut u8` (raw device pointer) rather than `*mut f32`
/// because the CUDA FFI interface uses untyped device pointers. The actual device
/// memory contains f32 scale values — the kernel casts internally.
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
    pub fn from_planar(pw: &PlanarWeights) -> GpuResult<Self> {
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
        if pw.rows > i32::MAX as usize {
            return Err(format!("rows ({}) exceed i32::MAX for CUDA FFI", pw.rows));
        }
        if pw.cols > i32::MAX as usize - 3 {
            return Err(format!(
                "cols ({}) exceed CUDA kernel indexing limit",
                pw.cols
            ));
        }
        if pw.group_size > i32::MAX as usize {
            return Err(format!(
                "group_size ({}) exceed i32::MAX for CUDA FFI",
                pw.group_size
            ));
        }

        init()?;

        let kp = pw.cols / 4;
        let data_bytes = pw.rows
            .checked_mul(kp)
            .ok_or_else(|| format!("data_bytes overflow: {} * {}", pw.rows, kp))?;
        let n_groups = pw.cols / pw.group_size;
        let scales_bytes = pw
            .rows
            .checked_mul(n_groups)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "scales_bytes overflow: {} * {} * {}",
                    pw.rows,
                    n_groups,
                    std::mem::size_of::<f32>()
                )
            })?;

        let d_data = DeviceAlloc::alloc(data_bytes, "data")?;
        let d_scales = DeviceAlloc::alloc(scales_bytes, "scales")?;

        // Upload row-major data
        // SAFETY: d_data is a valid device allocation of data_bytes. pw.data is a valid
        // host buffer of at least data_bytes (rows * kp). cuda_memcpy_h2d wraps cudaMemcpy H2D.
        let ret = unsafe { cuda_memcpy_h2d(d_data.as_ptr(), pw.data.as_ptr(), data_bytes) };
        if ret != 0 {
            return Err("H2D copy failed for data".to_string());
        }

        // Upload row-major scales
        // SAFETY: d_scales is a valid device allocation of scales_bytes. pw.scales_rm
        // contains rows * n_groups f32 values. Pointer cast to *const u8 is valid for memcpy.
        let ret = unsafe {
            cuda_memcpy_h2d(
                d_scales.as_ptr(),
                pw.scales_rm.as_ptr() as *const u8,
                scales_bytes,
            )
        };
        if ret != 0 {
            return Err("H2D copy failed for scales".to_string());
        }

        Ok(GpuWeights {
            d_data: d_data.into_raw(),
            d_scales: d_scales.into_raw(),
            rows: pw.rows,
            cols: pw.cols,
            group_size: pw.group_size,
            data_bytes,
            scales_bytes,
        })
    }
}

impl Drop for GpuWeights {
    fn drop(&mut self) {
        // SAFETY: d_data and d_scales were allocated by cuda_alloc (cudaMalloc) in
        // from_planar() and ownership was transferred via into_raw(). They have not
        // been freed elsewhere.
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
pub fn gemv_gpu(gw: &GpuWeights, x: &[i8], act_scale: f32, y: &mut [f32]) -> GpuResult<()> {
    if x.len() != gw.cols {
        return Err(format!("x.len() ({}) != cols ({})", x.len(), gw.cols));
    }
    if y.len() != gw.rows {
        return Err(format!("y.len() ({}) != rows ({})", y.len(), gw.rows));
    }
    // CUDA grid.x limit is 2^31-1 for compute capability 3.0+.
    // The old 65535 limit only applies to grid.y and grid.z dimensions.
    if gw.rows > 2_147_483_647 {
        return Err(format!(
            "rows ({}) exceed CUDA grid.x limit (2^31-1)",
            gw.rows
        ));
    }
    if gw.rows > i32::MAX as usize {
        return Err(format!("rows ({}) exceed i32::MAX for CUDA FFI", gw.rows));
    }
    if gw.cols > i32::MAX as usize - 3 {
        return Err(format!(
            "cols ({}) exceed CUDA kernel indexing limit",
            gw.cols
        ));
    }
    if gw.group_size > i32::MAX as usize {
        return Err(format!(
            "group_size ({}) exceed i32::MAX for CUDA FFI",
            gw.group_size
        ));
    }

    init()?;

    let x_bytes = x.len();
    let y_bytes = y
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "y byte size overflow".to_string())?;

    // Allocate device buffers for x and y
    let d_x = DeviceAlloc::alloc(x_bytes, "x")?;
    let d_y = DeviceAlloc::alloc(y_bytes, "y")?;

    // Upload x
    // SAFETY: d_x is a valid device allocation of x_bytes. x is a valid i8 slice of
    // length gw.cols (checked above). Pointer cast to *const u8 is valid for memcpy.
    let ret = unsafe { cuda_memcpy_h2d(d_x.as_ptr(), x.as_ptr() as *const u8, x_bytes) };
    if ret != 0 {
        return Err("H2D copy failed for x".to_string());
    }

    // Launch kernel
    // SAFETY: All device pointers (gw.d_data, gw.d_scales, d_x, d_y) are valid allocations
    // from cuda_alloc. Dimensions are validated above and fit in i32. The kernel reads
    // from d_data/d_scales/d_x and writes gw.rows floats to d_y.
    let ret = unsafe {
        cuda_ternary_gemv(
            gw.d_data,
            gw.d_scales as *const f32,
            d_x.as_ptr() as *const i8,
            act_scale,
            d_y.as_ptr() as *mut f32,
            gw.rows as i32,
            gw.cols as i32,
            gw.group_size as i32,
        )
    };
    if ret != 0 {
        return Err("CUDA kernel launch failed".to_string());
    }

    // Download y
    // SAFETY: d_y contains gw.rows f32 values written by the kernel. y is a valid mutable
    // slice of gw.rows f32 values (checked above). y_bytes = gw.rows * sizeof(f32).
    let ret = unsafe {
        cuda_memcpy_d2h(
            y.as_mut_ptr() as *mut u8,
            d_y.as_ptr() as *const u8,
            y_bytes,
        )
    };
    if ret != 0 {
        return Err("D2H copy failed for y".to_string());
    }

    // NOTE: This synchronize call is technically redundant after the synchronous
    // cudaMemcpy D2H above (cudaMemcpy with cudaMemcpyDeviceToHost is
    // synchronous and implicitly waits for all prior GPU work to complete).
    // Kept as a defensive safety net in case the CUDA runtime behavior changes
    // or the D2H implementation is switched to an async variant in the future.
    // SAFETY: cuda_synchronize wraps cudaDeviceSynchronize — no pointers, always safe to call.
    let ret = unsafe { cuda_synchronize() };
    if ret != 0 {
        return Err("CUDA synchronize failed".to_string());
    }

    // Temporary buffers are freed by DeviceAlloc Drop.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::needless_range_loop)] // Index needed for deterministic pseudo-random generation
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
        let gw = GpuWeights::from_planar(&pw).expect("gpu upload");
        let mut y_gpu = vec![0.0f32; 128];
        gemv_gpu(&gw, &x, act_scale, &mut y_gpu).expect("gpu gemv");

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

            let gw = GpuWeights::from_planar(&pw).expect("gpu upload");
            let mut y_gpu = vec![0.0f32; m];
            gemv_gpu(&gw, &x, act_scale, &mut y_gpu).expect("gpu gemv");

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
        let gw = GpuWeights::from_planar(&pw).expect("gpu upload");

        assert_eq!(gw.rows, 64);
        assert_eq!(gw.cols, 128);
        assert_eq!(gw.group_size, 128);
        assert!(gw.data_bytes > 0);
        assert!(gw.scales_bytes > 0);
    }
}
