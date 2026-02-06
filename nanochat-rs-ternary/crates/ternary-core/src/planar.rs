//! Planar SoA weight storage — THE runtime format.
//!
//! All runtime weight access uses planar SoA layout, not interleaved.
//! This matches the C `PlanarWeights` struct from ternary_final.c exactly.
//!
//! Layout:
//! - `data`: [rows * kp] row-major packed ternary bytes
//! - `data_colmaj`: [kp * rows_padded] column-major transpose, 128B aligned
//! - `scales_rm`: [rows * gprow] row-major per-group scales
//! - `scales_gm`: [gprow * rows_padded] group-major scales for SIMD
//! - `rows_padded`: round_up(rows, 64) for NT-load alignment

use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::pack::{pack_matrix, PackedMatrix};

// ============================================================
// AlignedVec — 128-byte aligned allocation
// ============================================================

/// A vector with guaranteed 128-byte alignment, suitable for AVX-512 NT-loads.
pub struct AlignedVec<T: Copy + Default> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

// Safety: AlignedVec owns its data
unsafe impl<T: Copy + Default + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Default + Sync> Sync for AlignedVec<T> {}

impl<T: Copy + Default> AlignedVec<T> {
    /// Allocate a zeroed, 128-byte aligned vector of `len` elements.
    pub fn new_zeroed(len: usize) -> Self {
        if len == 0 {
            return Self { ptr: NonNull::dangling(), len: 0, cap: 0 };
        }

        let size = std::mem::size_of::<T>() * len;
        let align = 128; // 128-byte alignment for AVX-512

        let layout = Layout::from_size_align(size, align).expect("invalid layout");

        // Safety: layout has nonzero size
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut T).expect("allocation failed");

        Self {
            ptr,
            len,
            cap: len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T: Copy + Default> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy + Default> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy + Default> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        let size = std::mem::size_of::<T>() * self.cap;
        let layout = Layout::from_size_align(size, 128).unwrap();
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T: Copy + Default + std::fmt::Debug> std::fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedVec(len={}, align=128)", self.len)
    }
}

impl<T: Copy + Default> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new = Self::new_zeroed(self.len);
        if self.len > 0 {
            new.copy_from_slice(self);
        }
        new
    }
}

fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) / align * align
}

// ============================================================
// PlanarWeights
// ============================================================

/// Planar SoA weight storage matching the C PlanarWeights struct.
///
/// This is THE runtime format for all GEMV kernels.
#[derive(Debug, Clone)]
pub struct PlanarWeights {
    /// Packed ternary bytes, row-major. [rows * kp] where kp = cols/4.
    pub data: AlignedVec<u8>,

    /// Packed ternary bytes, column-major. [kp * rows_padded].
    /// data_colmaj[c * rows_padded + r] = data[r * kp + c]
    pub data_colmaj: AlignedVec<u8>,

    /// Per-group scales, row-major. [rows * gprow] where gprow = cols/group_size.
    pub scales_rm: AlignedVec<f32>,

    /// Per-group scales, group-major (transposed). [gprow * rows_padded].
    /// scales_gm[g * rows_padded + r] = scales_rm[r * gprow + g]
    pub scales_gm: AlignedVec<f32>,

    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub rows_padded: usize,
}

impl PlanarWeights {
    /// Create PlanarWeights from row-major float weights.
    ///
    /// Quantizes to ternary, packs, and creates both row-major and column-major
    /// layouts with proper alignment.
    pub fn from_row_major(weights: &[f32], rows: usize, cols: usize, group_size: usize) -> Self {
        assert_eq!(weights.len(), rows * cols);
        assert!(cols % 4 == 0, "cols must be divisible by 4");
        assert!(cols % group_size == 0, "cols must be divisible by group_size");
        assert!(group_size % 4 == 0, "group_size must be divisible by 4");

        let pm = pack_matrix(weights, rows, cols, group_size);
        Self::from_packed_matrix(&pm)
    }

    /// Create PlanarWeights from an already-packed PackedMatrix.
    pub fn from_packed_matrix(pm: &PackedMatrix) -> Self {
        let rows = pm.rows;
        let cols = pm.cols;
        let group_size = pm.group_size;
        let kp = cols / 4;
        let gprow = cols / group_size;
        let rows_padded = round_up(rows, 64);

        // Row-major packed data
        let mut data = AlignedVec::new_zeroed(rows * kp);
        data[..pm.packed.len()].copy_from_slice(&pm.packed);

        // Column-major transpose
        let mut data_colmaj = AlignedVec::new_zeroed(kp * rows_padded);
        for r in 0..rows {
            for c in 0..kp {
                data_colmaj[c * rows_padded + r] = data[r * kp + c];
            }
        }

        // Row-major scales
        let mut scales_rm = AlignedVec::new_zeroed(rows * gprow);
        scales_rm[..pm.scales.len()].copy_from_slice(&pm.scales);

        // Group-major scales (transposed)
        let mut scales_gm = AlignedVec::new_zeroed(gprow * rows_padded);
        for r in 0..rows {
            for g in 0..gprow {
                scales_gm[g * rows_padded + r] = scales_rm[r * gprow + g];
            }
        }

        Self {
            data,
            data_colmaj,
            scales_rm,
            scales_gm,
            rows,
            cols,
            group_size,
            rows_padded,
        }
    }

    /// Create PlanarWeights from raw packed bytes and scales.
    pub fn from_packed(
        packed: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Self {
        let kp = cols / 4;
        let gprow = cols / group_size;
        assert_eq!(packed.len(), rows * kp);
        assert_eq!(scales.len(), rows * gprow);

        let pm = PackedMatrix {
            packed: packed.to_vec(),
            scales: scales.to_vec(),
            rows,
            cols,
            group_size,
        };
        Self::from_packed_matrix(&pm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_alignment() {
        let v: AlignedVec<u8> = AlignedVec::new_zeroed(256);
        assert_eq!(v.as_ptr() as usize % 128, 0, "not 128-byte aligned");
        assert_eq!(v.len(), 256);
    }

    #[test]
    fn test_aligned_vec_f32_alignment() {
        let v: AlignedVec<f32> = AlignedVec::new_zeroed(64);
        assert_eq!(v.as_ptr() as usize % 128, 0, "not 128-byte aligned");
        assert_eq!(v.len(), 64);
    }

    #[test]
    fn test_aligned_vec_zeroed() {
        let v: AlignedVec<u8> = AlignedVec::new_zeroed(128);
        for &b in v.iter() {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn test_aligned_vec_write_read() {
        let mut v: AlignedVec<f32> = AlignedVec::new_zeroed(4);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        v[3] = 4.0;
        assert_eq!(v[0], 1.0);
        assert_eq!(v[3], 4.0);
    }

    #[test]
    fn test_aligned_vec_empty() {
        let v: AlignedVec<u8> = AlignedVec::new_zeroed(0);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_planar_weights_basic() {
        let rows = 64;
        let cols = 128;
        let gs = 128;

        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();

        let pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

        assert_eq!(pw.rows, rows);
        assert_eq!(pw.cols, cols);
        assert_eq!(pw.group_size, gs);
        assert_eq!(pw.rows_padded, 64); // round_up(64, 64) = 64
        assert_eq!(pw.data.len(), rows * (cols / 4));
        assert_eq!(pw.data_colmaj.len(), (cols / 4) * pw.rows_padded);
        assert_eq!(pw.scales_rm.len(), rows * (cols / gs));
        assert_eq!(pw.scales_gm.len(), (cols / gs) * pw.rows_padded);
    }

    #[test]
    fn test_planar_weights_alignment() {
        let weights = vec![1.0f32; 128 * 128];
        let pw = PlanarWeights::from_row_major(&weights, 128, 128, 128);

        assert_eq!(pw.data.as_ptr() as usize % 128, 0);
        assert_eq!(pw.data_colmaj.as_ptr() as usize % 128, 0);
        assert_eq!(pw.scales_rm.as_ptr() as usize % 128, 0);
        assert_eq!(pw.scales_gm.as_ptr() as usize % 128, 0);
    }

    #[test]
    fn test_planar_weights_colmaj_transpose() {
        let rows = 4;
        let cols = 8;
        let gs = 4; // small group size for testing
        let kp = cols / 4;

        // All +1 weights
        let weights = vec![1.0f32; rows * cols];
        let pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

        // Verify transpose: data_colmaj[c * rows_padded + r] == data[r * kp + c]
        for r in 0..rows {
            for c in 0..kp {
                assert_eq!(
                    pw.data_colmaj[c * pw.rows_padded + r],
                    pw.data[r * kp + c],
                    "transpose mismatch at r={}, c={}",
                    r,
                    c
                );
            }
        }
    }

    #[test]
    fn test_planar_weights_scales_gm_transpose() {
        let rows = 4;
        let cols = 256;
        let gs = 128;
        let gprow = cols / gs;

        let mut weights = vec![0.0f32; rows * cols];
        // Make different groups have different scales
        for r in 0..rows {
            for c in 0..cols {
                weights[r * cols + c] = if c < 128 { 1.0 } else { 2.0 };
            }
        }
        let pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

        // Verify: scales_gm[g * rows_padded + r] == scales_rm[r * gprow + g]
        for r in 0..rows {
            for g in 0..gprow {
                assert_eq!(
                    pw.scales_gm[g * pw.rows_padded + r],
                    pw.scales_rm[r * gprow + g],
                    "scale transpose mismatch at r={}, g={}",
                    r,
                    g
                );
            }
        }
    }

    #[test]
    fn test_planar_weights_rows_padded() {
        // Non-aligned row count
        let weights = vec![1.0f32; 17 * 128];
        let pw = PlanarWeights::from_row_major(&weights, 17, 128, 128);
        assert_eq!(pw.rows_padded, 64); // round_up(17, 64) = 64

        let weights = vec![1.0f32; 65 * 128];
        let pw = PlanarWeights::from_row_major(&weights, 65, 128, 128);
        assert_eq!(pw.rows_padded, 128); // round_up(65, 64) = 128
    }

    #[test]
    fn test_aligned_vec_clone() {
        let mut v: AlignedVec<f32> = AlignedVec::new_zeroed(4);
        v[0] = 1.0;
        v[1] = 2.0;
        let v2 = v.clone();
        assert_eq!(v2[0], 1.0);
        assert_eq!(v2[1], 2.0);
        assert_eq!(v2.len(), 4);
        assert_eq!(v2.as_ptr() as usize % 128, 0);
    }

    #[test]
    fn test_aligned_vec_clone_empty() {
        let v: AlignedVec<u8> = AlignedVec::new_zeroed(0);
        let v2 = v.clone();
        assert!(v2.is_empty());
    }

    #[test]
    fn test_aligned_vec_debug() {
        let v: AlignedVec<u8> = AlignedVec::new_zeroed(16);
        let debug = format!("{:?}", v);
        assert!(debug.contains("AlignedVec"));
        assert!(debug.contains("16"));
    }

    #[test]
    fn test_aligned_vec_as_mut_ptr() {
        let mut v: AlignedVec<f32> = AlignedVec::new_zeroed(4);
        let ptr = v.as_mut_ptr();
        assert_eq!(ptr as usize % 128, 0);
        unsafe { *ptr = 42.0; }
        assert_eq!(v[0], 42.0);
    }

    #[test]
    fn test_aligned_vec_is_empty() {
        let v: AlignedVec<u8> = AlignedVec::new_zeroed(0);
        assert!(v.is_empty());

        let v2: AlignedVec<u8> = AlignedVec::new_zeroed(1);
        assert!(!v2.is_empty());
    }

    #[test]
    fn test_aligned_vec_deref_mut() {
        let mut v: AlignedVec<u8> = AlignedVec::new_zeroed(4);
        let slice: &mut [u8] = &mut v;
        slice[0] = 0xFF;
        assert_eq!(v[0], 0xFF);
    }

    #[test]
    fn test_aligned_vec_empty_deref_mut() {
        let mut v: AlignedVec<u8> = AlignedVec::new_zeroed(0);
        let slice: &mut [u8] = &mut v;
        assert!(slice.is_empty());
    }

    #[test]
    fn test_aligned_vec_empty_deref() {
        let v: AlignedVec<u8> = AlignedVec::new_zeroed(0);
        let slice: &[u8] = &v;
        assert!(slice.is_empty());
    }

    #[test]
    fn test_planar_weights_from_packed() {
        let rows = 4;
        let cols = 128;
        let gs = 128;

        let weights = vec![1.0f32; rows * cols];
        let pm = pack_matrix(&weights, rows, cols, gs);
        let pw = PlanarWeights::from_packed(&pm.packed, &pm.scales, rows, cols, gs);

        assert_eq!(pw.rows, rows);
        assert_eq!(pw.cols, cols);
        assert_eq!(pw.data.len(), rows * (cols / 4));
    }
}
