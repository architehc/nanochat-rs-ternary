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
// NUMA FFI declarations (Linux only, gated on has_numa cfg)
// ============================================================

#[cfg(all(target_os = "linux", has_numa))]
extern "C" {
    fn numa_available() -> libc::c_int;
    fn numa_alloc_onnode(size: libc::size_t, node: libc::c_int) -> *mut libc::c_void;
    fn numa_free(start: *mut libc::c_void, size: libc::size_t);
    fn numa_max_node() -> libc::c_int;
}

/// Check if NUMA is available at runtime.
///
/// Returns true if libnuma is linked and NUMA is active on the system.
/// Always returns false on non-Linux or when compiled without libnuma.
pub fn numa_is_available() -> bool {
    #[cfg(all(target_os = "linux", has_numa))]
    {
        // numa_available() returns 0 on success, -1 on failure
        unsafe { numa_available() >= 0 }
    }
    #[cfg(not(all(target_os = "linux", has_numa)))]
    {
        false
    }
}

/// Returns the maximum NUMA node index, or 0 if NUMA is not available.
pub fn numa_max_node_id() -> usize {
    #[cfg(all(target_os = "linux", has_numa))]
    {
        if numa_is_available() {
            let max = unsafe { numa_max_node() };
            if max >= 0 {
                max as usize
            } else {
                0
            }
        } else {
            0
        }
    }
    #[cfg(not(all(target_os = "linux", has_numa)))]
    {
        0
    }
}

// ============================================================
// AlignedVec — 128-byte aligned allocation
// ============================================================

/// Allocation source for AlignedVec: standard allocator or NUMA node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocSource {
    /// Standard aligned allocation via std::alloc
    Standard,
    /// NUMA-aware allocation on a specific node (Linux only)
    #[cfg(all(target_os = "linux", has_numa))]
    Numa { node: usize },
}

/// A vector with guaranteed 128-byte alignment, suitable for AVX-512 NT-loads.
///
/// Supports optional NUMA-aware allocation for dual-socket systems.
/// When allocated with `new_on_node()`, memory is placed on the specified
/// NUMA node for first-touch locality. Falls back to standard allocation
/// if NUMA is not available.
pub struct AlignedVec<T: Copy + Default> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    alloc_source: AllocSource,
}

// Safety: AlignedVec owns its data
unsafe impl<T: Copy + Default + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Default + Sync> Sync for AlignedVec<T> {}

impl<T: Copy + Default> AlignedVec<T> {
    /// Allocate a zeroed, 128-byte aligned vector of `len` elements.
    pub fn new_zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                cap: 0,
                alloc_source: AllocSource::Standard,
            };
        }

        let size = std::mem::size_of::<T>() * len;
        let align = 128; // 128-byte alignment for AVX-512

        let layout = Layout::from_size_align(size, align).expect("invalid layout");

        // Safety: layout has nonzero size
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut T).expect("allocation failed");

        // Advise huge pages for large buffers (>= 2MB)
        #[cfg(target_os = "linux")]
        {
            if size >= 2 * 1024 * 1024 {
                unsafe {
                    libc::madvise(raw as *mut libc::c_void, size, libc::MADV_HUGEPAGE);
                }
            }
        }

        Self {
            ptr,
            len,
            cap: len,
            alloc_source: AllocSource::Standard,
        }
    }

    /// Allocate a zeroed, 128-byte aligned vector of `len` elements on a specific NUMA node.
    ///
    /// On dual-socket EPYC systems, this pins the allocation to the specified socket's
    /// local memory for optimal bandwidth. The NUMA node index is typically 0 for socket 0
    /// and 1 for socket 1, but this depends on the system topology.
    ///
    /// Falls back to `new_zeroed()` if:
    /// - NUMA is not available at runtime
    /// - The platform is not Linux
    /// - The requested node exceeds the maximum node index
    /// - The NUMA allocation fails
    ///
    /// The resulting allocation maintains 128-byte alignment because `numa_alloc_onnode`
    /// returns page-aligned (4096+) memory via mmap, which exceeds our 128-byte requirement.
    pub fn new_on_node(len: usize, node: usize) -> Self {
        if len == 0 {
            return Self::new_zeroed(0);
        }

        #[cfg(all(target_os = "linux", has_numa))]
        {
            if !numa_is_available() {
                return Self::new_zeroed(len);
            }

            let max_node = numa_max_node_id();
            if node > max_node {
                return Self::new_zeroed(len);
            }

            let elem_size = std::mem::size_of::<T>();
            let size = elem_size * len;

            // numa_alloc_onnode returns page-aligned (4096+) memory via mmap,
            // which is automatically >= 128-byte aligned.
            let raw = unsafe { numa_alloc_onnode(size, node as libc::c_int) };
            if raw.is_null() {
                // NUMA allocation failed, fall back to standard
                return Self::new_zeroed(len);
            }

            // Zero the memory (Linux mmap typically returns zeroed pages,
            // but be explicit for safety)
            unsafe {
                std::ptr::write_bytes(raw as *mut u8, 0, size);
            }

            let ptr = NonNull::new(raw as *mut T).expect("NUMA alloc returned null");

            // Verify page alignment satisfies our 128-byte requirement
            debug_assert!(
                (raw as usize).is_multiple_of(128),
                "numa_alloc_onnode returned non-128-byte-aligned pointer: {:#x}",
                raw as usize
            );

            // Advise huge pages for large buffers (>= 2MB)
            if size >= 2 * 1024 * 1024 {
                unsafe {
                    libc::madvise(raw, size as libc::size_t, libc::MADV_HUGEPAGE);
                }
            }

            Self {
                ptr,
                len,
                cap: size, // Store byte size for numa_free in Drop
                alloc_source: AllocSource::Numa { node },
            }
        }

        #[cfg(not(all(target_os = "linux", has_numa)))]
        {
            let _ = node;
            Self::new_zeroed(len)
        }
    }

    /// Returns the NUMA node this allocation was placed on, if any.
    pub fn numa_node(&self) -> Option<usize> {
        #[cfg(all(target_os = "linux", has_numa))]
        {
            match self.alloc_source {
                AllocSource::Numa { node } => Some(node),
                _ => None,
            }
        }
        #[cfg(not(all(target_os = "linux", has_numa)))]
        {
            None
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
        match self.alloc_source {
            AllocSource::Standard => {
                let size = std::mem::size_of::<T>() * self.cap;
                let layout = Layout::from_size_align(size, 128).unwrap();
                unsafe {
                    alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
                }
            }
            #[cfg(all(target_os = "linux", has_numa))]
            AllocSource::Numa { .. } => {
                // For NUMA allocations, cap stores the byte size passed to numa_alloc_onnode
                unsafe {
                    numa_free(self.ptr.as_ptr() as *mut libc::c_void, self.cap);
                }
            }
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
        // Preserve NUMA placement when cloning
        let mut new = match self.alloc_source {
            AllocSource::Standard => Self::new_zeroed(self.len),
            #[cfg(all(target_os = "linux", has_numa))]
            AllocSource::Numa { node } => Self::new_on_node(self.len, node),
        };
        if self.len > 0 {
            new.copy_from_slice(self);
        }
        new
    }
}

fn round_up(x: usize, align: usize) -> usize {
    x.div_ceil(align) * align
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
        assert!(cols.is_multiple_of(4), "cols must be divisible by 4");
        assert!(
            cols.is_multiple_of(group_size),
            "cols must be divisible by group_size"
        );
        assert!(
            group_size.is_multiple_of(4),
            "group_size must be divisible by 4"
        );

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
        unsafe {
            *ptr = 42.0;
        }
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
    fn test_huge_pages_large_alloc() {
        // Allocate > 2MB to trigger MADV_HUGEPAGE advisory
        let n = 1024 * 1024; // 4MB of f32
        let v: AlignedVec<f32> = AlignedVec::new_zeroed(n);
        assert_eq!(v.len(), n);
        assert_eq!(v.as_ptr() as usize % 128, 0);
        // Verify memory is zeroed and accessible
        assert_eq!(v[0], 0.0);
        assert_eq!(v[n - 1], 0.0);
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

    // ============================================================
    // NUMA tests
    // ============================================================

    #[test]
    fn test_numa_available_check() {
        // Informational test: reports whether NUMA is available on this system
        let available = numa_is_available();
        let max_node = numa_max_node_id();
        println!("NUMA available: {}, max node: {}", available, max_node);
        // This test always passes — it just reports NUMA status
        if available {
            assert!(max_node < 256, "unreasonable max_node value: {}", max_node);
        } else {
            // When NUMA is not available, max_node should be 0
            assert_eq!(max_node, 0);
        }
    }

    #[test]
    fn test_alloc_on_node_0() {
        // Attempt NUMA allocation on node 0
        let v: AlignedVec<u8> = AlignedVec::new_on_node(256, 0);
        assert_eq!(v.len(), 256);
        // Must maintain 128-byte alignment regardless of NUMA availability
        assert_eq!(
            v.as_ptr() as usize % 128,
            0,
            "NUMA allocation not 128-byte aligned"
        );
        // Memory must be zeroed
        for &b in v.iter() {
            assert_eq!(b, 0, "NUMA allocation not zeroed");
        }

        if numa_is_available() {
            assert_eq!(v.numa_node(), Some(0));
        } else {
            // Fell back to standard allocation
            assert_eq!(v.numa_node(), None);
        }
    }

    #[test]
    fn test_numa_alloc_on_node_f32() {
        // Test NUMA allocation with f32 type
        let v: AlignedVec<f32> = AlignedVec::new_on_node(64, 0);
        assert_eq!(v.len(), 64);
        assert_eq!(v.as_ptr() as usize % 128, 0);
        // Verify zeroed
        for &val in v.iter() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_numa_alloc_write_read() {
        // Verify NUMA-allocated memory is read/writable
        let mut v: AlignedVec<f32> = AlignedVec::new_on_node(4, 0);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        v[3] = 4.0;
        assert_eq!(v[0], 1.0);
        assert_eq!(v[3], 4.0);
    }

    #[test]
    fn test_numa_alloc_empty() {
        let v: AlignedVec<u8> = AlignedVec::new_on_node(0, 0);
        assert!(v.is_empty());
        assert_eq!(v.numa_node(), None); // empty alloc is standard
    }

    #[test]
    fn test_numa_alloc_invalid_node_falls_back() {
        // Requesting a node beyond max should fall back to standard alloc
        let v: AlignedVec<u8> = AlignedVec::new_on_node(256, 999);
        assert_eq!(v.len(), 256);
        assert_eq!(v.as_ptr() as usize % 128, 0);
        // Should have fallen back to standard since node 999 doesn't exist
        assert_eq!(v.numa_node(), None);
    }

    #[test]
    fn test_numa_alloc_clone_preserves_node() {
        let mut v: AlignedVec<f32> = AlignedVec::new_on_node(4, 0);
        v[0] = 42.0;
        let v2 = v.clone();
        assert_eq!(v2[0], 42.0);
        assert_eq!(v2.len(), 4);
        assert_eq!(v2.as_ptr() as usize % 128, 0);
        // Clone should preserve the NUMA node
        assert_eq!(v.numa_node(), v2.numa_node());
    }

    #[test]
    fn test_numa_alloc_large_buffer() {
        // Test large NUMA allocation (triggers huge page advisory)
        let n = 1024 * 1024; // 4MB of f32
        let v: AlignedVec<f32> = AlignedVec::new_on_node(n, 0);
        assert_eq!(v.len(), n);
        assert_eq!(v.as_ptr() as usize % 128, 0);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[n - 1], 0.0);
    }
}
