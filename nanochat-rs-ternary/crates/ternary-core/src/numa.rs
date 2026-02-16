//! NUMA-aware memory allocation for dual-socket systems.
//!
//! Optimizes memory access patterns on multi-socket NUMA architectures
//! like dual AMD EPYC by allocating memory on the socket that will access it.
//!
//! Key techniques:
//! - Node-local allocation via mbind()
//! - CPU affinity detection via getcpu()
//! - First-touch allocation policy
//! - Huge pages for reduced TLB pressure

use std::alloc::{handle_alloc_error, Layout};
use std::sync::atomic::{AtomicBool, Ordering};

static NUMA_AVAILABLE: AtomicBool = AtomicBool::new(false);
static NUMA_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// NUMA allocator for node-local memory.
pub struct NumaAllocator;

impl NumaAllocator {
    /// Initialize NUMA subsystem and detect availability.
    pub fn init() {
        if NUMA_INITIALIZED.swap(true, Ordering::SeqCst) {
            return; // Already initialized
        }

        #[cfg(target_os = "linux")]
        {
            // Check if NUMA is available by reading /proc/self/numa_maps
            if std::path::Path::new("/sys/devices/system/node/node1").exists() {
                NUMA_AVAILABLE.store(true, Ordering::SeqCst);
                log::info!("NUMA subsystem detected and initialized");
            } else {
                log::info!("NUMA not available (single-socket system)");
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            log::info!("NUMA allocation not supported on this platform");
        }
    }

    /// Check if NUMA is available on this system.
    pub fn is_available() -> bool {
        NUMA_AVAILABLE.load(Ordering::SeqCst)
    }

    /// Get the current NUMA node for the calling thread.
    #[cfg(target_os = "linux")]
    pub fn current_node() -> i32 {
        unsafe {
            let mut cpu: libc::c_uint = 0;
            let mut node: libc::c_uint = 0;

            #[cfg(target_arch = "x86_64")]
            {
                // Use getcpu syscall
                if libc::syscall(
                    libc::SYS_getcpu,
                    &mut cpu,
                    &mut node,
                    std::ptr::null_mut::<libc::c_void>(),
                ) == 0
                {
                    return node as i32;
                }
            }

            // Fallback: assume node 0
            0
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn current_node() -> i32 {
        0
    }

    /// Allocate memory on a specific NUMA node.
    ///
    /// Falls back to standard allocation if NUMA is unavailable.
    #[cfg(target_os = "linux")]
    pub fn alloc_on_node(size: usize, node: i32) -> Option<*mut u8> {
        if !Self::is_available() {
            return None; // Let caller use standard allocation
        }

        unsafe {
            // Use mmap for large allocations
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return None;
            }

            // Bind to specific NUMA node using mbind
            #[cfg(target_arch = "x86_64")]
            {
                assert!(
                    (node as u32) < 64,
                    "NUMA node {} out of range for 64-bit nodemask",
                    node
                );
                let mut nodemask: u64 = 1u64 << (node as u32);
                let maxnode = 64;

                // MPOL_BIND = 2, MPOL_MF_STRICT = 1, MPOL_MF_MOVE = 2
                let result = libc::syscall(
                    237, // SYS_mbind on x86_64
                    ptr,
                    size,
                    2, // MPOL_BIND
                    &mut nodemask as *mut _ as *mut libc::c_void,
                    maxnode,
                    1 | 2, // MPOL_MF_STRICT | MPOL_MF_MOVE
                );

                if result != 0 {
                    log::warn!("mbind failed for node {}, using default policy", node);
                }
            }

            Some(ptr as *mut u8)
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn alloc_on_node(_size: usize, _node: i32) -> Option<*mut u8> {
        None
    }

    /// Deallocate NUMA-allocated memory.
    ///
    /// # Safety
    /// - `ptr` must have been allocated by `alloc_on_node` with the same `size`
    /// - `ptr` must not be used after calling this function
    /// - This function must not be called twice on the same pointer
    #[cfg(target_os = "linux")]
    pub unsafe fn dealloc(ptr: *mut u8, size: usize) {
        if !ptr.is_null() {
            libc::munmap(ptr as *mut libc::c_void, size);
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub unsafe fn dealloc(_ptr: *mut u8, _size: usize) {
        // No-op on non-Linux
    }

    /// Enable huge pages for the given memory region.
    ///
    /// Reduces TLB pressure for large weight matrices.
    #[cfg(target_os = "linux")]
    pub fn enable_huge_pages(ptr: *mut u8, size: usize) -> bool {
        unsafe {
            // MADV_HUGEPAGE = 14
            let result = libc::madvise(ptr as *mut libc::c_void, size, 14);
            result == 0
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn enable_huge_pages(_ptr: *mut u8, _size: usize) -> bool {
        false
    }
}

/// NUMA-aware vector with node-local allocation.
pub struct NumaVec<T> {
    ptr: *mut T,
    len: usize,
    node: i32,
    alloc_kind: AllocKind,
}

enum AllocKind {
    NumaMmap { size: usize },
    StdAlloc { layout: Layout },
    None,
}

impl<T: Copy + Default> NumaVec<T> {
    /// Create a new NUMA-aware vector on the specified node.
    pub fn new_on_node(capacity: usize, node: i32) -> Self {
        if capacity == 0 || std::mem::size_of::<T>() == 0 {
            return Self {
                ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
                len: capacity,
                node,
                alloc_kind: AllocKind::None,
            };
        }

        let elem_size = std::mem::size_of::<T>();
        let size = capacity
            .checked_mul(elem_size)
            .expect("NumaVec allocation size overflow");
        let required_align = std::mem::align_of::<T>().max(128);

        let alloc_std = |size: usize, required_align: usize| -> (*mut T, AllocKind) {
            let layout = Layout::from_size_align(size, required_align)
                .expect("invalid fallback layout for NumaVec");
            let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            (ptr, AllocKind::StdAlloc { layout })
        };

        let (ptr, alloc_kind) = if let Some(numa_ptr) = NumaAllocator::alloc_on_node(size, node) {
            // Enable huge pages for large allocations
            if size >= 2 * 1024 * 1024 {
                // 2MB threshold
                NumaAllocator::enable_huge_pages(numa_ptr, size);
            }
            if !(numa_ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
                log::warn!(
                    "NUMA allocation returned misaligned pointer for T (align={}), falling back to std alloc",
                    std::mem::align_of::<T>()
                );
                unsafe {
                    NumaAllocator::dealloc(numa_ptr, size);
                }
                alloc_std(size, required_align)
            } else {
                (numa_ptr as *mut T, AllocKind::NumaMmap { size })
            }
        } else {
            // Fallback to standard aligned allocation
            alloc_std(size, required_align)
        };

        // Validate pointer alignment before unsafe writes.
        if ptr.is_null() || !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            match &alloc_kind {
                AllocKind::NumaMmap { size } => unsafe {
                    NumaAllocator::dealloc(ptr as *mut u8, *size);
                },
                AllocKind::StdAlloc { layout } => unsafe {
                    std::alloc::dealloc(ptr as *mut u8, *layout);
                },
                AllocKind::None => {}
            }
            panic!(
                "NumaVec allocation produced invalid pointer (null or misaligned for align={})",
                std::mem::align_of::<T>()
            );
        }

        // Initialize with default values (first-touch policy)
        unsafe {
            for i in 0..capacity {
                std::ptr::write(ptr.add(i), T::default());
            }
        }

        Self {
            ptr,
            len: capacity,
            node,
            alloc_kind,
        }
    }

    /// Create on the current NUMA node.
    pub fn new(capacity: usize) -> Self {
        let node = NumaAllocator::current_node();
        Self::new_on_node(capacity, node)
    }

    /// Get slice of data.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get mutable slice of data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Get NUMA node this vector is allocated on.
    pub fn node(&self) -> i32 {
        self.node
    }
}

impl<T> Drop for NumaVec<T> {
    fn drop(&mut self) {
        match &self.alloc_kind {
            AllocKind::NumaMmap { size } => unsafe {
                NumaAllocator::dealloc(self.ptr as *mut u8, *size);
            },
            AllocKind::StdAlloc { layout } => unsafe {
                std::alloc::dealloc(self.ptr as *mut u8, *layout);
            },
            AllocKind::None => {}
        }
    }
}

unsafe impl<T: Send + Copy + Default> Send for NumaVec<T> {}
unsafe impl<T: Sync + Copy + Default> Sync for NumaVec<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_init() {
        NumaAllocator::init();
        // Should not panic
    }

    #[test]
    fn test_numa_availability() {
        NumaAllocator::init();
        let available = NumaAllocator::is_available();
        println!("NUMA available: {}", available);
        // Just check it doesn't panic
    }

    #[test]
    fn test_current_node() {
        NumaAllocator::init();
        let node = NumaAllocator::current_node();
        println!("Current NUMA node: {}", node);
        assert!(node >= 0);
    }

    #[test]
    fn test_numa_vec_creation() {
        NumaAllocator::init();
        let vec = NumaVec::<f32>::new(1024);
        assert_eq!(vec.as_slice().len(), 1024);
        assert!(vec.node() >= 0);
    }

    #[test]
    fn test_numa_vec_write_read() {
        NumaAllocator::init();
        let mut vec = NumaVec::<i32>::new(100);

        // Write
        for (i, val) in vec.as_mut_slice().iter_mut().enumerate() {
            *val = i as i32;
        }

        // Read
        for (i, &val) in vec.as_slice().iter().enumerate() {
            assert_eq!(val, i as i32);
        }
    }

    #[test]
    fn test_numa_vec_on_specific_node() {
        NumaAllocator::init();
        let vec = NumaVec::<u8>::new_on_node(2048, 0);
        assert_eq!(vec.node(), 0);
        assert_eq!(vec.as_slice().len(), 2048);
    }
}
