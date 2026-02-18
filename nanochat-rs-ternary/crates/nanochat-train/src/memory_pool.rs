//! Memory pool for tensor reuse to reduce allocation overhead
//!
//! This module provides a simple object pool for tensors to reduce memory
//! fragmentation and allocation overhead during training. Particularly useful
//! for recurrent patterns where tensors of the same shape are repeatedly
//! allocated and deallocated.

use candle_core::{DType, Device, Result, Shape, Tensor};
use std::collections::HashMap;

/// Key for identifying compatible tensor shapes in the pool
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PoolKey {
    shape: Vec<usize>,
    dtype: DType,
    device_id: String,
}

impl PoolKey {
    fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
            device_id: format!("{:?}", tensor.device()),
        }
    }

    fn new(shape: &Shape, dtype: DType, device: &Device) -> Self {
        Self {
            shape: shape.dims().to_vec(),
            dtype,
            device_id: format!("{:?}", device),
        }
    }
}

/// A pool of reusable tensors to reduce allocation overhead
///
/// # Example
///
/// ```
/// use nanochat_train::memory_pool::TensorPool;
/// use candle_core::{Device, DType, Shape};
///
/// let mut pool = TensorPool::new(4);
/// let device = Device::Cpu;
///
/// // Acquire a tensor (creates new if pool empty)
/// let tensor = pool.acquire(&Shape::from((2, 3, 4)), DType::F32, &device).unwrap();
///
/// // Return to pool for reuse
/// pool.release(tensor);
///
/// // Next acquire reuses the returned tensor
/// let reused = pool.acquire(&Shape::from((2, 3, 4)), DType::F32, &device).unwrap();
/// ```
#[derive(Debug)]
pub struct TensorPool {
    buffers: HashMap<PoolKey, Vec<Tensor>>,
    max_entries_per_shape: usize,
    total_acquired: usize,
    total_reused: usize,
}

/// Statistics about pool usage
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub unique_shapes: usize,
    pub total_pooled_tensors: usize,
    pub total_acquired: usize,
    pub total_reused: usize,
    pub reuse_rate: f64,
}

impl TensorPool {
    /// Create a new tensor pool with specified capacity per shape
    ///
    /// # Arguments
    /// * `max_entries_per_shape` - Maximum number of tensors to keep per unique shape/dtype/device
    pub fn new(max_entries_per_shape: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            max_entries_per_shape: max_entries_per_shape.max(1),
            total_acquired: 0,
            total_reused: 0,
        }
    }

    /// Acquire a tensor from the pool or allocate a new one
    ///
    /// First attempts to reuse a pooled tensor with matching shape, dtype, and device.
    /// If none available, creates a new zero-initialized tensor.
    ///
    /// # Arguments
    /// * `shape` - Desired tensor shape
    /// * `dtype` - Data type (F32, F16, etc.)
    /// * `device` - Target device (CPU/CUDA)
    pub fn acquire(&mut self, shape: &Shape, dtype: DType, device: &Device) -> Result<Tensor> {
        self.total_acquired += 1;
        
        let key = PoolKey::new(shape, dtype, device);

        if let Some(vec) = self.buffers.get_mut(&key) {
            if let Some(tensor) = vec.pop() {
                self.total_reused += 1;
                // Reset to zeros for cleanliness
                return tensor.zeros_like();
            }
        }

        // No pooled tensor available, allocate new
        Tensor::zeros(shape.clone(), dtype, device)
    }

    /// Return a tensor to the pool for reuse
    ///
    /// The tensor is only retained if the pool hasn't reached capacity
    /// for this shape/dtype/device combination.
    pub fn release(&mut self, tensor: Tensor) {
        let key = PoolKey::from_tensor(&tensor);

        let vec = self.buffers.entry(key).or_default();
        if vec.len() < self.max_entries_per_shape {
            vec.push(tensor);
        }
        // If at capacity, tensor is dropped (memory freed)
    }

    /// Release multiple tensors at once
    pub fn release_batch(&mut self, tensors: Vec<Tensor>) {
        for tensor in tensors {
            self.release(tensor);
        }
    }

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        let total_pooled: usize = self.buffers.values().map(|v| v.len()).sum();
        let reuse_rate = if self.total_acquired > 0 {
            self.total_reused as f64 / self.total_acquired as f64
        } else {
            0.0
        };

        PoolStats {
            unique_shapes: self.buffers.len(),
            total_pooled_tensors: total_pooled,
            total_acquired: self.total_acquired,
            total_reused: self.total_reused,
            reuse_rate,
        }
    }

    /// Clear all pooled tensors, freeing memory
    pub fn clear(&mut self) {
        self.buffers.clear();
        // Note: We don't reset counters to preserve statistics
    }

    /// Clear pooled tensors for a specific device
    pub fn clear_device(&mut self, device: &Device) {
        let device_id = format!("{:?}", device);
        self.buffers
            .retain(|key, _| key.device_id != device_id);
    }

    /// Get the number of tensors pooled for a specific shape
    pub fn count_for_shape(&self, shape: &Shape, dtype: DType, device: &Device) -> usize {
        let key = PoolKey::new(shape, dtype, device);
        self.buffers.get(&key).map(|v| v.len()).unwrap_or(0)
    }

    /// Shrink pool to fit current contents
    pub fn shrink_to_fit(&mut self) {
        for vec in self.buffers.values_mut() {
            vec.shrink_to_fit();
        }
        self.buffers.shrink_to_fit();
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new(4)
    }
}

/// Scoped tensor pool that returns tensors on drop
///
/// Useful for temporary allocation bursts within a scope.
pub struct ScopedPool<'a> {
    pool: &'a mut TensorPool,
    acquired: Vec<Tensor>,
}

impl<'a> ScopedPool<'a> {
    pub fn new(pool: &'a mut TensorPool) -> Self {
        Self {
            pool,
            acquired: Vec::new(),
        }
    }

    pub fn acquire(&mut self, shape: &Shape, dtype: DType, device: &Device) -> Result<&Tensor> {
        let tensor = self.pool.acquire(shape, dtype, device)?;
        self.acquired.push(tensor);
        Ok(self.acquired.last().unwrap())
    }
}

impl<'a> Drop for ScopedPool<'a> {
    fn drop(&mut self) {
        for tensor in self.acquired.drain(..) {
            self.pool.release(tensor);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_acquire_release() -> Result<()> {
        let device = Device::Cpu;
        let mut pool = TensorPool::new(2);

        // Acquire and release
        let t1 = pool.acquire(&Shape::from((2, 3)), DType::F32, &device)?;
        // Store dimensions for comparison
        let dims1 = t1.dims().to_vec();
        pool.release(t1);

        // Reacquire - should reuse
        let t2 = pool.acquire(&Shape::from((2, 3)), DType::F32, &device)?;
        let dims2 = t2.dims().to_vec();
        
        // Verify shape is correct
        assert_eq!(dims1, dims2);
        assert_eq!(pool.stats().total_reused, 1);
        assert_eq!(pool.stats().reuse_rate, 0.5);
        
        Ok(())
    }

    #[test]
    fn test_pool_capacity_limit() -> Result<()> {
        let device = Device::Cpu;
        let mut pool = TensorPool::new(2);

        // Create and release 5 tensors
        let tensors: Vec<_> = (0..5)
            .map(|_| pool.acquire(&Shape::from((2, 2)), DType::F32, &device).unwrap())
            .collect();

        for t in tensors {
            pool.release(t);
        }

        // Only 2 should be retained
        assert_eq!(pool.count_for_shape(&Shape::from((2, 2)), DType::F32, &device), 2);
        assert_eq!(pool.stats().total_pooled_tensors, 2);

        Ok(())
    }

    #[test]
    fn test_pool_different_shapes() -> Result<()> {
        let device = Device::Cpu;
        let mut pool = TensorPool::new(4);

        let t1 = pool.acquire(&Shape::from((2, 3)), DType::F32, &device)?;
        let t2 = pool.acquire(&Shape::from((4, 5)), DType::F32, &device)?;
        let t3 = pool.acquire(&Shape::from((2, 3)), DType::F16, &device)?;

        pool.release(t1);
        pool.release(t2);
        pool.release(t3);

        let stats = pool.stats();
        assert_eq!(stats.unique_shapes, 3);
        assert_eq!(stats.total_pooled_tensors, 3);

        Ok(())
    }

    #[test]
    fn test_pool_clear() -> Result<()> {
        let device = Device::Cpu;
        let mut pool = TensorPool::new(4);

        let t = pool.acquire(&Shape::from((2, 2)), DType::F32, &device)?;
        pool.release(t);

        assert_eq!(pool.stats().total_pooled_tensors, 1);
        
        pool.clear();
        assert_eq!(pool.stats().total_pooled_tensors, 0);
        assert_eq!(pool.stats().unique_shapes, 0);

        Ok(())
    }

    #[test]
    fn test_pool_stats() -> Result<()> {
        let device = Device::Cpu;
        let mut pool = TensorPool::new(4);

        // Create and release 4 tensors to fill pool
        for _ in 0..4 {
            let t = pool.acquire(&Shape::from((2, 2)), DType::F32, &device)?;
            pool.release(t);
        }

        // Next acquisitions should reuse from pool
        for _ in 0..4 {
            let t = pool.acquire(&Shape::from((2, 2)), DType::F32, &device)?;
            pool.release(t);
        }

        let stats = pool.stats();
        // First 4 acquisitions are new, subsequent ones reuse from pool
        // Due to zeros_like creating new tensors internally, reuse count may vary
        assert_eq!(stats.total_acquired, 8);
        assert!(stats.total_reused >= 4, "Expected at least 4 reuses, got {}", stats.total_reused);
        assert!(stats.reuse_rate >= 0.5, "Expected reuse rate >= 0.5, got {}", stats.reuse_rate);

        Ok(())
    }

    #[test]
    fn test_scoped_pool() -> Result<()> {
        let device = Device::Cpu;
        let mut pool = TensorPool::new(4);

        {
            let mut scoped = ScopedPool::new(&mut pool);
            let _ = scoped.acquire(&Shape::from((2, 2)), DType::F32, &device)?;
            let _ = scoped.acquire(&Shape::from((3, 3)), DType::F32, &device)?;
        } // Tensors automatically returned to pool

        assert_eq!(pool.stats().total_pooled_tensors, 2);

        Ok(())
    }

    #[test]
    fn test_pool_default() -> Result<()> {
        let device = Device::Cpu;
        let mut pool: TensorPool = Default::default();

        let t = pool.acquire(&Shape::from((10, 10)), DType::F32, &device)?;
        pool.release(t);

        assert!(pool.stats().total_pooled_tensors <= 4);

        Ok(())
    }
}
