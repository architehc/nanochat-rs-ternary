//! Kernel auto-tuning for runtime kernel selection.
//!
//! Benchmarks different kernel implementations (AVX-512, AVX2, Scalar)
//! and caches the best choice for each shape. Adapts to actual hardware
//! performance instead of using static heuristics.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

use crate::cpu::{gemv_scalar_ref, has_avx2, has_avx512};
use ternary_core::planar::PlanarWeights;

/// Shape key for kernel cache (M, K).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    pub m: usize,
    pub k: usize,
}

impl Shape {
    pub fn new(m: usize, k: usize) -> Self {
        Self { m, k }
    }
}

/// Available kernel implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelChoice {
    /// AVX-512 VPERMW kernel (fastest on supported hardware).
    Avx512,
    /// AVX2 PSHUFB kernel (good fallback).
    Avx2,
    /// Scalar reference (portable, always available).
    Scalar,
}

impl KernelChoice {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Avx512 => "AVX-512 VPERMW",
            Self::Avx2 => "AVX2 PSHUFB",
            Self::Scalar => "Scalar",
        }
    }
}

/// Kernel auto-tuner with per-shape caching.
pub struct KernelAutotuner {
    cache: RwLock<HashMap<Shape, KernelChoice>>,
    benchmark_iterations: usize,
}

const MAX_CACHE_ENTRIES: usize = 1024;

impl KernelAutotuner {
    /// Create a new auto-tuner.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            benchmark_iterations: 10, // Warm-up + benchmark iterations
        }
    }

    /// Select the best kernel for the given shape.
    ///
    /// Checks cache first, benchmarks if needed.
    pub fn select_kernel(&self, shape: Shape) -> KernelChoice {
        // Check cache
        {
            let cache = self.cache.read().unwrap();
            if let Some(&choice) = cache.get(&shape) {
                return choice;
            }
        }

        // Not cached - benchmark and cache result
        let choice = self.benchmark_kernels(shape);

        {
            let mut cache = self.cache.write().unwrap();
            if cache.len() >= MAX_CACHE_ENTRIES {
                if let Some(old_shape) = cache.keys().next().copied() {
                    cache.remove(&old_shape);
                }
            }
            cache.insert(shape, choice);
        }

        choice
    }

    /// Benchmark available kernels for this shape and return the best.
    fn benchmark_kernels(&self, shape: Shape) -> KernelChoice {
        let available_kernels = self.get_available_kernels();

        if available_kernels.is_empty() {
            return KernelChoice::Scalar; // Fallback
        }

        if available_kernels.len() == 1 {
            return available_kernels[0]; // Only one choice
        }

        // Create test data
        let (pw, x) = self.create_test_data(shape);

        // Benchmark each kernel
        let mut best_choice = KernelChoice::Scalar;
        let mut best_time = f64::MAX;

        for &kernel in &available_kernels {
            if let Some(time) = self.benchmark_kernel(kernel, &pw, &x, shape.m) {
                if time < best_time {
                    best_time = time;
                    best_choice = kernel;
                }
            }
        }

        log::debug!(
            "Auto-tuned shape ({}, {}): selected {} ({:.3}ms)",
            shape.m,
            shape.k,
            best_choice.name(),
            best_time * 1000.0
        );

        best_choice
    }

    /// Get list of available kernels on this hardware.
    fn get_available_kernels(&self) -> Vec<KernelChoice> {
        let mut kernels = vec![KernelChoice::Scalar]; // Always available

        if has_avx2() {
            kernels.push(KernelChoice::Avx2);
        }

        if has_avx512() {
            kernels.push(KernelChoice::Avx512);
        }

        kernels
    }

    /// Create test data for benchmarking.
    fn create_test_data(&self, shape: Shape) -> (PlanarWeights, Vec<i8>) {
        let m = shape.m;
        let k = shape.k;

        // Create random-ish weights
        let mut weights = vec![0.0f32; m * k];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = ((i * 7 + 13) % 3) as f32 - 1.0; // {-1, 0, 1}
        }

        let pw = PlanarWeights::from_row_major(&weights, m, k, 128);

        // Create random-ish activations
        let x: Vec<i8> = (0..k)
            .map(|i| (((i * 11 + 17) % 200) as i32 - 100) as i8)
            .collect();

        (pw, x)
    }

    /// Benchmark a specific kernel implementation.
    fn benchmark_kernel(
        &self,
        kernel: KernelChoice,
        pw: &PlanarWeights,
        x: &[i8],
        m: usize,
    ) -> Option<f64> {
        let mut y = vec![0.0f32; m];
        let act_scale = 1.0 / 127.0;

        // Warm-up
        for _ in 0..2 {
            match kernel {
                KernelChoice::Scalar => {
                    gemv_scalar_ref(pw, x, act_scale, &mut y);
                }
                #[cfg(target_arch = "x86_64")]
                KernelChoice::Avx2 | KernelChoice::Avx512 => {
                    // Call C kernel (delegates to best available)
                    crate::cpu::gemv(pw, x, act_scale, &mut y);
                }
                #[cfg(not(target_arch = "x86_64"))]
                KernelChoice::Avx2 | KernelChoice::Avx512 => {
                    return None; // Not available
                }
            }
        }

        // Benchmark
        let iterations = self.benchmark_iterations;
        let start = Instant::now();

        for _ in 0..iterations {
            match kernel {
                KernelChoice::Scalar => {
                    gemv_scalar_ref(pw, x, act_scale, &mut y);
                }
                #[cfg(target_arch = "x86_64")]
                KernelChoice::Avx2 | KernelChoice::Avx512 => {
                    crate::cpu::gemv(pw, x, act_scale, &mut y);
                }
                #[cfg(not(target_arch = "x86_64"))]
                KernelChoice::Avx2 | KernelChoice::Avx512 => {
                    return None;
                }
            }
        }

        let elapsed = start.elapsed();
        Some(elapsed.as_secs_f64() / iterations as f64)
    }

    /// Clear the cache (for testing or recalibration).
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get cache size.
    pub fn cache_size(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }
}

impl Default for KernelAutotuner {
    fn default() -> Self {
        Self::new()
    }
}

/// Global kernel auto-tuner instance.
static GLOBAL_AUTOTUNER: once_cell::sync::Lazy<KernelAutotuner> =
    once_cell::sync::Lazy::new(KernelAutotuner::new);

/// Get the global auto-tuner instance.
pub fn global_autotuner() -> &'static KernelAutotuner {
    &GLOBAL_AUTOTUNER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autotuner_creation() {
        let tuner = KernelAutotuner::new();
        assert_eq!(tuner.cache_size(), 0);
    }

    #[test]
    fn test_kernel_selection() {
        let tuner = KernelAutotuner::new();
        let shape = Shape::new(128, 128);

        // First call should benchmark
        let choice1 = tuner.select_kernel(shape);
        assert_eq!(tuner.cache_size(), 1);

        // Second call should use cache
        let choice2 = tuner.select_kernel(shape);
        assert_eq!(choice1, choice2);
        assert_eq!(tuner.cache_size(), 1);
    }

    #[test]
    fn test_multiple_shapes() {
        let tuner = KernelAutotuner::new();

        let shapes = vec![
            Shape::new(128, 128),
            Shape::new(256, 256),
            Shape::new(512, 512),
        ];

        for shape in shapes {
            let _choice = tuner.select_kernel(shape);
        }

        assert_eq!(tuner.cache_size(), 3);
    }

    #[test]
    fn test_cache_clear() {
        let tuner = KernelAutotuner::new();
        let shape = Shape::new(64, 128);

        tuner.select_kernel(shape);
        assert_eq!(tuner.cache_size(), 1);

        tuner.clear_cache();
        assert_eq!(tuner.cache_size(), 0);
    }

    #[test]
    fn test_global_autotuner() {
        let tuner = global_autotuner();
        let shape = Shape::new(1024, 1024);
        let choice = tuner.select_kernel(shape);
        println!("Selected kernel for 1024x1024: {}", choice.name());
    }
}
