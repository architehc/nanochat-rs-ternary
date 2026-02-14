//! GaLore 2: Memory-Efficient LLM Training by Gradient Low-Rank Projection
//! Based on: "GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection" (arXiv:2504.20437)

use candle_core::{Result, Tensor, DType, Device};
use candle_nn::Optimizer;
use std::collections::HashMap;

/// GaLore 2 Optimizer - Memory-efficient training with gradient low-rank projection
pub struct GaLore2Optimizer<OPT: Optimizer> {
    /// Base optimizer (Muon, Adam, etc.)
    base_optimizer: OPT,

    /// Rank for low-rank projection
    rank: usize,

    /// Update frequency (every N steps)
    update_freq: usize,

    /// Projection matrices for each parameter
    projections: HashMap<String, ProjectionPair>,

    /// Step counter
    step: usize,

    /// Scale factor for projected gradients
    scale: f64,

    /// Minimum dimension for applying GaLore
    min_dim: usize,
}

/// Projection matrices (P, Q) for low-rank gradient projection
#[derive(Clone)]
pub struct ProjectionPair {
    /// Right singular vectors (dim x rank)
    p: Tensor,

    /// Left singular vectors (dim x rank)
    q: Tensor,

    /// Last updated step
    last_updated: usize,
}

impl<OPT: Optimizer> GaLore2Optimizer<OPT> {
    /// Create new GaLore 2 optimizer
    pub fn new(base_optimizer: OPT, rank: usize, update_freq: usize) -> Self {
        Self {
            base_optimizer,
            rank,
            update_freq,
            projections: HashMap::new(),
            step: 0,
            scale: 1.0,
            min_dim: 256,  // Only apply to large matrices
        }
    }

    /// Set minimum dimension for applying GaLore
    pub fn with_min_dim(mut self, min_dim: usize) -> Self {
        self.min_dim = min_dim;
        self
    }

    /// Set scale factor for projected gradients
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Perform optimization step with gradient projection
    pub fn step(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        self.step += 1;

        // Update projections periodically
        if self.step % self.update_freq == 1 || self.projections.is_empty() {
            self.update_projections(gradients)?;
        }

        // Project gradients and apply base optimizer
        let projected_grads: HashMap<String, Tensor> = gradients
            .iter()
            .map(|(name, grad)| {
                let projected = self.project_gradient(name, grad).unwrap_or_else(|_| grad.clone());
                (name.clone(), projected)
            })
            .collect();

        self.base_optimizer.step(&projected_grads)
    }

    /// Project gradient to low-rank subspace
    fn project_gradient(&self, name: &str, grad: &Tensor) -> Result<Tensor> {
        // Only apply to large 2D matrices
        if grad.dims().len() != 2 {
            return Ok(grad.clone());
        }

        let (m, n) = grad.dims2()?;
        if m < self.min_dim || n < self.min_dim {
            return Ok(grad.clone());
        }

        // Get or create projection
        if let Some(proj) = self.projections.get(name) {
            // Project gradient: g_proj = Q^T @ g @ P
            // For efficiency, we compute: (Q^T @ g) @ P
            let temp = proj.q.matmul(grad)?;  // Q^T @ g
            let projected = temp.matmul(&proj.p)?;  // (Q^T @ g) @ P

            // Scale to maintain gradient magnitude
            Ok(projected.mul(self.scale)?)
        } else {
            Ok(grad.clone())
        }
    }

    /// Update projection matrices using SVD
    fn update_projections(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        for (name, grad) in gradients {
            // Only apply to large 2D matrices
            if grad.dims().len() != 2 {
                continue;
            }

            let (m, n) = grad.dims2()?;
            if m < self.min_dim || n < self.min_dim {
                continue;
            }

            // Compute SVD: grad = U @ S @ V^T
            // We use randomized SVD for efficiency
            let (u, _s, vt) = self.randomized_svd(grad, self.rank)?;

            // Store projection matrices
            let p = vt.t()?.narrow(0, 0, self.rank)?;  // V[:, :rank]
            let q = u.narrow(1, 0, self.rank)?;        // U[:, :rank]

            self.projections.insert(name.clone(), ProjectionPair {
                p,
                q,
                last_updated: self.step,
            });
        }

        Ok(())
    }

    /// Randomized SVD for efficiency (Halko et al. algorithm)
    fn randomized_svd(&self, a: &Tensor, rank: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let (m, n) = a.dims2()?;
        let device = a.device();

        // Step 1: Random projection
        let omega = Tensor::randn(0.0_f64, 1.0, (n, rank + 5), device)?;
        let y = a.matmul(&omega)?;  // m x (rank+5)

        // Step 2: QR decomposition
        let (q, _r) = y.qr()?;  // m x (rank+5)

        // Step 3: Project A
        let b = q.t()?.matmul(a)?;  // (rank+5) x n

        // Step 4: SVD of small matrix B
        let (u_tilde, s, vt) = b.svd()?;

        // Step 5: Reconstruct U
        let u = q.matmul(&u_tilde)?;

        Ok((u, s, vt))
    }

    /// Get memory savings statistics
    pub fn memory_stats(&self) -> GaLoreMemoryStats {
        let total_params: usize = self.projections.values()
            .map(|p| p.p.elem_count() + p.q.elem_count())
            .sum();

        let projected_count = self.projections.len();

        GaLoreMemoryStats {
            projected_layers: projected_count,
            projection_params: total_params,
            memory_saved_ratio: self.estimate_memory_savings(),
        }
    }

    fn estimate_memory_savings(&self) -> f64 {
        // Estimate memory savings from gradient compression
        // Full gradient: m * n * 4 bytes (FP32)
        // Projected gradient: (m + n) * rank * 4 bytes
        // Savings: 1 - (m + n) * rank / (m * n)

        let mut total_full = 0usize;
        let mut total_compressed = 0usize;

        for (name, proj) in &self.projections {
            if let Some(grad) = self.get_gradient_shape(name) {
                let (m, n) = grad;
                total_full += m * n;
                total_compressed += (m + n) * self.rank;
            }
        }

        if total_full == 0 {
            0.0
        } else {
            1.0 - (total_compressed as f64 / total_full as f64)
        }
    }

    fn get_gradient_shape(&self, _name: &str) -> Option<(usize, usize)> {
        // This would need to track gradient shapes
        // For now, return None
        None
    }
}

/// Memory statistics for GaLore
#[derive(Debug)]
pub struct GaLoreMemoryStats {
    pub projected_layers: usize,
    pub projection_params: usize,
    pub memory_saved_ratio: f64,
}

/// 8-bit quantized Muon optimizer states
pub struct QuantizedMuon {
    /// Base Muon optimizer
    base: MuonOptimizer,

    /// Block size for quantization
    block_size: usize,

    /// Quantized states
    quantized_states: HashMap<String, QuantizedState>,
}

pub struct QuantizedState {
    /// Quantized values (8-bit)
    values: Vec<u8>,

    /// Scale factors per block
    scales: Vec<f32>,

    /// Zero points per block
    zero_points: Vec<f32>,
}

impl QuantizedMuon {
    /// Quantize tensor to 8-bit
    pub fn quantize(&self, tensor: &Tensor) -> Result<QuantizedState> {
        let data = tensor.to_vec1::<f32>()?;
        let num_blocks = (data.len() + self.block_size - 1) / self.block_size;

        let mut values = Vec::new();
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();

        for block_idx in 0..num_blocks {
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(data.len());
            let block = &data[start..end];

            // Compute min/max for block
            let min = block.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Compute scale and zero point
            let scale = (max - min) / 255.0;
            let zero_point = min;

            scales.push(scale);
            zero_points.push(zero_point);

            // Quantize values
            for &val in block {
                let quantized = ((val - zero_point) / scale).round().clamp(0.0, 255.0) as u8;
                values.push(quantized);
            }
        }

        Ok(QuantizedState {
            values,
            scales,
            zero_points,
        })
    }

    /// Dequantize tensor from 8-bit
    pub fn dequantize(&self, state: &QuantizedState, shape: &[usize]) -> Result<Tensor> {
        let mut data = Vec::new();

        for (i, &val) in state.values.iter().enumerate() {
            let block_idx = i / self.block_size;
            let scale = state.scales[block_idx];
            let zero_point = state.zero_points[block_idx];

            let dequantized = val as f32 * scale + zero_point;
            data.push(dequantized);
        }

        Tensor::from_vec(data, shape, &Device::Cpu)
    }
}

/// Placeholder for MuonOptimizer - would integrate with existing implementation
pub struct MuonOptimizer;

impl Optimizer for MuonOptimizer {
    fn step(&mut self, _grads: &HashMap<String, Tensor>) -> Result<()> {
        // Implementation would be from existing nanochat-rs-ternary
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        0.001
    }
}
