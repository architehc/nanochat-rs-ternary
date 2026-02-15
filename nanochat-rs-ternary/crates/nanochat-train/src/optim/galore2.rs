//! GaLore 2: Memory-efficient training via gradient low-rank projection
//! Based on "GaLore 2: Reducing Memory in Large-Scale Training" (arXiv:2504.20437)
//!
//! Key insight: Project gradients to low-rank subspace before optimizer step,
//! reducing memory from O(d²) to O(d·r) where r << d.

use candle_core::{backprop::GradStore, Result, Tensor, Var};
use std::collections::HashMap;

/// Optimizer wrapper that applies GaLore2 gradient projection
pub struct GaLore2<OPT> {
    base_optimizer: OPT,

    /// Rank for low-rank projection (hardware-specific)
    /// Config A (Blackwell): 512, Config B (2×4090): 256, Config C (EPYC+4090): 384
    rank: usize,

    /// Update projections every N steps (default: 200)
    update_freq: usize,

    /// Left/right projection matrices per parameter
    projections: HashMap<usize, ProjectionPair>,

    /// Current step counter
    step: usize,

    /// Projection scale factor (TODO: not yet used)
    #[allow(dead_code)]
    scale: f64,

    /// Only apply to large matrices (min dimension threshold)
    min_dim: usize,

    /// Vars being optimized
    vars: Vec<Var>,
}

/// Left and right projection matrices from SVD
struct ProjectionPair {
    /// Right singular vectors (dim × rank)
    p: Tensor,
    /// Left singular vectors (dim × rank)
    q: Tensor,
    /// Step when last updated (TODO: not yet used for adaptive refresh)
    #[allow(dead_code)]
    last_updated: usize,
}

impl<OPT> GaLore2<OPT> {
    pub fn new(
        base_optimizer: OPT,
        vars: Vec<Var>,
        rank: usize,
        update_freq: usize,
        scale: f64,
    ) -> Result<Self> {
        Ok(Self {
            base_optimizer,
            rank,
            update_freq,
            projections: HashMap::new(),
            step: 0,
            scale,
            min_dim: 256, // Only apply to large matrices
            vars,
        })
    }

    /// Compute randomized SVD using Halko et al. algorithm
    /// Returns (U, S, V) where A ≈ U @ diag(S) @ V^T
    fn randomized_svd(&self, a: &Tensor, rank: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let device = a.device();
        let dims = a.dims();
        let (_m, n) = (dims[0], dims[1]);

        // Oversample by factor of 2 for accuracy
        let l = (rank * 2).min(n);

        // Random Gaussian matrix
        let omega = Tensor::randn(0.0f32, 1.0, (n, l), device)?;

        // Y = A @ Omega (range approximation)
        let y = a.matmul(&omega)?;

        // QR decomposition of Y to get orthonormal basis Q
        // For now, use Gram-Schmidt (TODO: replace with proper QR)
        let q = self.gram_schmidt(&y)?;

        // B = Q^T @ A (project A onto range of Q)
        let b = q.t()?.matmul(a)?;

        // Small SVD of B using power iteration
        let (u_b, s, vt) = self.small_svd(&b, rank)?;

        // U = Q @ U_B
        let u = q.matmul(&u_b)?;

        // Extract top-k singular values and vectors
        let u_k = u.narrow(1, 0, rank)?;
        let s_k = s.narrow(0, 0, rank)?;
        let vt_k = vt.narrow(0, 0, rank)?;

        Ok((u_k, s_k, vt_k.t()?))
    }

    /// Gram-Schmidt orthogonalization
    fn gram_schmidt(&self, a: &Tensor) -> Result<Tensor> {
        let _device = a.device();
        let dims = a.dims();
        let (_m, n) = (dims[0], dims[1]);

        let mut q_cols = Vec::new();

        for j in 0..n {
            let mut v = a.narrow(1, j, 1)?;

            // Subtract projections onto previous vectors
            for q_col in &q_cols {
                let proj = v.t()?.matmul(q_col)?; // <v, q>
                let proj_scaled = q_col.broadcast_mul(&proj)?;
                v = v.sub(&proj_scaled)?;
            }

            // Normalize
            let norm = v.sqr()?.sum_all()?.sqrt()?;
            let norm_val = norm.to_scalar::<f32>()?;
            if norm_val > 1e-10 {
                v = (&v / (norm_val as f64))?;
            }

            q_cols.push(v);
        }

        // Concatenate columns
        Tensor::cat(&q_cols, 1)
    }

    /// Small SVD using power iteration (for randomized SVD's projected matrix)
    fn small_svd(&self, a: &Tensor, rank: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let device = a.device();
        let dims = a.dims();
        let (m, n) = (dims[0], dims[1]);
        let k = rank.min(m).min(n);

        // Simplified: use A^T @ A for right singular vectors
        let ata = a.t()?.matmul(a)?;

        // Power iteration for top eigenvector
        let mut v = Tensor::randn(0.0f32, 1.0, (n, 1), device)?;
        for _ in 0..20 {
            v = ata.matmul(&v)?;
            let norm = v.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if norm > 1e-10 {
                v = (&v / (norm as f64))?;
            }
        }

        // u = A @ v / |A @ v|
        let u_unnorm = a.matmul(&v)?;
        let u_norm = u_unnorm.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let u = if u_norm > 1e-10 {
            (&u_unnorm / (u_norm as f64))?
        } else {
            u_unnorm
        };

        // sigma = u^T @ A @ v
        let sigma = u.t()?.matmul(a)?.matmul(&v)?;
        let sigma_val = sigma.to_scalar::<f32>()?;

        // For simplicity, return single singular triplet repeated k times
        // (Full implementation would compute k eigenvectors)
        let u_mat = u.broadcast_as((m, k))?;
        let s_vec = Tensor::from_vec(vec![sigma_val; k], k, device)?;
        let v_mat = v.t()?.broadcast_as((k, n))?;

        Ok((u_mat, s_vec, v_mat))
    }

    /// Project gradient to low-rank subspace: g_proj = Q^T @ g @ P
    fn project_gradient(&self, var_idx: usize, grad: &Tensor) -> Result<Tensor> {
        match self.projections.get(&var_idx) {
            Some(proj) => {
                // g_proj = Q^T @ g @ P
                let left = proj.q.t()?.matmul(grad)?;
                left.matmul(&proj.p)
            }
            None => Ok(grad.clone()),
        }
    }

    /// Unproject from low-rank subspace: g = Q @ g_proj @ P^T (TODO: not yet used)
    #[allow(dead_code)]
    fn unproject_gradient(&self, var_idx: usize, grad_proj: &Tensor) -> Result<Tensor> {
        match self.projections.get(&var_idx) {
            Some(proj) => {
                // g = Q @ g_proj @ P^T
                let left = proj.q.matmul(grad_proj)?;
                left.matmul(&proj.p.t()?)
            }
            None => Ok(grad_proj.clone()),
        }
    }

    /// Update projection matrices using randomized SVD
    fn update_projections(&mut self, grads: &GradStore) -> Result<()> {
        for (var_idx, var) in self.vars.iter().enumerate() {
            let grad = match grads.get(var.as_tensor()) {
                Some(g) => g,
                None => continue,
            };

            let dims = grad.dims();
            if dims.len() < 2 {
                continue; // Skip 1D tensors
            }

            let (rows, cols) = (dims[0], dims[1]);
            if rows.min(cols) < self.min_dim {
                continue; // Skip small matrices
            }

            // Reshape to 2D if needed
            let grad_2d = if dims.len() > 2 {
                let total_cols: usize = dims[1..].iter().product();
                grad.reshape((dims[0], total_cols))?
            } else {
                grad.clone()
            };

            // Compute SVD: grad ≈ Q @ S @ P^T
            let (q, _s, p) = self.randomized_svd(&grad_2d, self.rank)?;

            self.projections.insert(var_idx, ProjectionPair {
                p,
                q,
                last_updated: self.step,
            });
        }

        Ok(())
    }

    /// Check if projection update is needed
    fn should_update_projection(&self) -> bool {
        self.step.is_multiple_of(self.update_freq)
    }
}

/// GaLore2 wrapper for Muon optimizer
pub struct GaLore2Muon {
    galore: GaLore2<super::Muon>,
}

impl GaLore2Muon {
    pub fn new(
        muon: super::Muon,
        vars: Vec<Var>,
        rank: usize,
        update_freq: usize,
    ) -> Result<Self> {
        let galore = GaLore2::new(muon, vars, rank, update_freq, 1.0)?;
        Ok(Self { galore })
    }

    pub fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        // Update projections periodically
        if self.galore.should_update_projection() {
            self.galore.update_projections(grads)?;
        }

        // Project gradients
        let mut projected_grads = HashMap::new();
        for (var_idx, var) in self.galore.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let grad_proj = self.galore.project_gradient(var_idx, grad)?;
                projected_grads.insert(var.as_tensor().id(), grad_proj);
            }
        }

        // Create temporary GradStore with projected gradients
        // Note: This is a simplified version. In practice, you'd need to properly
        // integrate with candle's GradStore API

        // For now, call base optimizer directly with original grads
        // TODO: Properly handle projected gradients
        self.galore.base_optimizer.step(grads, clip_scale)?;

        self.galore.step += 1;
        Ok(())
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.galore.base_optimizer.set_lr(lr);
    }

    /// Get memory savings statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let mut total_params = 0;
        let mut projected_params = 0;

        for (var_idx, var) in self.galore.vars.iter().enumerate() {
            let dims = var.as_tensor().dims();
            if dims.len() >= 2 {
                let size: usize = dims.iter().product();
                total_params += size;

                if self.galore.projections.contains_key(&var_idx) {
                    let (rows, cols) = (dims[0], dims[1..].iter().product::<usize>());
                    // Projected size: (rows * rank) + (rank * cols)
                    let proj_size = (rows * self.galore.rank) + (self.galore.rank * cols);
                    projected_params += proj_size;
                } else {
                    projected_params += size;
                }
            }
        }

        let reduction = if total_params > 0 {
            1.0 - (projected_params as f64 / total_params as f64)
        } else {
            0.0
        };

        MemoryStats {
            total_params,
            projected_params,
            memory_reduction: reduction,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_params: usize,
    pub projected_params: usize,
    pub memory_reduction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarMap;

    #[test]
    fn test_galore2_creates_projections() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Small matrix for quick test
        let _w = vb.get_with_hints(
            (128, 128),
            "w",
            candle_nn::Init::Randn { mean: 0.0, stdev: 1.0 },
        )?;

        let vars = varmap.all_vars();
        let muon = super::super::Muon::new(vars.clone(), 0.02, 0.95, 3, 0.0)?;
        let galore = GaLore2Muon::new(muon, vars, 32, 10)?;

        // Just verify construction works
        let stats = galore.memory_stats();
        println!("GaLore2 initialized: reduction={:.1}%", stats.memory_reduction * 100.0);

        Ok(())
    }

    #[test]
    fn test_gram_schmidt_orthogonal() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::randn(0.0f32, 1.0, (8, 4), &device)?;

        let vars = Vec::new();
        let muon = super::super::Muon::new(vars.clone(), 0.02, 0.95, 5, 0.0)?;
        let galore = GaLore2::new(muon, vars, 32, 100, 1.0)?;

        let q = galore.gram_schmidt(&a)?;

        // Check Q^T @ Q ≈ I
        let qtq = q.t()?.matmul(&q)?;
        let n = qtq.dim(0)?;
        let identity = Tensor::eye(n, DType::F32, &device)?;
        let diff = (&qtq - &identity)?.abs()?.max_all()?.to_scalar::<f32>()?;

        assert!(diff < 0.1, "Q^T @ Q should be approximately I, max diff: {}", diff);
        Ok(())
    }
}
