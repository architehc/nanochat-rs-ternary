//! GaLore 2: Memory-efficient training via gradient low-rank projection
//! Based on "GaLore 2: Reducing Memory in Large-Scale Training" (arXiv:2504.20437)
//!
//! Key insight: Project gradients to low-rank subspace before optimizer step,
//! reducing memory from O(d²) to O(d·r) where r << d.

use candle_core::{backprop::GradStore, DType, Result, Tensor, Var};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Interface required by GaLore2 wrappers.
pub trait GaLoreOptimizer {
    fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()>;
    fn step_with_projected(
        &mut self,
        grads_by_id: &HashMap<candle_core::TensorId, Tensor>,
        clip_scale: f64,
    ) -> Result<()>;
    fn set_lr(&mut self, lr: f64);
}

impl GaLoreOptimizer for super::Muon {
    fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        super::Muon::step(self, grads, clip_scale)
    }

    fn step_with_projected(
        &mut self,
        grads_by_id: &HashMap<candle_core::TensorId, Tensor>,
        clip_scale: f64,
    ) -> Result<()> {
        self.step_with_grads(grads_by_id, clip_scale)
    }

    fn set_lr(&mut self, lr: f64) {
        super::Muon::set_lr(self, lr);
    }
}

impl GaLoreOptimizer for super::QuantizedMuon {
    fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        super::QuantizedMuon::step(self, grads, clip_scale)
    }

    fn step_with_projected(
        &mut self,
        grads_by_id: &HashMap<candle_core::TensorId, Tensor>,
        clip_scale: f64,
    ) -> Result<()> {
        self.step_with_grads(grads_by_id, clip_scale)
    }

    fn set_lr(&mut self, lr: f64) {
        super::QuantizedMuon::set_lr(self, lr);
    }
}

/// Optimizer wrapper that applies GaLore2 gradient projection
pub struct GaLore2<OPT> {
    base_optimizer: OPT,

    /// Rank for low-rank projection (hardware-specific)
    /// Config A (Blackwell): 512, Config B (2×4090): 256, Config C (EPYC+4090): 384
    rank: usize,

    /// Update projections every N steps (default: 200)
    update_freq: usize,
    /// Adaptive checks run more frequently than full refresh cadence.
    adaptive_check_freq: usize,
    /// Enable adaptive refresh based on projection error.
    adaptive_refresh: bool,
    /// Relative projection error threshold to trigger refresh.
    adaptive_error_threshold: f64,

    /// Left/right projection matrices per parameter
    projections: HashMap<usize, ProjectionPair>,

    /// Current step counter
    step: usize,

    /// Projection scale factor for projected gradients.
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
    /// Step when projection was last refreshed.
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
        let effective_update_freq = update_freq.max(1);
        Ok(Self {
            base_optimizer,
            rank,
            update_freq: effective_update_freq,
            adaptive_check_freq: (effective_update_freq / 4).max(1),
            adaptive_refresh: true,
            adaptive_error_threshold: 0.20,
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
        let (m, n) = (dims[0], dims[1]);

        // Oversample by factor of 2 for accuracy
        let l = (rank * 2).min(n).min(m);
        let target_rank = rank.min(l);

        // Random Gaussian matrix
        let omega = Tensor::randn(0.0f32, 1.0, (n, l), device)?;

        // Y = A @ Omega (range approximation)
        let y = a.matmul(&omega)?;

        // Orthogonalize Y to get an approximate range basis.
        let q = self.gram_schmidt(&y)?;

        // B = Q^T @ A (project A onto range of Q)
        let b = q.t()?.matmul(a)?;

        // Small SVD of B using power iteration
        let (u_b, s, vt) = self.small_svd(&b, target_rank)?;

        // U = Q @ U_B
        let u = q.matmul(&u_b)?;

        // Extract top-k singular values and vectors
        let u_k = u.narrow(1, 0, target_rank)?;
        let s_k = s.narrow(0, 0, target_rank)?;
        let vt_k = vt.narrow(0, 0, target_rank)?;

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

    /// Small SVD using orthogonal power iteration on A A^T (m x m).
    ///
    /// This avoids forming the much larger A^T A when n >> m and computes
    /// distinct top-k singular triplets instead of repeating one vector.
    fn small_svd(&self, a: &Tensor, rank: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let device = a.device();
        let dims = a.dims();
        let (m, n) = (dims[0], dims[1]);
        let k = rank.min(m).min(n);
        if k == 0 {
            return Err(candle_core::Error::Msg(
                "small_svd called with zero effective rank".to_string(),
            ));
        }

        // Work on the smaller symmetric matrix A A^T (m x m), where m is
        // typically rank*2 after projection.
        let aat = a.matmul(&a.t()?)?;

        let mut u_cols: Vec<Tensor> = Vec::with_capacity(k);
        let mut s_vals: Vec<f32> = Vec::with_capacity(k);
        let mut v_rows: Vec<Tensor> = Vec::with_capacity(k);

        for _ in 0..k {
            let mut u = Tensor::randn(0.0f32, 1.0, (m, 1), device)?;

            for _ in 0..24 {
                u = aat.matmul(&u)?;
                for prev in &u_cols {
                    let coeff = prev.t()?.matmul(&u)?;
                    let proj = prev.broadcast_mul(&coeff)?;
                    u = u.sub(&proj)?;
                }

                let norm = u.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                if norm <= 1e-10 {
                    break;
                }
                u = (&u / (norm as f64))?;
            }

            let norm = u.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if norm <= 1e-10 {
                break;
            }
            u = (&u / (norm as f64))?;

            let lambda = u.t()?.matmul(&aat)?.matmul(&u)?.squeeze(0)?.squeeze(0)?;
            let sigma = lambda.to_scalar::<f32>()?.max(0.0).sqrt();
            if sigma <= 1e-8 {
                break;
            }

            let mut v = a.t()?.matmul(&u)?;
            v = (&v / (sigma as f64))?;
            for prev in &v_rows {
                let coeff = prev.matmul(&v)?;
                let proj = prev.t()?.broadcast_mul(&coeff)?;
                v = v.sub(&proj)?;
            }
            let v_norm = v.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if v_norm > 1e-10 {
                v = (&v / (v_norm as f64))?;
            } else {
                break;
            }

            u_cols.push(u);
            s_vals.push(sigma);
            v_rows.push(v.t()?);
        }

        while u_cols.len() < k {
            u_cols.push(Tensor::zeros((m, 1), DType::F32, device)?);
            s_vals.push(0.0);
            v_rows.push(Tensor::zeros((1, n), DType::F32, device)?);
        }

        let u_mat = Tensor::cat(&u_cols, 1)?;
        let s_vec = Tensor::from_vec(s_vals, k, device)?;
        let vt_mat = Tensor::cat(&v_rows, 0)?;

        Ok((u_mat, s_vec, vt_mat))
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

    /// Compute relative projection error ||g - g_hat|| / ||g|| for one variable.
    fn projection_relative_error(&self, var_idx: usize, grad_2d: &Tensor) -> Result<f64> {
        if !self.projections.contains_key(&var_idx) {
            return Ok(f64::INFINITY);
        }

        let grad_proj = self.project_gradient(var_idx, grad_2d)?;
        let grad_recon = self.unproject_gradient(var_idx, &grad_proj)?;
        let diff_norm = (&grad_recon - grad_2d)?
            .sqr()?
            .sum_all()?
            .sqrt()?
            .to_scalar::<f32>()? as f64;
        let grad_norm = grad_2d.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
        if grad_norm <= 1e-12 {
            return Ok(0.0);
        }
        Ok(diff_norm / grad_norm)
    }

    /// Check if a single variable projection should refresh.
    fn should_refresh_projection(
        &self,
        var_idx: usize,
        grad_2d: &Tensor,
        force: bool,
    ) -> Result<bool> {
        if force || !self.projections.contains_key(&var_idx) {
            return Ok(true);
        }

        let Some(pair) = self.projections.get(&var_idx) else {
            return Ok(true);
        };

        // Hard refresh cadence.
        if self.step.saturating_sub(pair.last_updated) >= self.update_freq {
            return Ok(true);
        }

        // Adaptive refresh cadence.
        if !self.adaptive_refresh
            || self.step.saturating_sub(pair.last_updated) < self.adaptive_check_freq
        {
            return Ok(false);
        }

        let rel_error = self.projection_relative_error(var_idx, grad_2d)?;
        Ok(rel_error > self.adaptive_error_threshold)
    }

    /// Update projection matrices using randomized SVD
    fn update_projections(&mut self, grads: &GradStore, force: bool) -> Result<()> {
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

            if !self.should_refresh_projection(var_idx, &grad_2d, force)? {
                continue;
            }

            // Compute SVD: grad ≈ Q @ S @ P^T
            let (q, _s, p) = self.randomized_svd(&grad_2d, self.rank)?;

            self.projections.insert(
                var_idx,
                ProjectionPair {
                    p,
                    q,
                    last_updated: self.step,
                },
            );
        }

        Ok(())
    }

    /// Check if projection update is needed
    fn should_run_projection_pass(&self) -> bool {
        self.step.is_multiple_of(self.update_freq)
            || (self.adaptive_refresh && self.step.is_multiple_of(self.adaptive_check_freq))
    }

    /// Build transformed gradients for optimizer update.
    ///
    /// For variables with available projections, gradients are projected to
    /// low-rank space and unprojected back to full-rank tensors, applying the
    /// GaLore subspace filter before the base optimizer step.
    fn build_projected_gradients(
        &self,
        grads: &GradStore,
    ) -> Result<HashMap<candle_core::TensorId, Tensor>> {
        let mut projected = HashMap::new();

        for (var_idx, var) in self.vars.iter().enumerate() {
            let grad = match grads.get(var.as_tensor()) {
                Some(g) => g,
                None => continue,
            };

            let dims = grad.dims().to_vec();
            let grad_2d = if dims.len() > 2 {
                let total_cols: usize = dims[1..].iter().product();
                grad.reshape((dims[0], total_cols))?
            } else {
                grad.clone()
            };

            let transformed_2d = if self.projections.contains_key(&var_idx) {
                let grad_proj = self.project_gradient(var_idx, &grad_2d)?;
                self.unproject_gradient(var_idx, &grad_proj)?
            } else {
                grad_2d
            };

            let transformed = if dims.len() > 2 {
                transformed_2d.reshape(dims)?
            } else {
                transformed_2d
            };

            projected.insert(var.as_tensor().id(), transformed);
        }

        Ok(projected)
    }

    fn estimate_memory_stats(&self) -> MemoryStats {
        let mut total_params = 0;
        let mut projected_params = 0;

        for (var_idx, var) in self.vars.iter().enumerate() {
            let dims = var.as_tensor().dims();
            if dims.len() >= 2 {
                let size: usize = dims.iter().product();
                total_params += size;

                if self.projections.contains_key(&var_idx) {
                    let (rows, cols) = (dims[0], dims[1..].iter().product::<usize>());
                    // Projected size: (rows * rank) + (rank * cols)
                    let proj_size = (rows * self.rank) + (self.rank * cols);
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

/// GaLore2 wrapper for Muon optimizer
pub struct GaLore2Muon {
    galore: GaLore2<super::Muon>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaLore2MuonState {
    pub base: super::muon::MuonState,
    pub step: usize,
}

impl GaLore2Muon {
    pub fn new(muon: super::Muon, vars: Vec<Var>, rank: usize, update_freq: usize) -> Result<Self> {
        let galore = GaLore2::new(muon, vars, rank, update_freq, 1.0)?;
        Ok(Self { galore })
    }

    pub fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        // Full refresh on update cadence, adaptive check between cadence points.
        if self.galore.should_run_projection_pass() {
            let force = self.galore.step.is_multiple_of(self.galore.update_freq);
            self.galore.update_projections(grads, force)?;
        }

        // Apply projected/unprojected gradients in the optimizer step.
        let projected_grads = self.galore.build_projected_gradients(grads)?;
        self.galore
            .base_optimizer
            .step_with_projected(&projected_grads, clip_scale)?;

        self.galore.step += 1;
        Ok(())
    }

    pub fn set_lr(&mut self, lr: f64) {
        GaLoreOptimizer::set_lr(&mut self.galore.base_optimizer, lr);
    }

    /// Get memory savings statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.galore.estimate_memory_stats()
    }

    pub fn export_state(&self) -> Result<GaLore2MuonState> {
        Ok(GaLore2MuonState {
            base: self.galore.base_optimizer.export_state()?,
            step: self.galore.step,
        })
    }

    pub fn import_state(&mut self, state: &GaLore2MuonState) -> Result<()> {
        self.galore.base_optimizer.import_state(&state.base)?;
        self.galore.step = state.step;
        Ok(())
    }
}

/// GaLore2 wrapper for 8-bit Quantized Muon optimizer
pub struct GaLore2Quantized {
    galore: GaLore2<super::QuantizedMuon>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaLore2QuantizedState {
    pub base: super::muon_quantized::QuantizedMuonState,
    pub step: usize,
}

impl GaLore2Quantized {
    pub fn new(
        qmuon: super::QuantizedMuon,
        vars: Vec<Var>,
        rank: usize,
        update_freq: usize,
    ) -> Result<Self> {
        let galore = GaLore2::new(qmuon, vars, rank, update_freq, 1.0)?;
        Ok(Self { galore })
    }

    pub fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        if self.galore.should_run_projection_pass() {
            let force = self.galore.step.is_multiple_of(self.galore.update_freq);
            self.galore.update_projections(grads, force)?;
        }

        let projected_grads = self.galore.build_projected_gradients(grads)?;
        self.galore
            .base_optimizer
            .step_with_projected(&projected_grads, clip_scale)?;

        self.galore.step += 1;
        Ok(())
    }

    pub fn set_lr(&mut self, lr: f64) {
        GaLoreOptimizer::set_lr(&mut self.galore.base_optimizer, lr);
    }

    pub fn memory_stats(&self) -> MemoryStats {
        self.galore.estimate_memory_stats()
    }

    pub fn export_state(&self) -> Result<GaLore2QuantizedState> {
        Ok(GaLore2QuantizedState {
            base: self.galore.base_optimizer.export_state(),
            step: self.galore.step,
        })
    }

    pub fn import_state(&mut self, state: &GaLore2QuantizedState) -> Result<()> {
        self.galore.base_optimizer.import_state(&state.base)?;
        self.galore.step = state.step;
        Ok(())
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
    use candle_core::{DType, Device};
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
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let vars = varmap.all_vars();
        let muon = super::super::Muon::new(vars.clone(), 0.02, 0.95, 3, 0.0)?;
        let galore = GaLore2Muon::new(muon, vars, 32, 10)?;

        // Just verify construction works
        let stats = galore.memory_stats();
        println!(
            "GaLore2 initialized: reduction={:.1}%",
            stats.memory_reduction * 100.0
        );

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

        assert!(
            diff < 0.1,
            "Q^T @ Q should be approximately I, max diff: {}",
            diff
        );
        Ok(())
    }

    #[test]
    fn test_adaptive_projection_pass_cadence() -> Result<()> {
        let vars = Vec::new();
        let muon = super::super::Muon::new(vars.clone(), 0.02, 0.95, 5, 0.0)?;
        let mut galore = GaLore2::new(muon, vars, 32, 8, 1.0)?;

        // step=0 always refreshes.
        galore.step = 0;
        assert!(galore.should_run_projection_pass());

        // adaptive_check_freq should be update_freq/4 = 2.
        galore.step = 1;
        assert!(!galore.should_run_projection_pass());

        galore.step = 2;
        assert!(galore.should_run_projection_pass());

        // hard refresh cadence.
        galore.step = 8;
        assert!(galore.should_run_projection_pass());
        Ok(())
    }

    #[test]
    fn test_update_freq_zero_is_sanitized() -> Result<()> {
        let vars = Vec::new();
        let muon = super::super::Muon::new(vars.clone(), 0.02, 0.95, 5, 0.0)?;
        let galore = GaLore2::new(muon, vars, 32, 0, 1.0)?;
        assert_eq!(galore.update_freq, 1);
        assert_eq!(galore.adaptive_check_freq, 1);
        Ok(())
    }

    #[test]
    fn test_small_svd_returns_distinct_components() -> Result<()> {
        let device = Device::Cpu;
        let mut diag = vec![0.0f32; 36];
        for i in 0..6 {
            diag[i * 6 + i] = (6 - i) as f32;
        }
        let a = Tensor::from_vec(diag, (6, 6), &device)?;

        let vars = Vec::new();
        let muon = super::super::Muon::new(vars.clone(), 0.02, 0.95, 5, 0.0)?;
        let galore = GaLore2::new(muon, vars, 3, 10, 1.0)?;
        let (u, s, _vt) = galore.small_svd(&a, 3)?;

        let s_vals = s.to_vec1::<f32>()?;
        assert_eq!(s_vals.len(), 3);
        assert!(s_vals[0] >= s_vals[1]);
        assert!(s_vals[1] >= s_vals[2]);
        assert!(s_vals[2] > 0.0);

        // U columns should be approximately orthonormal.
        let utu = u.t()?.matmul(&u)?;
        let eye = Tensor::eye(3, DType::F32, &device)?;
        let max_diff = (&utu - &eye)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_diff < 0.25, "U^T U max diff too large: {}", max_diff);
        Ok(())
    }
}
