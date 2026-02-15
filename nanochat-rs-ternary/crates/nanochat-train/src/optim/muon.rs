//! Muon optimizer: Nesterov momentum + Newton-Schulz orthogonalization.

use candle_core::{backprop::GradStore, DType, Result, Tensor, Var};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Newton-Schulz orthogonalization (quintic polynomial iteration).
///
/// Computes the polar factor of G (nearest orthogonal matrix).
/// Quintic coefficients (3.4445, -4.7750, 2.0315) are tuned for fast convergence.
pub fn newton_schulz_orthogonalize(g: &Tensor, ns_steps: usize) -> Result<Tensor> {
    let (a, b, c) = (3.4445f64, -4.7750f64, 2.0315f64);

    let mut x = g.to_dtype(DType::F32)?;
    let dims = x.dims();
    assert!(dims.len() >= 2);
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];

    let transposed = rows > cols;
    if transposed {
        x = x.t()?;
    }

    // Normalize by Frobenius norm
    let norm = x.sqr()?.sum_all()?.sqrt()?;
    let norm_val = norm.to_scalar::<f32>()?;
    if norm_val > 1e-7 {
        x = (&x / (norm_val as f64))?;
    }

    // Quintic NS iteration
    for _ in 0..ns_steps {
        let a_mat = x.matmul(&x.t()?)?; // X @ X^T
        let a_sq = a_mat.matmul(&a_mat)?;
        let term1 = (&a_mat * b)?;
        let term2 = (&a_sq * c)?;
        let b_mat = (&term1 + &term2)?; // b*A + c*A²
        let term_a = (&x * a)?;
        let term_b = b_mat.matmul(&x)?;
        x = (&term_a + &term_b)?; // a*X + B@X
    }

    if transposed {
        x = x.t()?;
    }

    // Scale by aspect ratio
    let scale = (rows.max(1) as f64 / cols.max(1) as f64).sqrt().max(1.0);
    x = (&x * scale)?;

    Ok(x)
}

/// Muon optimizer for 2D+ parameters.
pub struct Muon {
    vars: Vec<Var>,
    momentum_buffers: Vec<Tensor>,
    pub lr: f64,
    beta: f64,
    ns_steps: usize,
    weight_decay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorState {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuonState {
    pub momentum_buffers: Vec<TensorState>,
    pub lr: f64,
    pub beta: f64,
    pub ns_steps: usize,
    pub weight_decay: f64,
}

impl Muon {
    pub fn new(vars: Vec<Var>, lr: f64, beta: f64, ns_steps: usize, wd: f64) -> Result<Self> {
        let momentum_buffers = vars
            .iter()
            .map(|v| Tensor::zeros_like(v.as_tensor()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            momentum_buffers,
            lr,
            beta,
            ns_steps,
            weight_decay: wd,
        })
    }

    pub fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        for (i, var) in self.vars.iter().enumerate() {
            let grad = match grads.get(var.as_tensor()) {
                Some(g) => g,
                None => continue,
            };

            let grad = (grad * clip_scale)?;

            // EMA momentum: buf = beta * buf + (1 - beta) * grad
            let buf = &self.momentum_buffers[i];
            let buf_scaled = (buf * self.beta)?;
            let grad_scaled = (&grad * (1.0 - self.beta))?;
            let new_buf = (&buf_scaled + &grad_scaled)?;
            self.momentum_buffers[i] = new_buf.clone();

            let update = if var.as_tensor().dims().len() >= 2 {
                // Nesterov: update = beta * buf + (1 - beta) * grad
                let nest_grad = (&grad * (1.0 - self.beta))?;
                let nest_buf = (&new_buf * self.beta)?;
                let nesterov = (&nest_grad + &nest_buf)?;
                // Reshape to 2D for orthogonalization
                let orig_shape = nesterov.dims().to_vec();
                let rows = orig_shape[0];
                let cols: usize = orig_shape[1..].iter().product();
                let nesterov_2d = nesterov.reshape((rows, cols))?;
                let orth = newton_schulz_orthogonalize(&nesterov_2d, self.ns_steps)?;
                orth.reshape(orig_shape)?
            } else {
                // 1D: plain momentum
                new_buf
            };

            // Weight decay (multiplicative)
            if self.weight_decay > 0.0 {
                let decayed = (var.as_tensor() * (1.0 - self.lr * self.weight_decay))?;
                var.set(&decayed)?;
            }

            // Apply update: w = w - lr * update
            let scaled_update = (&update * self.lr)?;
            let new_val = var.as_tensor().sub(&scaled_update)?;
            var.set(&new_val)?;
        }
        Ok(())
    }

    /// Step with externally provided gradients keyed by tensor id.
    ///
    /// Used by gradient-transform wrappers (e.g. GaLore2) that need to override
    /// the gradient tensor before the optimizer update.
    pub fn step_with_grads(
        &mut self,
        grads_by_id: &HashMap<candle_core::TensorId, Tensor>,
        clip_scale: f64,
    ) -> Result<()> {
        for (i, var) in self.vars.iter().enumerate() {
            let grad = match grads_by_id.get(&var.as_tensor().id()) {
                Some(g) => g,
                None => continue,
            };

            let grad = (grad * clip_scale)?;

            // EMA momentum: buf = beta * buf + (1 - beta) * grad
            let buf = &self.momentum_buffers[i];
            let buf_scaled = (buf * self.beta)?;
            let grad_scaled = (&grad * (1.0 - self.beta))?;
            let new_buf = (&buf_scaled + &grad_scaled)?;
            self.momentum_buffers[i] = new_buf.clone();

            let update = if var.as_tensor().dims().len() >= 2 {
                // Nesterov: update = beta * buf + (1 - beta) * grad
                let nest_grad = (&grad * (1.0 - self.beta))?;
                let nest_buf = (&new_buf * self.beta)?;
                let nesterov = (&nest_grad + &nest_buf)?;
                // Reshape to 2D for orthogonalization
                let orig_shape = nesterov.dims().to_vec();
                let rows = orig_shape[0];
                let cols: usize = orig_shape[1..].iter().product();
                let nesterov_2d = nesterov.reshape((rows, cols))?;
                let orth = newton_schulz_orthogonalize(&nesterov_2d, self.ns_steps)?;
                orth.reshape(orig_shape)?
            } else {
                // 1D: plain momentum
                new_buf
            };

            // Weight decay (multiplicative)
            if self.weight_decay > 0.0 {
                let decayed = (var.as_tensor() * (1.0 - self.lr * self.weight_decay))?;
                var.set(&decayed)?;
            }

            // Apply update: w = w - lr * update
            let scaled_update = (&update * self.lr)?;
            let new_val = var.as_tensor().sub(&scaled_update)?;
            var.set(&new_val)?;
        }
        Ok(())
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    pub fn export_state(&self) -> Result<MuonState> {
        let mut buffers = Vec::with_capacity(self.momentum_buffers.len());
        for buf in &self.momentum_buffers {
            buffers.push(TensorState {
                shape: buf.dims().to_vec(),
                data: buf.flatten_all()?.to_vec1::<f32>()?,
            });
        }

        Ok(MuonState {
            momentum_buffers: buffers,
            lr: self.lr,
            beta: self.beta,
            ns_steps: self.ns_steps,
            weight_decay: self.weight_decay,
        })
    }

    pub fn import_state(&mut self, state: &MuonState) -> Result<()> {
        if state.momentum_buffers.len() != self.momentum_buffers.len() {
            return Err(candle_core::Error::Msg(format!(
                "Muon state mismatch: expected {} momentum buffers, got {}",
                self.momentum_buffers.len(),
                state.momentum_buffers.len()
            )));
        }

        let mut restored = Vec::with_capacity(state.momentum_buffers.len());
        for (idx, snap) in state.momentum_buffers.iter().enumerate() {
            let expected_shape = self.vars[idx].as_tensor().dims().to_vec();
            if snap.shape != expected_shape {
                return Err(candle_core::Error::Msg(format!(
                    "Muon state shape mismatch at index {}: expected {:?}, got {:?}",
                    idx, expected_shape, snap.shape
                )));
            }
            let expected_len: usize = expected_shape.iter().product();
            if snap.data.len() != expected_len {
                return Err(candle_core::Error::Msg(format!(
                    "Muon state data length mismatch at index {}: expected {}, got {}",
                    idx,
                    expected_len,
                    snap.data.len()
                )));
            }
            restored.push(Tensor::from_vec(
                snap.data.clone(),
                snap.shape.as_slice(),
                self.vars[idx].device(),
            )?);
        }

        self.momentum_buffers = restored;
        self.lr = state.lr;
        self.beta = state.beta;
        self.ns_steps = state.ns_steps;
        self.weight_decay = state.weight_decay;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_newton_schulz_orthogonal() -> Result<()> {
        let device = Device::Cpu;
        // Use square matrix so aspect-ratio scaling = 1.0
        let g = Tensor::randn(0.0f32, 1.0, (16, 16), &device)?;
        let orth = newton_schulz_orthogonalize(&g, 5)?;

        assert_eq!(orth.dims(), &[16, 16]);

        // Check approximate orthonormality: orth^T @ orth should be ~I
        let product = orth.t()?.matmul(&orth)?;
        let n = product.dim(0)?;
        let identity = Tensor::eye(n, DType::F32, &device)?;
        let diff = (&product - &identity)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        // Relaxed tolerance — NS with 5 steps is approximate
        assert!(diff < 0.5, "orth^T @ orth should be ~I, max diff: {}", diff);
        Ok(())
    }

    #[test]
    fn test_muon_step_updates_params() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w = vb.get_with_hints(
            (16, 8),
            "w",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let orig = w.to_vec2::<f32>()?;

        // Simulate gradient
        let x = Tensor::randn(0.0f32, 1.0, (1, 8), &device)?;
        let y = x.matmul(&w.t()?)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let vars = varmap.all_vars();
        let mut muon = Muon::new(vars, 0.02, 0.95, 5, 0.0)?;
        muon.step(&grads, 1.0)?;

        let updated = w.to_vec2::<f32>()?;
        let changed = orig
            .iter()
            .flatten()
            .zip(updated.iter().flatten())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Parameters should have changed after Muon step");
        Ok(())
    }

    #[test]
    fn test_muon_momentum_accumulates() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w = vb.get_with_hints(
            (8, 4),
            "w",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let vars = varmap.all_vars();
        let mut muon = Muon::new(vars, 0.01, 0.95, 3, 0.0)?;

        // Run 3 steps
        for _ in 0..3 {
            let x = Tensor::randn(0.0f32, 1.0, (1, 4), &device)?;
            let y = x.matmul(&w.t()?)?;
            let loss = y.sum_all()?;
            let grads = loss.backward()?;
            muon.step(&grads, 1.0)?;
        }

        // Momentum buffer should be non-zero
        let buf_norm = muon.momentum_buffers[0]
            .sqr()?
            .sum_all()?
            .sqrt()?
            .to_scalar::<f32>()?;
        assert!(buf_norm > 0.0, "Momentum buffer should be non-zero");
        Ok(())
    }

    #[test]
    fn test_muon_1d_no_orthogonalize() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w = vb.get_with_hints(16, "w", candle_nn::Init::Const(1.0))?;

        let orig = w.to_vec1::<f32>()?;

        // Create a simple computation with the 1D param
        let x = Tensor::randn(0.0f32, 1.0, (1, 16), &device)?;
        let y = x.broadcast_mul(&w)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let vars = varmap.all_vars();
        let mut muon = Muon::new(vars, 0.01, 0.95, 5, 0.0)?;
        muon.step(&grads, 1.0)?;

        let updated = w.to_vec1::<f32>()?;
        let changed = orig
            .iter()
            .zip(updated.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "1D param should still update");
        Ok(())
    }
}
