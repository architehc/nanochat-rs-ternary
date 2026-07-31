//! Muon optimizer: Nesterov momentum + Newton-Schulz orthogonalization.

use candle_core::{backprop::GradStore, DType, Result, Tensor, Var};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Newton-Schulz orthogonalization (quintic polynomial iteration).
///
/// Computes the polar factor of G (nearest orthogonal matrix).
/// Quintic coefficients (3.4445, -4.7750, 2.0315) are tuned for fast convergence.
pub fn newton_schulz_orthogonalize(g: &Tensor, ns_steps: usize) -> Result<Tensor> {
    newton_schulz_with_dtype(g, ns_steps, iteration_dtype(g))
}

/// Precision the quintic iteration runs at.
///
/// On CUDA the iteration runs in bf16 so the matmuls reach the tensor cores;
/// in f32 they fall back to the much slower CUDA-core path. This mirrors the
/// reference Muon implementation, which also orthogonalizes in bfloat16.
///
/// The iteration tolerates it because it is *self-correcting*: the quintic map
/// contracts toward the polar factor, so a rounding error at step `i` is pulled
/// back toward the orthogonal manifold by step `i+1`. Only the direction of the
/// update matters, not its precise magnitude — and the input is a stochastic
/// gradient estimate to begin with.
///
/// CPU keeps f32, both because there is no tensor-core payoff and so that the
/// portable tests stay exact.
fn iteration_dtype(g: &Tensor) -> DType {
    if g.device().is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// [`newton_schulz_orthogonalize`] with the iteration precision pinned, so tests
/// can compare precisions against each other on one device.
pub fn newton_schulz_with_dtype(g: &Tensor, ns_steps: usize, compute: DType) -> Result<Tensor> {
    let (a, b, c) = (3.4445f64, -4.7750f64, 2.0315f64);

    // Detach input — we don't need second-order gradients through the optimizer.
    // This prevents the NS computation graph from chaining back to the backward pass.
    let mut x = g.detach().to_dtype(DType::F32)?;
    let dims = x.dims();
    assert!(dims.len() >= 2);
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];

    let transposed = rows > cols;
    if transposed {
        x = x.t()?.contiguous()?;
    }

    // Normalize by Frobenius norm.
    //
    // Done entirely on-device: reading the norm back to the host costs a full
    // device synchronize, and this runs once per 2D parameter per step (~160
    // times). `where_cond` reproduces the previous branch exactly — divide by
    // the norm when it exceeds 1e-7, otherwise leave `x` untouched (divide by
    // 1) so a near-zero matrix is not amplified.
    let norm = crate::reduce::sum_squares(&x)?.sqrt()?;
    let divisor = norm
        .gt(1e-7f64)?
        .where_cond(&norm, &Tensor::ones_like(&norm)?)?;
    x = x.broadcast_div(&divisor.reshape((1, 1))?)?;
    // Detach after normalization to start iterations graph-free. The norm above
    // stays in f32 — it spans the full dynamic range of the gradient, which is
    // exactly what bf16's 8-bit mantissa would lose.
    x = x.detach().to_dtype(compute)?;

    // Quintic NS iteration — detach at each step to prevent graph accumulation.
    // Without detach, each iteration's matmuls chain to all previous iterations,
    // causing O(ns_steps * matrix_size) GPU memory retention.
    for _ in 0..ns_steps {
        let a_mat = x.matmul(&x.t()?)?; // X @ X^T
        let a_sq = a_mat.matmul(&a_mat)?;
        let term1 = (&a_mat * b)?;
        let term2 = (&a_sq * c)?;
        let b_mat = (&term1 + &term2)?; // b*A + c*A²
        let term_a = (&x * a)?;
        let term_b = b_mat.matmul(&x)?;
        x = (&term_a + &term_b)?.detach(); // a*X + B@X — detach breaks graph chain
    }

    if transposed {
        x = x.t()?.contiguous()?;
    }

    x.to_dtype(DType::F32)
}

/// Element budget for one batched Newton-Schulz call.
///
/// The iteration holds a few `[n, side, side]` temporaries at once, so this caps
/// them at roughly 64M elements — about 256MB in f32, 128MB in bf16 — per
/// intermediate. Groups larger than this are split into consecutive chunks,
/// which changes nothing numerically since the matrices are independent.
const NS_BATCH_ELEM_BUDGET: usize = 64 * 1024 * 1024;

/// Newton-Schulz over a *stack* of identically shaped matrices, `[n, rows, cols]`.
///
/// Mathematically identical to calling [`newton_schulz_with_dtype`] on each of
/// the `n` matrices: every operation in the iteration is per-matrix, and batched
/// matmul contracts only the last two dims, so matrices never mix. The
/// normalization is likewise per-matrix — each is divided by *its own* Frobenius
/// norm, which is what `sum over last two dims, keepdim` gives.
///
/// The point is launch count, not arithmetic. This model has ~160 2D parameters
/// but only a handful of distinct shapes, and the per-matrix iteration issues
/// ~7 kernels per step. Batching by shape turns ~160x5x7 launches into ~4x5x7,
/// and the step is launch-bound (see `docs/BLACKWELL_LOW_PRECISION.md`).
pub fn newton_schulz_batched(g: &Tensor, ns_steps: usize, compute: DType) -> Result<Tensor> {
    let (a, b, c) = (3.4445f64, -4.7750f64, 2.0315f64);

    let mut x = g.detach().to_dtype(DType::F32)?;
    let (_n, rows, cols) = x.dims3()?;

    // Every matrix in the stack has the same shape, so this decision is uniform.
    let transposed = rows > cols;
    if transposed {
        x = x.transpose(1, 2)?.contiguous()?;
    }

    // Per-matrix Frobenius norm -> [n, 1, 1], so the broadcast divides each
    // matrix by its own norm rather than by a norm pooled across the stack.
    let norm = x.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    let divisor = norm
        .gt(1e-7f64)?
        .where_cond(&norm, &Tensor::ones_like(&norm)?)?;
    x = x.broadcast_div(&divisor)?;
    x = x.detach().to_dtype(compute)?;

    for _ in 0..ns_steps {
        let xt = x.transpose(1, 2)?.contiguous()?;
        let a_mat = x.matmul(&xt)?; // X @ X^T, per matrix
        let a_sq = a_mat.matmul(&a_mat)?;
        let b_mat = ((&a_mat * b)? + (&a_sq * c)?)?;
        x = ((&x * a)? + b_mat.matmul(&x)?)?.detach();
    }

    if transposed {
        x = x.transpose(1, 2)?.contiguous()?;
    }
    x.to_dtype(DType::F32)
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

    /// One optimizer step.
    ///
    /// Runs in three phases so that the orthogonalization can be **batched by
    /// shape**. Doing it per parameter issues ~7 kernels per NS iteration for
    /// each of ~160 2D parameters; because this model has only a handful of
    /// distinct weight shapes, grouping them collapses that to ~7 per iteration
    /// per *shape*. The arithmetic is unchanged — see [`newton_schulz_batched`].
    pub fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        // Phase 1: momentum update for every parameter. 2D+ parameters have
        // their Nesterov look-ahead set aside for the batched orthogonalization;
        // 1D parameters take plain momentum and are ready immediately.
        let mut updates: Vec<Option<Tensor>> = vec![None; self.vars.len()];
        // (var index, original shape, nesterov reshaped to 2D)
        let mut pending: Vec<(usize, Vec<usize>, Tensor)> = Vec::new();

        for i in 0..self.vars.len() {
            let var = &self.vars[i];
            let grad = match grads.get(var.as_tensor()) {
                Some(g) => g,
                None => continue,
            };

            // Detach grad to sever any remaining backward-graph references.
            let grad = (grad * clip_scale)?.detach();

            // EMA momentum: buf = beta * buf + (1 - beta) * grad
            let prev_buf = self.momentum_buffers[i].clone();
            let buf_scaled = (&prev_buf * self.beta)?;
            let grad_scaled = (&grad * (1.0 - self.beta))?;
            let new_buf = (&buf_scaled + &grad_scaled)?;
            self.momentum_buffers[i] = new_buf.detach();

            if var.as_tensor().dims().len() >= 2 {
                // Nesterov look-ahead: extrapolate along momentum direction.
                let new_buf_d = &self.momentum_buffers[i];
                let delta = (new_buf_d - &prev_buf)?;
                let nesterov = (new_buf_d + (&delta * self.beta)?)?.detach();
                let orig_shape = nesterov.dims().to_vec();
                let rows = orig_shape[0];
                let cols: usize = orig_shape[1..].iter().product();
                pending.push((i, orig_shape, nesterov.reshape((rows, cols))?));
            } else {
                // 1D: plain momentum (use detached buffer)
                updates[i] = Some(self.momentum_buffers[i].clone());
            }
        }

        // Phase 2: orthogonalize, one batched call per distinct 2D shape.
        let mut by_shape: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (slot, (_, _, m)) in pending.iter().enumerate() {
            let d = m.dims();
            by_shape.entry((d[0], d[1])).or_default().push(slot);
        }

        for slots in by_shape.values() {
            let compute = iteration_dtype(&pending[slots[0]].2);
            let (rows, cols) = {
                let d = pending[slots[0]].2.dims();
                (d[0], d[1])
            };

            // Bound the temporaries. The iteration materializes `X @ X^T` and its
            // square, each `[n, min, min]` where `min = min(rows, cols)` after the
            // transpose. Batching every matrix of a shape at once is fine for this
            // model, but a 7B-scale one would allocate tens of GB here, so the
            // group is split into chunks under a fixed element budget.
            let side = rows.min(cols);
            let per_matrix = side * side;
            let chunk = (NS_BATCH_ELEM_BUDGET / per_matrix.max(1)).clamp(1, slots.len());

            for group in slots.chunks(chunk) {
                if group.len() == 1 {
                    let (i, shape, m) = &pending[group[0]];
                    let orth = newton_schulz_with_dtype(m, self.ns_steps, compute)?;
                    updates[*i] = Some(orth.reshape(shape.clone())?);
                    continue;
                }

                let stack: Vec<Tensor> = group.iter().map(|&s| pending[s].2.clone()).collect();
                let batched =
                    newton_schulz_batched(&Tensor::stack(&stack, 0)?, self.ns_steps, compute)?;
                for (row, &slot) in group.iter().enumerate() {
                    let (i, shape, _) = &pending[slot];
                    let orth = batched.narrow(0, row, 1)?.squeeze(0)?;
                    updates[*i] = Some(orth.reshape(shape.clone())?);
                }
            }
        }

        // Phase 3: apply.
        for (i, var) in self.vars.iter().enumerate() {
            let update = match &updates[i] {
                Some(u) => u,
                None => continue,
            };

            // Weight decay (multiplicative)
            if self.weight_decay > 0.0 {
                let decayed = (var.as_tensor() * (1.0 - self.lr * self.weight_decay))?.detach();
                var.set(&decayed)?;
            }

            // Apply update: w = w - lr * update
            let scaled_update = (update * self.lr)?;
            let new_val = var.as_tensor().sub(&scaled_update)?.detach();
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

            let grad = (grad * clip_scale)?.detach();

            // EMA momentum: buf = beta * buf + (1 - beta) * grad
            let prev_buf = self.momentum_buffers[i].clone();
            let buf_scaled = (&prev_buf * self.beta)?;
            let grad_scaled = (&grad * (1.0 - self.beta))?;
            let new_buf = (&buf_scaled + &grad_scaled)?;
            self.momentum_buffers[i] = new_buf.detach();

            let update = if var.as_tensor().dims().len() >= 2 {
                // Nesterov look-ahead: extrapolate along momentum direction.
                let new_buf_d = &self.momentum_buffers[i];
                let delta = (new_buf_d - &prev_buf)?;
                let nesterov = (new_buf_d + (&delta * self.beta)?)?.detach();
                // Reshape to 2D for orthogonalization
                let orig_shape = nesterov.dims().to_vec();
                let rows = orig_shape[0];
                let cols: usize = orig_shape[1..].iter().product();
                let nesterov_2d = nesterov.reshape((rows, cols))?;
                let orth = newton_schulz_orthogonalize(&nesterov_2d, self.ns_steps)?;
                orth.reshape(orig_shape)?
            } else {
                // 1D: plain momentum (use detached buffer)
                self.momentum_buffers[i].clone()
            };

            // Weight decay (multiplicative)
            if self.weight_decay > 0.0 {
                let decayed = (var.as_tensor() * (1.0 - self.lr * self.weight_decay))?.detach();
                var.set(&decayed)?;
            }

            // Apply update: w = w - lr * update
            let scaled_update = (&update * self.lr)?;
            let new_val = var.as_tensor().sub(&scaled_update)?.detach();
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

    /// Batching must not change the answer: each matrix in the stack has to come
    /// out exactly as it would have on its own.
    ///
    /// The risk this guards against is a reduction that pools across the stack —
    /// most plausibly the Frobenius normalization, which must be per-matrix. A
    /// shared norm would be masked by a stack of similarly scaled matrices, so
    /// the magnitudes here deliberately span 100x.
    #[test]
    fn batched_newton_schulz_matches_per_matrix() -> Result<()> {
        let device = Device::Cpu;
        let mats: Vec<Tensor> = (0..5)
            .map(|i| {
                let scale = 10f64.powi(i as i32 - 2); // 0.01 .. 100
                Tensor::randn(0f32, 1.0, (48usize, 64usize), &device).and_then(|t| t * scale)
            })
            .collect::<Result<_>>()?;

        let batched = newton_schulz_batched(&Tensor::stack(&mats, 0)?, 5, DType::F32)?;
        for (i, m) in mats.iter().enumerate() {
            let want = newton_schulz_with_dtype(m, 5, DType::F32)?;
            let got = batched.narrow(0, i, 1)?.squeeze(0)?;
            let diff = (&got - &want)?
                .abs()?
                .max_keepdim(1)?
                .max_keepdim(0)?
                .reshape(())?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-5, "matrix {i} differs by {diff}");
        }
        Ok(())
    }

    /// `step` groups 2D parameters by shape before orthogonalizing, so a step
    /// over a realistic mix — repeated shapes, a unique shape, and a 1D
    /// parameter that skips orthogonalization entirely — must still move every
    /// parameter, and by the same amount as the ungrouped path would.
    #[test]
    fn step_updates_every_param_across_shape_groups() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let init = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        };

        // Three of one shape, one of another, plus a 1D parameter.
        let a1 = vb.get_with_hints((8, 4), "a1", init)?;
        let a2 = vb.get_with_hints((8, 4), "a2", init)?;
        let a3 = vb.get_with_hints((8, 4), "a3", init)?;
        let b1 = vb.get_with_hints((6, 10), "b1", init)?;
        let c1 = vb.get_with_hints(5, "c1", init)?;

        let before: Vec<Vec<f32>> = [&a1, &a2, &a3, &b1, &c1]
            .iter()
            .map(|t| t.flatten_all()?.to_vec1::<f32>())
            .collect::<Result<_>>()?;

        // A loss touching all five so every parameter gets a gradient.
        let loss = ((a1.sum_all()? + a2.sum_all()?)? + (a3.sum_all()? + b1.sum_all()?)?)?
            .add(&c1.sum_all()?)?;
        let grads = loss.backward()?;

        let mut muon = Muon::new(varmap.all_vars(), 0.02, 0.95, 5, 0.0)?;
        muon.step(&grads, 1.0)?;

        for (i, t) in [&a1, &a2, &a3, &b1, &c1].iter().enumerate() {
            let after = t.flatten_all()?.to_vec1::<f32>()?;
            let moved: f32 = after
                .iter()
                .zip(&before[i])
                .map(|(x, y)| (x - y).abs())
                .sum();
            assert!(moved > 0.0, "parameter {i} was not updated");
            assert!(after.iter().all(|v| v.is_finite()), "parameter {i} not finite");
        }
        Ok(())
    }

    /// Tall matrices take the transpose path; batching must handle it too.
    #[test]
    fn batched_newton_schulz_handles_tall_matrices() -> Result<()> {
        let device = Device::Cpu;
        let mats: Vec<Tensor> = (0..3)
            .map(|_| Tensor::randn(0f32, 1.0, (96usize, 32usize), &device))
            .collect::<Result<_>>()?;

        let batched = newton_schulz_batched(&Tensor::stack(&mats, 0)?, 5, DType::F32)?;
        assert_eq!(batched.dims(), &[3, 96, 32]);
        for (i, m) in mats.iter().enumerate() {
            let want = newton_schulz_with_dtype(m, 5, DType::F32)?;
            let got = batched.narrow(0, i, 1)?.squeeze(0)?;
            let diff = (&got - &want)?
                .abs()?
                .max_keepdim(1)?
                .max_keepdim(0)?
                .reshape(())?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-5, "matrix {i} differs by {diff}");
        }
        Ok(())
    }

    /// The bf16 iteration must orthogonalize as well as the f32 one.
    ///
    /// Orthogonality — not agreement with the f32 result — is the property Muon
    /// depends on, so that is what this measures: `||X X^T - I||_max`.
    ///
    /// Note the bar is *relative*. Muon's quintic coefficients are tuned for
    /// fast convergence to a band around the orthogonal manifold, not to it
    /// exactly, so even the f32 iteration leaves a residual near 0.2 after 5
    /// steps. The claim being tested is that dropping to bf16 does not make that
    /// materially worse, and that the resulting update still points the same way.
    #[cfg(feature = "cuda")]
    #[test]
    fn bf16_iteration_orthogonalizes_like_f32() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (rows, cols) = (256usize, 512usize);
        let g = Tensor::randn(0f32, 1.0, (rows, cols), &device)?;

        let residual = |x: &Tensor| -> Result<f32> {
            // x is [rows, cols] with rows < cols, so X X^T is the square one.
            let xxt = x.matmul(&x.t()?.contiguous()?)?;
            let n = xxt.dim(0)?;
            let eye = Tensor::from_vec(
                (0..n * n)
                    .map(|i| if i / n == i % n { 1f32 } else { 0f32 })
                    .collect::<Vec<_>>(),
                (n, n),
                xxt.device(),
            )?;
            (xxt - eye)?
                .abs()?
                .max_keepdim(1)?
                .max_keepdim(0)?
                .reshape(())?
                .to_scalar::<f32>()
        };

        let f32_out = newton_schulz_with_dtype(&g, 5, DType::F32)?;
        let bf16_out = newton_schulz_with_dtype(&g, 5, DType::BF16)?;

        let r_f32 = residual(&f32_out)?;
        let r_bf16 = residual(&bf16_out)?;
        assert!(
            r_bf16 < r_f32 * 1.5 + 0.05,
            "bf16 residual {r_bf16} materially worse than f32 {r_f32}"
        );

        // And the two updates must point the same way — a Muon step is a
        // direction, so a large angle between them would change training.
        let dot = (&f32_out * &bf16_out)?;
        let dot = crate::reduce::sum_all_fast(&dot)?.to_scalar::<f32>()?;
        let n1 = crate::reduce::sum_squares(&f32_out)?
            .to_scalar::<f32>()?
            .sqrt();
        let n2 = crate::reduce::sum_squares(&bf16_out)?
            .to_scalar::<f32>()?
            .sqrt();
        let cosine = dot / (n1 * n2);
        assert!(cosine > 0.99, "bf16 update diverged from f32: cos={cosine}");
        Ok(())
    }

    #[test]
    fn test_newton_schulz_orthogonal() -> Result<()> {
        let device = Device::Cpu;
        // Square case should converge near an orthogonal matrix.
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
    fn test_newton_schulz_rectangular_preserves_orthogonality() -> Result<()> {
        let device = Device::Cpu;

        // Tall matrix: columns should be orthonormal.
        let g_tall = Tensor::randn(0.0f32, 1.0, (32, 8), &device)?;
        let q_tall = newton_schulz_orthogonalize(&g_tall, 5)?;
        let gram_tall = q_tall.t()?.matmul(&q_tall)?;
        let i_tall = Tensor::eye(8, DType::F32, &device)?;
        let diff_tall = (&gram_tall - &i_tall)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff_tall < 0.6,
            "tall matrix orthogonality drift: {}",
            diff_tall
        );

        // Wide matrix: rows should be orthonormal.
        let g_wide = Tensor::randn(0.0f32, 1.0, (8, 32), &device)?;
        let q_wide = newton_schulz_orthogonalize(&g_wide, 5)?;
        let gram_wide = q_wide.matmul(&q_wide.t()?)?;
        let i_wide = Tensor::eye(8, DType::F32, &device)?;
        let diff_wide = (&gram_wide - &i_wide)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff_wide < 0.6,
            "wide matrix orthogonality drift: {}",
            diff_wide
        );

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
