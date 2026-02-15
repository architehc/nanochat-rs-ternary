//! Lion optimizer: sign-based update with EMA momentum.

use candle_core::{backprop::GradStore, DType, Result, Tensor, Var};
use serde::{Deserialize, Serialize};

use super::muon::TensorState;

/// Lion optimizer.
///
/// Update rule:
///   1. Weight decay: p *= (1 - lr * wd)
///   2. Update direction: update = sign(beta1*m + (1-beta1)*grad)
///   3. Apply: p -= lr * update
///   4. Momentum: m = beta2*m + (1-beta2)*grad
pub struct Lion {
    vars: Vec<Var>,
    exp_avg: Vec<Tensor>,
    pub lr: f64,
    beta1: f64,
    beta2: f64,
    weight_decay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LionState {
    pub exp_avg: Vec<TensorState>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
}

impl Lion {
    pub fn new(vars: Vec<Var>, lr: f64, beta1: f64, beta2: f64, wd: f64) -> Result<Self> {
        let exp_avg = vars
            .iter()
            .map(|v| Tensor::zeros_like(v.as_tensor()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            exp_avg,
            lr,
            beta1,
            beta2,
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
            let m = &self.exp_avg[i];

            // 1. Weight decay (multiplicative)
            if self.weight_decay > 0.0 {
                let decayed = (var.as_tensor() * (1.0 - self.lr * self.weight_decay))?;
                var.set(&decayed)?;
            }

            // 2. Update = sign(beta1*m + (1-beta1)*grad)
            let m_scaled = (m * self.beta1)?;
            let g_scaled = (&grad * (1.0 - self.beta1))?;
            let interp = (&m_scaled + &g_scaled)?;
            let update = sign(&interp)?;

            // 3. Apply: p -= lr * update
            let scaled_update = (&update * self.lr)?;
            let new_val = var.as_tensor().sub(&scaled_update)?;
            var.set(&new_val)?;

            // 4. Update momentum: m = beta2*m + (1-beta2)*grad
            let m2_scaled = (m * self.beta2)?;
            let g2_scaled = (&grad * (1.0 - self.beta2))?;
            self.exp_avg[i] = (&m2_scaled + &g2_scaled)?;
        }
        Ok(())
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    pub fn export_state(&self) -> Result<LionState> {
        let mut exp_avg = Vec::with_capacity(self.exp_avg.len());
        for tensor in &self.exp_avg {
            exp_avg.push(TensorState {
                shape: tensor.dims().to_vec(),
                data: tensor.flatten_all()?.to_vec1::<f32>()?,
            });
        }

        Ok(LionState {
            exp_avg,
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            weight_decay: self.weight_decay,
        })
    }

    pub fn import_state(&mut self, state: &LionState) -> Result<()> {
        if state.exp_avg.len() != self.exp_avg.len() {
            return Err(candle_core::Error::Msg(format!(
                "Lion state mismatch: expected {} EMA tensors, got {}",
                self.exp_avg.len(),
                state.exp_avg.len()
            )));
        }

        let mut restored = Vec::with_capacity(state.exp_avg.len());
        for (idx, snap) in state.exp_avg.iter().enumerate() {
            let expected_shape = self.vars[idx].as_tensor().dims().to_vec();
            if snap.shape != expected_shape {
                return Err(candle_core::Error::Msg(format!(
                    "Lion state shape mismatch at index {}: expected {:?}, got {:?}",
                    idx, expected_shape, snap.shape
                )));
            }
            let expected_len: usize = expected_shape.iter().product();
            if snap.data.len() != expected_len {
                return Err(candle_core::Error::Msg(format!(
                    "Lion state data length mismatch at index {}: expected {}, got {}",
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

        self.exp_avg = restored;
        self.lr = state.lr;
        self.beta1 = state.beta1;
        self.beta2 = state.beta2;
        self.weight_decay = state.weight_decay;
        Ok(())
    }
}

/// Element-wise sign function: -1 for negative, 0 for zero, +1 for positive.
fn sign(t: &Tensor) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(t)?;
    let pos = t.gt(&zeros)?.to_dtype(DType::F32)?;
    let neg = t.lt(&zeros)?.to_dtype(DType::F32)?;
    &pos - &neg
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_lion_step_updates_params() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w = vb.get_with_hints(16, "w", candle_nn::Init::Const(1.0))?;

        let orig = w.to_vec1::<f32>()?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 16), &device)?;
        let y = x.broadcast_mul(&w)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let vars = varmap.all_vars();
        let mut lion = Lion::new(vars, 1e-3, 0.9, 0.99, 0.0)?;
        lion.step(&grads, 1.0)?;

        let updated = w.to_vec1::<f32>()?;
        let changed = orig
            .iter()
            .zip(updated.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Parameters should change after Lion step");
        Ok(())
    }

    #[test]
    fn test_lion_sign_direction() -> Result<()> {
        let device = Device::Cpu;
        // Test sign function
        let t = Tensor::new(&[-2.0f32, -0.5, 0.0, 0.5, 2.0], &device)?;
        let s = sign(&t)?;
        let vals = s.to_vec1::<f32>()?;
        assert_eq!(vals, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_lion_weight_decay() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w = vb.get_with_hints(8, "w", candle_nn::Init::Const(1.0))?;

        let x = Tensor::zeros((1, 8), DType::F32, &device)?;
        let y = x.broadcast_mul(&w)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        // With wd and zero gradients, weight should decrease
        let vars = varmap.all_vars();
        let mut lion = Lion::new(vars, 0.1, 0.9, 0.99, 1.0)?;
        lion.step(&grads, 1.0)?;

        let vals = w.to_vec1::<f32>()?;
        for &v in &vals {
            assert!(v < 1.0, "Weight should decrease with decay: {}", v);
        }
        Ok(())
    }

    #[test]
    fn test_lion_state_export_import_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w = vb.get_with_hints(8, "w", candle_nn::Init::Const(1.0))?;

        // Create a non-zero optimizer state.
        let x = Tensor::randn(0.0f32, 1.0, (1, 8), &device)?;
        let y = x.broadcast_mul(&w)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let vars = varmap.all_vars();
        let mut lion = Lion::new(vars.clone(), 1e-3, 0.9, 0.99, 0.0)?;
        lion.step(&grads, 1.0)?;
        let state = lion.export_state()?;

        let mut restored = Lion::new(vars, 1e-4, 0.8, 0.95, 0.1)?;
        restored.import_state(&state)?;
        let restored_state = restored.export_state()?;

        assert_eq!(restored_state.exp_avg.len(), state.exp_avg.len());
        assert_eq!(restored_state.lr, state.lr);
        assert_eq!(restored_state.beta1, state.beta1);
        assert_eq!(restored_state.beta2, state.beta2);
        assert_eq!(restored_state.weight_decay, state.weight_decay);
        Ok(())
    }
}
