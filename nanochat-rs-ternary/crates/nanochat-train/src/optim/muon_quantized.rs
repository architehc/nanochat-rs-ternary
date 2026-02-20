//! 8-bit Quantized Muon optimizer
//! Based on "8-bit Optimizers for Efficient Training" (arXiv:2509.23106)
//!
//! Key insight: Quantize optimizer states (momentum buffers) to INT8,
//! reducing memory by 75% with minimal impact on convergence.

use candle_core::{backprop::GradStore, Result, Tensor, Var};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantization configuration for optimizer states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    /// Use block-wise quantization (128 elements per block)
    pub block_size: usize,
    /// Use stochastic rounding
    pub stochastic_rounding: bool,
    /// Dynamic range adjustment
    pub dynamic_range: bool,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            stochastic_rounding: true,
            dynamic_range: true,
        }
    }
}

/// Quantized state: INT8 values + per-block FP32 scales
struct QuantizedState {
    /// Quantized values (INT8)
    values: Vec<i8>,
    /// Per-block scale factors
    scales: Vec<f32>,
    /// Original shape
    shape: Vec<usize>,
    /// Block size for quantization
    block_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedBufferState {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub shape: Vec<usize>,
    pub block_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedMuonState {
    pub momentum_buffers: Vec<QuantizedBufferState>,
    pub lr: f64,
    pub beta: f64,
    pub ns_steps: usize,
    pub weight_decay: f64,
    pub quant_config: QuantConfig,
}

impl QuantizedState {
    /// Quantize FP32 tensor to INT8 with per-block absmax scaling
    fn from_tensor(tensor: &Tensor, config: &QuantConfig) -> Result<Self> {
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        let shape = tensor.dims().to_vec();
        let n = data.len();

        let mut values = Vec::with_capacity(n);
        let mut scales = Vec::new();

        // Process in blocks
        for chunk in data.chunks(config.block_size) {
            // Compute absmax for this block
            let absmax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if absmax > 1e-10 { absmax / 127.0 } else { 1.0 };
            scales.push(scale);

            // Quantize to INT8
            let inv_scale = if absmax > 1e-10 { 127.0 / absmax } else { 0.0 };
            for &val in chunk {
                let quantized = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
                values.push(quantized);
            }
        }

        Ok(Self {
            values,
            scales,
            shape,
            block_size: config.block_size,
        })
    }

    /// Dequantize back to FP32 tensor
    fn to_tensor(&self, device: &candle_core::Device) -> Result<Tensor> {
        let mut data = Vec::with_capacity(self.values.len());

        for (block_idx, chunk) in self.values.chunks(self.block_size).enumerate() {
            let scale = self.scales[block_idx];
            for &val in chunk {
                data.push(val as f32 * scale);
            }
        }

        let tensor = Tensor::from_vec(data, self.shape.as_slice(), device)?;
        Ok(tensor)
    }

    /// In-place update with EMA: self = beta * self + (1 - beta) * new
    #[cfg(test)]
    fn ema_update(&mut self, new: &Tensor, beta: f64, config: &QuantConfig) -> Result<()> {
        // Dequantize current state
        let current = self.to_tensor(new.device())?;

        // Compute EMA
        let current_scaled = (&current * beta)?;
        let new_scaled = (new * (1.0 - beta))?;
        let updated = (&current_scaled + &new_scaled)?;

        // Re-quantize
        let new_state = Self::from_tensor(&updated, config)?;
        self.values = new_state.values;
        self.scales = new_state.scales;

        Ok(())
    }

    /// Re-quantize state directly from a tensor (no additional EMA step).
    fn assign_from_tensor(&mut self, new: &Tensor, config: &QuantConfig) -> Result<()> {
        let new_state = Self::from_tensor(new, config)?;
        self.values = new_state.values;
        self.scales = new_state.scales;
        Ok(())
    }
}

/// 8-bit Quantized Muon optimizer
pub struct QuantizedMuon {
    vars: Vec<Var>,
    /// Quantized momentum buffers (INT8 + scales)
    momentum_buffers: Vec<QuantizedState>,
    pub lr: f64,
    beta: f64,
    ns_steps: usize,
    weight_decay: f64,
    quant_config: QuantConfig,
}

impl QuantizedMuon {
    pub fn new(vars: Vec<Var>, lr: f64, beta: f64, ns_steps: usize, wd: f64) -> Result<Self> {
        Self::new_with_config(vars, lr, beta, ns_steps, wd, QuantConfig::default())
    }

    pub fn new_with_config(
        vars: Vec<Var>,
        lr: f64,
        beta: f64,
        ns_steps: usize,
        wd: f64,
        quant_config: QuantConfig,
    ) -> Result<Self> {
        let momentum_buffers = vars
            .iter()
            .map(|v| {
                let zeros = Tensor::zeros_like(v.as_tensor())?;
                QuantizedState::from_tensor(&zeros, &quant_config)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            vars,
            momentum_buffers,
            lr,
            beta,
            ns_steps,
            weight_decay: wd,
            quant_config,
        })
    }

    pub fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        for (i, var) in self.vars.iter().enumerate() {
            let grad = match grads.get(var.as_tensor()) {
                Some(g) => g,
                None => continue,
            };

            let grad = (grad * clip_scale)?;

            // Dequantize momentum buffer
            let prev_buf = self.momentum_buffers[i].to_tensor(var.device())?;

            // EMA momentum: buf = beta * buf + (1 - beta) * grad
            let buf_scaled = (&prev_buf * self.beta)?;
            let grad_scaled = (&grad * (1.0 - self.beta))?;
            let new_buf = (&buf_scaled + &grad_scaled)?;

            // Re-quantize and store updated momentum buffer.
            self.momentum_buffers[i].assign_from_tensor(&new_buf, &self.quant_config)?;

            let update = if var.as_tensor().dims().len() >= 2 {
                // Nesterov look-ahead: extrapolate momentum in the update direction
                // nesterov = new_buf + beta * (new_buf - prev_buf)
                let delta = (&new_buf - &prev_buf)?;
                let nesterov = (&new_buf + &((&delta) * self.beta)?)?;

                // Reshape to 2D for orthogonalization
                let orig_shape = nesterov.dims().to_vec();
                let rows = orig_shape[0];
                let cols: usize = orig_shape[1..].iter().product();
                let nesterov_2d = nesterov.reshape((rows, cols))?;
                let orth = super::muon::newton_schulz_orthogonalize(&nesterov_2d, self.ns_steps)?;
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

            // Dequantize momentum buffer
            let prev_buf = self.momentum_buffers[i].to_tensor(var.device())?;

            // EMA momentum: buf = beta * buf + (1 - beta) * grad
            let buf_scaled = (&prev_buf * self.beta)?;
            let grad_scaled = (&grad * (1.0 - self.beta))?;
            let new_buf = (&buf_scaled + &grad_scaled)?;

            // Re-quantize and store updated momentum buffer.
            self.momentum_buffers[i].assign_from_tensor(&new_buf, &self.quant_config)?;

            let update = if var.as_tensor().dims().len() >= 2 {
                // Nesterov look-ahead: extrapolate momentum in the update direction
                // nesterov = new_buf + beta * (new_buf - prev_buf)
                let delta = (&new_buf - &prev_buf)?;
                let nesterov = (&new_buf + &((&delta) * self.beta)?)?;

                // Reshape to 2D for orthogonalization
                let orig_shape = nesterov.dims().to_vec();
                let rows = orig_shape[0];
                let cols: usize = orig_shape[1..].iter().product();
                let nesterov_2d = nesterov.reshape((rows, cols))?;
                let orth = super::muon::newton_schulz_orthogonalize(&nesterov_2d, self.ns_steps)?;
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

    pub fn export_state(&self) -> QuantizedMuonState {
        QuantizedMuonState {
            momentum_buffers: self
                .momentum_buffers
                .iter()
                .map(|state| QuantizedBufferState {
                    values: state.values.clone(),
                    scales: state.scales.clone(),
                    shape: state.shape.clone(),
                    block_size: state.block_size,
                })
                .collect(),
            lr: self.lr,
            beta: self.beta,
            ns_steps: self.ns_steps,
            weight_decay: self.weight_decay,
            quant_config: self.quant_config.clone(),
        }
    }

    pub fn import_state(&mut self, state: &QuantizedMuonState) -> Result<()> {
        if state.momentum_buffers.len() != self.momentum_buffers.len() {
            return Err(candle_core::Error::Msg(format!(
                "Quantized Muon state mismatch: expected {} buffers, got {}",
                self.momentum_buffers.len(),
                state.momentum_buffers.len()
            )));
        }

        let mut restored = Vec::with_capacity(state.momentum_buffers.len());
        for (idx, snap) in state.momentum_buffers.iter().enumerate() {
            if snap.block_size == 0 {
                return Err(candle_core::Error::Msg(format!(
                    "Quantized Muon state invalid block_size=0 at index {}",
                    idx
                )));
            }

            let expected_shape = self.vars[idx].as_tensor().dims().to_vec();
            if snap.shape != expected_shape {
                return Err(candle_core::Error::Msg(format!(
                    "Quantized Muon shape mismatch at index {}: expected {:?}, got {:?}",
                    idx, expected_shape, snap.shape
                )));
            }

            let expected_len: usize = expected_shape.iter().product();
            if snap.values.len() != expected_len {
                return Err(candle_core::Error::Msg(format!(
                    "Quantized Muon values length mismatch at index {}: expected {}, got {}",
                    idx,
                    expected_len,
                    snap.values.len()
                )));
            }

            let expected_scales = expected_len.div_ceil(snap.block_size);
            if snap.scales.len() != expected_scales {
                return Err(candle_core::Error::Msg(format!(
                    "Quantized Muon scales length mismatch at index {}: expected {}, got {}",
                    idx,
                    expected_scales,
                    snap.scales.len()
                )));
            }

            restored.push(QuantizedState {
                values: snap.values.clone(),
                scales: snap.scales.clone(),
                shape: snap.shape.clone(),
                block_size: snap.block_size,
            });
        }

        self.momentum_buffers = restored;
        self.lr = state.lr;
        self.beta = state.beta;
        self.ns_steps = state.ns_steps;
        self.weight_decay = state.weight_decay;
        self.quant_config = state.quant_config.clone();
        Ok(())
    }

    /// Get memory savings statistics
    pub fn memory_stats(&self) -> QuantMemoryStats {
        let mut fp32_bytes = 0usize;
        let mut int8_bytes = 0usize;
        let mut scale_bytes = 0usize;

        for state in &self.momentum_buffers {
            let n_elements = state.values.len();
            fp32_bytes += n_elements * 4; // Original FP32

            int8_bytes += n_elements; // Quantized INT8
            scale_bytes += state.scales.len() * 4; // FP32 scales
        }

        let total_quantized = int8_bytes + scale_bytes;
        let reduction = 1.0 - (total_quantized as f64 / fp32_bytes as f64);

        QuantMemoryStats {
            fp32_bytes,
            int8_bytes,
            scale_bytes,
            total_quantized,
            memory_reduction: reduction,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantMemoryStats {
    pub fp32_bytes: usize,
    pub int8_bytes: usize,
    pub scale_bytes: usize,
    pub total_quantized: usize,
    pub memory_reduction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_quantized_state_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let original = Tensor::randn(0.0f32, 1.0, (1024,), &device)?;

        let config = QuantConfig::default();
        let quantized = QuantizedState::from_tensor(&original, &config)?;
        let reconstructed = quantized.to_tensor(&device)?;

        let orig_data = original.to_vec1::<f32>()?;
        let recon_data = reconstructed.to_vec1::<f32>()?;

        // Check quantization error is reasonable
        let max_diff = orig_data
            .iter()
            .zip(recon_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // With block-wise absmax quantization, error should be < 1% of range
        assert!(max_diff < 0.1, "Quantization error too large: {}", max_diff);
        Ok(())
    }

    #[test]
    fn test_quantized_muon_updates_params() -> Result<()> {
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
        let mut qmuon = QuantizedMuon::new(vars, 0.02, 0.95, 5, 0.0)?;
        qmuon.step(&grads, 1.0)?;

        let updated = w.to_vec2::<f32>()?;
        let changed = orig
            .iter()
            .flatten()
            .zip(updated.iter().flatten())
            .any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(
            changed,
            "Parameters should have changed after quantized step"
        );
        Ok(())
    }

    #[test]
    fn test_memory_reduction() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create some large parameters
        let _w1 = vb.get_with_hints(
            (4096, 4096),
            "w1",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;
        let _w2 = vb.get_with_hints(
            (4096, 11008),
            "w2",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let vars = varmap.all_vars();
        let qmuon = QuantizedMuon::new(vars, 0.02, 0.95, 5, 0.0)?;

        let stats = qmuon.memory_stats();
        println!("Memory reduction: {:.2}%", stats.memory_reduction * 100.0);
        println!("FP32: {} MB", stats.fp32_bytes / 1024 / 1024);
        println!("INT8+scales: {} MB", stats.total_quantized / 1024 / 1024);

        // Should get ~75% reduction (4x smaller INT8, plus small overhead for scales)
        assert!(
            stats.memory_reduction > 0.70,
            "Expected >70% memory reduction"
        );
        Ok(())
    }

    #[test]
    fn test_ema_update_accumulates() -> Result<()> {
        let device = Device::Cpu;
        let config = QuantConfig::default();

        // Start with zeros
        let zeros = Tensor::zeros((128,), DType::F32, &device)?;
        let mut state = QuantizedState::from_tensor(&zeros, &config)?;

        // Add some values via EMA
        for i in 0..5 {
            let data = vec![i as f32; 128];
            let new = Tensor::from_vec(data, 128, &device)?;
            state.ema_update(&new, 0.9, &config)?;
        }

        // State should be non-zero
        let result = state.to_tensor(&device)?;
        let sum = result.sum_all()?.to_scalar::<f32>()?;
        assert!(sum > 0.0, "EMA state should accumulate");
        Ok(())
    }
}
