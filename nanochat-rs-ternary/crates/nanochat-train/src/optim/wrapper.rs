//! Optimizer wrapper for conditional optimizer selection based on config.

use candle_core::{backprop::GradStore, Result, Var};

use super::{GaLore2Muon, Muon, QuantizedMuon};

/// Wrapper enum for different Muon optimizer variants
pub enum MuonOptimizer {
    /// Standard FP32 Muon
    Standard(Muon),
    /// 8-bit quantized Muon (75% memory reduction)
    Quantized(QuantizedMuon),
    /// GaLore2 wrapper around standard Muon (50-65% memory reduction)
    GaLore(GaLore2Muon),
}

impl MuonOptimizer {
    /// Create optimizer based on config flags
    #[allow(clippy::too_many_arguments)] // Config constructor needs all optimizer parameters
    pub fn from_config(
        vars: Vec<Var>,
        lr: f64,
        beta: f64,
        ns_steps: usize,
        wd: f64,
        use_8bit: bool,
        use_galore: bool,
        galore_rank: usize,
        galore_update_freq: usize,
    ) -> Result<Self> {
        match (use_8bit, use_galore) {
            (false, false) => {
                // Standard FP32 Muon
                let muon = Muon::new(vars, lr, beta, ns_steps, wd)?;
                Ok(MuonOptimizer::Standard(muon))
            }
            (true, false) => {
                // 8-bit quantized Muon
                let qmuon = QuantizedMuon::new(vars, lr, beta, ns_steps, wd)?;
                Ok(MuonOptimizer::Quantized(qmuon))
            }
            (false, true) => {
                // GaLore2 with standard Muon
                let muon = Muon::new(vars.clone(), lr, beta, ns_steps, wd)?;
                let galore = GaLore2Muon::new(muon, vars, galore_rank, galore_update_freq)?;
                Ok(MuonOptimizer::GaLore(galore))
            }
            (true, true) => {
                // GaLore2 with quantized Muon (maximum memory savings)
                let qmuon = QuantizedMuon::new(vars.clone(), lr, beta, ns_steps, wd)?;
                // TODO: Create GaLore2<QuantizedMuon> variant
                // For now, just use quantized (8-bit gives most of the savings)
                Ok(MuonOptimizer::Quantized(qmuon))
            }
        }
    }

    /// Execute optimizer step
    pub fn step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<()> {
        match self {
            MuonOptimizer::Standard(m) => m.step(grads, clip_scale),
            MuonOptimizer::Quantized(qm) => qm.step(grads, clip_scale),
            MuonOptimizer::GaLore(gm) => gm.step(grads, clip_scale),
        }
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f64) {
        match self {
            MuonOptimizer::Standard(m) => m.set_lr(lr),
            MuonOptimizer::Quantized(qm) => qm.set_lr(lr),
            MuonOptimizer::GaLore(gm) => gm.set_lr(lr),
        }
    }

    /// Get memory statistics (optional, for logging)
    pub fn memory_stats(&self) -> OptimizerMemoryStats {
        match self {
            MuonOptimizer::Standard(_) => OptimizerMemoryStats {
                variant: "Standard Muon",
                memory_reduction: 0.0,
                details: "No quantization".to_string(),
            },
            MuonOptimizer::Quantized(qm) => {
                let stats = qm.memory_stats();
                OptimizerMemoryStats {
                    variant: "8-bit Quantized Muon",
                    memory_reduction: stats.memory_reduction,
                    details: format!(
                        "FP32: {} MB, INT8: {} MB",
                        stats.fp32_bytes / 1024 / 1024,
                        stats.total_quantized / 1024 / 1024
                    ),
                }
            }
            MuonOptimizer::GaLore(gm) => {
                let stats = gm.memory_stats();
                OptimizerMemoryStats {
                    variant: "GaLore2 Muon",
                    memory_reduction: stats.memory_reduction,
                    details: format!(
                        "Total: {}, Projected: {}",
                        stats.total_params, stats.projected_params
                    ),
                }
            }
        }
    }
}

/// Memory statistics for logging
pub struct OptimizerMemoryStats {
    pub variant: &'static str,
    pub memory_reduction: f64,
    pub details: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarMap;

    #[test]
    fn test_optimizer_wrapper_standard() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _w = vb.get_with_hints(
            (16, 8),
            "w",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let vars = varmap.all_vars();
        let opt = MuonOptimizer::from_config(vars, 0.02, 0.95, 5, 0.0, false, false, 256, 200)?;

        match opt {
            MuonOptimizer::Standard(_) => Ok(()),
            _ => panic!("Expected Standard variant"),
        }
    }

    #[test]
    fn test_optimizer_wrapper_quantized() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _w = vb.get_with_hints(
            (16, 8),
            "w",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let vars = varmap.all_vars();
        let opt = MuonOptimizer::from_config(vars, 0.02, 0.95, 5, 0.0, true, false, 256, 200)?;

        match opt {
            MuonOptimizer::Quantized(_) => Ok(()),
            _ => panic!("Expected Quantized variant"),
        }
    }

    #[test]
    fn test_optimizer_wrapper_galore() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _w = vb.get_with_hints(
            (256, 256),
            "w",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let vars = varmap.all_vars();
        let opt = MuonOptimizer::from_config(vars, 0.02, 0.95, 5, 0.0, false, true, 64, 200)?;

        match opt {
            MuonOptimizer::GaLore(_) => Ok(()),
            _ => panic!("Expected GaLore variant"),
        }
    }

    #[test]
    fn test_optimizer_step() -> Result<()> {
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

        let vars = varmap.all_vars();
        let mut opt = MuonOptimizer::from_config(vars, 0.02, 0.95, 5, 0.0, true, false, 256, 200)?;

        // Generate gradients
        let x = candle_core::Tensor::randn(0.0f32, 1.0, (1, 8), &device)?;
        let y = x.matmul(&w.t()?)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        // Step should work
        opt.step(&grads, 1.0)?;
        opt.set_lr(0.01);

        Ok(())
    }
}
