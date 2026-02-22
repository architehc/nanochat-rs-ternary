//! Differentiable Wave Field Attention for training with Candle autograd.
//!
//! Supports three convolution modes for gradient flow:
//! - FFT: Complex<f32> via wave_fft::candle_fft (original)
//! - FWHT: Integer add/sub only via wave_fft::candle_fwht
//! - Haar DWT: Integer add/sub only via wave_fft::candle_haar

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::layers::BitLinearSTE;

// Re-export from nanochat_model::config via the public type.
// We define it locally to avoid adding nanochat-model as a dependency.
pub use nanochat_model_config::ConvolveMode;

mod nanochat_model_config {
    /// Convolution mode for wave field attention (mirrors nanochat_model::config::ConvolveMode).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ConvolveMode {
        Fft,
        Fwht,
        Haar,
    }
}

/// Differentiable wave field attention for training.
///
/// Ternary projections use BitLinearSTE (STE through quantization).
/// Physics params (omega, alpha_raw, phi) are FP32 Vars with direct gradients.
pub struct WaveFieldAttentionTrain {
    pub scatter_proj: BitLinearSTE, // dim -> n_heads * head_dim
    pub gate_proj: BitLinearSTE,    // dim -> n_heads
    pub out_proj: BitLinearSTE,     // n_heads * head_dim -> dim
    pub omega: Tensor,              // [n_heads] FP32 — frequency
    pub alpha_raw: Tensor,          // [n_heads] FP32 — raw damping (softplus in forward)
    pub phi: Tensor,                // [n_heads] FP32 — phase
    pub coupling_logits: Option<Tensor>, // [n_heads, n_heads] FP32
    pub n_heads: usize,
    pub head_dim: usize,
    pub field_size: usize,
    pub convolve_mode: ConvolveMode,
    pub haar_levels: Option<usize>,
}

impl WaveFieldAttentionTrain {
    pub fn new(
        dim: usize,
        n_heads: usize,
        head_dim: usize,
        field_size: usize,
        group_size: usize,
        use_head_coupling: bool,
        convolve_mode: ConvolveMode,
        haar_levels: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let total_proj = n_heads * head_dim;

        let scatter_proj = BitLinearSTE::new(dim, total_proj, group_size, vb.pp("scatter"))?;
        let gate_proj = BitLinearSTE::new(dim, n_heads, group_size, vb.pp("gate"))?;
        let out_proj = BitLinearSTE::new(total_proj, dim, group_size, vb.pp("out"))?;

        // Physics params created via VarBuilder so they're tracked in VarMap
        // (required for optimizer management and checkpoint save/restore).
        // Uniform init provides symmetry breaking across heads.
        let omega = vb.get_with_hints(
            n_heads,
            "omega",
            candle_nn::Init::Uniform { lo: 0.3, up: 4.0 },
        )?;

        let alpha_raw = vb.get_with_hints(
            n_heads,
            "alpha_raw",
            candle_nn::Init::Uniform { lo: -3.0, up: 0.5 },
        )?;

        let phi = vb.get_with_hints(
            n_heads,
            "phi",
            candle_nn::Init::Uniform { lo: 0.0, up: std::f64::consts::PI },
        )?;

        let coupling_logits = if use_head_coupling {
            // Initialize as zeros — softmax of zeros = uniform coupling.
            // During training, diagonal elements will learn to dominate.
            let logits = vb.get_with_hints(
                (n_heads, n_heads),
                "coupling_logits",
                candle_nn::Init::Const(0.0),
            )?;
            Some(logits)
        } else {
            None
        };

        Ok(Self {
            scatter_proj,
            gate_proj,
            out_proj,
            omega,
            alpha_raw,
            phi,
            coupling_logits,
            n_heads,
            head_dim,
            field_size,
            convolve_mode,
            haar_levels,
        })
    }

    /// Forward pass: x [batch, seq_len, dim] -> [batch, seq_len, dim]
    ///
    /// 1. Scatter tokens onto per-head wave fields via bilinear interpolation
    /// 2. Generate causal wave kernel from physics params
    /// 3. FFT convolution (autograd-compatible via CustomOp2)
    /// 4. Optional inter-head coupling
    /// 5. Gather per-token values from convolved fields
    /// 6. Content gate + output projection
    pub fn forward(&self, x: &Tensor, max_seq_len: usize) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (batch, seq_len, _dim) = (dims[0], dims[1], dims[2]);
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let field_size = self.field_size;
        let total_proj = n_heads * head_dim;

        // 1. Project to scatter space: [batch, seq, n_heads * head_dim]
        let scattered = self.scatter_proj.forward(x)?;

        // 2. Reshape to [batch, seq, n_heads, head_dim]
        let scattered = scattered.reshape((batch, seq_len, n_heads, head_dim))?;

        // 3. Build fields via bilinear scatter
        let stride = if max_seq_len > 1 {
            (field_size - 1) as f32 / (max_seq_len - 1) as f32
        } else {
            0.0
        };

        // Build scatter matrix: [seq_len, field_size]
        // Clamp positions to [0, field_size - 1] to prevent OOB when seq_len > max_seq_len
        let max_pos = (field_size - 1) as f32;
        let positions: Vec<f32> = (0..seq_len)
            .map(|t| (t as f32 * stride).clamp(0.0, max_pos))
            .collect();

        let mut scatter_weights = vec![0.0f32; seq_len * field_size];
        for t in 0..seq_len {
            let pos = positions[t];
            let idx_lo = (pos.floor() as usize).min(field_size - 1);
            let idx_hi = (idx_lo + 1).min(field_size - 1);
            let frac = pos - idx_lo as f32;
            scatter_weights[t * field_size + idx_lo] += 1.0 - frac;
            if idx_hi != idx_lo {
                scatter_weights[t * field_size + idx_hi] += frac;
            }
        }
        let scatter_mat = Tensor::from_vec(
            scatter_weights,
            (seq_len, field_size),
            x.device(),
        )?;

        // scattered: [batch, seq, n_heads, head_dim]
        // -> fields [batch, n_heads, field_size, head_dim]
        // = scatter_mat^T @ scattered (per batch, per head)
        let scattered_t = scattered.transpose(1, 2)?; // [batch, n_heads, seq, head_dim]
        let scattered_flat = scattered_t.reshape((batch * n_heads, seq_len, head_dim))?;
        let scatter_mat_t = scatter_mat.t()?; // [field_size, seq_len]
        // Expand scatter_mat_t for batched matmul
        let scatter_mat_t_exp = scatter_mat_t
            .unsqueeze(0)?
            .broadcast_as((batch * n_heads, field_size, seq_len))?
            .contiguous()?;
        let fields = scatter_mat_t_exp.matmul(&scattered_flat)?;

        // 4. Generate wave kernel from physics params
        // kernel[h, t] = exp(-softplus(alpha_raw[h]) * t) * cos(omega[h] * t + phi[h])
        let alpha = softplus(&self.alpha_raw)?; // [n_heads], always positive

        let t_vals: Vec<f32> = (0..field_size).map(|t| t as f32).collect();
        let t_tensor = Tensor::from_vec(t_vals, field_size, x.device())?;

        let alpha_2d = alpha.unsqueeze(1)?;      // [n_heads, 1]
        let omega_2d = self.omega.unsqueeze(1)?;  // [n_heads, 1]
        let phi_2d = self.phi.unsqueeze(1)?;      // [n_heads, 1]
        let t_2d = t_tensor.unsqueeze(0)?;        // [1, field_size]

        // damping: exp(-alpha * t)
        let neg_alpha_t = alpha_2d.broadcast_mul(&t_2d)?.neg()?;
        let damping = neg_alpha_t.exp()?;

        // oscillation: cos(omega * t + phi)
        let phase = omega_2d.broadcast_mul(&t_2d)?.broadcast_add(&phi_2d)?;
        let oscillation = phase.cos()?;

        // kernel: [n_heads, field_size]
        let kernel = damping.mul(&oscillation)?;

        // 5. Transform-domain convolution per head (dispatched by convolve_mode)
        let fields = fields.reshape((batch, n_heads, field_size, head_dim))?;

        let mut convolved_heads = Vec::with_capacity(n_heads);
        for h in 0..n_heads {
            let field_h = fields.narrow(1, h, 1)?.squeeze(1)?; // [batch, field_size, head_dim]
            let kernel_h = kernel.narrow(0, h, 1)?.squeeze(0)?.contiguous()?; // [field_size]

            // Transpose to [batch, head_dim, field_size], flatten to [batch*head_dim, field_size]
            let field_t = field_h.transpose(1, 2)?.contiguous()?; // [batch, head_dim, field_size]
            let field_flat = field_t.reshape((batch * head_dim, field_size))?;

            // Dispatch convolution based on mode
            let conv_flat = match self.convolve_mode {
                ConvolveMode::Fft => {
                    wave_fft::candle_fft::fft_convolve_with_grad(
                        &field_flat, &kernel_h, field_size,
                    )?
                }
                ConvolveMode::Fwht => {
                    wave_fft::candle_fwht::fwht_convolve_with_grad(
                        &field_flat, &kernel_h, field_size,
                    )?
                }
                ConvolveMode::Haar => {
                    let levels = self.haar_levels
                        .unwrap_or_else(|| (field_size as f64).log2() as usize);
                    wave_fft::candle_haar::haar_convolve_with_grad(
                        &field_flat, &kernel_h, field_size, levels,
                    )?
                }
            }; // [batch*head_dim, field_size]

            // Reshape back: [batch, head_dim, field_size] -> transpose -> [batch, field_size, head_dim]
            let conv_h = conv_flat
                .reshape((batch, head_dim, field_size))?
                .transpose(1, 2)?
                .contiguous()?; // [batch, field_size, head_dim]
            convolved_heads.push(conv_h.unsqueeze(1)?); // [batch, 1, field_size, head_dim]
        }
        let convolved = Tensor::cat(&convolved_heads, 1)?; // [batch, n_heads, field_size, head_dim]

        // 6. Optional head coupling
        let convolved = if let Some(ref logits) = self.coupling_logits {
            let weights = candle_nn::ops::softmax(logits, D::Minus1)?;
            let c_flat = convolved.reshape((batch, n_heads, field_size * head_dim))?;
            let weights_exp = weights
                .unsqueeze(0)?
                .broadcast_as((batch, n_heads, n_heads))?
                .contiguous()?;
            let coupled = weights_exp.matmul(&c_flat)?;
            coupled.reshape((batch, n_heads, field_size, head_dim))?
        } else {
            convolved
        };

        // 7. Gather per-token values from convolved fields
        let convolved_flat = convolved.reshape((batch * n_heads, field_size, head_dim))?;
        let scatter_mat_exp = scatter_mat
            .unsqueeze(0)?
            .broadcast_as((batch * n_heads, seq_len, field_size))?
            .contiguous()?;
        let gathered = scatter_mat_exp.matmul(&convolved_flat)?;
        let gathered = gathered.reshape((batch, n_heads, seq_len, head_dim))?;
        let gathered = gathered.transpose(1, 2)?; // [batch, seq_len, n_heads, head_dim]
        let gathered = gathered.reshape((batch, seq_len, total_proj))?;

        // 8. Content gate: sigmoid(gate_proj(x)) * gathered
        let gate = self.gate_proj.forward(x)?; // [batch, seq_len, n_heads]
        let gate = candle_nn::ops::sigmoid(&gate)?;
        let gate = gate
            .unsqueeze(3)?
            .broadcast_as((batch, seq_len, n_heads, head_dim))?
            .contiguous()?
            .reshape((batch, seq_len, total_proj))?;
        let gated = gathered.mul(&gate)?;

        // 9. Output projection
        self.out_proj.forward(&gated)
    }

    /// Collect linear weight parameters (for Muon optimizer).
    pub fn linear_params(&self) -> Vec<&Tensor> {
        vec![
            self.scatter_proj.weight(),
            self.gate_proj.weight(),
            self.out_proj.weight(),
        ]
    }

    /// Collect physics + coupling parameters (for Lion optimizer, same group as mHC).
    pub fn physics_params(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.omega, &self.alpha_raw, &self.phi];
        if let Some(ref logits) = self.coupling_logits {
            params.push(logits);
        }
        params
    }
}

/// Numerically stable softplus: log(1 + exp(x)).
///
/// For large x (>20), exp(x) overflows f32. But softplus(x) ≈ x when x >> 0,
/// so we use a piecewise approach: softplus(x) = x + log(1 + exp(-|x|)).
/// This avoids overflow since exp(-|x|) ≤ 1 for all x.
fn softplus(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = x + log(1 + exp(-|x|))  [numerically stable for all x]
    //
    // Derivation: log(1 + exp(x)) = log(exp(x)(exp(-x) + 1))
    //           = x + log(1 + exp(-x))
    // For x < 0: use original formula log(1 + exp(x)) since exp(x) < 1
    // Combined: max(x, 0) + log(1 + exp(-|x|))
    let abs_x = x.abs()?;
    let neg_abs = abs_x.neg()?;
    let exp_neg = neg_abs.exp()?;
    let ones = x.ones_like()?;
    let log_term = (ones + exp_neg)?.log()?;
    let relu_x = x.relu()?;
    relu_x + log_term
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn make_wave_field(
        dim: usize,
        n_heads: usize,
        head_dim: usize,
        field_size: usize,
    ) -> Result<(WaveFieldAttentionTrain, VarMap)> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let wf = WaveFieldAttentionTrain::new(
            dim, n_heads, head_dim, field_size,
            dim, // group_size = dim for test
            true, // use_head_coupling
            ConvolveMode::Fft,
            None,
            vb.pp("wf"),
        )?;
        Ok((wf, varmap))
    }

    #[test]
    fn test_forward_shape() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field(dim, n_heads, head_dim, field_size)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 0.1, (2, 8, dim), &device)?;
        let out = wf.forward(&x, 32)?;
        assert_eq!(out.dims(), &[2, 8, dim]);
        Ok(())
    }

    #[test]
    fn test_all_params_have_gradients() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field(dim, n_heads, head_dim, field_size)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 0.1, (1, 4, dim), &device)?;
        let out = wf.forward(&x, 32)?;
        let loss = out.sum_all()?;
        let grads = loss.backward()?;

        // Linear params
        assert!(grads.get(wf.scatter_proj.weight()).is_some(), "scatter grad missing");
        assert!(grads.get(wf.gate_proj.weight()).is_some(), "gate grad missing");
        assert!(grads.get(wf.out_proj.weight()).is_some(), "out grad missing");

        // Physics params
        assert!(grads.get(&wf.omega).is_some(), "omega grad missing");
        assert!(grads.get(&wf.alpha_raw).is_some(), "alpha_raw grad missing");
        assert!(grads.get(&wf.phi).is_some(), "phi grad missing");

        // Coupling
        if let Some(ref logits) = wf.coupling_logits {
            assert!(grads.get(logits).is_some(), "coupling grad missing");
        }

        Ok(())
    }

    #[test]
    fn test_physics_params_receive_nonzero_gradients() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field(dim, n_heads, head_dim, field_size)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 0.1, (1, 4, dim), &device)?;
        let out = wf.forward(&x, 32)?;
        let loss = out.sum_all()?;
        let grads = loss.backward()?;

        let omega_grad = grads.get(&wf.omega).unwrap();
        let omega_norm = omega_grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(omega_norm > 0.0, "omega gradient should be non-zero");

        let phi_grad = grads.get(&wf.phi).unwrap();
        let phi_norm = phi_grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(phi_norm > 0.0, "phi gradient should be non-zero");

        Ok(())
    }

    #[test]
    fn test_alpha_stays_positive() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field(dim, n_heads, head_dim, field_size)?;

        // softplus of alpha_raw should always be positive
        let alpha = softplus(&wf.alpha_raw)?;
        let alpha_vec: Vec<f32> = alpha.to_vec1()?;
        for (h, &a) in alpha_vec.iter().enumerate() {
            assert!(a > 0.0, "alpha[{}] should be positive, got {}", h, a);
        }
        Ok(())
    }

    #[test]
    fn test_coupling_softmax_rows_sum_to_one() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field(dim, n_heads, head_dim, field_size)?;

        if let Some(ref logits) = wf.coupling_logits {
            let weights = candle_nn::ops::softmax(logits, D::Minus1)?;
            let row_sums = weights.sum(D::Minus1)?;
            let row_sums_vec: Vec<f32> = row_sums.to_vec1()?;
            for (h, sum) in row_sums_vec.iter().enumerate() {
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "row {} sum should be 1.0, got {}",
                    h, sum
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_single_training_step() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, varmap) = make_wave_field(dim, n_heads, head_dim, field_size)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 0.1, (1, 4, dim), &device)?;
        let out = wf.forward(&x, 32)?;
        let loss = out.sqr()?.mean_all()?;
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(loss_val.is_finite(), "loss should be finite");

        let _grads = loss.backward()?;

        // VarMap has: 3 projection weights + 1 coupling_logits + 3 physics (omega, alpha_raw, phi) = 7
        let vars = varmap.all_vars();
        assert_eq!(vars.len(), 7, "should have 7 vars (3 projections + coupling + 3 physics), got {}", vars.len());

        Ok(())
    }
}
