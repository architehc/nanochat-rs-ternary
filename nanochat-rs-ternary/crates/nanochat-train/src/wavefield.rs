//! Differentiable Wave Field Attention for training with Candle autograd.
//!
//! Supports three transform-domain mixing modes for gradient flow:
//! - **FFT:** Circular (shift) convolution via `wave_fft::candle_fft`. Uses `Complex<f32>`.
//! - **FWHT:** XOR convolution `c[k] = (1/N) Σ a[i]·b[i⊕k]` via `wave_fft::candle_fwht`.
//!   Integer add/sub only. Pair-swap mixing at power-of-2 distances, not cyclic shifts.
//! - **Haar:** Diagonal wavelet-basis scaling `IHaar(Haar(s) ⊙ Haar(k))` via
//!   `wave_fft::candle_haar`. Integer add/sub only. Scale-selective filtering,
//!   NOT shift-invariant.
//!
//! **Training-specific:** All three modes use pure Candle tensor ops (reshape + narrow +
//! cat butterflies). Ops stay on whatever device the input lives on (CPU or CUDA) —
//! no device transfers needed. Autograd gradients come for free.

use std::cell::RefCell;

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::layers::BitLinearSTE;

/// Cached scatter matrices for wavefield attention.
/// Both the base and expanded versions are stored to avoid re-allocating
/// ~24MB of GPU memory per forward pass per layer.
struct ScatterCache {
    seq_len: usize,
    max_seq_len: usize,
    batch_n_heads: usize,
    /// [batch*n_heads, seq_len, field_size] — for gather step
    scatter_exp: Tensor,
    /// [batch*n_heads, field_size, seq_len] — for scatter step
    scatter_t_exp: Tensor,
}

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
    pub gate_proj: BitLinearSTE,    // dim -> n_heads * head_dim (per-head_dim gating)
    pub out_proj: BitLinearSTE,     // n_heads * head_dim -> dim
    pub omega: Tensor,              // [n_heads] FP32 — frequency
    pub alpha_raw: Tensor,          // [n_heads] FP32 — raw damping (softplus in forward)
    pub phi: Tensor,                // [n_heads] FP32 — phase
    pub coupling_logits: Option<Tensor>, // [n_heads, n_heads] FP32
    /// Direct Haar-domain kernel coefficients. When present (Haar mode),
    /// these bypass the time-domain kernel generation (omega/alpha/phi)
    /// and give the model independent control over every wavelet scale.
    /// Shape: [n_heads, field_size] FP32.
    pub kernel_haar_coeffs: Option<Tensor>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub field_size: usize,
    pub convolve_mode: ConvolveMode,
    pub haar_levels: Option<usize>,
    /// When true, gate_proj outputs [n_heads * head_dim] (per-element gating).
    /// When false (legacy), gate_proj outputs [n_heads] (per-head gating, broadcast across head_dim).
    pub per_elem_gate: bool,
    /// Cached t_tensor [field_size] — same every call, precomputed once.
    t_tensor: Tensor,
    /// Cached expanded scatter matrices keyed by (seq_len, max_seq_len, batch*n_heads).
    /// Stores: scatter_mat_exp [BN, seq, field] and scatter_mat_t_exp [BN, field, seq]
    /// These are the biggest per-call allocations (~24MB total).
    scatter_cache: RefCell<Option<ScatterCache>>,
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
        per_elem_gate: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert!(
            field_size >= 2,
            "WaveField field_size must be >= 2 (got {}). \
             field_size=1 collapses all tokens to a single position, \
             making the wave field stateless.",
            field_size
        );

        let total_proj = n_heads * head_dim;

        let scatter_proj = BitLinearSTE::new(dim, total_proj, group_size, vb.pp("scatter"))?;
        // per_elem_gate=true: [dim -> n_heads*head_dim] fine-grained gating (new)
        // per_elem_gate=false: [dim -> n_heads] per-head gating, broadcast (legacy)
        let gate_out = if per_elem_gate { total_proj } else { n_heads };
        let gate_proj = BitLinearSTE::new(dim, gate_out, group_size, vb.pp("gate"))?;
        let out_proj = BitLinearSTE::new(total_proj, dim, group_size, vb.pp("out"))?;

        // Physics params created via VarBuilder so they're tracked in VarMap
        // (required for optimizer management and checkpoint save/restore).
        // Uniform init provides symmetry breaking across heads.
        // For Haar mode with direct kernel coefficients, omega/alpha_raw/phi are
        // still created (for checkpoint compatibility) but not used in forward().
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

        // For Haar mode with direct coefficients: learn kernel in the Haar domain.
        // This gives the model independent control over every wavelet scale,
        // bypassing the time-domain parameterization (omega/alpha/phi).
        // Only enabled when per_elem_gate=true (wavefield_haar_direct config).
        let kernel_haar_coeffs = if convolve_mode == ConvolveMode::Haar && per_elem_gate {
            let coeffs = vb.get_with_hints(
                (n_heads, field_size),
                "kernel_haar_coeffs",
                candle_nn::Init::Uniform { lo: -0.5, up: 0.5 },
            )?;
            Some(coeffs)
        } else {
            None
        };

        // Precompute t_tensor [0, 1, ..., field_size-1] — constant, detached.
        let t_vals: Vec<f32> = (0..field_size).map(|t| t as f32).collect();
        let device = omega.device();
        let t_tensor = Tensor::from_vec(t_vals, field_size, device)?.detach();

        Ok(Self {
            scatter_proj,
            gate_proj,
            out_proj,
            omega,
            alpha_raw,
            phi,
            coupling_logits,
            kernel_haar_coeffs,
            n_heads,
            head_dim,
            field_size,
            convolve_mode,
            haar_levels,
            per_elem_gate,
            t_tensor,
            scatter_cache: RefCell::new(None),
        })
    }

    /// Build expanded scatter matrices for the given dimensions.
    /// Returns ScatterCache with both [BN, seq, field] and [BN, field, seq] variants.
    /// All tensors are detached (constant data, no gradients needed).
    fn build_scatter_cache(
        &self,
        seq_len: usize,
        max_seq_len: usize,
        batch_n_heads: usize,
        device: &candle_core::Device,
    ) -> Result<ScatterCache> {
        let field_size = self.field_size;
        let stride = if max_seq_len > 1 {
            (field_size - 1) as f32 / (max_seq_len - 1) as f32
        } else {
            0.0
        };
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
        let mat = Tensor::from_vec(scatter_weights, (seq_len, field_size), device)?.detach();

        // Pre-expand for batched matmul — avoids allocating these every forward pass
        let scatter_exp = mat
            .unsqueeze(0)?
            .broadcast_as((batch_n_heads, seq_len, field_size))?
            .contiguous()?
            .detach();
        let scatter_t_exp = mat
            .t()?
            .contiguous()?
            .unsqueeze(0)?
            .broadcast_as((batch_n_heads, field_size, seq_len))?
            .contiguous()?
            .detach();

        Ok(ScatterCache {
            seq_len,
            max_seq_len,
            batch_n_heads,
            scatter_exp,
            scatter_t_exp,
        })
    }

    /// Forward pass: x [batch, seq_len, dim] -> [batch, seq_len, dim]
    ///
    /// 1. Scatter tokens onto per-head wave fields via bilinear interpolation
    /// 2. Generate wave kernel from physics params (or use direct Haar coefficients)
    /// 3. Batched transform-domain convolution (all heads at once)
    /// 4. Optional inter-head coupling
    /// 5. Gather per-token values from convolved fields
    /// 6. Per-head_dim content gate + output projection
    ///
    /// **Bidirectional note (Haar mode):** Haar wave field layers perform
    /// scale-selective filtering — diagonal scaling in the wavelet basis — which
    /// is inherently bidirectional. The Haar DWT does NOT perform positional
    /// mixing; it only weights wavelet scales (coarse vs fine features). Causal
    /// structure is enforced by the standard attention layers in the model, not
    /// by the wave field. This is by design: the wave field provides
    /// frequency-domain feature selection while attention provides causal
    /// sequential reasoning.
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

        // 3. Build fields via bilinear scatter (cached — same for given dims)
        let bn = batch * n_heads;
        let (scatter_exp, scatter_t_exp) = {
            let cache = self.scatter_cache.borrow();
            let hit = cache.as_ref().map_or(false, |c| {
                c.seq_len == seq_len && c.max_seq_len == max_seq_len && c.batch_n_heads == bn
            });
            if hit {
                let c = cache.as_ref().unwrap();
                (c.scatter_exp.clone(), c.scatter_t_exp.clone())
            } else {
                drop(cache);
                let new_cache = self.build_scatter_cache(seq_len, max_seq_len, bn, x.device())?;
                let exp = new_cache.scatter_exp.clone();
                let t_exp = new_cache.scatter_t_exp.clone();
                *self.scatter_cache.borrow_mut() = Some(new_cache);
                (exp, t_exp)
            }
        };

        // scattered: [batch, seq, n_heads, head_dim]
        // -> fields [batch, n_heads, field_size, head_dim]
        // = scatter_mat^T @ scattered (per batch, per head)
        let scattered_t = scattered.transpose(1, 2)?; // [batch, n_heads, seq, head_dim]
        let scattered_flat = scattered_t.reshape((bn, seq_len, head_dim))?;
        // Use cached expanded transpose (detached — no graph accumulation)
        let fields = scatter_t_exp.matmul(&scattered_flat)?;

        // 4. Generate wave kernel from physics params
        // kernel[h, t] = exp(-softplus(alpha_raw[h]) * t) * cos(omega[h] * t + phi[h])
        // For Haar mode with direct kernel_haar_coeffs, we skip this and use
        // the learned Haar-domain coefficients directly (see step 5).
        let kernel = if self.kernel_haar_coeffs.is_none() {
            let alpha = softplus(&self.alpha_raw)?; // [n_heads], always positive

            let alpha_2d = alpha.unsqueeze(1)?;      // [n_heads, 1]
            let omega_2d = self.omega.unsqueeze(1)?;  // [n_heads, 1]
            let phi_2d = self.phi.unsqueeze(1)?;      // [n_heads, 1]
            let t_2d = self.t_tensor.unsqueeze(0)?;   // [1, field_size] (precomputed)

            // damping: exp(-alpha * t)
            let neg_alpha_t = alpha_2d.broadcast_mul(&t_2d)?.neg()?;
            let damping = neg_alpha_t.exp()?;

            // oscillation: cos(omega * t + phi)
            let phase = omega_2d.broadcast_mul(&t_2d)?.broadcast_add(&phi_2d)?;
            let oscillation = phase.cos()?;

            // kernel: [n_heads, field_size]
            Some(damping.mul(&oscillation)?)
        } else {
            None // Haar direct coefficients path — no time-domain kernel needed
        };

        // 5. Batched transform-domain convolution (all heads at once)
        let fields = fields.reshape((batch, n_heads, field_size, head_dim))?;

        // Reshape fields to [batch * n_heads * head_dim, field_size] for batched convolution
        let fields_for_conv = fields
            .transpose(2, 3)? // [batch, n_heads, head_dim, field_size]
            .contiguous()?
            .reshape((batch * n_heads * head_dim, field_size))?;

        let convolved_flat = match self.convolve_mode {
            ConvolveMode::Fft | ConvolveMode::Fwht => {
                // kernel: [n_heads, field_size] -> expand to [batch * n_heads * head_dim, field_size]
                let kernel = kernel.as_ref().unwrap();
                let kernel_expanded = kernel
                    .unsqueeze(1)? // [n_heads, 1, field_size]
                    .broadcast_as((n_heads, head_dim, field_size))?
                    .contiguous()?
                    .reshape((n_heads * head_dim, field_size))?;
                let kernel_batched = kernel_expanded
                    .unsqueeze(0)? // [1, n_heads * head_dim, field_size]
                    .broadcast_as((batch, n_heads * head_dim, field_size))?
                    .contiguous()?
                    .reshape((batch * n_heads * head_dim, field_size))?;

                match self.convolve_mode {
                    ConvolveMode::Fft => {
                        wave_fft::candle_fft::fft_convolve_with_grad(
                            &fields_for_conv, &kernel_batched, field_size,
                        )?
                    }
                    ConvolveMode::Fwht => {
                        wave_fft::candle_fwht::fwht_convolve_with_grad(
                            &fields_for_conv, &kernel_batched, field_size,
                        )?
                    }
                    _ => unreachable!(),
                }
            }
            ConvolveMode::Haar => {
                let levels = self.haar_levels
                    .unwrap_or_else(|| (field_size as f64).log2() as usize);

                if let Some(ref coeffs) = self.kernel_haar_coeffs {
                    // Direct Haar coefficients path: coeffs are already in Haar domain.
                    // Use haar_scale_with_grad which does IHaar(Haar(signal) * coeffs)
                    // without transforming the coefficients.
                    let coeffs_expanded = coeffs
                        .unsqueeze(1)? // [n_heads, 1, field_size]
                        .broadcast_as((n_heads, head_dim, field_size))?
                        .contiguous()?
                        .reshape((n_heads * head_dim, field_size))?;
                    let coeffs_batched = coeffs_expanded
                        .unsqueeze(0)? // [1, n_heads * head_dim, field_size]
                        .broadcast_as((batch, n_heads * head_dim, field_size))?
                        .contiguous()?
                        .reshape((batch * n_heads * head_dim, field_size))?;
                    wave_fft::candle_haar::haar_scale_with_grad(
                        &fields_for_conv, &coeffs_batched, field_size, levels,
                    )?
                } else {
                    // Fallback: time-domain kernel path (Haar transform of both signal and kernel)
                    let kernel = kernel.as_ref().unwrap();
                    let kernel_expanded = kernel
                        .unsqueeze(1)?
                        .broadcast_as((n_heads, head_dim, field_size))?
                        .contiguous()?
                        .reshape((n_heads * head_dim, field_size))?;
                    let kernel_batched = kernel_expanded
                        .unsqueeze(0)?
                        .broadcast_as((batch, n_heads * head_dim, field_size))?
                        .contiguous()?
                        .reshape((batch * n_heads * head_dim, field_size))?;
                    wave_fft::candle_haar::haar_convolve_with_grad(
                        &fields_for_conv, &kernel_batched, field_size, levels,
                    )?
                }
            }
        }; // [batch * n_heads * head_dim, field_size]

        // Reshape back: [batch * n_heads * head_dim, field_size] -> [batch, n_heads, field_size, head_dim]
        let convolved = convolved_flat
            .reshape((batch, n_heads, head_dim, field_size))?
            .transpose(2, 3)? // [batch, n_heads, field_size, head_dim]
            .contiguous()?;

        // 6. Optional head coupling
        let convolved = if let Some(ref logits) = self.coupling_logits {
            // Manual softmax: avoids candle_nn CustomOp which may lack CUDA
            let max_logits = logits.max(D::Minus1)?.unsqueeze(D::Minus1)?;
            let shifted = logits.broadcast_sub(&max_logits)?;
            let exp_shifted = shifted.exp()?;
            let sum_exp = exp_shifted.sum(D::Minus1)?.unsqueeze(D::Minus1)?;
            let weights = exp_shifted.broadcast_div(&sum_exp)?;
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

        // 7. Gather per-token values from convolved fields (using cached scatter_exp)
        let convolved_flat = convolved.reshape((bn, field_size, head_dim))?;
        let gathered = scatter_exp.matmul(&convolved_flat)?;
        let gathered = gathered.reshape((batch, n_heads, seq_len, head_dim))?;
        let gathered = gathered.transpose(1, 2)?; // [batch, seq_len, n_heads, head_dim]
        let gathered = gathered.reshape((batch, seq_len, total_proj))?;

        // 8. Content gate: sigmoid(gate_proj(x)) * gathered
        let gate = self.gate_proj.forward(x)?;
        // Manual sigmoid: 1 / (1 + exp(-x)) — avoids candle_nn CustomOp which lacks CUDA
        let gate = gate.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
        // Legacy per-head gate [batch, seq, n_heads] needs broadcast to [batch, seq, n_heads*head_dim]
        let gate = if !self.per_elem_gate {
            let gate = gate.reshape((batch, seq_len, n_heads, 1))?;
            gate.broadcast_as((batch, seq_len, n_heads, head_dim))?
                .reshape((batch, seq_len, total_proj))?
        } else {
            gate
        };
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
    /// Includes kernel_haar_coeffs when present (Haar mode) — these are physics-like
    /// params (wavelet-scale weights), not linear projection weights.
    pub fn physics_params(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.omega, &self.alpha_raw, &self.phi];
        if let Some(ref logits) = self.coupling_logits {
            params.push(logits);
        }
        if let Some(ref coeffs) = self.kernel_haar_coeffs {
            params.push(coeffs);
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
    // Use affine(1.0, 1.0) instead of allocating ones_like + add
    let log_term = exp_neg.affine(1.0, 1.0)?.log()?;
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
            true, // per_elem_gate
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

    fn make_wave_field_haar(
        dim: usize,
        n_heads: usize,
        head_dim: usize,
        field_size: usize,
    ) -> Result<(WaveFieldAttentionTrain, VarMap)> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let levels = (field_size as f64).log2() as usize;
        let wf = WaveFieldAttentionTrain::new(
            dim, n_heads, head_dim, field_size,
            dim, // group_size = dim for test
            true, // use_head_coupling
            ConvolveMode::Haar,
            Some(levels),
            true, // per_elem_gate
            vb.pp("wf"),
        )?;
        Ok((wf, varmap))
    }

    #[test]
    fn test_haar_direct_coeffs_forward_shape() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field_haar(dim, n_heads, head_dim, field_size)?;

        // kernel_haar_coeffs should be present in Haar mode
        assert!(wf.kernel_haar_coeffs.is_some(), "Haar mode should have kernel_haar_coeffs");

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 0.1, (2, 8, dim), &device)?;
        let out = wf.forward(&x, 32)?;
        assert_eq!(out.dims(), &[2, 8, dim]);
        Ok(())
    }

    #[test]
    fn test_haar_direct_coeffs_have_gradients() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field_haar(dim, n_heads, head_dim, field_size)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 0.1, (1, 4, dim), &device)?;
        let out = wf.forward(&x, 32)?;
        let loss = out.sum_all()?;
        let grads = loss.backward()?;

        // Direct Haar coefficients should have gradients
        let coeffs = wf.kernel_haar_coeffs.as_ref().unwrap();
        let grad = grads.get(coeffs);
        assert!(grad.is_some(), "kernel_haar_coeffs should have gradient");

        let grad_norm = grad.unwrap().sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(grad_norm > 0.0, "kernel_haar_coeffs gradient should be non-zero");

        // omega/alpha_raw/phi should NOT have gradients (not used in Haar direct path)
        // (They are created but not part of the forward computation graph.)
        assert!(grads.get(&wf.omega).is_none(), "omega should not have gradient in Haar direct path");

        Ok(())
    }

    #[test]
    fn test_haar_training_step_var_count() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, varmap) = make_wave_field_haar(dim, n_heads, head_dim, field_size)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 0.1, (1, 4, dim), &device)?;
        let out = wf.forward(&x, 32)?;
        let loss = out.sqr()?.mean_all()?;
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(loss_val.is_finite(), "loss should be finite");

        let _grads = loss.backward()?;

        // VarMap has: 3 projections + coupling + 3 physics + kernel_haar_coeffs = 8
        let vars = varmap.all_vars();
        assert_eq!(vars.len(), 8, "Haar mode should have 8 vars, got {}", vars.len());

        Ok(())
    }

    #[test]
    fn test_haar_coeffs_in_physics_params() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field_haar(dim, n_heads, head_dim, field_size)?;

        let physics = wf.physics_params();
        // Should include: omega, alpha_raw, phi, coupling_logits, kernel_haar_coeffs = 5
        assert_eq!(physics.len(), 5, "Haar physics params should include kernel_haar_coeffs");

        let linear = wf.linear_params();
        // Should NOT include kernel_haar_coeffs (only scatter, gate, out = 3)
        assert_eq!(linear.len(), 3, "linear params should not include kernel_haar_coeffs");

        Ok(())
    }

    #[test]
    fn test_fft_no_haar_coeffs() -> Result<()> {
        let dim = 64;
        let n_heads = 4;
        let head_dim = 16;
        let field_size = 32;
        let (wf, _varmap) = make_wave_field(dim, n_heads, head_dim, field_size)?;

        // FFT mode should NOT have kernel_haar_coeffs
        assert!(wf.kernel_haar_coeffs.is_none(), "FFT mode should not have kernel_haar_coeffs");

        let physics = wf.physics_params();
        // omega, alpha_raw, phi, coupling_logits = 4
        assert_eq!(physics.len(), 4, "FFT physics params count");

        Ok(())
    }

    #[test]
    #[should_panic(expected = "field_size must be >= 2")]
    fn test_field_size_one_panics() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // field_size=1 should panic during construction
        let _ = WaveFieldAttentionTrain::new(
            64,  // dim
            4,   // n_heads
            16,  // head_dim
            1,   // field_size — degenerate, should panic
            64,  // group_size
            false,
            ConvolveMode::Haar,
            Some(0),
            false,
            vb.pp("wave"),
        );
    }
}
