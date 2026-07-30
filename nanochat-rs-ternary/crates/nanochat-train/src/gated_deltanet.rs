//! Gated DeltaNet training module (ICLR 2025, arXiv:2412.06464).
//!
//! Linear recurrent attention with delta rule updates, gated output,
//! and Mamba2-style alpha decay gate. Uses Candle autograd for training.
//!
//! Core algorithm per timestep t. The state is stored **value-major**,
//! `S = Σ_τ err_τ @ k_τ^T`, so both the prediction and the read are plain
//! matmuls with no transpose. This matches the inference implementation in
//! `nanochat-model/src/deltanet.rs` — keep the two orientations in sync.
//!
//! ```text
//! g_t = -A_log.exp() * softplus(a_proj(x) + dt_bias)   // always negative
//! beta_t = sigmoid(b_proj(x))                           // update gate
//! S_t = S_{t-1} * exp(g_t) + (beta_t * (v_t - S_{t-1} @ k_t)) @ k_t^T  // delta update
//! o_t = S_t @ q_t                                       // read output
//! y_t = wo(RMSNorm(o_t) * SiLU(gate_t))                 // gated output
//! ```
//!
//! Because K is L2-normalized, `S @ k_t` recovers the value currently stored
//! under key `k_t`, so a repeated key is *overwritten* rather than accumulated.
//! That overwrite property is what makes this the delta rule rather than plain
//! linear attention; `test_delta_rule_overwrites_repeated_key` pins it down.

#[cfg(test)]
use candle_core::DType;
use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::layers::{BitLinearSTE, RMSNormTrain};

/// Log-space decay rate at init. `exp(0.5) ≈ 1.6487`.
const A_LOG_INIT: f64 = 0.5;

/// Softplus bias at init, chosen so per-token decay ≈ 0.989. See `new`.
const DT_BIAS_INIT: f64 = -5.0;

/// Gated DeltaNet attention for training.
///
/// Recurrent over sequence positions. All projections use BitLinearSTE
/// for ternary quantization-aware training.
pub struct GatedDeltaNetTrain {
    pub wq: BitLinearSTE,
    pub wk: BitLinearSTE,
    pub wv: BitLinearSTE,
    pub wo: BitLinearSTE,
    pub a_proj: BitLinearSTE,    // dim -> n_heads (alpha gate input)
    pub b_proj: BitLinearSTE,    // dim -> n_heads (beta gate)
    pub g_proj: BitLinearSTE,    // dim -> n_heads * head_dim (output gate)
    pub a_log: Tensor,           // [n_heads] learnable log-space decay rate
    pub dt_bias: Tensor,         // [n_heads] learnable bias for softplus
    pub out_norm: RMSNormTrain,  // RMSNorm on output before gating
    pub n_heads: usize,
    pub head_dim: usize,
}

impl GatedDeltaNetTrain {
    pub fn new(
        dim: usize,
        n_heads: usize,
        group_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert!(n_heads > 0, "n_heads must be non-zero");
        assert!(
            dim % n_heads == 0,
            "dim ({}) must be divisible by n_heads ({})",
            dim,
            n_heads
        );
        let head_dim = dim / n_heads;

        let wq = BitLinearSTE::new(dim, n_heads * head_dim, group_size, vb.pp("wq"))?;
        let wk = BitLinearSTE::new(dim, n_heads * head_dim, group_size, vb.pp("wk"))?;
        let wv = BitLinearSTE::new(dim, n_heads * head_dim, group_size, vb.pp("wv"))?;
        let wo = BitLinearSTE::new(n_heads * head_dim, dim, group_size, vb.pp("wo"))?;
        let a_proj = BitLinearSTE::new(dim, n_heads, group_size, vb.pp("a_proj"))?;
        let b_proj = BitLinearSTE::new(dim, n_heads, group_size, vb.pp("b_proj"))?;
        let g_proj = BitLinearSTE::new(dim, n_heads * head_dim, group_size, vb.pp("g_proj"))?;

        // Decay at init must be near 1.0 (remember by default, learn to forget)
        // — the Mamba2/GDN convention. Per-token decay is
        //     exp(-exp(a_log) * softplus(a_proj(x) + dt_bias))
        // and a_proj(x) is ~0 at init, so dt_bias sets the operating point:
        //     exp(A_LOG_INIT) * softplus(DT_BIAS_INIT)
        //       = 1.6487 * softplus(-5.0) = 1.6487 * 0.006715 = 0.01107
        //     decay = exp(-0.01107) = 0.989   (state half-life ~63 tokens)
        // dt_bias = 0.0 would give softplus(0) = ln2 and decay = 0.32, i.e. 68%
        // of the state discarded per token — under two tokens of memory, which
        // starves the recurrent layers of any long-range capacity at init.
        let a_log = vb.get_with_hints(n_heads, "a_log", candle_nn::Init::Const(A_LOG_INIT))?;
        let dt_bias = vb.get_with_hints(n_heads, "dt_bias", candle_nn::Init::Const(DT_BIAS_INIT))?;

        let out_norm = RMSNormTrain::new(head_dim, vb.pp("out_norm"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            a_proj,
            b_proj,
            g_proj,
            a_log,
            dt_bias,
            out_norm,
            n_heads,
            head_dim,
        })
    }

    /// Forward pass: x [batch, seq_len, dim] -> [batch, seq_len, dim]
    ///
    /// Recurrent through sequence positions. Each position updates the
    /// per-head state matrix S: [head_dim, head_dim].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;

        // Project Q, K, V: [batch, seq, dim] -> [batch, seq, n_heads * head_dim]
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Reshape to [batch, seq, n_heads, head_dim]
        let q = q.reshape((batch, seq_len, n_heads, head_dim))?;
        let k = k.reshape((batch, seq_len, n_heads, head_dim))?;
        let v = v.reshape((batch, seq_len, n_heads, head_dim))?;

        // L2-normalize Q and K per head
        let q = l2_normalize_last_dim(&q)?;
        let k = l2_normalize_last_dim(&k)?;

        // Alpha gate: g = -exp(a_log) * softplus(a_proj(x) + dt_bias)
        // a_proj(x): [batch, seq, n_heads]
        let a_raw = self.a_proj.forward(x)?; // [batch, seq, n_heads]
        let a_log_exp = self.a_log.exp()?; // [n_heads]
        let dt_bias = self.dt_bias.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, n_heads]
        let a_input = a_raw.broadcast_add(&dt_bias)?; // [batch, seq, n_heads]
        let softplus_a = softplus(&a_input)?; // [batch, seq, n_heads]
        let a_log_bcast = a_log_exp.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, n_heads]
        let g = softplus_a.broadcast_mul(&a_log_bcast)?.neg()?; // [batch, seq, n_heads], always negative
        let decay = g.exp()?; // [batch, seq, n_heads], in (0, 1)

        // Beta gate: beta = sigmoid(b_proj(x))
        // Manual sigmoid: 1 / (1 + exp(-x)) — candle_nn::ops::sigmoid has no CUDA kernel
        let beta_raw = self.b_proj.forward(x)?; // [batch, seq, n_heads]
        let beta = beta_raw.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;

        // Output gate: gate = g_proj(x)
        let gate = self.g_proj.forward(x)?; // [batch, seq, dim]
        let gate = silu(&gate)?;

        // Recurrent loop over sequence positions
        // State S: [batch, n_heads, head_dim, head_dim] — initialized to zero
        let device = x.device();
        let mut s = Tensor::zeros(
            (batch, n_heads, head_dim, head_dim),
            candle_core::DType::F32,
            device,
        )?;

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extract per-position tensors: [batch, n_heads, head_dim] or [batch, n_heads]
            // contiguous() needed because narrow+squeeze produces non-contiguous views
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [batch, n_heads, head_dim]
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [batch, n_heads, head_dim]
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [batch, n_heads, head_dim]
            let decay_t = decay.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [batch, n_heads]
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [batch, n_heads]

            let (s_next, o_t) = delta_rule_step(&s, &q_t, &k_t, &v_t, &beta_t, &decay_t)?;
            s = s_next;
            outputs.push(o_t);
        }

        // Stack outputs: [batch, seq_len, n_heads, head_dim]
        let output = Tensor::stack(&outputs, 1)?; // [batch, seq, n_heads, head_dim]

        // Apply RMSNorm per head (on head_dim dimension)
        let output = self.out_norm.forward(&output)?; // [batch, seq, n_heads, head_dim]

        // Apply output gating: output * SiLU(gate)
        // gate: [batch, seq, dim] -> [batch, seq, n_heads, head_dim]
        let gate = gate.reshape((batch, seq_len, n_heads, head_dim))?;
        let output = (output * gate)?;

        // Reshape to [batch, seq, dim] and project
        let output = output.reshape((batch, seq_len, dim))?;
        self.wo.forward(&output)
    }

    /// Collect linear weight parameters (for Muon optimizer).
    pub fn linear_params(&self) -> Vec<&Tensor> {
        vec![
            self.wq.weight(),
            self.wk.weight(),
            self.wv.weight(),
            self.wo.weight(),
            self.a_proj.weight(),
            self.b_proj.weight(),
            self.g_proj.weight(),
        ]
    }

    /// Collect scalar/1D parameters (for Lion optimizer).
    pub fn scalar_params(&self) -> Vec<&Tensor> {
        vec![&self.a_log, &self.dt_bias]
    }

    /// Collect norm parameters (for Lion optimizer).
    pub fn norm_params(&self) -> Vec<&Tensor> {
        vec![self.out_norm.weight()]
    }
}

/// One gated delta-rule step for every (batch, head) pair.
///
/// State is value-major: `S = Σ_τ err_τ @ k_τ^T`, shape `[batch, n_heads, head_dim, head_dim]`.
/// `q_t`, `k_t`, `v_t` are `[batch, n_heads, head_dim]`; `beta_t`, `decay_t` are `[batch, n_heads]`.
/// Returns `(S_t, o_t)`.
///
/// Split out of `forward` so the delta-rule invariants can be tested with
/// controlled beta/decay instead of whatever the random projections emit.
fn delta_rule_step(
    s: &Tensor,
    q_t: &Tensor,
    k_t: &Tensor,
    v_t: &Tensor,
    beta_t: &Tensor,
    decay_t: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Decay state: S *= decay_t
    // decay_t: [batch, n_heads] -> [batch, n_heads, 1, 1] for broadcasting
    let decay_t_4d = decay_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
    let s = s.broadcast_mul(&decay_t_4d)?;

    // Prediction error: error = v_t - S @ k_t
    // Value-major S makes this the value currently stored under k_t (K is
    // L2-normalized), which is what the delta rule must subtract.
    let k_t_col = k_t.unsqueeze(D::Minus1)?; // [batch, n_heads, head_dim, 1]
    let sk = s.matmul(&k_t_col)?.squeeze(D::Minus1)?; // [batch, n_heads, head_dim]
    let error = (v_t - &sk)?; // [batch, n_heads, head_dim]

    // Update: S += outer(beta_t * error, k_t)
    let beta_t_3d = beta_t.unsqueeze(D::Minus1)?; // [batch, n_heads, 1]
    let scaled_error = error.broadcast_mul(&beta_t_3d)?; // [batch, n_heads, head_dim]
    let scaled_error_col = scaled_error.unsqueeze(D::Minus1)?; // [batch, n_heads, head_dim, 1]
    let k_t_row = k_t.unsqueeze(2)?; // [batch, n_heads, 1, head_dim]
    let update = scaled_error_col.matmul(&k_t_row)?; // [batch, n_heads, head_dim, head_dim]
    let s = (&s + &update)?;

    // Read: o_t = S @ q_t
    let q_t_col = q_t.unsqueeze(D::Minus1)?; // [batch, n_heads, head_dim, 1]
    let o_t = s.matmul(&q_t_col)?.squeeze(D::Minus1)?; // [batch, n_heads, head_dim]

    Ok((s, o_t))
}

/// L2-normalize along the last dimension.
fn l2_normalize_last_dim(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let norm = (norm + 1e-8f64)?; // epsilon for stability
    x.broadcast_div(&norm)
}

/// Softplus activation: log(1 + exp(x)), numerically stable.
///
/// Uses `max(x, 0) + log(1 + exp(-|x|))`. The naive `log(1 + exp(x))` overflows
/// to `inf` for x >~ 88; the forward pass survives (decay -> 0) but the backward
/// pass then evaluates `0 * inf` and yields NaN gradients. `a_proj` sums `dim`
/// ternary terms, so large inputs are reachable during an instability. The
/// inference twin in `nanochat-model/src/deltanet.rs` guards the same way.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let log1p_exp = (x.abs()?.neg()?.exp()? + 1.0f64)?.log()?;
    relu.add(&log1p_exp)
}

/// SiLU activation: x * sigmoid(x)
/// Uses candle_nn::ops::silu which has a fused CUDA kernel (unlike standalone sigmoid).
fn silu(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_gated_deltanet_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let dn = GatedDeltaNetTrain::new(64, 4, 64, vb.pp("test"))?;

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device)?;
        let y = dn.forward(&x)?;
        assert_eq!(y.dims(), &[2, 8, 64]);
        Ok(())
    }

    #[test]
    fn test_gated_deltanet_gradient_flows() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let dn = GatedDeltaNetTrain::new(64, 4, 64, vb.pp("test"))?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 64), &device)?;
        let y = dn.forward(&x)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        // Check gradients flow to key parameters
        let wq_grad = grads.get(dn.wq.weight()).expect("wq should have gradient");
        let gn = wq_grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(gn > 0.0, "wq gradient should be non-zero, got {}", gn);

        let a_log_grad = grads.get(&dn.a_log).expect("a_log should have gradient");
        let gn = a_log_grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(gn > 0.0, "a_log gradient should be non-zero, got {}", gn);

        Ok(())
    }

    #[test]
    fn test_gated_deltanet_param_counts() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let dn = GatedDeltaNetTrain::new(64, 4, 64, vb.pp("test"))?;

        assert_eq!(dn.linear_params().len(), 7); // wq, wk, wv, wo, a_proj, b_proj, g_proj
        assert_eq!(dn.scalar_params().len(), 2); // a_log, dt_bias
        assert_eq!(dn.norm_params().len(), 1); // out_norm
        Ok(())
    }

    #[test]
    fn test_l2_normalize() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[3.0f32, 4.0], &device)?.unsqueeze(0)?;
        let y = l2_normalize_last_dim(&x)?;
        let vals = y.flatten_all()?.to_vec1::<f32>()?;
        // 3/5 = 0.6, 4/5 = 0.8
        assert!((vals[0] - 0.6).abs() < 1e-4);
        assert!((vals[1] - 0.8).abs() < 1e-4);
        Ok(())
    }

    /// Build the [1, 1, D] per-step tensors `delta_rule_step` expects.
    fn step_inputs(
        vals: &[f32],
        device: &Device,
    ) -> Result<Tensor> {
        Tensor::new(vals, device)?.reshape((1, 1, vals.len()))
    }

    /// The delta rule must OVERWRITE a repeated key, not accumulate onto it.
    ///
    /// This is the property that separates DeltaNet from plain linear
    /// attention. Writing (k, v1) then (k, v2) with beta=1 and decay=1 must
    /// read back v2; an implementation that predicts with `S^T @ k` instead of
    /// `S @ k` under this value-major state layout reads back v1 + v2.
    #[test]
    fn test_delta_rule_overwrites_repeated_key() -> Result<()> {
        let device = Device::Cpu;
        let d = 4;

        // Unit key (K is L2-normalized in forward), two distinct values.
        let k = step_inputs(&[1.0, 0.0, 0.0, 0.0], &device)?;
        let v1 = step_inputs(&[0.0, 1.0, 0.0, 0.0], &device)?;
        let v2 = step_inputs(&[0.0, 0.0, 1.0, 0.0], &device)?;
        let beta = Tensor::new(&[1.0f32], &device)?.reshape((1, 1))?; // full update
        let decay = Tensor::new(&[1.0f32], &device)?.reshape((1, 1))?; // no forgetting

        let mut s = Tensor::zeros((1, 1, d, d), DType::F32, &device)?;
        for v in [&v1, &v2] {
            let (s_next, _) = delta_rule_step(&s, &k, &k, v, &beta, &decay)?;
            s = s_next;
        }

        // Read with q = k: must be v2, not v1 + v2.
        let (_, o) = delta_rule_step(&s, &k, &k, &v2, &beta, &decay)?;
        let got = o.flatten_all()?.to_vec1::<f32>()?;
        let want = v2.flatten_all()?.to_vec1::<f32>()?;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "component {i}: repeated key was not overwritten — got {got:?}, want {want:?}"
            );
        }
        Ok(())
    }

    /// Writing (k, v) with beta=1 then reading with q=k must return v exactly.
    #[test]
    fn test_delta_rule_recalls_written_value() -> Result<()> {
        let device = Device::Cpu;
        let k = step_inputs(&[0.0, 1.0, 0.0, 0.0], &device)?;
        let v = step_inputs(&[0.5, -1.5, 2.0, 0.25], &device)?;
        let beta = Tensor::new(&[1.0f32], &device)?.reshape((1, 1))?;
        let decay = Tensor::new(&[1.0f32], &device)?.reshape((1, 1))?;

        let s = Tensor::zeros((1, 1, 4, 4), DType::F32, &device)?;
        let (s, _) = delta_rule_step(&s, &k, &k, &v, &beta, &decay)?;
        let (_, o) = delta_rule_step(&s, &k, &k, &v, &beta, &decay)?;

        let got = o.flatten_all()?.to_vec1::<f32>()?;
        let want = v.flatten_all()?.to_vec1::<f32>()?;
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "recall failed — got {got:?}, want {want:?}");
        }
        Ok(())
    }

    /// An orthogonal key must not disturb a value stored under another key.
    #[test]
    fn test_delta_rule_orthogonal_keys_independent() -> Result<()> {
        let device = Device::Cpu;
        let k1 = step_inputs(&[1.0, 0.0, 0.0, 0.0], &device)?;
        let k2 = step_inputs(&[0.0, 1.0, 0.0, 0.0], &device)?;
        let v1 = step_inputs(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let v2 = step_inputs(&[-1.0, 0.0, 1.0, 0.0], &device)?;
        let beta = Tensor::new(&[1.0f32], &device)?.reshape((1, 1))?;
        let decay = Tensor::new(&[1.0f32], &device)?.reshape((1, 1))?;

        let s = Tensor::zeros((1, 1, 4, 4), DType::F32, &device)?;
        let (s, _) = delta_rule_step(&s, &k1, &k1, &v1, &beta, &decay)?;
        let (s, _) = delta_rule_step(&s, &k2, &k2, &v2, &beta, &decay)?;

        // Read k1 back — v2's write used an orthogonal key, so v1 must survive.
        let (_, o) = delta_rule_step(&s, &k1, &k1, &v1, &beta, &decay)?;
        let got = o.flatten_all()?.to_vec1::<f32>()?;
        let want = v1.flatten_all()?.to_vec1::<f32>()?;
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "interference — got {got:?}, want {want:?}");
        }
        Ok(())
    }

    /// decay=0 must clear the state entirely.
    #[test]
    fn test_delta_rule_decay_forgets() -> Result<()> {
        let device = Device::Cpu;
        let k = step_inputs(&[1.0, 0.0, 0.0, 0.0], &device)?;
        let v = step_inputs(&[0.0, 5.0, 0.0, 0.0], &device)?;
        let zero_v = step_inputs(&[0.0, 0.0, 0.0, 0.0], &device)?;
        let beta_on = Tensor::new(&[1.0f32], &device)?.reshape((1, 1))?;
        let beta_off = Tensor::new(&[0.0f32], &device)?.reshape((1, 1))?;
        let keep = Tensor::new(&[1.0f32], &device)?.reshape((1, 1))?;
        let forget = Tensor::new(&[0.0f32], &device)?.reshape((1, 1))?;

        let s = Tensor::zeros((1, 1, 4, 4), DType::F32, &device)?;
        let (s, _) = delta_rule_step(&s, &k, &k, &v, &beta_on, &keep)?;
        // decay=0, beta=0: state wiped, nothing written.
        let (_, o) = delta_rule_step(&s, &k, &k, &zero_v, &beta_off, &forget)?;
        let got = o.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            got.iter().all(|g| g.abs() < 1e-6),
            "decay=0 should clear state, got {got:?}"
        );
        Ok(())
    }

    /// Init must remember by default: decay ≈ 0.989, not ≈ 0.32.
    #[test]
    fn test_init_decay_is_near_one() -> Result<()> {
        let device = Device::Cpu;
        let a_log = Tensor::new(&[A_LOG_INIT as f32], &device)?;
        let dt_bias = Tensor::new(&[DT_BIAS_INIT as f32], &device)?;
        let decay = softplus(&dt_bias)?
            .broadcast_mul(&a_log.exp()?)?
            .neg()?
            .exp()?
            .to_vec1::<f32>()?[0];
        assert!(
            decay > 0.95,
            "init decay {decay} is too aggressive — recurrent layers start with \
             almost no memory (dt_bias=0.0 would give ~0.32)"
        );
        assert!(decay < 1.0, "decay must be strictly below 1.0, got {decay}");
        Ok(())
    }

    /// Softplus must not overflow to inf for large inputs (NaN gradients).
    #[test]
    fn test_softplus_large_input_is_finite() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[100.0f32, 500.0, -100.0], &device)?;
        let y = softplus(&x)?.to_vec1::<f32>()?;
        assert!(y.iter().all(|v| v.is_finite()), "softplus overflowed: {y:?}");
        // softplus(x) ≈ x for large positive x
        assert!((y[0] - 100.0).abs() < 1e-3, "got {}", y[0]);
        assert!((y[1] - 500.0).abs() < 1e-2, "got {}", y[1]);
        // softplus(x) ≈ 0 for large negative x
        assert!(y[2].abs() < 1e-6, "got {}", y[2]);
        Ok(())
    }

    /// Gradients must stay finite through a large-magnitude alpha input.
    #[test]
    fn test_softplus_large_input_gradient_is_finite() -> Result<()> {
        let device = Device::Cpu;
        let x = candle_core::Var::from_tensor(&Tensor::new(&[120.0f32], &device)?)?;
        let loss = softplus(x.as_tensor())?.sum_all()?;
        let grads = loss.backward()?;
        let g = grads
            .get(&x)
            .expect("softplus input should have gradient")
            .to_vec1::<f32>()?;
        assert!(g[0].is_finite(), "softplus gradient is not finite: {:?}", g);
        Ok(())
    }

    /// CPU/CUDA agreement for the whole module, forward and backward.
    ///
    /// Guards two things that only break on GPU:
    ///  - op coverage. `softplus` uses relu/abs/neg/exp/log; candle does not
    ///    have a CUDA kernel for every op (this module already avoids
    ///    `candle_nn::ops::sigmoid` for exactly that reason), and a missing
    ///    kernel surfaces only when a tensor is on the device.
    ///  - numerical agreement, so a GPU training run and a CPU export/eval
    ///    path cannot silently diverge.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gated_deltanet_cpu_cuda_agreement() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no CUDA device available ({e})");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        let (dim, heads, batch, seq) = (64, 4, 2, 8);

        // Same weights on both devices: build on CPU, then copy the varmap data.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &cpu);
        let dn_cpu = GatedDeltaNetTrain::new(dim, heads, 64, vb.pp("t"))?;

        let varmap_gpu = VarMap::new();
        let vb_gpu = VarBuilder::from_varmap(&varmap_gpu, DType::F32, &cuda);
        let dn_gpu = GatedDeltaNetTrain::new(dim, heads, 64, vb_gpu.pp("t"))?;
        {
            // Mirror every CPU tensor onto the GPU model so outputs are comparable.
            let src = varmap.data().lock().unwrap();
            let dst = varmap_gpu.data().lock().unwrap();
            for (name, var) in src.iter() {
                let target = dst
                    .get(name)
                    .unwrap_or_else(|| panic!("gpu varmap missing {name}"));
                target.set(&var.as_tensor().to_device(&cuda)?)?;
            }
        }

        let x_cpu = Tensor::randn(0.0f32, 1.0, (batch, seq, dim), &cpu)?;
        let x_gpu = x_cpu.to_device(&cuda)?;

        // Forward must run on the device at all, then agree with CPU.
        let y_cpu = dn_cpu.forward(&x_cpu)?;
        let y_gpu = dn_gpu.forward(&x_gpu)?;
        assert_eq!(y_gpu.dims(), &[batch, seq, dim]);

        let a = y_cpu.flatten_all()?.to_vec1::<f32>()?;
        let b = y_gpu.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(
            b.iter().all(|v| v.is_finite()),
            "CUDA forward produced non-finite values"
        );
        assert!(
            max_diff < 1e-3,
            "CPU/CUDA forward disagree: max_diff={max_diff}"
        );

        // Backward must also run on the device (this is where a missing kernel
        // for abs/relu in the softplus path would surface).
        let loss = y_gpu.sum_all()?;
        let grads = loss.backward()?;
        for (label, t) in [
            ("wq", dn_gpu.wq.weight()),
            ("a_log", &dn_gpu.a_log),
            ("dt_bias", &dn_gpu.dt_bias),
        ] {
            let g = grads
                .get(t)
                .unwrap_or_else(|| panic!("{label} has no gradient on CUDA"));
            let norm = g.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            assert!(norm.is_finite(), "{label} CUDA gradient is not finite");
        }
        println!("CPU/CUDA agreement: max_diff={max_diff:.2e}");
        Ok(())
    }

    /// softplus must survive large inputs on the GPU too — the overflow path
    /// is where NaN gradients would appear during a real training instability.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_softplus_cuda_large_input() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no CUDA device available ({e})");
                return Ok(());
            }
        };
        let x = candle_core::Var::from_tensor(&Tensor::new(&[120.0f32, 0.0, -60.0], &cuda)?)?;
        let y = softplus(x.as_tensor())?;
        let vals = y.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "CUDA softplus overflowed: {vals:?}"
        );
        assert!((vals[0] - 120.0).abs() < 1e-2, "got {}", vals[0]);
        assert!((vals[1] - std::f32::consts::LN_2).abs() < 1e-4, "got {}", vals[1]);

        let grads = y.sum_all()?.backward()?;
        let g = grads.get(&x).expect("no gradient").to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        assert!(
            g.iter().all(|v| v.is_finite()),
            "CUDA softplus gradient not finite: {g:?}"
        );
        Ok(())
    }

    #[test]
    fn test_softplus() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &device)?;
        let y = softplus(&x)?;
        let vals = y.to_vec1::<f32>()?;
        // softplus(0) = ln(2) ≈ 0.6931
        assert!((vals[0] - 0.6931).abs() < 1e-3);
        // softplus(1) = ln(1+e) ≈ 1.3133
        assert!((vals[1] - 1.3133).abs() < 1e-3);
        Ok(())
    }
}
