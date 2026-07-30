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

/// Chunk length for the chunkwise-parallel (WY) recurrence.
///
/// The sequential form issues O(seq_len) `[head_dim, head_dim]` state matmuls;
/// the chunked form issues O(seq_len / CHUNK) of them and replaces the rest with
/// `[CHUNK, CHUNK]` and `[CHUNK, head_dim]` matmuls. Larger is fewer state
/// updates but more intra-chunk work, which grows as CHUNK^2.
pub const DEFAULT_CHUNK_SIZE: usize = 64;

/// Sub-block size for the two-level triangular solve.
///
/// Diagonal blocks this size are inverted by repeated squaring of the Neumann
/// series, which is exact here but diverges for large blocks. Measured f32
/// residual `||(I+N)T - I||max` against plain forward substitution, chunk 64:
///
/// ```text
///   regime                    cond(I+N)   fwdsub    B=8      B=16
///   realistic |k.k| ~ 0.09      4.3e0     8.9e-8   8.2e-8   7.5e-8
///   moderate  |n|   ~ 0.3       1.5e2     9.5e-7   9.5e-7   1.9e-6
///   identical keys, beta = 1    8.2e1     0.0      0.0      0.0
/// ```
///
/// B=8 matches forward substitution everywhere reachable; B=16 starts to drift.
/// (A synthetic `|n| ~ 1` matrix has cond 3.7e8 and defeats *every* f32 method
/// including forward substitution, but it is not reachable: `n` entries are
/// `beta * decay * (k_s . k_t)` with L2-normalized keys.)
const SOLVE_BLOCK: usize = 8;

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
    /// Uses the chunkwise-parallel (WY) recurrence with [`DEFAULT_CHUNK_SIZE`].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_chunk(x, DEFAULT_CHUNK_SIZE)
    }

    /// Forward pass with an explicit chunk length.
    ///
    /// `chunk_size <= 1` selects the sequential reference path (one
    /// `delta_rule_step` per timestep). Any larger value uses the chunkwise
    /// form, which is mathematically equivalent — see
    /// `test_chunked_matches_sequential`.
    pub fn forward_with_chunk(&self, x: &Tensor, chunk_size: usize) -> Result<Tensor> {
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

        // Recurrence: [batch, seq, n_heads, head_dim] out, either path.
        let output = if chunk_size <= 1 {
            sequential_recurrence(&q, &k, &v, &beta, &decay)?
        } else {
            // The chunked form batches over (batch, n_heads), so move heads next
            // to batch: [b, seq, h, d] -> [b, h, seq, d].
            let to_bh = |t: &Tensor| t.transpose(1, 2)?.contiguous();
            let out_bh = chunked_recurrence(
                &to_bh(&q)?,
                &to_bh(&k)?,
                &to_bh(&v)?,
                &beta.transpose(1, 2)?.contiguous()?,
                &g.transpose(1, 2)?.contiguous()?,
                chunk_size,
            )?;
            out_bh.transpose(1, 2)?.contiguous()?
        };

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

/// Sequential reference recurrence: one `delta_rule_step` per timestep.
///
/// Inputs are `[batch, seq, n_heads, head_dim]` (`beta`/`decay` are
/// `[batch, seq, n_heads]`); output is `[batch, seq, n_heads, head_dim]`.
/// Kept as the definition the chunked form is checked against.
fn sequential_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    decay: &Tensor,
) -> Result<Tensor> {
    let (batch, seq_len, n_heads, head_dim) = q.dims4()?;
    let mut s = Tensor::zeros(
        (batch, n_heads, head_dim, head_dim),
        q.dtype(),
        q.device(),
    )?;
    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        // contiguous() needed because narrow+squeeze produces non-contiguous views
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
        let decay_t = decay.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
        let (s_next, o_t) = delta_rule_step(&s, &q_t, &k_t, &v_t, &beta_t, &decay_t)?;
        s = s_next;
        outputs.push(o_t);
    }
    Tensor::stack(&outputs, 1)
}

/// Chunkwise-parallel Gated DeltaNet recurrence via the WY representation.
///
/// Mathematically identical to [`sequential_recurrence`], but the expensive
/// `[head_dim, head_dim]` state update runs once per *chunk* instead of once per
/// timestep. Inputs are `[batch, n_heads, seq, head_dim]`; `beta` and `g` are
/// `[batch, n_heads, seq]`, where `g = log(decay) <= 0`.
///
/// # Derivation
///
/// With state stored value-major as in [`delta_rule_step`], one step is
///
/// ```text
/// S_t = a_t S_{t-1} + u_t k_t^T,   u_t = b_t (v_t - a_t S_{t-1} k_t)
/// o_t = S_t q_t
/// ```
///
/// Let `Gcum_t = sum_{s<=t} g_s` inside the chunk, so `Gamma_t = exp(Gcum_t)` is
/// the cumulative decay and `Gamma_t / Gamma_s = exp(Gcum_t - Gcum_s) <= 1` for
/// `t >= s`. Unrolling `S_{t-1}` and substituting gives a linear system in the
/// per-step updates `u_t` that is *triangular*, because `u_t` depends only on
/// `u_s` for `s < t`:
///
/// ```text
/// u_t = b_t [ v_t - Gamma_t S_0 k_t - sum_{s<t} exp(Gcum_t - Gcum_s)(k_s . k_t) u_s ]
/// ```
///
/// Writing `L[t,s] = exp(Gcum_t - Gcum_s)(k_s . k_t)` (strictly lower) and
/// `N = diag(b) L`:
///
/// ```text
/// (I + N) U = B (V - D K S_0^T)        =>  U = (I + N)^-1 B (V - D K S_0^T)
/// O        = D (Q S_0^T) + M U,            M[t,s] = exp(Gcum_t-Gcum_s)(k_s . q_t), s <= t
/// S_C      = Gamma_C S_0 + (R U)^T K,      R = diag(exp(Gcum_C - Gcum_t))
/// ```
///
/// Every exponent used is `<= 0`, so no term can overflow; the anti-causal half
/// of the difference matrix is clamped to `0` before `exp` and then masked away,
/// which keeps it finite instead of `inf * 0 = NaN`.
fn chunked_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    chunk_size: usize,
) -> Result<Tensor> {
    let (batch, n_heads, seq_len, head_dim) = q.dims4()?;
    let device = q.device();
    let dtype = q.dtype();

    let mut s = Tensor::zeros((batch, n_heads, head_dim, head_dim), dtype, device)?;
    let mut outputs = Vec::with_capacity(seq_len.div_ceil(chunk_size));
    let mut start = 0usize;

    while start < seq_len {
        let c = (seq_len - start).min(chunk_size);

        let qc = q.narrow(2, start, c)?.contiguous()?; // [b, h, c, d]
        let kc = k.narrow(2, start, c)?.contiguous()?;
        let vc = v.narrow(2, start, c)?.contiguous()?;
        let bc = beta.narrow(2, start, c)?.unsqueeze(D::Minus1)?.contiguous()?; // [b, h, c, 1]
        let gc = g.narrow(2, start, c)?.unsqueeze(D::Minus1)?.contiguous()?; // [b, h, c, 1]

        // Cumulative log-decay within the chunk, as a matmul with a lower
        // triangular ones matrix (differentiable, no cumsum dependency).
        let incl = tri_mask(c, false, dtype, device)?.reshape((1, 1, c, c))?;
        let strict = tri_mask(c, true, dtype, device)?.reshape((1, 1, c, c))?;
        let gcum = incl.broadcast_matmul(&gc)?; // [b, h, c, 1]
        let gamma = gcum.exp()?; // Gamma_t in (0, 1]

        // dr[t,s] = exp(Gcum_t - Gcum_s). Clamped at 0 so the anti-causal half
        // (where the exponent is positive) stays finite before masking.
        let dr = gcum
            .broadcast_sub(&gcum.transpose(2, 3)?)?
            .minimum(0.0)?
            .exp()?; // [b, h, c, c]

        // N = diag(beta) * (dr .* K K^T) restricted to s < t.
        let kkt = kc.matmul(&kc.transpose(2, 3)?)?;
        let n = (&dr * &kkt)?
            .broadcast_mul(&strict)?
            .broadcast_mul(&bc)?;

        // Solve (I + N) U = B (V - Gamma K S^T) for U.
        let ks = kc.matmul(&s.transpose(2, 3)?)?; // [b, h, c, d]
        let rhs = (&vc - &ks.broadcast_mul(&gamma)?)?.broadcast_mul(&bc)?;
        let u = solve_unit_lower(&n, &rhs, c)?;

        // O = Gamma (Q S^T) + M U, with M causal-inclusive.
        let qkt = qc.matmul(&kc.transpose(2, 3)?)?;
        let m = (&dr * &qkt)?.broadcast_mul(&incl)?;
        let qs = qc.matmul(&s.transpose(2, 3)?)?;
        outputs.push((qs.broadcast_mul(&gamma)? + m.matmul(&u)?)?);

        // S <- Gamma_C S + sum_t exp(Gcum_C - Gcum_t) u_t k_t^T
        let g_last = gcum.narrow(2, c - 1, 1)?; // [b, h, 1, 1]
        let r = g_last.broadcast_sub(&gcum)?.minimum(0.0)?.exp()?; // [b, h, c, 1]
        let ru = u.broadcast_mul(&r)?;
        s = (s.broadcast_mul(&g_last.exp()?)? + ru.transpose(2, 3)?.matmul(&kc)?)?;

        start += c;
    }

    Tensor::cat(&outputs, 2)
}

/// Solve `(I + n) X = rhs` for `X`, with `n` strictly lower triangular over its
/// last two dims. `n` is `[b, h, c, c]`, `rhs` is `[b, h, c, d]`.
///
/// Two-level: the `c` diagonal blocks of size [`SOLVE_BLOCK`] are inverted
/// *together* by repeated squaring (exact at that size), then the solution is
/// carried across blocks by forward substitution. This turns `c` sequential
/// row-steps into `c / SOLVE_BLOCK` block-steps, which matters because each step
/// is a separate kernel launch and the per-row work is far too small to occupy a
/// GPU — the launch overhead, not the arithmetic, was the bottleneck.
///
/// Falls back to per-row substitution when `c` is not a whole number of blocks,
/// which only happens for a ragged final chunk.
fn solve_unit_lower(n: &Tensor, rhs: &Tensor, c: usize) -> Result<Tensor> {
    if c < 2 * SOLVE_BLOCK || !c.is_multiple_of(SOLVE_BLOCK) {
        return solve_unit_lower_rowwise(n, rhs, c);
    }
    let (batch, heads, _, _) = n.dims4()?;
    let blk = SOLVE_BLOCK;
    let n_blocks = c / blk;

    // Invert every diagonal block in one batched pass. Candle's matmul does not
    // batch over more than one leading dim, so the block index is folded into a
    // flat batch rather than kept as a separate axis.
    let mut diag = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        diag.push(n.narrow(2, i * blk, blk)?.narrow(3, i * blk, blk)?.contiguous()?);
    }
    let stacked = Tensor::stack(&diag, 2)?; // [b, h, n_blocks, blk, blk]
    let inverted = neumann_inverse(
        &stacked.reshape((batch * heads * n_blocks, blk, blk))?,
        blk,
    )?
    .reshape((batch, heads, n_blocks, blk, blk))?;

    // Block forward substitution:
    //   X_i = Dinv_i (rhs_i - n[i, :i*blk] X_{:i*blk})
    let mut acc: Option<Tensor> = None;
    for i in 0..n_blocks {
        let d_inv = inverted.narrow(2, i, 1)?.squeeze(2)?.contiguous()?; // [b, h, blk, blk]
        let rhs_i = rhs.narrow(2, i * blk, blk)?.contiguous()?; // [b, h, blk, d]
        let corrected = match &acc {
            None => rhs_i,
            Some(prev) => {
                let coeff = n
                    .narrow(2, i * blk, blk)?
                    .narrow(3, 0, i * blk)?
                    .contiguous()?; // [b, h, blk, i*blk]
                (rhs_i - coeff.matmul(prev)?)?
            }
        };
        let x_i = d_inv.matmul(&corrected)?;
        acc = Some(match acc {
            None => x_i,
            Some(prev) => Tensor::cat(&[&prev, &x_i], 2)?,
        });
    }
    acc.ok_or_else(|| candle_core::Error::Msg("empty chunk".into()))
}

/// Per-row forward substitution: `X_t = rhs_t - n[t, :t] X_{:t}`.
///
/// Exact and shape-agnostic, but one kernel launch per row. Used for ragged
/// chunks and as the reference the blocked path is tested against.
fn solve_unit_lower_rowwise(n: &Tensor, rhs: &Tensor, c: usize) -> Result<Tensor> {
    let mut acc: Option<Tensor> = None;
    for t in 0..c {
        let rhs_t = rhs.narrow(2, t, 1)?.contiguous()?; // [b, h, 1, d]
        let row = match &acc {
            None => rhs_t,
            Some(prev) => {
                let coeff = n.narrow(2, t, 1)?.narrow(3, 0, t)?.contiguous()?; // [b, h, 1, t]
                (rhs_t - coeff.matmul(prev)?)?
            }
        };
        acc = Some(match acc {
            None => row,
            Some(prev) => Tensor::cat(&[&prev, &row], 2)?,
        });
    }
    acc.ok_or_else(|| candle_core::Error::Msg("empty chunk".into()))
}

/// `(I + n)^-1` for strictly lower triangular blocks over the last two dims,
/// by repeated squaring of the Neumann series:
/// `Y_{j+1} = (I + P_j) Y_j`, `P_{j+1} = P_j^2`, so `Y_j = sum_{i<2^j} (-n)^i`.
///
/// Exact once the reach passes the block size, since `n^blk = 0`. Only valid for
/// *small* blocks — the partial sums grow combinatorially, which is why this is
/// applied per [`SOLVE_BLOCK`] rather than to the whole chunk.
/// Takes `[flat_batch, blk, blk]`.
fn neumann_inverse(n: &Tensor, blk: usize) -> Result<Tensor> {
    let eye = identity(blk, n.dtype(), n.device())?.reshape((1, blk, blk))?;
    let mut y = eye.broadcast_sub(n)?; // I - n
    let mut p = n.matmul(n)?; // (-n)^2
    let mut reach = 2usize;
    while reach < blk {
        y = (&y + p.matmul(&y)?)?;
        p = p.matmul(&p)?;
        reach *= 2;
    }
    Ok(y)
}

/// `[c, c]` identity.
fn identity(c: usize, dtype: candle_core::DType, device: &candle_core::Device) -> Result<Tensor> {
    let mut data = vec![0f32; c * c];
    for i in 0..c {
        data[i * c + i] = 1.0;
    }
    Tensor::from_vec(data, (c, c), device)?.to_dtype(dtype)
}

/// `[c, c]` causal mask: 1 where `s < t` (strict) or `s <= t` (inclusive).
fn tri_mask(c: usize, strict: bool, dtype: candle_core::DType, device: &candle_core::Device) -> Result<Tensor> {
    let mut data = vec![0f32; c * c];
    for t in 0..c {
        for s in 0..c {
            let keep = if strict { s < t } else { s <= t };
            if keep {
                data[t * c + s] = 1.0;
            }
        }
    }
    Tensor::from_vec(data, (c, c), device)?.to_dtype(dtype)
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

    /// Max abs difference between two tensors of the same shape.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        assert_eq!(a.dims(), b.dims(), "shape mismatch");
        let a = a.flatten_all()?.to_vec1::<f32>()?;
        let b = b.flatten_all()?.to_vec1::<f32>()?;
        Ok(a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max))
    }

    /// The chunked WY recurrence must match the sequential reference.
    ///
    /// This is the correctness contract for the whole chunked path: it is only a
    /// valid optimization if it computes the same function. Swept over chunk
    /// sizes including ones that do not divide the sequence length, so the
    /// ragged final chunk is covered too.
    #[test]
    fn test_chunked_matches_sequential() -> Result<()> {
        let device = Device::Cpu;
        for &(dim, heads, batch, seq) in &[(64, 4, 2, 32), (32, 2, 1, 17), (128, 8, 2, 40)] {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let dn = GatedDeltaNetTrain::new(dim, heads, 64, vb.pp("t"))?;
            let x = Tensor::randn(0.0f32, 1.0, (batch, seq, dim), &device)?;

            let reference = dn.forward_with_chunk(&x, 1)?;
            for &chunk in &[2usize, 3, 8, 16, 64, 128] {
                let got = dn.forward_with_chunk(&x, chunk)?;
                let diff = max_abs_diff(&reference, &got)?;
                assert!(
                    diff < 2e-3,
                    "chunk={chunk} dim={dim} seq={seq}: diverged from sequential by {diff}"
                );
            }
        }
        Ok(())
    }

    /// Equivalence must hold in the adversarial regime that breaks the
    /// Neumann-series shortcut: identical keys with beta driven to 1, where the
    /// intra-chunk triangular system is maximally coupled.
    #[test]
    fn test_chunked_matches_sequential_identical_keys() -> Result<()> {
        let device = Device::Cpu;
        let (batch, heads, seq, d) = (1, 2, 32, 8);

        // Every key the same direction; beta = 1; no decay.
        let mut kdata = vec![0f32; batch * heads * seq * d];
        for i in 0..(batch * heads * seq) {
            kdata[i * d] = 1.0; // unit vector along axis 0
        }
        let k = Tensor::from_vec(kdata, (batch, heads, seq, d), &device)?;
        let q = k.clone();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, d), &device)?;
        let beta = Tensor::ones((batch, heads, seq), DType::F32, &device)?;
        let g = Tensor::zeros((batch, heads, seq), DType::F32, &device)?; // decay = 1

        // Sequential reference wants [b, seq, h, d].
        let to_bshd = |t: &Tensor| t.transpose(1, 2)?.contiguous();
        let decay = g.exp()?;
        let reference = sequential_recurrence(
            &to_bshd(&q)?,
            &to_bshd(&k)?,
            &to_bshd(&v)?,
            &beta.transpose(1, 2)?.contiguous()?,
            &decay.transpose(1, 2)?.contiguous()?,
        )?;

        for &chunk in &[4usize, 8, 16, 32] {
            let got = chunked_recurrence(&q, &k, &v, &beta, &g, chunk)?
                .transpose(1, 2)?
                .contiguous()?;
            let diff = max_abs_diff(&reference, &got)?;
            assert!(
                diff < 2e-3,
                "identical-keys chunk={chunk}: diverged by {diff} (this is the case \
                 where a Neumann-series inverse fails)"
            );
        }
        Ok(())
    }

    /// Build a strictly lower triangular `n` with entries scaled by `scale`.
    fn strict_lower(c: usize, scale: f64, device: &Device) -> Result<Tensor> {
        let full = (Tensor::randn(0.0f32, 1.0, (1, 1, c, c), device)? * scale)?;
        let mask = tri_mask(c, true, DType::F32, device)?.reshape((1, 1, c, c))?;
        full.broadcast_mul(&mask)
    }

    /// The solve must actually satisfy `(I + n) X = rhs`.
    #[test]
    fn test_solve_unit_lower_residual() -> Result<()> {
        let device = Device::Cpu;
        for &c in &[8usize, 16, 24, 64] {
            // 0.09 is the realistic magnitude of beta * decay * (k_s . k_t) with
            // L2-normalized keys in a high-dimensional head.
            let n = strict_lower(c, 0.09, &device)?;
            let rhs = Tensor::randn(0.0f32, 1.0, (1, 1, c, 5), &device)?;
            let x = solve_unit_lower(&n, &rhs, c)?;

            let eye = identity(c, DType::F32, &device)?.reshape((1, 1, c, c))?;
            let residual = max_abs_diff(&(&n + &eye)?.matmul(&x)?, &rhs)?;
            assert!(residual < 1e-4, "c={c}: solve residual {residual}");
        }
        Ok(())
    }

    /// The blocked solve must agree with per-row substitution, which is the
    /// reference it replaces. Covers block-aligned and ragged `c`, and a range
    /// of coupling strengths.
    #[test]
    fn test_blocked_solve_matches_rowwise() -> Result<()> {
        let device = Device::Cpu;
        for &c in &[16usize, 24, 32, 64] {
            for &scale in &[0.05f64, 0.09, 0.3] {
                let n = strict_lower(c, scale, &device)?;
                let rhs = Tensor::randn(0.0f32, 1.0, (2, 3, c, 7), &device)?;
                let n = n.broadcast_as((2, 3, c, c))?.contiguous()?;

                let blocked = solve_unit_lower(&n, &rhs, c)?;
                let rowwise = solve_unit_lower_rowwise(&n, &rhs, c)?;
                let diff = max_abs_diff(&blocked, &rowwise)?;
                assert!(
                    diff < 1e-4,
                    "c={c} scale={scale}: blocked solve differs from rowwise by {diff}"
                );
            }
        }
        Ok(())
    }

    /// The adversarial case that defeats a whole-chunk Neumann series must still
    /// be handled: identical keys with beta = 1 makes every off-diagonal entry 1.
    #[test]
    fn test_blocked_solve_identical_keys() -> Result<()> {
        let device = Device::Cpu;
        let c = 32;
        let ones = Tensor::ones((1, 1, c, c), DType::F32, &device)?;
        let mask = tri_mask(c, true, DType::F32, &device)?.reshape((1, 1, c, c))?;
        let n = ones.broadcast_mul(&mask)?;
        let rhs = Tensor::randn(0.0f32, 1.0, (1, 1, c, 4), &device)?;

        let blocked = solve_unit_lower(&n, &rhs, c)?;
        let rowwise = solve_unit_lower_rowwise(&n, &rhs, c)?;
        assert!(
            max_abs_diff(&blocked, &rowwise)? < 1e-3,
            "blocked solve diverged on the all-ones (identical keys, beta=1) case"
        );

        let eye = identity(c, DType::F32, &device)?.reshape((1, 1, c, c))?;
        let residual = max_abs_diff(&(&n + &eye)?.matmul(&blocked)?, &rhs)?;
        assert!(residual < 1e-3, "identical-keys solve residual {residual}");
        Ok(())
    }

    /// `neumann_inverse` must be exact at the block size it is actually used at.
    #[test]
    fn test_neumann_inverse_exact_at_block_size() -> Result<()> {
        let device = Device::Cpu;
        let blk = SOLVE_BLOCK;
        let n = strict_lower(blk, 1.0, &device)?.reshape((1, blk, blk))?;
        let inv = neumann_inverse(&n, blk)?;
        let eye = identity(blk, DType::F32, &device)?.reshape((1, blk, blk))?;
        let residual = max_abs_diff(&(&n + &eye)?.matmul(&inv)?, &eye)?;
        assert!(
            residual < 1e-4,
            "neumann inverse residual {residual} at block size {blk}"
        );
        Ok(())
    }

    /// Chunking must not break autograd, and must give the same gradients as the
    /// sequential path.
    #[test]
    fn test_chunked_gradients_match_sequential() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let dn = GatedDeltaNetTrain::new(64, 4, 64, vb.pp("t"))?;
        let x = Tensor::randn(0.0f32, 1.0, (1, 24, 64), &device)?;

        let grad_norms = |chunk: usize| -> Result<Vec<f32>> {
            let y = dn.forward_with_chunk(&x, chunk)?;
            let grads = y.sum_all()?.backward()?;
            let mut out = Vec::new();
            for t in [dn.wq.weight(), dn.wv.weight(), &dn.a_log, &dn.dt_bias] {
                let g = grads.get(t).expect("missing gradient");
                out.push(g.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?);
            }
            Ok(out)
        };

        let seq_norms = grad_norms(1)?;
        let chunk_norms = grad_norms(8)?;
        for (i, (a, b)) in seq_norms.iter().zip(chunk_norms.iter()).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "non-finite gradient");
            assert!(*a > 0.0, "param {i} had zero gradient in sequential path");
            let rel = (a - b).abs() / a.max(1e-6);
            assert!(
                rel < 2e-2,
                "param {i}: gradient norm differs between paths, {a} vs {b} (rel {rel})"
            );
        }
        Ok(())
    }

    /// A single chunk covering the whole sequence is the pure-parallel limit and
    /// must still match.
    #[test]
    fn test_chunked_single_chunk_whole_sequence() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let dn = GatedDeltaNetTrain::new(32, 2, 32, vb.pp("t"))?;
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, 32), &device)?;
        let reference = dn.forward_with_chunk(&x, 1)?;
        let single = dn.forward_with_chunk(&x, 16)?;
        let diff = max_abs_diff(&reference, &single)?;
        assert!(diff < 2e-3, "single-chunk path diverged by {diff}");
        Ok(())
    }

    /// Strong decay must not destabilize the chunked form: the cumulative-decay
    /// ratios stay bounded by construction, and outputs must remain finite.
    #[test]
    fn test_chunked_strong_decay_is_finite() -> Result<()> {
        let device = Device::Cpu;
        let (batch, heads, seq, d) = (1, 2, 40, 16);
        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, d), &device)?;
        let k = l2_normalize_last_dim(&q)?;
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, d), &device)?;
        let beta = Tensor::ones((batch, heads, seq), DType::F32, &device)?;
        // g = -20 per step: cumulative decay underflows hard within a chunk.
        let g = (Tensor::ones((batch, heads, seq), DType::F32, &device)? * -20.0)?;

        let out = chunked_recurrence(&q, &k, &v, &beta, &g, 16)?;
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|x| x.is_finite()),
            "strong decay produced non-finite output"
        );

        let decay = g.exp()?;
        let to_bshd = |t: &Tensor| t.transpose(1, 2)?.contiguous();
        let reference = sequential_recurrence(
            &to_bshd(&q)?,
            &to_bshd(&k)?,
            &to_bshd(&v)?,
            &beta.transpose(1, 2)?.contiguous()?,
            &decay.transpose(1, 2)?.contiguous()?,
        )?;
        let diff = max_abs_diff(&reference, &out.transpose(1, 2)?.contiguous()?)?;
        assert!(diff < 2e-3, "strong-decay chunked diverged by {diff}");
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
