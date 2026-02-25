//! Shared loop block for LoopLM (arXiv:2510.25741).
//!
//! Instead of N unique layers, LoopLM uses a shallow stack of local layers
//! plus a shared middle layer that loops multiple times:
//!
//! Input → [Local Layers] → [Shared Loop Layer × L iterations] → [Local Layers] → Output

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::TrainConfig;
use crate::layers::{BitLinearSTE, RMSNormTrain};
use crate::mhc::MhcLiteN2Train;

/// Shared loop block: recurrent transformer layer with global state accumulation.
///
/// Architecture per LoopLM paper Section 3.2:
/// - Local transformations: Q/K/V projections, FFN (unique per iteration)
/// - Global recurrent updates: Gate mechanisms (g_qk, g_ffn) that mix current with accumulated state
pub struct SharedLoopBlock {
    // Local attention projections
    pub wq: BitLinearSTE,
    pub wk: BitLinearSTE,
    pub wv: BitLinearSTE,
    pub wo: BitLinearSTE,

    // Global attention gate (controls recurrent QK mixing)
    pub g_qk: BitLinearSTE,

    // Local FFN projections (SwiGLU)
    pub w_gate: BitLinearSTE,
    pub w_up: BitLinearSTE,
    pub w_down: BitLinearSTE,

    // Global FFN gate (controls recurrent FFN output mixing)
    pub g_ffn: BitLinearSTE,

    // Norms
    pub norm_attn: RMSNormTrain,
    pub norm_ffn: RMSNormTrain,

    // mHC residual connection handlers
    pub mhc_attn: MhcLiteN2Train,
    pub mhc_ffn: MhcLiteN2Train,

    // Config
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_rep: usize,
    pub ffn_dim: usize,
}

impl SharedLoopBlock {
    pub fn new(cfg: &TrainConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = dim / n_heads;
        let n_rep = n_heads / n_kv_heads;
        let ffn_dim = cfg.ffn_dim();

        // Local attention projections
        let wq = BitLinearSTE::new(dim, dim, cfg.group_size, vb.pp("wq"))?;
        let wk = BitLinearSTE::new(dim, n_kv_heads * head_dim, cfg.group_size, vb.pp("wk"))?;
        let wv = BitLinearSTE::new(dim, n_kv_heads * head_dim, cfg.group_size, vb.pp("wv"))?;
        let wo = BitLinearSTE::new(dim, dim, cfg.group_size, vb.pp("wo"))?;

        // Global attention gate
        let g_qk = BitLinearSTE::new(dim, dim, cfg.group_size, vb.pp("g_qk"))?;

        // Local FFN projections
        let w_gate = BitLinearSTE::new(dim, ffn_dim, cfg.group_size, vb.pp("w_gate"))?;
        let w_up = BitLinearSTE::new(dim, ffn_dim, cfg.group_size, vb.pp("w_up"))?;
        let w_down = BitLinearSTE::new(ffn_dim, dim, cfg.group_size, vb.pp("w_down"))?;

        // Global FFN gate
        let g_ffn = BitLinearSTE::new(dim, dim, cfg.group_size, vb.pp("g_ffn"))?;

        // Norms
        let norm_attn = RMSNormTrain::new(dim, vb.pp("norm_attn"))?;
        let norm_ffn = RMSNormTrain::new(dim, vb.pp("norm_ffn"))?;

        // mHC residual connection handlers
        let mhc_attn = MhcLiteN2Train::new(vb.pp("mhc_attn"))?;
        let mhc_ffn = MhcLiteN2Train::new(vb.pp("mhc_ffn"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            g_qk,
            w_gate,
            w_up,
            w_down,
            g_ffn,
            norm_attn,
            norm_ffn,
            mhc_attn,
            mhc_ffn,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rep,
            ffn_dim,
        })
    }

    /// Forward pass for ONE loop iteration (simplified - no RoPE for MVP).
    ///
    /// Args:
    /// - x_expanded: [batch, seq, dim * n_streams] - mHC-expanded hidden state
    /// - global_state: Option<&Tensor> - accumulated global state [batch, seq, dim]
    ///
    /// Returns:
    /// - x_expanded_out: [batch, seq, dim * n_streams] - Updated hidden state
    /// - global_state_out: [batch, seq, dim] - Updated global recurrent state
    pub fn forward(
        &self,
        x_expanded: &Tensor,
        global_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // ========== Attention Sub-Layer ==========

        // 1. mHC prepare: collapse expanded input to single stream
        let x = self.mhc_attn.prepare_input(x_expanded, self.dim)?;

        // 2. Pre-norm
        let x_norm = self.norm_attn.forward(&x)?;

        // 3. Local Q/K/V projections
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;

        // 4. Simple scaled dot-product attention (no RoPE for MVP)
        let attn_out = self.compute_attention(&q, &k, &v)?;

        // 5. Output projection
        let attn_out = self.wo.forward(&attn_out)?;

        // 6. Global gate: mix with accumulated global state
        let gated_attn = if let Some(g_state) = global_state {
            let gate = self.g_qk.forward(&x_norm)?;
            let gate = candle_nn::ops::sigmoid(&gate)?;

            // Mix: gate * attn_out + (1 - gate) * global_state
            let one = gate.ones_like()?;
            let inv_gate = (&one - &gate)?;
            ((&gate * &attn_out)? + (&inv_gate * g_state)?)?
        } else {
            attn_out.clone()
        };

        // 7. mHC apply: add residual
        let x_expanded = self.mhc_attn.apply(x_expanded, &gated_attn, self.dim)?;

        // ========== FFN Sub-Layer ==========

        // 1. mHC prepare
        let x = self.mhc_ffn.prepare_input(&x_expanded, self.dim)?;

        // 2. Pre-norm
        let x_norm = self.norm_ffn.forward(&x)?;

        // 3. SwiGLU FFN
        let gate = self.w_gate.forward(&x_norm)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.w_up.forward(&x_norm)?;
        let ffn_hidden = (&gate * &up)?;
        let ffn_out = self.w_down.forward(&ffn_hidden)?;

        // 4. Global FFN gate
        let gated_ffn = if let Some(g_state) = global_state {
            let gate = self.g_ffn.forward(&x_norm)?;
            let gate = candle_nn::ops::sigmoid(&gate)?;

            let one = gate.ones_like()?;
            let inv_gate = (&one - &gate)?;
            ((&gate * &ffn_out)? + (&inv_gate * g_state)?)?
        } else {
            ffn_out.clone()
        };

        // 5. mHC apply
        let x_expanded_out = self.mhc_ffn.apply(&x_expanded, &gated_ffn, self.dim)?;

        // 6. Update global state: average of attention and FFN outputs
        let global_state_out = ((&gated_attn + &gated_ffn)? * 0.5)?;

        Ok((x_expanded_out, global_state_out))
    }

    /// Compute multi-head attention (simplified).
    fn compute_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = q.dims3()?;

        // Reshape: [batch, seq, n_heads, head_dim]
        let q = q.reshape((batch, seq, self.n_heads, self.head_dim))?;
        let k = k.reshape((batch, seq, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq, self.n_kv_heads, self.head_dim))?;

        // Transpose: [batch, n_heads, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // GQA: repeat K/V if needed
        let k = if self.n_rep > 1 {
            k.repeat(&[1, self.n_rep, 1, 1])?
        } else {
            k
        };
        let v = if self.n_rep > 1 {
            v.repeat(&[1, self.n_rep, 1, 1])?
        } else {
            v
        };

        // Attention scores: Q @ K^T / sqrt(head_dim)
        let k_t = k.transpose(D::Minus1, D::Minus2)?;
        let scores = q.matmul(&k_t)?;
        let scale = (self.head_dim as f64).sqrt();
        let scores = (scores / scale)?;

        // Causal mask (for training sequences)
        let scores = self.apply_causal_mask(&scores)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Output: [batch, n_heads, seq, head_dim]
        let out = attn_weights.matmul(&v)?;

        // Reshape back: [batch, seq, dim]
        let out = out.transpose(1, 2)?.contiguous()?;
        out.reshape((batch, seq, self.dim))
    }

    /// Apply causal mask to attention scores.
    fn apply_causal_mask(&self, scores: &Tensor) -> Result<Tensor> {
        let (_, _, seq, _) = scores.dims4()?;

        // Create causal mask: upper triangular matrix of -inf
        let mask: Vec<f32> = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();

        let mask = Tensor::from_vec(mask, (seq, seq), scores.device())?;
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, seq]

        scores.broadcast_add(&mask)
    }

    /// Collect all linear (BitLinearSTE) parameters.
    pub fn linear_params(&self) -> Vec<&Tensor> {
        vec![
            self.wq.weight(),
            self.wk.weight(),
            self.wv.weight(),
            self.wo.weight(),
            self.g_qk.weight(),
            self.g_ffn.weight(),
            self.w_gate.weight(),
            self.w_up.weight(),
            self.w_down.weight(),
        ]
    }

    /// Collect all mHC parameters.
    pub fn mhc_params(&self) -> Vec<&Tensor> {
        let mut params = self.mhc_attn.params().into_iter().collect::<Vec<_>>();
        params.extend(self.mhc_ffn.params());
        params
    }

    /// Collect all norm parameters.
    pub fn norm_params(&self) -> Vec<&Tensor> {
        vec![self.norm_attn.weight(), self.norm_ffn.weight()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainConfig;
    use candle_core::{DType, Device, IndexOp};
    use candle_nn::VarMap;

    fn test_config() -> TrainConfig {
        TrainConfig {
            dim: 64,
            n_layers: 3,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.0,
            vocab_size: 256,
            max_seq_len: 32,
            group_size: 64,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,
            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 10,
            total_steps: 100,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 3,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
            wavefield_physics_lr: 5e-4,
            wavefield_warmup_delay: 0,
            wavefield_haar_direct: true,
        }
    }

    #[test]
    fn test_shared_loop_block_construction() -> Result<()> {
        let cfg = test_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = SharedLoopBlock::new(&cfg, vb.pp("loop_block"))?;

        assert_eq!(block.dim, 64);
        assert_eq!(block.n_heads, 4);
        assert_eq!(block.head_dim, 16);

        Ok(())
    }

    #[test]
    fn test_shared_loop_block_forward_single_iteration() -> Result<()> {
        let cfg = test_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = SharedLoopBlock::new(&cfg, vb.pp("loop_block"))?;

        let batch = 2;
        let seq = 4;
        let dim = cfg.dim;
        let n_streams = cfg.mhc_n_streams;

        // Create input: [batch, seq, dim * n_streams]
        let x_expanded = Tensor::randn(0f32, 1.0, (batch, seq, dim * n_streams), &device)?;

        // First iteration (no global state)
        let (x_out, global_state) = block.forward(&x_expanded, None)?;

        // Check output shapes
        assert_eq!(x_out.dims(), &[batch, seq, dim * n_streams]);
        assert_eq!(global_state.dims(), &[batch, seq, dim]);

        Ok(())
    }

    #[test]
    fn test_shared_loop_block_multiple_iterations() -> Result<()> {
        let cfg = test_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = SharedLoopBlock::new(&cfg, vb.pp("loop_block"))?;

        let batch = 1;
        let seq = 2;
        let dim = cfg.dim;
        let n_streams = cfg.mhc_n_streams;

        let x_expanded = Tensor::randn(0f32, 1.0, (batch, seq, dim * n_streams), &device)?;

        // Iteration 1
        let (mut x_state, mut global_state) = block.forward(&x_expanded, None)?;

        // Iteration 2: with global state
        (x_state, global_state) = block.forward(&x_state, Some(&global_state))?;

        // Iteration 3
        (x_state, global_state) = block.forward(&x_state, Some(&global_state))?;

        // Output shapes should remain consistent
        assert_eq!(x_state.dims(), &[batch, seq, dim * n_streams]);
        assert_eq!(global_state.dims(), &[batch, seq, dim]);

        Ok(())
    }

    #[test]
    fn test_causal_mask_prevents_future_attention() -> Result<()> {
        let cfg = test_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = SharedLoopBlock::new(&cfg, vb.pp("loop_block"))?;

        // Create attention scores: [batch=1, n_heads=4, seq=3, seq=3]
        let scores = Tensor::ones((1, 4, 3, 3), DType::F32, &device)?;

        let masked = block.apply_causal_mask(&scores)?;

        // After softmax, future positions should have zero weight
        let weights = candle_nn::ops::softmax_last_dim(&masked)?;

        // Check that position 0 can only attend to itself
        let weights_0 = weights.i((0, 0, 0))?; // First query
        let w0 = weights_0.i(0)?.to_scalar::<f32>()?;
        let w1 = weights_0.i(1)?.to_scalar::<f32>()?;
        let w2 = weights_0.i(2)?.to_scalar::<f32>()?;

        assert!((w0 - 1.0).abs() < 1e-5, "Should attend only to position 0");
        assert!(w1.abs() < 1e-5, "Should not attend to future position 1");
        assert!(w2.abs() < 1e-5, "Should not attend to future position 2");

        Ok(())
    }
}
