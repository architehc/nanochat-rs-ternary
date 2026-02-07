//! Multi-head attention with RoPE for training.

use candle_core::{Result, Tensor, D};
#[cfg(test)]
use candle_core::DType;
use candle_nn::VarBuilder;

use crate::layers::BitLinearSTE;

/// Precompute RoPE cos/sin frequency tables.
pub fn precompute_rope_freqs(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half_dim), device)?;
    let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;
    let angles = positions.matmul(&freqs)?; // [max_seq_len, half_dim]
    let cos = angles.cos()?;
    let sin = angles.sin()?;
    Ok((cos, sin))
}

/// Apply rotary position embedding to Q and K tensors.
/// q, k: [batch, n_heads, seq_len, head_dim]
/// cos, sin: [seq_len, half_dim] (sliced from precomputed table)
fn apply_rotary_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let head_dim = q.dim(D::Minus1)?;
    let half = head_dim / 2;

    // Split into even/odd halves
    let q0 = q.narrow(D::Minus1, 0, half)?;
    let q1 = q.narrow(D::Minus1, half, half)?;
    let k0 = k.narrow(D::Minus1, 0, half)?;
    let k1 = k.narrow(D::Minus1, half, half)?;

    // Reshape cos/sin for broadcasting: [1, 1, seq_len, half_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    let q_rot0 = (q0.broadcast_mul(&cos)? - q1.broadcast_mul(&sin)?)?;
    let q_rot1 = (q0.broadcast_mul(&sin)? + q1.broadcast_mul(&cos)?)?;
    let k_rot0 = (k0.broadcast_mul(&cos)? - k1.broadcast_mul(&sin)?)?;
    let k_rot1 = (k0.broadcast_mul(&sin)? + k1.broadcast_mul(&cos)?)?;

    let q_out = Tensor::cat(&[&q_rot0, &q_rot1], D::Minus1)?;
    let k_out = Tensor::cat(&[&k_rot0, &k_rot1], D::Minus1)?;

    Ok((q_out, k_out))
}

/// Training-mode multi-head attention (no KV cache).
pub struct AttentionTrain {
    pub wq: BitLinearSTE,
    pub wk: BitLinearSTE,
    pub wv: BitLinearSTE,
    pub wo: BitLinearSTE,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionTrain {
    pub fn new(dim: usize, n_heads: usize, n_kv_heads: usize, group_size: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / n_heads;
        let wq = BitLinearSTE::new(dim, n_heads * head_dim, group_size, vb.pp("wq"))?;
        let wk = BitLinearSTE::new(dim, n_kv_heads * head_dim, group_size, vb.pp("wk"))?;
        let wv = BitLinearSTE::new(dim, n_kv_heads * head_dim, group_size, vb.pp("wv"))?;
        let wo = BitLinearSTE::new(n_heads * head_dim, dim, group_size, vb.pp("wo"))?;
        Ok(Self { wq, wk, wv, wo, n_heads, n_kv_heads, head_dim })
    }

    /// Forward pass: x [batch, seq_len, dim] -> [batch, seq_len, dim]
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _dim) = x.dims3()?;

        // Project Q, K, V
        let q = self.wq.forward(x)?; // [batch, seq, n_heads*head_dim]
        let k = self.wk.forward(x)?; // [batch, seq, n_kv_heads*head_dim]
        let v = self.wv.forward(x)?; // [batch, seq, n_kv_heads*head_dim]

        // Reshape to [batch, n_heads, seq, head_dim]
        let q = q.reshape((batch, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = apply_rotary_emb(&q, &k, cos, sin)?;

        // GQA: repeat K,V if needed
        let (k, v) = if self.n_kv_heads < self.n_heads {
            let n_rep = self.n_heads / self.n_kv_heads;
            let k = k.repeat(&[1, n_rep, 1, 1])?;
            let v = v.repeat(&[1, n_rep, 1, 1])?;
            (k, v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention with causal mask
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

        // Causal mask: upper triangular -inf
        let mask = causal_mask(seq_len, scores.device())?;
        let scores = scores.broadcast_add(&mask)?;

        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let v = v.contiguous()?;
        let attn_out = attn_weights.matmul(&v)?; // [batch, n_heads, seq, head_dim]

        // Reshape and project output
        let attn_out = attn_out.transpose(1, 2)?.reshape((batch, seq_len, self.n_heads * self.head_dim))?;
        self.wo.forward(&attn_out)
    }

    /// Collect weight tensors for param groups.
    pub fn linear_params(&self) -> Vec<&Tensor> {
        vec![self.wq.weight(), self.wk.weight(), self.wv.weight(), self.wo.weight()]
    }
}

/// Create causal attention mask: upper triangle filled with -inf, diagonal and below = 0.
fn causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask_data, (seq_len, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_attention_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = AttentionTrain::new(64, 4, 4, 64, vb.pp("test"))?;

        let (cos, sin) = precompute_rope_freqs(16, 32, 10000.0, &device)?;
        let cos_slice = cos.narrow(0, 0, 8)?;
        let sin_slice = sin.narrow(0, 0, 8)?;

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device)?;
        let y = attn.forward(&x, &cos_slice, &sin_slice)?;
        assert_eq!(y.dims(), &[2, 8, 64]);
        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let mask = causal_mask(4, &device)?;
        let vals = mask.to_vec2::<f32>()?;
        // Diagonal and below should be 0.0
        assert_eq!(vals[0][0], 0.0);
        assert_eq!(vals[1][1], 0.0);
        assert_eq!(vals[2][1], 0.0);
        // Above diagonal should be -inf
        assert!(vals[0][1].is_infinite() && vals[0][1] < 0.0);
        assert!(vals[0][3].is_infinite() && vals[0][3] < 0.0);
        Ok(())
    }

    #[test]
    fn test_rope_preserves_norm() -> Result<()> {
        let device = Device::Cpu;
        let (cos, sin) = precompute_rope_freqs(16, 8, 10000.0, &device)?;
        let q = Tensor::randn(0.0f32, 1.0, (1, 4, 8, 16), &device)?;
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 8, 16), &device)?;
        let (q_rot, _) = apply_rotary_emb(&q, &k, &cos, &sin)?;

        // RoPE should approximately preserve L2 norm
        let orig_norm = q.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let rot_norm = q_rot.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!((orig_norm - rot_norm).abs() / orig_norm < 0.01,
            "RoPE should preserve norm: {} vs {}", orig_norm, rot_norm);
        Ok(())
    }

    #[test]
    fn test_attention_gradient_flows() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = AttentionTrain::new(64, 4, 4, 64, vb.pp("test"))?;

        let (cos, sin) = precompute_rope_freqs(16, 8, 10000.0, &device)?;
        let cos_slice = cos.narrow(0, 0, 4)?;
        let sin_slice = sin.narrow(0, 0, 4)?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 64), &device)?;
        let y = attn.forward(&x, &cos_slice, &sin_slice)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let grad = grads.get(attn.wq.weight()).expect("wq should have gradient");
        let gn = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(gn > 0.0, "wq gradient should be non-zero");
        Ok(())
    }
}
