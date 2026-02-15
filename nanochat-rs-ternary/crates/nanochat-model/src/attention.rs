//! Multi-Head Attention with RoPE (Rotary Position Embeddings).
//!
//! Supports MHA and GQA. All projections are BitLinear (ternary).
//! KV-cache support for autoregressive inference.

use crate::bitlinear::BitLinear;
use crate::config::ModelConfig;

/// RoPE frequency table, precomputed.
#[derive(Debug, Clone)]
pub struct RopeFreqs {
    /// cos values: [max_seq_len, head_dim/2]
    pub cos: Vec<f32>,
    /// sin values: [max_seq_len, head_dim/2]
    pub sin: Vec<f32>,
    pub max_seq_len: usize,
    pub half_dim: usize,
}

impl RopeFreqs {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_dim + i] = angle.cos();
                sin[pos * half_dim + i] = angle.sin();
            }
        }

        Self {
            cos,
            sin,
            max_seq_len,
            half_dim,
        }
    }

    /// Apply RoPE to q or k in-place. `data` is [n_heads, head_dim], pos is the position offset.
    pub fn apply(&self, data: &mut [f32], n_heads: usize, head_dim: usize, pos: usize) {
        assert!(pos < self.max_seq_len);
        let half = head_dim / 2;
        let cos_row = &self.cos[pos * self.half_dim..pos * self.half_dim + half];
        let sin_row = &self.sin[pos * self.half_dim..pos * self.half_dim + half];

        for h in 0..n_heads {
            let base = h * head_dim;
            for i in 0..half {
                let x0 = data[base + i];
                let x1 = data[base + half + i];
                data[base + i] = x0 * cos_row[i] - x1 * sin_row[i];
                data[base + half + i] = x1 * cos_row[i] + x0 * sin_row[i];
            }
        }
    }
}

/// KV cache for one layer.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Key cache: [max_seq_len, n_kv_heads * head_dim]
    pub k: Vec<f32>,
    /// Value cache: [max_seq_len, n_kv_heads * head_dim]
    pub v: Vec<f32>,
    pub len: usize,
    pub kv_dim: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        Self {
            k: vec![0.0; max_seq_len * kv_dim],
            v: vec![0.0; max_seq_len * kv_dim],
            len: 0,
            kv_dim,
        }
    }

    pub fn append_kv(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert_eq!(v.len(), self.kv_dim);
        let offset = self.len * self.kv_dim;
        self.k[offset..offset + self.kv_dim].copy_from_slice(k);
        self.v[offset..offset + self.kv_dim].copy_from_slice(v);
        self.len += 1;
    }
}

/// Multi-head attention (supports GQA via n_kv_heads < n_heads).
#[derive(Debug)]
pub struct Attention {
    pub wq: BitLinear,
    pub wk: BitLinear,
    pub wv: BitLinear,
    pub wo: BitLinear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_rep: usize,
}

impl Attention {
    /// Create attention with random weights (for testing).
    pub fn new_random(config: &ModelConfig) -> Self {
        let hd = config.head_dim();
        let dim = config.dim;
        let kv_dim = config.n_kv_heads * hd;
        let gs = config.group_size;

        Self {
            wq: BitLinear::from_float(&random_weights(dim, dim, 1), dim, dim, gs),
            wk: BitLinear::from_float(&random_weights(kv_dim, dim, 2), kv_dim, dim, gs),
            wv: BitLinear::from_float(&random_weights(kv_dim, dim, 3), kv_dim, dim, gs),
            wo: BitLinear::from_float(&random_weights(dim, dim, 4), dim, dim, gs),
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: hd,
            n_rep: config.n_rep(),
        }
    }

    /// Forward pass for a single token at position `pos`.
    ///
    /// x:     [dim] input
    /// cache: KV cache for this layer
    /// rope:  precomputed RoPE frequencies
    /// pos:   current position
    /// out:   [dim] output
    pub fn forward(
        &self,
        x: &[f32],
        cache: &mut KvCache,
        rope: &RopeFreqs,
        pos: usize,
        out: &mut [f32],
    ) {
        let dim = self.wq.cols;
        let kv_dim = self.n_kv_heads * self.head_dim;

        // Project Q, K, V
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];

        self.wq.forward(x, &mut q);
        self.wk.forward(x, &mut k);
        self.wv.forward(x, &mut v);

        // Apply RoPE to Q and K
        rope.apply(&mut q, self.n_heads, self.head_dim, pos);
        rope.apply(&mut k, self.n_kv_heads, self.head_dim, pos);

        // Append to KV cache
        cache.append_kv(&k, &v);

        // Attention: for each head, compute scaled dot-product over cached KV
        let seq_len = cache.len;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let mut attn_out = vec![0.0f32; dim];

        for h in 0..self.n_heads {
            let kv_h = h / self.n_rep; // GQA: multiple Q heads share one KV head
            let q_offset = h * self.head_dim;
            let kv_offset = kv_h * self.head_dim;

            // Compute attention scores
            let mut scores = vec![0.0f32; seq_len];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_base = t * kv_dim + kv_offset;
                let mut dot = 0.0f32;
                for d in 0..self.head_dim {
                    dot += q[q_offset + d] * cache.k[k_base + d];
                }
                *score = dot * scale;
            }

            // Softmax
            softmax_inplace(&mut scores);

            // Weighted sum of values
            let out_offset = h * self.head_dim;
            for (t, &w) in scores.iter().enumerate() {
                let v_base = t * kv_dim + kv_offset;
                for d in 0..self.head_dim {
                    attn_out[out_offset + d] += w * cache.v[v_base + d];
                }
            }
        }

        // Output projection
        self.wo.forward(&attn_out, out);
    }

    /// Batched forward pass for prefill: process `seq_len` tokens with causal attention.
    ///
    /// x_batch:   [seq_len * dim] — flattened input tokens
    /// seq_len:   number of tokens to process
    /// cache:     KV cache (will be filled with all seq_len positions)
    /// rope:      RoPE frequencies
    /// start_pos: starting position offset (0 for fresh prefill)
    /// out_batch: [seq_len * dim] — flattened output
    pub fn forward_batch(
        &self,
        x_batch: &[f32],
        seq_len: usize,
        cache: &mut KvCache,
        rope: &RopeFreqs,
        start_pos: usize,
        out_batch: &mut [f32],
    ) {
        let dim = self.wq.cols;
        let kv_dim = self.n_kv_heads * self.head_dim;

        // 1. Project all Q, K, V for all tokens
        let mut all_q = vec![0.0f32; seq_len * dim];
        let mut all_k = vec![0.0f32; seq_len * kv_dim];
        let mut all_v = vec![0.0f32; seq_len * kv_dim];

        self.wq.forward_batch(x_batch, seq_len, &mut all_q);
        self.wk.forward_batch(x_batch, seq_len, &mut all_k);
        self.wv.forward_batch(x_batch, seq_len, &mut all_v);

        // 2. Apply RoPE and fill KV cache for each position
        for t in 0..seq_len {
            let pos = start_pos + t;
            let q_slice = &mut all_q[t * dim..(t + 1) * dim];
            let k_slice = &mut all_k[t * kv_dim..(t + 1) * kv_dim];
            let v_slice = &all_v[t * kv_dim..(t + 1) * kv_dim];

            rope.apply(q_slice, self.n_heads, self.head_dim, pos);
            rope.apply(k_slice, self.n_kv_heads, self.head_dim, pos);

            cache.append_kv(k_slice, v_slice);
        }

        // 3. Causal attention: query at position t attends to [0..start_pos+t+1]
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        for t in 0..seq_len {
            let causal_len = start_pos + t + 1; // number of KV positions visible
            let q_base = t * dim;
            let out_base = t * dim;

            let mut attn_out = vec![0.0f32; dim];

            for h in 0..self.n_heads {
                let kv_h = h / self.n_rep;
                let q_offset = q_base + h * self.head_dim;
                let kv_offset = kv_h * self.head_dim;

                // Compute attention scores
                let mut scores = vec![0.0f32; causal_len];
                for (s, score) in scores.iter_mut().enumerate() {
                    let k_base = s * kv_dim + kv_offset;
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += all_q[q_offset + d] * cache.k[k_base + d];
                    }
                    *score = dot * scale;
                }

                // Softmax
                softmax_inplace(&mut scores);

                // Weighted sum of values
                let h_out_offset = h * self.head_dim;
                for (s, &w) in scores.iter().enumerate() {
                    let v_base = s * kv_dim + kv_offset;
                    for d in 0..self.head_dim {
                        attn_out[h_out_offset + d] += w * cache.v[v_base + d];
                    }
                }
            }

            // Output projection for this token
            self.wo
                .forward(&attn_out, &mut out_batch[out_base..out_base + dim]);
        }
    }
}

/// In-place softmax.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Generate deterministic pseudo-random weights for testing.
fn random_weights(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; rows * cols];
    for (i, val) in w.iter_mut().enumerate() {
        let v = ((i as u32).wrapping_add(seed * 7919)).wrapping_mul(2654435761) >> 16;
        *val = (v % 200) as f32 / 100.0 - 1.0;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_freqs() {
        let rope = RopeFreqs::new(64, 128, 10000.0);
        assert_eq!(rope.cos.len(), 128 * 32);
        // Position 0 should have cos=1, sin=0
        for i in 0..32 {
            assert!((rope.cos[i] - 1.0).abs() < 1e-5);
            assert!(rope.sin[i].abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_apply() {
        let rope = RopeFreqs::new(4, 16, 10000.0);
        let mut data = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, dim=4
        let orig = data.clone();
        rope.apply(&mut data, 1, 4, 0);
        // At position 0, rotation is identity
        for i in 0..4 {
            assert!((data[i] - orig[i]).abs() < 1e-5, "pos 0 should be identity");
        }
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KvCache::new(16, 2, 4);
        assert_eq!(cache.len, 0);
        cache.append_kv(&[1.0; 8], &[2.0; 8]);
        assert_eq!(cache.len, 1);
        assert_eq!(cache.k[0], 1.0);
        assert_eq!(cache.v[0], 2.0);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_attention_forward_batch_matches_sequential() {
        let config = ModelConfig::d20();
        let attn = Attention::new_random(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let dim = config.dim;
        let seq_len = 4;

        // Create input: seq_len tokens
        let mut x_batch = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            for d in 0..dim {
                x_batch[t * dim + d] = ((t * dim + d) as f32 / (seq_len * dim) as f32) - 0.5;
            }
        }

        // Sequential forward (reference)
        let mut cache_seq = KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim());
        let mut out_seq_all = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            let x = &x_batch[t * dim..(t + 1) * dim];
            let out = &mut out_seq_all[t * dim..(t + 1) * dim];
            attn.forward(x, &mut cache_seq, &rope, t, out);
        }

        // Batched forward
        let mut cache_batch =
            KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim());
        let mut out_batch = vec![0.0f32; seq_len * dim];
        attn.forward_batch(
            &x_batch,
            seq_len,
            &mut cache_batch,
            &rope,
            0,
            &mut out_batch,
        );

        // Compare: each token's output should match
        for t in 0..seq_len {
            let max_diff: f32 = (0..dim)
                .map(|d| (out_seq_all[t * dim + d] - out_batch[t * dim + d]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-5,
                "token {}: batched vs sequential max_diff={} (should be < 1e-5)",
                t,
                max_diff
            );
        }

        // KV cache lengths should match
        assert_eq!(cache_seq.len, cache_batch.len);
    }

    #[test]
    fn test_attention_forward() {
        let config = ModelConfig::d20();
        let attn = Attention::new_random(&config);
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let mut cache = KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim());

        let x = vec![0.1f32; config.dim];
        let mut out = vec![0.0f32; config.dim];

        // Single token at position 0
        attn.forward(&x, &mut cache, &rope, 0, &mut out);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite attention output"
        );
        assert!(out.iter().any(|&v| v != 0.0), "all-zero attention output");
    }
}
