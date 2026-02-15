//! Token embeddings (FP32, not quantized to ternary).

/// Token embedding table.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Weight matrix [vocab_size, dim], row-major.
    pub weight: Vec<f32>,
    pub vocab_size: usize,
    pub dim: usize,
}

impl Embedding {
    /// Create embedding with zeros (for loading from checkpoint).
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        Self {
            weight: vec![0.0; vocab_size * dim],
            vocab_size,
            dim,
        }
    }

    /// Create embedding with small random init (for testing).
    pub fn new_random(vocab_size: usize, dim: usize, seed: u64) -> Self {
        let mut weight = vec![0.0f32; vocab_size * dim];
        // Simple LCG PRNG for deterministic init
        let mut state = seed;
        for w in weight.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to roughly [-0.02, 0.02]
            *w = ((state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.04;
        }
        Self {
            weight,
            vocab_size,
            dim,
        }
    }

    /// Look up a single token embedding.
    pub fn forward_token(&self, token_id: u32, out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        let token_idx = token_id as usize;
        if token_idx >= self.vocab_size {
            out.fill(0.0);
            return;
        }
        let offset = token_idx * self.dim;
        out.copy_from_slice(&self.weight[offset..offset + self.dim]);
    }

    /// Look up embeddings for a sequence of tokens.
    /// Output: [seq_len * dim], row-major.
    pub fn forward(&self, token_ids: &[u32], out: &mut [f32]) {
        let seq_len = token_ids.len();
        assert_eq!(out.len(), seq_len * self.dim);
        for (t, &tid) in token_ids.iter().enumerate() {
            self.forward_token(tid, &mut out[t * self.dim..(t + 1) * self.dim]);
        }
    }

    /// Use embedding weights as LM head (weight tying).
    /// Computes logits[v] = sum_d(x[d] * weight[v * dim + d]) for all v.
    /// This is an FP32 matmul (no ternary quantization).
    pub fn forward_as_lm_head(&self, x: &[f32], logits: &mut [f32]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(logits.len(), self.vocab_size);
        for (v, logit) in logits.iter_mut().enumerate().take(self.vocab_size) {
            let row = &self.weight[v * self.dim..(v + 1) * self.dim];
            let mut sum = 0.0f32;
            for d in 0..self.dim {
                sum += x[d] * row[d];
            }
            *logit = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() {
        let mut emb = Embedding::new(4, 3);
        // Set row 2 to known values
        emb.weight[6] = 1.0;
        emb.weight[7] = 2.0;
        emb.weight[8] = 3.0;

        let mut out = vec![0.0; 3];
        emb.forward_token(2, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_embedding_lookup_oob_returns_zero() {
        let emb = Embedding::new_random(4, 3, 42);
        let mut out = vec![9.0; 3];
        emb.forward_token(99, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_embedding_sequence() {
        let mut emb = Embedding::new(4, 2);
        emb.weight = vec![
            0.0, 1.0, // token 0
            2.0, 3.0, // token 1
            4.0, 5.0, // token 2
            6.0, 7.0, // token 3
        ];

        let tokens = [3, 1, 0];
        let mut out = vec![0.0; 6];
        emb.forward(&tokens, &mut out);
        assert_eq!(out, vec![6.0, 7.0, 2.0, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn test_embedding_random_init() {
        let emb = Embedding::new_random(100, 64, 42);
        assert_eq!(emb.weight.len(), 100 * 64);
        // Check values are in reasonable range
        for &w in &emb.weight {
            assert!(w.abs() < 0.1, "weight {} out of range", w);
        }
        // Check not all zeros
        assert!(emb.weight.iter().any(|&w| w != 0.0));
    }
}
