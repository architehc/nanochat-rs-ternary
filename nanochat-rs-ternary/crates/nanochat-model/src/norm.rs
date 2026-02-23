//! RMSNorm — Root Mean Square Layer Normalization.

use ternary_core::planar::AlignedVec;

/// RMSNorm parameters: just a learned scale vector.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: AlignedVec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    /// Create RMSNorm with all-ones weight (identity init).
    pub fn new(dim: usize) -> Self {
        let mut weight = AlignedVec::new_zeroed(dim);
        for w in weight.iter_mut() {
            *w = 1.0;
        }
        Self {
            weight,
            eps: 1e-6,
        }
    }

    /// Apply RMSNorm in-place: x = x / rms(x) * weight
    pub fn forward_inplace(&self, x: &mut [f32]) {
        let dim = self.weight.len();
        assert_eq!(x.len(), dim);

        // Compute 1/sqrt(mean(x^2) + eps)
        let mut ss = 0.0f32;
        for &v in x.iter() {
            ss += v * v;
        }
        let rms_inv = 1.0 / (ss / dim as f32 + self.eps).sqrt();

        // Normalize and scale
        for (x_val, &w) in x.iter_mut().zip(self.weight.iter()).take(dim) {
            *x_val = *x_val * rms_inv * w;
        }
    }

    /// Apply RMSNorm, writing result to `out`.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let dim = self.weight.len();
        assert_eq!(x.len(), dim);
        assert_eq!(out.len(), dim);

        let mut ss = 0.0f32;
        for &v in x.iter() {
            ss += v * v;
        }
        let rms_inv = 1.0 / (ss / dim as f32 + self.eps).sqrt();

        for i in 0..dim {
            out[i] = x[i] * rms_inv * self.weight[i];
        }
    }

    /// Batched RMSNorm: process `seq_len` tokens.
    ///
    /// x_batch:   [seq_len * dim]
    /// out_batch: [seq_len * dim]
    pub fn forward_batch(&self, x_batch: &[f32], seq_len: usize, out_batch: &mut [f32]) {
        let dim = self.weight.len();
        assert_eq!(x_batch.len(), seq_len * dim);
        assert_eq!(out_batch.len(), seq_len * dim);

        for t in 0..seq_len {
            let x = &x_batch[t * dim..(t + 1) * dim];
            let out = &mut out_batch[t * dim..(t + 1) * dim];
            self.forward(x, out);
        }
    }

    /// Clone the layer and place weight allocations on a specific NUMA node.
    pub fn clone_to_node(&self, node: usize) -> Self {
        Self {
            weight: self.weight.clone_to_node(node),
            eps: self.eps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_identity_weight() {
        let norm = RMSNorm::new(4);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let x_orig = x.clone();
        norm.forward_inplace(&mut x);

        // After RMSNorm with weight=1, output should be normalized
        let rms = (x_orig.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-6).sqrt();
        for i in 0..4 {
            let expected = x_orig[i] / rms;
            assert!(
                (x[i] - expected).abs() < 1e-5,
                "idx {}: got {}, expected {}",
                i,
                x[i],
                expected
            );
        }
    }

    #[test]
    fn test_rmsnorm_with_weights() {
        let mut norm = RMSNorm::new(4);
        for w in norm.weight.iter_mut() {
            *w = 2.0;
        }

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        norm.forward(&x, &mut out);

        // rms(1,1,1,1) ≈ 1.0, so output ≈ 2.0 each
        for &v in &out {
            assert!((v - 2.0).abs() < 0.01, "got {}", v);
        }
    }

    #[test]
    fn test_rmsnorm_forward_vs_inplace() {
        let norm = RMSNorm::new(8);
        let x = vec![0.5, -1.0, 2.0, -0.3, 1.5, 0.0, -2.0, 0.7];

        let mut x_inplace = x.clone();
        norm.forward_inplace(&mut x_inplace);

        let mut x_out = vec![0.0; 8];
        norm.forward(&x, &mut x_out);

        for i in 0..8 {
            assert!(
                (x_inplace[i] - x_out[i]).abs() < 1e-7,
                "mismatch at {}: {} vs {}",
                i,
                x_inplace[i],
                x_out[i]
            );
        }
    }
}
