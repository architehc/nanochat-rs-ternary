// n2.rs — MhcLiteN2: 2-stream mHC-lite with 1 learnable residual parameter
//
// Only 2 permutation matrices exist for n=2:
//   P0 = [[1,0],[0,1]]  (identity)
//   P1 = [[0,1],[1,0]]  (swap)
//
// H_res = alpha * I + (1-alpha) * J
// where alpha = sigmoid(logit), so H_res is always doubly stochastic.
//
// Total learnable params per layer: 1 (res) + 2 (pre) + 2 (pre_bias) + 2 (post) + 2 (post_bias) = 9

use crate::sigmoid;
use std::fmt;

/// Error while deserializing `MhcLiteN2` from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum N2DecodeError {
    TooShort { expected: usize, actual: usize },
}

impl fmt::Display for N2DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            N2DecodeError::TooShort { expected, actual } => {
                write!(
                    f,
                    "N2 decode failed: expected at least {} bytes, got {} bytes",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for N2DecodeError {}

#[derive(Clone, Debug)]
pub struct MhcLiteN2 {
    /// Residual mixing: sigmoid(alpha_logit) interpolates I vs swap
    pub alpha_logit: f32,

    /// Pre-projection: how streams feed into layer F
    /// h_pre = sigmoid(pre_logits + pre_bias), shape [2]
    pub pre_logits: [f32; 2],
    pub pre_bias: [f32; 2],

    /// Post-projection: how layer output distributes back to streams
    /// h_post = 2 * sigmoid(post_logits + post_bias), shape [2]
    pub post_logits: [f32; 2],
    pub post_bias: [f32; 2],
}

impl MhcLiteN2 {
    /// Create with identity initialization (alpha=1 -> pure identity residual)
    pub fn new_identity() -> Self {
        Self {
            // Large positive logit -> sigmoid ~ 1 -> identity matrix
            alpha_logit: 5.0,
            // Equal mixing for pre/post, biased toward identity-like behavior
            pre_logits: [0.0, 0.0],
            pre_bias: [0.5, 0.5],
            post_logits: [0.0, 0.0],
            post_bias: [0.5, 0.5],
        }
    }

    /// Load from serialized weights (e.g., from PyTorch checkpoint)
    pub fn from_weights(
        alpha_logit: f32,
        pre_logits: [f32; 2],
        pre_bias: [f32; 2],
        post_logits: [f32; 2],
        post_bias: [f32; 2],
    ) -> Self {
        Self {
            alpha_logit,
            pre_logits,
            pre_bias,
            post_logits,
            post_bias,
        }
    }

    /// Compute H_res (2x2 doubly stochastic matrix)
    /// Guaranteed: all entries >= 0, rows sum to 1, cols sum to 1
    #[inline]
    pub fn h_res(&self) -> [[f32; 2]; 2] {
        let alpha = sigmoid(self.alpha_logit);
        let beta = 1.0 - alpha;
        [[alpha, beta], [beta, alpha]]
    }

    /// Compute H_pre (2-element non-negative vector)
    #[inline]
    pub fn h_pre(&self) -> [f32; 2] {
        [
            sigmoid(self.pre_logits[0] + self.pre_bias[0]),
            sigmoid(self.pre_logits[1] + self.pre_bias[1]),
        ]
    }

    /// Compute H_post (2-element non-negative vector, 2x scaled)
    #[inline]
    pub fn h_post(&self) -> [f32; 2] {
        [
            2.0 * sigmoid(self.post_logits[0] + self.post_bias[0]),
            2.0 * sigmoid(self.post_logits[1] + self.post_bias[1]),
        ]
    }

    /// Prepare layer input: mix 2 streams -> 1 stream
    /// Input:  x = [batch, 2*C] (two streams concatenated)
    /// Output: [batch, C]
    pub fn prepare_input(&self, x: &[f32], dim_c: usize) -> Vec<f32> {
        let h_pre = self.h_pre();
        let batch = x.len() / (2 * dim_c);
        let mut out = vec![0.0f32; batch * dim_c];

        for b in 0..batch {
            let s0 = &x[b * 2 * dim_c..b * 2 * dim_c + dim_c]; // stream 0
            let s1 = &x[b * 2 * dim_c + dim_c..b * 2 * dim_c + 2 * dim_c]; // stream 1
            let o = &mut out[b * dim_c..(b + 1) * dim_c];

            for i in 0..dim_c {
                o[i] = h_pre[0] * s0[i] + h_pre[1] * s1[i];
            }
        }
        out
    }

    /// Apply residual update with training-matching semantics.
    ///
    /// FIXED: Apply residual to each stream FIRST, then mix streams.
    /// Old formula: x_out = H_res @ x + H_post^T * layer_output
    /// New formula: x_out = H_res @ (x + H_post^T * layer_output)
    ///
    /// This matches the training code and prevents identity bypass.
    ///
    /// x:            [batch, 2*C]  (two streams)
    /// layer_output: [batch, C]    (single stream from layer F)
    /// Returns:      [batch, 2*C]  (two streams)
    pub fn apply(&self, x: &[f32], layer_output: &[f32], dim_c: usize) -> Vec<f32> {
        let h_res = self.h_res();
        let h_post = self.h_post();
        let batch = x.len() / (2 * dim_c);
        let mut out = vec![0.0f32; batch * 2 * dim_c];

        for b in 0..batch {
            let s0 = &x[b * 2 * dim_c..b * 2 * dim_c + dim_c];
            let s1 = &x[b * 2 * dim_c + dim_c..b * 2 * dim_c + 2 * dim_c];
            let ly = &layer_output[b * dim_c..(b + 1) * dim_c];

            let (o0, o1) = out[b * 2 * dim_c..b * 2 * dim_c + 2 * dim_c].split_at_mut(dim_c);

            for i in 0..dim_c {
                // FIXED: Apply residual to each stream FIRST
                let s0_with_res = s0[i] + h_post[0] * ly[i];
                let s1_with_res = s1[i] + h_post[1] * ly[i];

                // THEN mix streams with H_res
                o0[i] = h_res[0][0] * s0_with_res + h_res[0][1] * s1_with_res;
                o1[i] = h_res[1][0] * s0_with_res + h_res[1][1] * s1_with_res;
            }
        }
        out
    }

    /// Initialize expanded state from single-stream input
    /// Input:  [batch, C]
    /// Output: [batch, 2*C] (duplicate into both streams)
    pub fn expand_input(x: &[f32], dim_c: usize) -> Vec<f32> {
        let batch = x.len() / dim_c;
        let mut out = vec![0.0f32; batch * 2 * dim_c];
        for b in 0..batch {
            let src = &x[b * dim_c..(b + 1) * dim_c];
            // Stream 0 = input, Stream 1 = input (both start identical)
            out[b * 2 * dim_c..b * 2 * dim_c + dim_c].copy_from_slice(src);
            out[b * 2 * dim_c + dim_c..b * 2 * dim_c + 2 * dim_c].copy_from_slice(src);
        }
        out
    }

    /// Collapse expanded state back to single stream
    /// Input:  [batch, 2*C]
    /// Output: [batch, C] (average the two streams)
    pub fn collapse_output(x: &[f32], dim_c: usize) -> Vec<f32> {
        let batch = x.len() / (2 * dim_c);
        let mut out = vec![0.0f32; batch * dim_c];
        for b in 0..batch {
            let s0 = &x[b * 2 * dim_c..b * 2 * dim_c + dim_c];
            let s1 = &x[b * 2 * dim_c + dim_c..b * 2 * dim_c + 2 * dim_c];
            let o = &mut out[b * dim_c..(b + 1) * dim_c];
            for i in 0..dim_c {
                o[i] = 0.5 * (s0[i] + s1[i]);
            }
        }
        out
    }

    /// Deserialize N=2 parameters from 36 bytes (little-endian f32)
    ///
    /// Layout: [alpha_logit][pre_logits x2][pre_bias x2][post_logits x2][post_bias x2]
    pub fn from_bytes(data: &[u8]) -> Result<Self, N2DecodeError> {
        if data.len() < 36 {
            return Err(N2DecodeError::TooShort {
                expected: 36,
                actual: data.len(),
            });
        }
        let f = |offset: usize| -> f32 {
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        Ok(Self {
            alpha_logit: f(0),
            pre_logits: [f(4), f(8)],
            pre_bias: [f(12), f(16)],
            post_logits: [f(20), f(24)],
            post_bias: [f(28), f(32)],
        })
    }

    /// Serialize N=2 parameters to 36 bytes (little-endian f32)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(36);
        out.extend_from_slice(&self.alpha_logit.to_le_bytes());
        for v in &self.pre_logits {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.pre_bias {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.post_logits {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.post_bias {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::verify_doubly_stochastic_2x2;

    #[test]
    fn test_n2_identity_init() {
        let mhc = MhcLiteN2::new_identity();
        let h = mhc.h_res();
        // Should be close to identity
        assert!((h[0][0] - 1.0).abs() < 0.02);
        assert!(h[0][1].abs() < 0.02);
        assert!(h[1][0].abs() < 0.02);
        assert!((h[1][1] - 1.0).abs() < 0.02);
        verify_doubly_stochastic_2x2(&h, 1e-6).unwrap();
    }

    #[test]
    fn test_n2_all_logits_doubly_stochastic() {
        // Test that H_res is doubly stochastic for arbitrary logits
        for logit in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let mhc = MhcLiteN2 {
                alpha_logit: logit,
                pre_logits: [0.0; 2],
                pre_bias: [0.0; 2],
                post_logits: [0.0; 2],
                post_bias: [0.0; 2],
            };
            let h = mhc.h_res();
            verify_doubly_stochastic_2x2(&h, 1e-6).unwrap();
        }
    }

    #[test]
    fn test_n2_forward_pass() {
        let mhc = MhcLiteN2::new_identity();
        let dim_c = 4;

        // Input: single stream [1, 4]
        let x_single = vec![1.0, 2.0, 3.0, 4.0];

        // Expand to 2 streams
        let x = MhcLiteN2::expand_input(&x_single, dim_c);
        assert_eq!(x.len(), 8);

        // Prepare input for layer
        let layer_in = mhc.prepare_input(&x, dim_c);
        assert_eq!(layer_in.len(), 4);

        // Simulate layer output (identity for test)
        let layer_out = layer_in.clone();

        // Apply mHC update
        let x_out = mhc.apply(&x, &layer_out, dim_c);
        assert_eq!(x_out.len(), 8);

        // Collapse back
        let y = MhcLiteN2::collapse_output(&x_out, dim_c);
        assert_eq!(y.len(), 4);

        // With identity init, output should be close to input + layer_contribution
        // (won't be exact due to sigmoid scaling)
        for &val in y.iter() {
            assert!(val.is_finite(), "Output {} is not finite", val);
        }
    }

    #[test]
    fn test_serialization_roundtrip_n2() {
        let mhc = MhcLiteN2 {
            alpha_logit: 1.5,
            pre_logits: [0.3, -0.7],
            pre_bias: [0.1, 0.2],
            post_logits: [-0.5, 0.8],
            post_bias: [0.4, -0.1],
        };
        let bytes = mhc.to_bytes();
        assert_eq!(bytes.len(), 36);
        let mhc2 = MhcLiteN2::from_bytes(&bytes).unwrap();
        assert!((mhc.alpha_logit - mhc2.alpha_logit).abs() < 1e-7);
        for i in 0..2 {
            assert!((mhc.pre_logits[i] - mhc2.pre_logits[i]).abs() < 1e-7);
            assert!((mhc.pre_bias[i] - mhc2.pre_bias[i]).abs() < 1e-7);
            assert!((mhc.post_logits[i] - mhc2.post_logits[i]).abs() < 1e-7);
            assert!((mhc.post_bias[i] - mhc2.post_bias[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_n2_from_bytes_too_short() {
        let data = vec![0u8; 35]; // one byte short
        assert!(MhcLiteN2::from_bytes(&data).is_err());
    }

    #[test]
    fn test_n2_expand_collapse_roundtrip() {
        let dim_c = 8;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expanded = MhcLiteN2::expand_input(&x, dim_c);
        let collapsed = MhcLiteN2::collapse_output(&expanded, dim_c);
        for i in 0..dim_c {
            assert!(
                (x[i] - collapsed[i]).abs() < 1e-6,
                "Mismatch at {}: {} vs {}",
                i,
                x[i],
                collapsed[i]
            );
        }
    }

    #[test]
    fn test_n2_from_weights() {
        let mhc = MhcLiteN2::from_weights(2.0, [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]);
        assert!((mhc.alpha_logit - 2.0).abs() < 1e-7);
        assert!((mhc.pre_logits[0] - 0.1).abs() < 1e-7);
        assert!((mhc.post_bias[1] - 0.8).abs() < 1e-7);
    }
}
