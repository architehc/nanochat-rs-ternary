// n4.rs — MhcLiteN4: 4-stream full BvN parameterization (24 permutation matrices)
//
// Birkhoff-von-Neumann theorem: any doubly stochastic matrix is a convex
// combination of permutation matrices. For n=4, there are 4! = 24 permutations.
//
// H_res = sum_{k=0}^{23} theta_k * P_k
// where theta = softmax(logits), P_k are the 24 permutation matrices of S_4.
//
// This is EXACT -- no approximation gap, no Sinkhorn iterations.
//
// Total learnable params per layer: 24 (res) + 4 (pre) + 4 (pre_bias) + 4 (post) + 4 (post_bias) = 40

use crate::{sigmoid, softmax_24};
use std::fmt;

/// Error while deserializing `MhcLiteN4` from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum N4DecodeError {
    TooShort { expected: usize, actual: usize },
}

impl fmt::Display for N4DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            N4DecodeError::TooShort { expected, actual } => {
                write!(
                    f,
                    "N4 decode failed: expected at least {} bytes, got {} bytes",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for N4DecodeError {}

/// All 24 permutation matrices of S_4, stored as index arrays.
/// perm[k] = [p0, p1, p2, p3] means row i maps to column perm[k][i].
pub const PERMS_S4: [[usize; 4]; 24] = [
    [0, 1, 2, 3], // identity
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0], // full reversal
];

#[derive(Clone, Debug)]
pub struct MhcLiteN4 {
    /// BvN logits: softmax -> convex combination weights over 24 permutations
    pub res_logits: [f32; 24],

    /// Pre-projection logits and bias
    pub pre_logits: [f32; 4],
    pub pre_bias: [f32; 4],

    /// Post-projection logits and bias
    pub post_logits: [f32; 4],
    pub post_bias: [f32; 4],
}

impl MhcLiteN4 {
    /// Identity initialization: all weight on the identity permutation (index 0)
    pub fn new_identity() -> Self {
        let mut res_logits = [0.0f32; 24];
        // Heavy weight on identity permutation at index 0
        res_logits[0] = 10.0;

        Self {
            res_logits,
            pre_logits: [0.0; 4],
            pre_bias: [0.5; 4],
            post_logits: [0.0; 4],
            post_bias: [0.5; 4],
        }
    }

    /// Load from serialized weights
    pub fn from_weights(
        res_logits: [f32; 24],
        pre_logits: [f32; 4],
        pre_bias: [f32; 4],
        post_logits: [f32; 4],
        post_bias: [f32; 4],
    ) -> Self {
        Self {
            res_logits,
            pre_logits,
            pre_bias,
            post_logits,
            post_bias,
        }
    }

    /// Compute H_res: 4x4 doubly stochastic matrix (exact)
    /// theta = softmax(res_logits)
    /// H_res = sum_k theta_k * P_k
    pub fn h_res(&self) -> [[f32; 4]; 4] {
        let theta = softmax_24(&self.res_logits);
        let mut mat = [[0.0f32; 4]; 4];

        for k in 0..24 {
            let perm = &PERMS_S4[k];
            let w = theta[k];
            for i in 0..4 {
                mat[i][perm[i]] += w;
            }
        }
        mat
    }

    /// Compute H_pre: 4-element non-negative vector
    pub fn h_pre(&self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = sigmoid(self.pre_logits[i] + self.pre_bias[i]);
        }
        out
    }

    /// Compute H_post: 4-element non-negative vector, 2x scaled
    pub fn h_post(&self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = 2.0 * sigmoid(self.post_logits[i] + self.post_bias[i]);
        }
        out
    }

    /// Prepare layer input: mix 4 streams -> 1 stream
    /// Input:  [batch, 4*C]
    /// Output: [batch, C]
    pub fn prepare_input(&self, x: &[f32], dim_c: usize) -> Vec<f32> {
        let h_pre = self.h_pre();
        let batch = x.len() / (4 * dim_c);
        let mut out = vec![0.0f32; batch * dim_c];

        for b in 0..batch {
            let base = b * 4 * dim_c;
            let o = &mut out[b * dim_c..(b + 1) * dim_c];

            for i in 0..dim_c {
                o[i] = h_pre[0] * x[base + i]
                    + h_pre[1] * x[base + dim_c + i]
                    + h_pre[2] * x[base + 2 * dim_c + i]
                    + h_pre[3] * x[base + 3 * dim_c + i];
            }
        }
        out
    }

    /// Apply residual update with training-matching semantics.
    ///
    /// Applies stream-local residual first, then mixes streams:
    /// x_out = H_res @ (x + H_post^T * layer_output)
    ///
    /// x:            [batch, 4*C]
    /// layer_output: [batch, C]
    /// Returns:      [batch, 4*C]
    pub fn apply(&self, x: &[f32], layer_output: &[f32], dim_c: usize) -> Vec<f32> {
        let h_res = self.h_res();
        let h_post = self.h_post();
        assert!(dim_c > 0, "dim_c must be > 0");
        assert!(
            x.len().is_multiple_of(4 * dim_c),
            "x.len() ({}) must be divisible by 4*dim_c ({})",
            x.len(),
            4 * dim_c
        );
        let batch = x.len() / (4 * dim_c);
        assert_eq!(
            layer_output.len(),
            batch * dim_c,
            "layer_output.len() ({}) must equal batch*dim_c ({})",
            layer_output.len(),
            batch * dim_c
        );
        let mut out = vec![0.0f32; batch * 4 * dim_c];

        for b in 0..batch {
            let x_base = b * 4 * dim_c;
            let ly = &layer_output[b * dim_c..(b + 1) * dim_c];

            for i in 0..dim_c {
                let mut with_res = [0.0f32; 4];
                for s in 0..4 {
                    with_res[s] = x[x_base + s * dim_c + i] + h_post[s] * ly[i];
                }
                for s in 0..4 {
                    let o = &mut out[x_base + s * dim_c..x_base + (s + 1) * dim_c];
                    let mut val = 0.0f32;
                    for (t, wr) in with_res.iter().enumerate() {
                        val += h_res[s][t] * wr;
                    }
                    o[i] = val;
                }
            }
        }
        out
    }

    /// Initialize: duplicate single stream to n=4 streams
    pub fn expand_input(x: &[f32], dim_c: usize) -> Vec<f32> {
        let batch = x.len() / dim_c;
        let mut out = vec![0.0f32; batch * 4 * dim_c];
        for b in 0..batch {
            let src = &x[b * dim_c..(b + 1) * dim_c];
            for s in 0..4 {
                out[b * 4 * dim_c + s * dim_c..b * 4 * dim_c + (s + 1) * dim_c]
                    .copy_from_slice(src);
            }
        }
        out
    }

    /// Collapse: average 4 streams -> 1
    pub fn collapse_output(x: &[f32], dim_c: usize) -> Vec<f32> {
        let batch = x.len() / (4 * dim_c);
        let mut out = vec![0.0f32; batch * dim_c];
        for b in 0..batch {
            let o = &mut out[b * dim_c..(b + 1) * dim_c];
            for s in 0..4 {
                let src = &x[b * 4 * dim_c + s * dim_c..b * 4 * dim_c + (s + 1) * dim_c];
                for i in 0..dim_c {
                    o[i] += 0.25 * src[i];
                }
            }
        }
        out
    }

    /// Deserialize N=4 parameters from 160 bytes (little-endian f32)
    ///
    /// Layout: [res_logits: 24*f32][pre_logits: 4*f32][pre_bias: 4*f32]
    ///         [post_logits: 4*f32][post_bias: 4*f32]
    pub fn from_bytes(data: &[u8]) -> Result<Self, N4DecodeError> {
        if data.len() < 160 {
            return Err(N4DecodeError::TooShort {
                expected: 160,
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

        let mut res_logits = [0.0f32; 24];
        for (i, rl) in res_logits.iter_mut().enumerate() {
            *rl = f(i * 4);
        }

        let mut pre_logits = [0.0f32; 4];
        let mut pre_bias = [0.0f32; 4];
        let mut post_logits = [0.0f32; 4];
        let mut post_bias = [0.0f32; 4];

        for i in 0..4 {
            pre_logits[i] = f(96 + i * 4);
            pre_bias[i] = f(112 + i * 4);
            post_logits[i] = f(128 + i * 4);
            post_bias[i] = f(144 + i * 4);
        }

        Ok(Self {
            res_logits,
            pre_logits,
            pre_bias,
            post_logits,
            post_bias,
        })
    }

    /// Serialize N=4 parameters to 160 bytes (little-endian f32)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(160);
        for v in &self.res_logits {
            out.extend_from_slice(&v.to_le_bytes());
        }
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
    use crate::verify::verify_doubly_stochastic;

    #[test]
    fn test_n4_identity_init() {
        let mhc = MhcLiteN4::new_identity();
        let h = mhc.h_res();
        // Should be close to identity
        for (i, row) in h.iter().enumerate() {
            assert!((row[i] - 1.0).abs() < 0.01, "Diagonal [{i}] = {}", row[i]);
        }
        verify_doubly_stochastic(&h, 1e-5).unwrap();
    }

    #[test]
    fn test_n4_random_logits_doubly_stochastic() {
        // BvN guarantees exactness for ANY logits
        let mhc = MhcLiteN4 {
            res_logits: [
                1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.5, -0.5, 1.5, -1.5, 0.3, -0.3, 0.7, -0.7, 1.2,
                -1.2, 0.1, -0.1, 2.0, -2.0, 0.8, -0.8, 1.8, -1.8,
            ],
            pre_logits: [0.0; 4],
            pre_bias: [0.0; 4],
            post_logits: [0.0; 4],
            post_bias: [0.0; 4],
        };
        let h = mhc.h_res();
        verify_doubly_stochastic(&h, 1e-6).unwrap();
    }

    #[test]
    fn test_n4_composite_gain_bounded() {
        // Key mHC property: composite gain stays bounded
        let mut matrices = Vec::new();
        for seed in 0..64 {
            let mut logits = [0.0f32; 24];
            for (i, logit) in logits.iter_mut().enumerate() {
                // Pseudo-random logits
                *logit = ((seed * 24 + i) as f32 * 0.7).sin() * 3.0;
            }
            let mhc = MhcLiteN4 {
                res_logits: logits,
                pre_logits: [0.0; 4],
                pre_bias: [0.0; 4],
                post_logits: [0.0; 4],
                post_bias: [0.0; 4],
            };
            matrices.push(mhc.h_res());
        }

        let gain = crate::verify::composite_amax_gain(&matrices);
        // For 64 doubly stochastic matrices, gain should be <= 1.0 (exact)
        // mHC-lite gives exact DS, so composite is also DS, gain exactly 1.0
        assert!(gain <= 1.0 + 1e-4, "Composite gain {} exceeds bound", gain);
    }

    #[test]
    fn test_n4_forward_pass() {
        let mhc = MhcLiteN4::new_identity();
        let dim_c = 4;

        // Input: single stream [1, 4]
        let x_single = vec![1.0, 2.0, 3.0, 4.0];

        // Expand to 4 streams
        let x = MhcLiteN4::expand_input(&x_single, dim_c);
        assert_eq!(x.len(), 16);

        // Prepare input for layer
        let layer_in = mhc.prepare_input(&x, dim_c);
        assert_eq!(layer_in.len(), 4);

        // Simulate layer output (identity for test)
        let layer_out = layer_in.clone();

        // Apply mHC update
        let x_out = mhc.apply(&x, &layer_out, dim_c);
        assert_eq!(x_out.len(), 16);

        // Collapse back
        let y = MhcLiteN4::collapse_output(&x_out, dim_c);
        assert_eq!(y.len(), 4);

        // Output should be finite
        for &val in y.iter() {
            assert!(val.is_finite(), "Output {} is not finite", val);
        }
    }

    #[test]
    fn test_n4_apply_matches_residual_first_semantics() {
        let mut res_logits = [-100.0f32; 24];
        res_logits[0] = 0.0; // identity
        res_logits[6] = 0.0; // swap stream 0/1
        let mhc = MhcLiteN4::from_weights(
            res_logits,
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0, 2.0, 0.0, -2.0],
        );

        let x = vec![1.0f32, 2.0, 3.0, 4.0]; // [batch=1, 4*C], C=1
        let layer_output = vec![10.0f32];
        let out = mhc.apply(&x, &layer_output, 1);

        let h_res = mhc.h_res();
        let h_post = mhc.h_post();
        for s in 0..4 {
            let expected: f32 = (0..4)
                .map(|t| h_res[s][t] * (x[t] + h_post[t] * layer_output[0]))
                .sum();
            assert!(
                (out[s] - expected).abs() < 1e-5,
                "stream {} mismatch: got {}, expected {}",
                s,
                out[s],
                expected
            );
        }
    }

    #[test]
    fn test_serialization_roundtrip_n4() {
        let mhc = MhcLiteN4::new_identity();
        let bytes = mhc.to_bytes();
        assert_eq!(bytes.len(), 160);
        let mhc2 = MhcLiteN4::from_bytes(&bytes).unwrap();
        for i in 0..24 {
            assert!((mhc.res_logits[i] - mhc2.res_logits[i]).abs() < 1e-7);
        }
        for i in 0..4 {
            assert!((mhc.pre_logits[i] - mhc2.pre_logits[i]).abs() < 1e-7);
            assert!((mhc.pre_bias[i] - mhc2.pre_bias[i]).abs() < 1e-7);
            assert!((mhc.post_logits[i] - mhc2.post_logits[i]).abs() < 1e-7);
            assert!((mhc.post_bias[i] - mhc2.post_bias[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_n4_from_bytes_too_short() {
        let data = vec![0u8; 159]; // one byte short
        assert!(MhcLiteN4::from_bytes(&data).is_err());
    }

    #[test]
    fn test_n4_expand_collapse_roundtrip() {
        let dim_c = 4;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let expanded = MhcLiteN4::expand_input(&x, dim_c);
        let collapsed = MhcLiteN4::collapse_output(&expanded, dim_c);
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
    fn test_perms_s4_are_valid_permutations() {
        // Each perm should be a permutation of [0,1,2,3]
        for (k, perm) in PERMS_S4.iter().enumerate() {
            let mut sorted = *perm;
            sorted.sort();
            assert_eq!(
                sorted,
                [0, 1, 2, 3],
                "PERMS_S4[{}] is not a valid permutation",
                k
            );
        }
        // Should have exactly 24 distinct permutations
        let mut set: Vec<[usize; 4]> = PERMS_S4.to_vec();
        set.sort();
        set.dedup();
        assert_eq!(
            set.len(),
            24,
            "PERMS_S4 does not have 24 distinct permutations"
        );
    }

    #[test]
    fn test_n4_from_weights() {
        let res_logits = [1.0; 24];
        let pre_logits = [0.1, 0.2, 0.3, 0.4];
        let pre_bias = [0.5, 0.6, 0.7, 0.8];
        let post_logits = [-0.1, -0.2, -0.3, -0.4];
        let post_bias = [-0.5, -0.6, -0.7, -0.8];
        let mhc = MhcLiteN4::from_weights(res_logits, pre_logits, pre_bias, post_logits, post_bias);
        assert_eq!(mhc.res_logits, [1.0; 24]);
        assert!((mhc.pre_logits[0] - 0.1).abs() < 1e-7);
        assert!((mhc.post_bias[3] - (-0.8)).abs() < 1e-7);
    }

    #[test]
    fn test_n4_h_pre() {
        let mhc = MhcLiteN4::new_identity();
        let h_pre = mhc.h_pre();
        assert_eq!(h_pre.len(), 4);
        // With default logits=0, bias=0.5: sigmoid(0.5) ≈ 0.622
        for &v in &h_pre {
            assert!(v > 0.0, "h_pre should be positive, got {}", v);
            assert!(v < 1.0, "h_pre should be < 1.0, got {}", v);
            assert!(
                (v - 0.6225).abs() < 0.01,
                "h_pre ≈ sigmoid(0.5) ≈ 0.622, got {}",
                v
            );
        }
    }

    #[test]
    fn test_n4_h_post() {
        let mhc = MhcLiteN4::new_identity();
        let h_post = mhc.h_post();
        assert_eq!(h_post.len(), 4);
        // With default logits=0, bias=0.5: 2*sigmoid(0.5) ≈ 1.245
        for &v in &h_post {
            assert!(v > 0.0, "h_post should be positive, got {}", v);
            assert!(v < 2.0, "h_post should be < 2.0, got {}", v);
            assert!(
                (v - 1.245).abs() < 0.01,
                "h_post ≈ 2*sigmoid(0.5) ≈ 1.245, got {}",
                v
            );
        }
    }

    #[test]
    fn test_n4_many_random_logits_doubly_stochastic() {
        // Property test: 1000 random logit vectors should all produce valid DS matrices
        for seed in 0..1000 {
            let mut logits = [0.0f32; 24];
            for (i, logit) in logits.iter_mut().enumerate() {
                *logit = ((seed * 24 + i) as f32 * 0.7).sin() * 5.0;
            }
            let mhc = MhcLiteN4 {
                res_logits: logits,
                pre_logits: [0.0; 4],
                pre_bias: [0.0; 4],
                post_logits: [0.0; 4],
                post_bias: [0.0; 4],
            };
            let h = mhc.h_res();
            verify_doubly_stochastic(&h, 1e-5).unwrap_or_else(|e| panic!("Seed {}: {}", seed, e));
        }
    }
}
