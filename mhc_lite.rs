// mhc_lite.rs — Manifold-Constrained Hyper-Connections (mHC-lite variant)
//
// Exact Birkhoff-von-Neumann parameterization of doubly stochastic matrices.
// No Sinkhorn-Knopp iterations needed — doubly stochastic by construction.
//
// Reference: mHC (arXiv 2512.24880), mHC-lite (arXiv 2601.05732)
//
// For nanochat-rs ternary inference:
//   - mHC matrices are FP32 (tiny: 24 params/layer for n=4, 1 param/layer for n=2)
//   - All linear layers remain ternary 2-bit packed
//   - Overhead: <0.00004% of layer compute
//
// Two implementations:
//   MhcLiteN2 — 2-stream, 1 learnable param, recommended starting point
//   MhcLiteN4 — 4-stream, full BvN with 24 permutation matrices

#![allow(dead_code)]

// ============================================================================
// N=2: The Minimal mHC (Recommended Starting Point)
// ============================================================================
//
// Only 2 permutation matrices exist for n=2:
//   P0 = [[1,0],[0,1]]  (identity)
//   P1 = [[0,1],[1,0]]  (swap)
//
// H_res = alpha * I + (1-alpha) * J
// where alpha = sigmoid(logit), so H_res is always doubly stochastic.
//
// Total learnable params per layer: 1 (res) + 2 (pre) + 2 (pre_bias) + 2 (post) + 2 (post_bias) = 9

#[derive(Clone, Debug)]
pub struct MhcLiteN2 {
    // Residual mixing: sigmoid(alpha_logit) interpolates I vs swap
    pub alpha_logit: f32,

    // Pre-projection: how streams feed into layer F
    // h_pre = sigmoid(pre_logits + pre_bias), shape [2]
    pub pre_logits: [f32; 2],
    pub pre_bias: [f32; 2],

    // Post-projection: how layer output distributes back to streams
    // h_post = 2 * sigmoid(post_logits + post_bias), shape [2]
    pub post_logits: [f32; 2],
    pub post_bias: [f32; 2],
}

impl MhcLiteN2 {
    /// Create with identity initialization (alpha=1 → pure identity residual)
    pub fn new_identity() -> Self {
        Self {
            // Large positive logit → sigmoid ≈ 1 → identity matrix
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

    /// Prepare layer input: mix 2 streams → 1 stream
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

    /// Apply residual update:
    /// x_out = H_res @ x + H_post^T * layer_output
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

            let o0 = &mut out[b * 2 * dim_c..b * 2 * dim_c + dim_c];
            let o1 = &mut out[b * 2 * dim_c + dim_c..b * 2 * dim_c + 2 * dim_c];

            for i in 0..dim_c {
                // Residual mixing
                o0[i] = h_res[0][0] * s0[i] + h_res[0][1] * s1[i] + h_post[0] * ly[i];
                o1[i] = h_res[1][0] * s0[i] + h_res[1][1] * s1[i] + h_post[1] * ly[i];
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
}

// ============================================================================
// N=4: Full BvN Parameterization (24 Permutation Matrices)
// ============================================================================
//
// Birkhoff-von-Neumann theorem: any doubly stochastic matrix is a convex
// combination of permutation matrices. For n=4, there are 4! = 24 permutations.
//
// H_res = Σ_{k=0}^{23} θ_k · P_k
// where θ = softmax(logits), P_k are the 24 permutation matrices of S_4.
//
// This is EXACT — no approximation gap, no Sinkhorn iterations.
//
// Total learnable params per layer: 24 (res) + 4 (pre) + 4 (pre_bias) + 4 (post) + 4 (post_bias) = 40

/// All 24 permutation matrices of S_4, stored as index arrays.
/// perm[k] = [p0, p1, p2, p3] means row i maps to column perm[k][i].
const PERMS_S4: [[usize; 4]; 24] = [
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
    /// BvN logits: softmax → convex combination weights over 24 permutations
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
    /// θ = softmax(res_logits)
    /// H_res = Σ_k θ_k · P_k
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
    #[inline]
    pub fn h_pre(&self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = sigmoid(self.pre_logits[i] + self.pre_bias[i]);
        }
        out
    }

    /// Compute H_post: 4-element non-negative vector, 2x scaled
    #[inline]
    pub fn h_post(&self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = 2.0 * sigmoid(self.post_logits[i] + self.post_bias[i]);
        }
        out
    }

    /// Prepare layer input: mix 4 streams → 1 stream
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

    /// Apply residual update:
    /// x_out[s] = Σ_t H_res[s][t] * x[t] + H_post[s] * layer_output
    ///
    /// x:            [batch, 4*C]
    /// layer_output: [batch, C]
    /// Returns:      [batch, 4*C]
    pub fn apply(&self, x: &[f32], layer_output: &[f32], dim_c: usize) -> Vec<f32> {
        let h_res = self.h_res();
        let h_post = self.h_post();
        let batch = x.len() / (4 * dim_c);
        let mut out = vec![0.0f32; batch * 4 * dim_c];

        for b in 0..batch {
            let x_base = b * 4 * dim_c;
            let ly = &layer_output[b * dim_c..(b + 1) * dim_c];

            for s in 0..4 {
                let o = &mut out[x_base + s * dim_c..x_base + (s + 1) * dim_c];
                for i in 0..dim_c {
                    let mut val = h_post[s] * ly[i];
                    for t in 0..4 {
                        val += h_res[s][t] * x[x_base + t * dim_c + i];
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

    /// Collapse: average 4 streams → 1
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
}

// ============================================================================
// Verification: Doubly Stochastic Property Checker
// ============================================================================

/// Verify that a matrix is doubly stochastic within tolerance
pub fn verify_doubly_stochastic(mat: &[[f32; 4]; 4], tol: f32) -> Result<(), String> {
    // Check non-negativity
    for i in 0..4 {
        for j in 0..4 {
            if mat[i][j] < -tol {
                return Err(format!(
                    "Negative entry: mat[{}][{}] = {}",
                    i, j, mat[i][j]
                ));
            }
        }
    }

    // Check row sums = 1
    for i in 0..4 {
        let row_sum: f32 = mat[i].iter().sum();
        if (row_sum - 1.0).abs() > tol {
            return Err(format!("Row {} sum = {} (expected 1.0)", i, row_sum));
        }
    }

    // Check column sums = 1
    for j in 0..4 {
        let col_sum: f32 = (0..4).map(|i| mat[i][j]).sum();
        if (col_sum - 1.0).abs() > tol {
            return Err(format!("Col {} sum = {} (expected 1.0)", j, col_sum));
        }
    }

    Ok(())
}

/// Verify n=2 matrix
pub fn verify_doubly_stochastic_2x2(mat: &[[f32; 2]; 2], tol: f32) -> Result<(), String> {
    for i in 0..2 {
        for j in 0..2 {
            if mat[i][j] < -tol {
                return Err(format!(
                    "Negative entry: mat[{}][{}] = {}",
                    i, j, mat[i][j]
                ));
            }
        }
    }
    let r0: f32 = mat[0][0] + mat[0][1];
    let r1: f32 = mat[1][0] + mat[1][1];
    let c0: f32 = mat[0][0] + mat[1][0];
    let c1: f32 = mat[0][1] + mat[1][1];

    if (r0 - 1.0).abs() > tol || (r1 - 1.0).abs() > tol {
        return Err(format!("Row sums: [{}, {}] (expected 1.0)", r0, r1));
    }
    if (c0 - 1.0).abs() > tol || (c1 - 1.0).abs() > tol {
        return Err(format!("Col sums: [{}, {}] (expected 1.0)", c0, c1));
    }
    Ok(())
}

/// Compute composite gain (Amax Gain Magnitude) for a sequence of H_res matrices.
/// This is the diagnostic metric from the mHC paper.
/// Amax = max(max_row_sum, max_col_sum) of the composite product.
pub fn composite_amax_gain(matrices: &[[[f32; 4]; 4]]) -> f32 {
    let mut composite = [[0.0f32; 4]; 4];
    // Start with identity
    for i in 0..4 {
        composite[i][i] = 1.0;
    }

    for mat in matrices {
        let prev = composite;
        composite = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    composite[i][j] += prev[i][k] * mat[k][j];
                }
            }
        }
    }

    // Amax = max of (max row sum, max col sum)
    let max_row_sum = (0..4)
        .map(|i| composite[i].iter().map(|v| v.abs()).sum::<f32>())
        .fold(0.0f32, f32::max);

    let max_col_sum = (0..4)
        .map(|j| (0..4).map(|i| composite[i][j].abs()).sum::<f32>())
        .fold(0.0f32, f32::max);

    f32::max(max_row_sum, max_col_sum)
}

// ============================================================================
// Utilities
// ============================================================================

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax over 24 elements
fn softmax_24(logits: &[f32; 24]) -> [f32; 24] {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [0.0f32; 24];
    let mut sum = 0.0f32;

    for i in 0..24 {
        out[i] = (logits[i] - max_val).exp();
        sum += out[i];
    }

    let inv_sum = 1.0 / sum;
    for i in 0..24 {
        out[i] *= inv_sum;
    }
    out
}

// ============================================================================
// Weight I/O: Load mHC parameters from binary checkpoint
// ============================================================================

/// Serialized format for a single layer's mHC-lite N=4 parameters.
/// Total: 40 * 4 = 160 bytes per layer.
///
/// Layout: [res_logits: 24*f32][pre_logits: 4*f32][pre_bias: 4*f32]
///         [post_logits: 4*f32][post_bias: 4*f32]
impl MhcLiteN4 {
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 160 {
            return None;
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
        for i in 0..24 {
            res_logits[i] = f(i * 4);
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

        Some(Self {
            res_logits,
            pre_logits,
            pre_bias,
            post_logits,
            post_bias,
        })
    }

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

/// Serialized format for N=2: 36 bytes per layer
impl MhcLiteN2 {
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 36 {
            return None;
        }
        let f = |offset: usize| -> f32 {
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        Some(Self {
            alpha_logit: f(0),
            pre_logits: [f(4), f(8)],
            pre_bias: [f(12), f(16)],
            post_logits: [f(20), f(24)],
            post_bias: [f(28), f(32)],
        })
    }

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_n4_identity_init() {
        let mhc = MhcLiteN4::new_identity();
        let h = mhc.h_res();
        // Should be close to identity
        for i in 0..4 {
            assert!(
                (h[i][i] - 1.0).abs() < 0.01,
                "Diagonal [{i}] = {}",
                h[i][i]
            );
        }
        verify_doubly_stochastic(&h, 1e-5).unwrap();
    }

    #[test]
    fn test_n4_random_logits_doubly_stochastic() {
        // BvN guarantees exactness for ANY logits
        let mhc = MhcLiteN4 {
            res_logits: [
                1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.5, -0.5, 1.5, -1.5, 0.3, -0.3, 0.7, -0.7,
                1.2, -1.2, 0.1, -0.1, 2.0, -2.0, 0.8, -0.8, 1.8, -1.8,
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
            for i in 0..24 {
                // Pseudo-random logits
                logits[i] = ((seed * 24 + i) as f32 * 0.7).sin() * 3.0;
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

        let gain = composite_amax_gain(&matrices);
        // For 64 doubly stochastic matrices, gain should be ≤ 1.0 (exact)
        // mHC-lite gives exact DS, so composite is also DS, gain exactly 1.0
        assert!(
            gain <= 1.0 + 1e-4,
            "Composite gain {} exceeds bound",
            gain
        );
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
        for i in 0..4 {
            assert!(y[i].is_finite(), "Output {} is not finite", y[i]);
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
        let mhc2 = MhcLiteN2::from_bytes(&bytes).unwrap();
        assert!((mhc.alpha_logit - mhc2.alpha_logit).abs() < 1e-7);
        for i in 0..2 {
            assert!((mhc.pre_logits[i] - mhc2.pre_logits[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_serialization_roundtrip_n4() {
        let mhc = MhcLiteN4::new_identity();
        let bytes = mhc.to_bytes();
        let mhc2 = MhcLiteN4::from_bytes(&bytes).unwrap();
        for i in 0..24 {
            assert!((mhc.res_logits[i] - mhc2.res_logits[i]).abs() < 1e-7);
        }
    }
}
