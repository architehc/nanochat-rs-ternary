// mhc-lite — Manifold-Constrained Hyper-Connections (mHC-lite variant)
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

pub mod analysis;
pub mod io;
pub mod n2;
pub mod n4;
pub mod verify;

pub use analysis::{AdaptiveInit, LayerStats, MhcAnalyzer, ModelAnalysis};
pub use io::{load_mhc_file, save_mhc_file, MhcFileHeader, MhcLayerParams};
pub use n2::MhcLiteN2;
pub use n4::MhcLiteN4;
pub use verify::{composite_amax_gain, verify_doubly_stochastic, verify_doubly_stochastic_2x2};

// ============================================================================
// Shared Utilities
// ============================================================================

/// Sigmoid activation function.
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax over 24 elements.
pub(crate) fn softmax_24(logits: &[f32; 24]) -> [f32; 24] {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        // Degenerate case (all -inf/NaN): keep DS invariants by falling back to uniform.
        eprintln!(
            "WARNING: softmax_24 received degenerate input (max={:.6}), returning uniform distribution (possible upstream bug)",
            max_val
        );
        return [1.0 / 24.0; 24];
    }

    let mut out = [0.0f32; 24];
    let mut sum = 0.0f32;

    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        let shifted = l - max_val;
        *o = if shifted.is_finite() {
            shifted.exp()
        } else {
            0.0
        };
        sum += *o;
    }

    // Guard against division by zero or near-zero denominator.
    // An epsilon of 1e-30 prevents inf from extremely small positive sums
    // while being small enough to not affect normal softmax precision.
    const SOFTMAX_EPS: f32 = 1e-30;

    if !sum.is_finite() || sum <= 0.0 {
        eprintln!(
            "WARNING: softmax_24 exp sum is degenerate (sum={:.6}), returning uniform distribution",
            sum
        );
        return [1.0 / 24.0; 24];
    }

    let inv_sum = 1.0 / (sum + SOFTMAX_EPS);
    for o in out.iter_mut() {
        *o *= inv_sum;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::softmax_24;

    #[test]
    fn test_softmax_24_all_neg_inf_is_finite_and_normalized() {
        let logits = [f32::NEG_INFINITY; 24];
        let out = softmax_24(&logits);
        let sum: f32 = out.iter().sum();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
    }

    #[test]
    fn test_softmax_24_all_nan_is_finite_and_normalized() {
        let logits = [f32::NAN; 24];
        let out = softmax_24(&logits);
        let sum: f32 = out.iter().sum();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
    }
}
