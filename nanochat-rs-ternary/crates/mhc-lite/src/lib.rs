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

pub mod n2;
pub mod n4;
pub mod verify;
pub mod io;

pub use n2::MhcLiteN2;
pub use n4::MhcLiteN4;
pub use verify::{verify_doubly_stochastic, verify_doubly_stochastic_2x2, composite_amax_gain};
pub use io::{MhcFileHeader, load_mhc_file, save_mhc_file, MhcLayerParams};

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
    let mut out = [0.0f32; 24];
    let mut sum = 0.0f32;

    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        *o = (l - max_val).exp();
        sum += *o;
    }

    let inv_sum = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv_sum;
    }
    out
}
