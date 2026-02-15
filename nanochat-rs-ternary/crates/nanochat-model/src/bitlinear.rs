//! BitLinear — Ternary linear layer for inference.
//!
//! Quantizes activations to INT8 on the fly, then dispatches to ternary GEMV.

use ternary_core::planar::PlanarWeights;
use ternary_kernels::cpu;

/// Ternary linear layer (inference only).
///
/// Weights are packed ternary in PlanarWeights format.
/// Activations are quantized to INT8 per-token before GEMV.
#[derive(Debug, Clone)]
pub struct BitLinear {
    pub pw: PlanarWeights,
    pub rows: usize, // output dim
    pub cols: usize, // input dim
}

impl BitLinear {
    /// Create from pre-packed PlanarWeights.
    pub fn new(pw: PlanarWeights) -> Self {
        let rows = pw.rows;
        let cols = pw.cols;
        Self { pw, rows, cols }
    }

    /// Create from row-major float weights (for testing).
    /// Quantizes to ternary and packs.
    pub fn from_float(weights: &[f32], rows: usize, cols: usize, group_size: usize) -> Self {
        let pw = PlanarWeights::from_row_major(weights, rows, cols, group_size);
        Self::new(pw)
    }

    /// Forward pass: quantize activations -> ternary GEMV -> output.
    ///
    /// x:   [cols] float activations
    /// out: [rows] output buffer
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);

        // Per-token absmax quantization to INT8
        let (x_q, act_scale) = quantize_activations_i8(x);

        // Ternary GEMV
        cpu::gemv(&self.pw, &x_q, act_scale, out);
    }

    /// Batched forward pass for prefill: process `seq_len` tokens.
    ///
    /// x_batch:   [seq_len * cols] — flattened token activations
    /// seq_len:   number of tokens
    /// out_batch: [seq_len * rows] — flattened output buffer
    pub fn forward_batch(&self, x_batch: &[f32], seq_len: usize, out_batch: &mut [f32]) {
        assert_eq!(x_batch.len(), seq_len * self.cols);
        assert_eq!(out_batch.len(), seq_len * self.rows);

        for t in 0..seq_len {
            let x = &x_batch[t * self.cols..(t + 1) * self.cols];
            let out = &mut out_batch[t * self.rows..(t + 1) * self.rows];
            self.forward(x, out);
        }
    }
}

/// Quantize activations to INT8 using per-token absmax scaling.
///
/// Returns (quantized_i8, scale) where scale = absmax / 127.
fn quantize_activations_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let absmax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if absmax < 1e-8 {
        return (vec![0i8; x.len()], 0.0);
    }

    let scale = absmax / 127.0;
    let inv_scale = 127.0 / absmax;
    let x_q: Vec<i8> = x
        .iter()
        .map(|&v| {
            let q = (v * inv_scale).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();

    (x_q, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_activations() {
        let x = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let (x_q, scale) = quantize_activations_i8(&x);

        // absmax = 1.0, scale = 1/127
        assert!((scale - 1.0 / 127.0).abs() < 1e-6);
        assert_eq!(x_q[0], 127);
        assert_eq!(x_q[1], -127);
        assert_eq!(x_q[4], 0);
    }

    #[test]
    fn test_quantize_activations_zero() {
        let x = vec![0.0, 0.0, 0.0];
        let (x_q, scale) = quantize_activations_i8(&x);
        assert_eq!(scale, 0.0);
        assert!(x_q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_bitlinear_forward() {
        let rows = 128;
        let cols = 128;
        let gs = 128;

        // Create weights — all +1
        let weights = vec![1.0f32; rows * cols];
        let bl = BitLinear::from_float(&weights, rows, cols, gs);

        // Input: constant 0.5
        let x = vec![0.5f32; cols];
        let mut out = vec![0.0f32; rows];
        bl.forward(&x, &mut out);

        // Output should be finite and non-zero
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {}", v);
        }
        assert!(
            out.iter().any(|&v| v != 0.0),
            "all-zero output — GEMV not working"
        );
    }

    #[test]
    fn test_bitlinear_forward_batch() {
        let rows = 128;
        let cols = 128;
        let gs = 128;
        let seq_len = 4;

        let weights = vec![1.0f32; rows * cols];
        let bl = BitLinear::from_float(&weights, rows, cols, gs);

        // Create batch input
        let mut x_batch = vec![0.0f32; seq_len * cols];
        for t in 0..seq_len {
            for c in 0..cols {
                x_batch[t * cols + c] = (t as f32 + 1.0) * 0.1;
            }
        }

        // Batched forward
        let mut out_batch = vec![0.0f32; seq_len * rows];
        bl.forward_batch(&x_batch, seq_len, &mut out_batch);

        // Compare with per-token forward
        for t in 0..seq_len {
            let x = &x_batch[t * cols..(t + 1) * cols];
            let mut out_single = vec![0.0f32; rows];
            bl.forward(x, &mut out_single);

            let batch_out = &out_batch[t * rows..(t + 1) * rows];
            for r in 0..rows {
                assert!(
                    (batch_out[r] - out_single[r]).abs() < 1e-6,
                    "token {}, row {}: batch={} vs single={}",
                    t,
                    r,
                    batch_out[r],
                    out_single[r]
                );
            }
        }
    }

    #[test]
    fn test_bitlinear_matches_scalar_ref() {
        let rows = 128;
        let cols = 128;
        let gs = 128;

        // Random-ish weights
        let mut weights = vec![0.0f32; rows * cols];
        #[allow(clippy::needless_range_loop)]
        // Index needed for deterministic pseudo-random generation
        for i in 0..weights.len() {
            let v = ((i as u32).wrapping_mul(2654435761) >> 16) % 200;
            weights[i] = v as f32 / 100.0 - 1.0;
        }
        let bl = BitLinear::from_float(&weights, rows, cols, gs);

        // Input
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 / cols as f32) - 0.5).collect();

        // Forward via BitLinear
        let mut out_bl = vec![0.0f32; rows];
        bl.forward(&x, &mut out_bl);

        // Forward via scalar ref with same quantized activations
        let (x_q, act_scale) = quantize_activations_i8(&x);
        let mut out_ref = vec![0.0f32; rows];
        cpu::gemv_scalar_ref(&bl.pw, &x_q, act_scale, &mut out_ref);

        let max_diff: f32 = out_bl
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "BitLinear vs scalar ref max diff: {}",
            max_diff
        );
    }
}
