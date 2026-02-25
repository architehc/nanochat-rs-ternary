//! BitLinear — Ternary linear layer for inference.
//!
//! Quantizes activations to INT8 on the fly, then dispatches to ternary GEMV or GEMM.

use ternary_core::planar::PlanarWeights;
use ternary_kernels::cpu;

/// Ternary linear layer (inference only).
///
/// Weights are packed ternary in PlanarWeights format.
/// Activations are quantized to INT8 per-token before GEMV.
#[derive(Debug)]
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
        let mut x_q_workspace = vec![0i8; self.cols];
        self.forward_with_workspace(x, &mut x_q_workspace, out);
    }

    /// Forward pass with caller-provided INT8 workspace.
    ///
    /// Reuses `x_q_workspace` for activation quantization to avoid per-call
    /// heap allocations in hot inference paths.
    pub fn forward_with_workspace(&self, x: &[f32], x_q_workspace: &mut [i8], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        assert_eq!(x_q_workspace.len(), self.cols);

        // Per-token absmax quantization to INT8
        let act_scale = quantize_activations_i8_into(x, x_q_workspace);
        self.forward_quantized(x_q_workspace, act_scale, out);
    }

    /// Forward pass for already-quantized activations.
    pub fn forward_quantized(&self, x_q: &[i8], act_scale: f32, out: &mut [f32]) {
        assert_eq!(x_q.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        cpu::gemv(&self.pw, x_q, act_scale, out);
    }

    /// Batched forward pass for prefill: process `seq_len` tokens.
    pub fn forward_batch(&self, x_batch: &[f32], seq_len: usize, out_batch: &mut [f32]) {
        let mut x_q_batch = vec![0i8; seq_len * self.cols];
        let mut scales_batch = vec![0.0f32; seq_len];
        self.forward_batch_with_workspaces(x_batch, seq_len, out_batch, &mut x_q_batch, &mut scales_batch);
    }

    /// Batched forward pass with caller-provided quantization and scale workspaces.
    pub fn forward_batch_with_workspaces(
        &self,
        x_batch: &[f32],
        seq_len: usize,
        out_batch: &mut [f32],
        x_q_batch: &mut [i8],
        scales_batch: &mut [f32],
    ) {
        assert_eq!(x_batch.len(), seq_len * self.cols);
        assert_eq!(out_batch.len(), seq_len * self.rows);
        assert_eq!(x_q_batch.len(), seq_len * self.cols);
        assert_eq!(scales_batch.len(), seq_len);

        for t in 0..seq_len {
            let x = &x_batch[t * self.cols..(t + 1) * self.cols];
            let x_q = &mut x_q_batch[t * self.cols..(t + 1) * self.cols];
            scales_batch[t] = quantize_activations_i8_into(x, x_q);
        }

        // Call parallel GEMM kernel
        cpu::gemm(&self.pw, x_q_batch, scales_batch, out_batch);
    }

    /// Clone the layer and place the weight allocations on a specific NUMA node.
    pub fn clone_to_node(&self, node: usize) -> Self {
        Self {
            pw: self.pw.clone_to_node(node),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Clone for BitLinear {
    fn clone(&self) -> Self {
        Self {
            pw: self.pw.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

/// Quantize activations to INT8 using a caller-provided output buffer.
pub fn quantize_activations_i8_into(x: &[f32], x_q_out: &mut [i8]) -> f32 {
    assert_eq!(x_q_out.len(), x.len());
    let absmax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if absmax < 1e-8 {
        x_q_out.fill(0);
        return 0.0;
    }

    let scale = absmax / 127.0;
    let inv_scale = 127.0 / absmax;
    for (dst, &v) in x_q_out.iter_mut().zip(x.iter()) {
        let q = (v * inv_scale).round();
        *dst = q.clamp(-127.0, 127.0) as i8;
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantize_activations_i8(x: &[f32]) -> (Vec<i8>, f32) {
        let mut x_q = vec![0i8; x.len()];
        let scale = quantize_activations_i8_into(x, &mut x_q);
        (x_q, scale)
    }

    #[test]
    fn test_quantize_activations() {
        let x = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let (x_q, scale) = quantize_activations_i8(&x);

        assert!((scale - 1.0 / 127.0).abs() < 1e-6);
        assert_eq!(x_q[0], 127);
        assert_eq!(x_q[1], -127);
        assert_eq!(x_q[4], 0);
    }

    #[test]
    fn test_bitlinear_forward() {
        let rows = 128;
        let cols = 128;
        let gs = 128;
        let weights = vec![1.0f32; rows * cols];
        let bl = BitLinear::from_float(&weights, rows, cols, gs);
        let x = vec![0.5f32; cols];
        let mut out = vec![0.0f32; rows];
        bl.forward(&x, &mut out);
        for &v in &out {
            assert!(v.is_finite());
        }
        assert!(out.iter().any(|&v| v != 0.0));
    }
}
