//! Ternary quantization functions for QAT training.
//!
//! Ports the Python `ternary_qat.py` quantization logic to Candle tensors.

use candle_core::{Result, Tensor};

/// Per-group absmean quantization: weights -> ternary {-1,0,+1} + scales.
///
/// Algorithm (BitNet b1.58):
///   1. Reshape weights into groups of `group_size`
///   2. Per-group scale = mean(|w|), clamped to avoid div-by-zero
///   3. Normalize: w_scaled = w / scale
///   4. Round + clip to {-1, 0, +1}
///
/// Returns (w_ternary, scales) where w_ternary has same shape as input.
pub fn absmean_quantize(w: &Tensor, group_size: usize) -> Result<(Tensor, Tensor)> {
    let orig_shape = w.dims().to_vec();
    let numel = w.elem_count();

    // Flatten and pad if needed
    let (w_flat, pad_len) = if !numel.is_multiple_of(group_size) {
        let pad_len = group_size - (numel % group_size);
        let w_flat = w.reshape((numel,))?;
        let zeros = Tensor::zeros((pad_len,), w.dtype(), w.device())?;
        let w_padded = Tensor::cat(&[&w_flat, &zeros], 0)?;
        (w_padded, pad_len)
    } else {
        (w.reshape((numel,))?, 0)
    };

    let n_groups = w_flat.elem_count() / group_size;
    let w_grouped = w_flat.reshape((n_groups, group_size))?;

    // Per-group scale = mean(|w|), clamped
    let scales = w_grouped
        .abs()?
        .mean_keepdim(1)?
        .clamp(1e-8, f64::MAX)?
        .squeeze(1)?; // [n_groups]

    // Normalize, round, clip to ternary
    let scales_expanded = scales.unsqueeze(1)?; // [n_groups, 1]
    let w_scaled = w_grouped.broadcast_div(&scales_expanded)?;
    let w_ternary = w_scaled.round()?.clamp(-1.0, 1.0)?;

    // Remove padding and reshape
    let w_ternary = if pad_len > 0 {
        w_ternary.reshape((n_groups * group_size,))?.narrow(0, 0, numel)?.reshape(orig_shape)?
    } else {
        w_ternary.reshape((n_groups * group_size,))?.reshape(orig_shape)?
    };

    Ok((w_ternary, scales))
}

/// Dequantize: ternary + scales -> approximate FP32.
pub fn dequantize_ternary(
    w_ternary: &Tensor,
    scales: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let orig_shape = w_ternary.dims().to_vec();
    let numel = w_ternary.elem_count();

    let (w_flat, pad_len) = if !numel.is_multiple_of(group_size) {
        let pad_len = group_size - (numel % group_size);
        let w_flat = w_ternary.reshape((numel,))?;
        let zeros = Tensor::zeros((pad_len,), w_ternary.dtype(), w_ternary.device())?;
        (Tensor::cat(&[&w_flat, &zeros], 0)?, pad_len)
    } else {
        (w_ternary.reshape((numel,))?, 0)
    };

    let n_groups = w_flat.elem_count() / group_size;
    let w_grouped = w_flat.reshape((n_groups, group_size))?;
    let scales_expanded = scales.unsqueeze(1)?;
    let w_recon = w_grouped.broadcast_mul(&scales_expanded)?;

    if pad_len > 0 {
        w_recon.reshape((n_groups * group_size,))?.narrow(0, 0, numel)?.reshape(orig_shape)
    } else {
        w_recon.reshape((n_groups * group_size,))?.reshape(orig_shape)
    }
}

/// Per-token absmax activation quantization to simulated INT8.
///
/// For each token (last dimension), find absmax and scale to [-127, 127].
/// Returns (x_q, scales) where x_q has same shape, scales has shape [..., 1].
pub fn per_token_absmax_quantize(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let last_dim = x.dims().len() - 1;
    // absmax over last dimension, keepdim
    let absmax = x.abs()?.max_keepdim(last_dim)?.clamp(1e-8, f64::MAX)?;
    let scales = (&absmax / 127.0)?;
    let inv_scales = (127.0 / &absmax)?;
    let x_q = x.broadcast_mul(&inv_scales)?.round()?.clamp(-127.0, 127.0)?;
    Ok((x_q, scales))
}

/// Dequantize INT8 activations back to approximate FP32.
pub fn dequantize_activations(x_q: &Tensor, scales: &Tensor) -> Result<Tensor> {
    x_q.broadcast_mul(scales)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_absmean_produces_ternary() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0.0f32, 0.5, (256, 128), &device)?;
        let (w_t, scales) = absmean_quantize(&w, 128)?;

        // Check shape preserved
        assert_eq!(w_t.dims(), &[256, 128]);

        // Check values are ternary: all values should be in {-1, 0, 1}
        let w_flat = w_t.flatten_all()?.to_vec1::<f32>()?;
        for &v in &w_flat {
            assert!(
                (v - (-1.0)).abs() < 1e-6 || (v - 0.0).abs() < 1e-6 || (v - 1.0).abs() < 1e-6,
                "Non-ternary value: {}", v
            );
        }

        // Scales should be positive
        let s = scales.to_vec1::<f32>()?;
        for &v in &s {
            assert!(v > 0.0, "Scale should be positive: {}", v);
        }

        Ok(())
    }

    #[test]
    fn test_absmean_roundtrip_error() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0.0f32, 0.5, (128, 128), &device)?;
        let (w_t, scales) = absmean_quantize(&w, 128)?;
        let w_recon = dequantize_ternary(&w_t, &scales, 128)?;

        // Error should be bounded — ternary can't represent fine values
        let diff = (&w - &w_recon)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 2.0, "Max reconstruction error too large: {}", diff);

        // Shape preserved
        assert_eq!(w_recon.dims(), &[128, 128]);
        Ok(())
    }

    #[test]
    fn test_per_token_absmax_range() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (4, 16, 256), &device)?;
        let (x_q, scales) = per_token_absmax_quantize(&x)?;

        // Shape preserved
        assert_eq!(x_q.dims(), &[4, 16, 256]);
        assert_eq!(scales.dims(), &[4, 16, 1]);

        // Values should be in [-127, 127]
        let max_val = x_q.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_val <= 127.0 + 1e-6, "Quantized value out of range: {}", max_val);

        // Scales should be positive
        let s_flat = scales.flatten_all()?.to_vec1::<f32>()?;
        for &v in &s_flat {
            assert!(v > 0.0, "Scale should be positive: {}", v);
        }

        Ok(())
    }

    #[test]
    fn test_absmean_handles_padding() -> Result<()> {
        let device = Device::Cpu;
        // 300 elements, not divisible by 128
        let w = Tensor::randn(0.0f32, 1.0, (300,), &device)?;
        let (w_t, scales) = absmean_quantize(&w, 128)?;

        // Shape should be preserved
        assert_eq!(w_t.dims(), &[300]);

        // Should have ceil(300/128) = 3 groups
        assert_eq!(scales.dims(), &[3]);

        // Values should still be ternary
        let w_flat = w_t.to_vec1::<f32>()?;
        for &v in &w_flat {
            assert!(
                (v - (-1.0)).abs() < 1e-6 || (v - 0.0).abs() < 1e-6 || (v - 1.0).abs() < 1e-6,
                "Non-ternary value: {}", v
            );
        }

        Ok(())
    }

    #[test]
    fn test_dequantize_activations_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 64), &device)?;
        let (x_q, scales) = per_token_absmax_quantize(&x)?;
        let x_deq = dequantize_activations(&x_q, &scales)?;

        // Error should be bounded by quantization step size
        let diff = (&x - &x_deq)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let max_scale = scales.max_all()?.to_scalar::<f32>()?;
        assert!(diff < max_scale * 1.5, "Dequantization error too large: {}", diff);

        Ok(())
    }
}
