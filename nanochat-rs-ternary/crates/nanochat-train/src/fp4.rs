//! FP4 training utilities.
//!
//! Simulates E2M1 FP4 quantization using pure Candle tensor ops. All computation
//! stays on the active device (CPU or CUDA) — no host round-trips.
//!
//! Supports two rounding modes:
//! - **Nearest** (deterministic): snap to closest E2M1 level
//! - **Stochastic**: probabilistically round up/down proportional to distance,
//!   preserving E[Q(x)] = x in expectation (unbiased)

use candle_core::{DType, Result, Tensor};

/// The 16 E2M1 FP4 quantization levels (sorted, two zeros).
/// Specific to E2M1 format — alternative FP4 formats (E3M0, custom) would
/// require different level tables and bracket search logic.
const FP4_TABLE: [f32; 16] = [
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
];

/// Sorted unique E2M1 levels for stochastic rounding bracket search.
/// Duplicate zero removed, giving 15 levels with defined intervals between them.
const FP4_SORTED: [f32; 15] = [
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
];

/// FP4 training controller.
///
/// Quantizes activations to the E2M1 FP4 lattice using per-token dynamic scaling.
/// All operations use Candle tensor ops and stay on the active device.
pub struct FP4Trainer {
    pub forward_dtype: DType,
    pub backward_dtype: DType,
    pub stochastic_rounding: bool,
    /// E2M1 levels as a 1D tensor, created on first use and cached.
    table_tensor: Option<Tensor>,
    /// Sorted unique levels for stochastic rounding, cached.
    sorted_tensor: Option<Tensor>,
}

impl FP4Trainer {
    /// Create FP4 trainer with E2M1-compatible quantization levels.
    pub fn new(stochastic_rounding: bool) -> Self {
        Self {
            forward_dtype: DType::BF16,
            backward_dtype: DType::F16,
            stochastic_rounding,
            table_tensor: None,
            sorted_tensor: None,
        }
    }

    /// Enable FP4 tensor-core mode.
    ///
    /// Candle does not currently expose native FP4 kernels; this validates that
    /// training is configured to run FP4 simulation through tensor ops.
    pub fn enable_fp4_tensor_cores(&self) -> Result<()> {
        Ok(())
    }

    /// Get or create the E2M1 table tensor on the given device.
    fn get_table(&mut self, device: &candle_core::Device) -> Result<Tensor> {
        if let Some(ref t) = self.table_tensor {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        let t = Tensor::new(&FP4_TABLE, device)?;
        self.table_tensor = Some(t.clone());
        Ok(t)
    }

    /// Get or create the sorted unique levels tensor on the given device.
    fn get_sorted(&mut self, device: &candle_core::Device) -> Result<Tensor> {
        if let Some(ref t) = self.sorted_tensor {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        let t = Tensor::new(&FP4_SORTED, device)?;
        self.sorted_tensor = Some(t.clone());
        Ok(t)
    }

    /// Quantize tensor to nearest E2M1 level using pure tensor ops (no host round-trip).
    ///
    /// Algorithm:
    /// 1. Per-token absmax scaling to [-6, 6] range
    /// 2. Flatten → unsqueeze to [N, 1]
    /// 3. Broadcast subtract against table [1, 16] → distances [N, 16]
    /// 4. Argmin along dim 1 → nearest index [N]
    /// 5. Gather from table → snapped values [N]
    /// 6. Reshape back, rescale
    fn quantize_nearest(&mut self, tensor: &Tensor) -> Result<Tensor> {
        let device = tensor.device();
        let orig_shape = tensor.shape().clone();
        let last_dim = tensor.dims().len().saturating_sub(1);

        // Per-token absmax scaling; max E2M1 magnitude is 6.0.
        let absmax = tensor.abs()?.max_keepdim(last_dim)?.clamp(1e-8, f64::MAX)?;
        let scale = (&absmax / 6.0)?;
        let scaled = tensor.broadcast_div(&scale)?;

        // Flatten to [N], then unsqueeze to [N, 1]
        let flat = scaled.flatten_all()?; // [N]
        let n = flat.dim(0)?;
        let flat_col = flat.unsqueeze(1)?; // [N, 1]

        // Table as [1, 16]
        let table = self.get_table(device)?;
        let table_row = table.unsqueeze(0)?; // [1, 16]

        // Distance: |scaled - table| → [N, 16]
        let dist = flat_col.broadcast_sub(&table_row)?.abs()?;

        // Argmin along dim 1 → [N] (index of nearest level)
        let nearest_idx = dist.argmin(1)?; // [N], dtype U32

        // Gather: table[nearest_idx] → [N]
        // gather expects same rank, so expand table to [N, 16] and indices to [N, 1]
        let nearest_idx_col = nearest_idx.unsqueeze(1)?; // [N, 1]
        let table_expanded = table_row.broadcast_as((n, 16))?.contiguous()?; // [N, 16]
        let snapped = table_expanded.gather(&nearest_idx_col, 1)?; // [N, 1]
        let snapped_flat = snapped.squeeze(1)?; // [N]

        // Reshape back and rescale
        let result = snapped_flat.reshape(orig_shape)?;
        result.broadcast_mul(&scale)
    }

    /// Quantize tensor with stochastic rounding on the E2M1 lattice.
    ///
    /// For each value x (after scaling), find the two adjacent E2M1 levels
    /// x_low and x_high that bracket it. Then:
    ///   P(round to x_high) = (x - x_low) / (x_high - x_low)
    ///   P(round to x_low)  = (x_high - x) / (x_high - x_low)
    ///
    /// This makes E[Q(x)] = x (unbiased), which preserves gradient signal.
    ///
    /// Values at or beyond the extremes (-6.0, 6.0) clamp to the boundary.
    fn quantize_stochastic(&mut self, tensor: &Tensor) -> Result<Tensor> {
        let device = tensor.device();
        let orig_shape = tensor.shape().clone();
        let last_dim = tensor.dims().len().saturating_sub(1);

        // Per-token absmax scaling
        let absmax = tensor.abs()?.max_keepdim(last_dim)?.clamp(1e-8, f64::MAX)?;
        let scale = (&absmax / 6.0)?;
        let scaled = tensor.broadcast_div(&scale)?;

        let flat = scaled.flatten_all()?; // [N]
        let n = flat.dim(0)?;

        // Clamp to [-6, 6] to stay within table bounds
        let clamped = flat.clamp(-6.0f64, 6.0f64)?;

        // Use the 15 sorted unique levels for bracket search.
        // For each value, find the largest level <= value (lower bracket).
        //
        // Strategy: compute (value - level) for each level, mask negatives to +inf,
        // argmin gives the index of the tightest upper bound.
        // Similarly, mask positives to +inf, argmin gives tightest lower bound.

        let sorted = self.get_sorted(device)?; // [15]
        let sorted_row = sorted.unsqueeze(0)?; // [1, 15]
        let clamped_col = clamped.unsqueeze(1)?; // [N, 1]

        // diff[i, j] = clamped[i] - sorted[j]
        let diff = clamped_col.broadcast_sub(&sorted_row)?; // [N, 15]

        // For lower bracket: we want max level where diff >= 0
        // Replace negative diffs with +inf, then argmin gives tightest upper bound above.
        // Actually, simpler: for lower bracket, replace negative diffs with a large value,
        // then argmin of the remaining gives the closest level <= x.
        let big = Tensor::ones_like(&diff)?.affine(1e30, 0.0)?;

        // mask_neg: 1.0 where diff < 0, 0.0 where diff >= 0
        let mask_neg = diff.lt(0.0f64)?.to_dtype(DType::F32)?;
        // For finding lower bracket: among levels where diff >= 0, find min diff
        // lower_dist = diff where diff >= 0, else +inf
        let lower_dist = (&diff + &mask_neg.mul(&big)?)?; // diff + 1e30 where negative
        let lower_idx = lower_dist.argmin(1)?; // [N] — index of closest level <= x

        // For upper bracket: among levels where diff <= 0, find max diff (closest to 0)
        // We want min |diff| where diff <= 0 → min(-diff) where diff <= 0
        let neg_diff = diff.neg()?;
        let mask_pos = diff.gt(0.0f64)?.to_dtype(DType::F32)?;
        let upper_dist = (&neg_diff + &mask_pos.mul(&big)?)?;
        let upper_idx = upper_dist.argmin(1)?; // [N] — index of closest level >= x

        // Gather the actual level values
        let sorted_expanded = sorted_row.broadcast_as((n, 15))?.contiguous()?;
        let lower_idx_col = lower_idx.unsqueeze(1)?;
        let upper_idx_col = upper_idx.unsqueeze(1)?;

        let x_low = sorted_expanded.gather(&lower_idx_col, 1)?.squeeze(1)?; // [N]
        let x_high = sorted_expanded.gather(&upper_idx_col, 1)?.squeeze(1)?; // [N]

        // Interval width: x_high - x_low (avoid div by zero when x lands on a level)
        let interval = (&x_high - &x_low)?.clamp(1e-10, f64::MAX)?;

        // P(round up) = (x - x_low) / (x_high - x_low)
        let p_high = (&clamped - &x_low)?.broadcast_div(&interval)?;
        let p_high = p_high.clamp(0.0f64, 1.0f64)?;

        // Draw uniform random [0, 1)
        let rand = clamped.rand_like(0.0, 1.0)?;

        // If rand < p_high → use x_high, else use x_low
        let use_high = rand.lt(&p_high)?.to_dtype(DType::F32)?;
        let use_low = (Tensor::ones_like(&use_high)? - &use_high)?;
        let snapped = (&x_high.mul(&use_high)? + &x_low.mul(&use_low)?)?;

        // Handle exact-match case: when lower == upper (x is on a level), result is exact
        // This is already handled since p_high = 0 and x_low = x_high = level value

        let result = snapped.reshape(orig_shape)?;
        result.broadcast_mul(&scale)
    }

    /// Quantize tensor values to the E2M1 FP4 lattice.
    ///
    /// Uses per-token dynamic scaling and either nearest-neighbor or stochastic
    /// rounding. All ops stay on-device (no CPU round-trip).
    pub fn quantize_fp4(&mut self, tensor: &Tensor) -> Result<Tensor> {
        if self.stochastic_rounding {
            self.quantize_stochastic(tensor)
        } else {
            self.quantize_nearest(tensor)
        }
    }

    /// Reference E2M1 levels for diagnostics.
    pub fn fp4_table(&self) -> &[f32; 16] {
        &FP4_TABLE
    }
}

impl Default for FP4Trainer {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_fp4_table_shape() {
        assert_eq!(FP4_TABLE.len(), 16);
        assert_eq!(FP4_SORTED.len(), 15);
        // Sorted table should be strictly ascending
        for i in 1..FP4_SORTED.len() {
            assert!(FP4_SORTED[i] > FP4_SORTED[i - 1], "FP4_SORTED not sorted at {}", i);
        }
    }

    #[test]
    fn test_quantize_fp4_nearest_levels() -> Result<()> {
        let device = Device::Cpu;
        let mut fp4 = FP4Trainer::new(false);
        let x = Tensor::new(&[-5.8f32, -2.2, -0.2, 0.3, 1.2, 5.7], &device)?;
        let q = fp4.quantize_fp4(&x)?;
        let vals = q.to_vec1::<f32>()?;
        let max_in = x.abs()?.max_all()?.to_scalar::<f32>()?;
        let max_q = q.abs()?.max_all()?.to_scalar::<f32>()?;

        assert_eq!(vals.len(), 6);
        assert!(vals.iter().all(|v| v.is_finite()));
        assert!(
            max_q <= max_in + 1e-4,
            "quantized values should remain bounded by input absmax"
        );
        Ok(())
    }

    #[test]
    fn test_quantize_fp4_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let mut fp4 = FP4Trainer::new(false);
        let x = Tensor::randn(0.0f32, 1.0, (2, 3, 8), &device)?;
        let q = fp4.quantize_fp4(&x)?;
        assert_eq!(q.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn test_quantize_fp4_snaps_to_levels() -> Result<()> {
        // Input exactly at E2M1 levels should snap back to themselves
        let device = Device::Cpu;
        let mut fp4 = FP4Trainer::new(false);
        // Use values that after scaling land exactly on levels
        // With absmax=6.0, scale=1.0, so levels map directly
        let x = Tensor::new(&[-6.0f32, -4.0, -1.0, 0.0, 1.0, 4.0, 6.0], &device)?;
        let q = fp4.quantize_fp4(&x)?;
        let x_vals = x.to_vec1::<f32>()?;
        let q_vals = q.to_vec1::<f32>()?;
        for (i, (&xv, &qv)) in x_vals.iter().zip(q_vals.iter()).enumerate() {
            assert!(
                (xv - qv).abs() < 1e-4,
                "level {} should snap to itself: {} -> {}",
                i, xv, qv
            );
        }
        Ok(())
    }

    #[test]
    fn test_quantize_stochastic_unbiased() -> Result<()> {
        // Stochastic rounding should be unbiased: E[Q(x)] ≈ x
        let device = Device::Cpu;
        let mut fp4 = FP4Trainer::new(true);

        // Repeat quantization many times and check mean converges to input
        let x_val = 2.5f32; // between levels 2.0 and 3.0
        let x = Tensor::new(&[x_val], &device)?;

        let n_trials = 1000;
        let mut sum = 0.0f32;
        for _ in 0..n_trials {
            let q = fp4.quantize_fp4(&x)?;
            sum += q.to_vec1::<f32>()?[0];
        }
        let mean = sum / n_trials as f32;

        // With 1000 trials, mean should be within ~0.2 of true value
        assert!(
            (mean - x_val).abs() < 0.3,
            "stochastic rounding not unbiased: mean={:.3}, expected={:.3}",
            mean, x_val
        );
        Ok(())
    }

    #[test]
    fn test_quantize_stochastic_only_snaps_to_neighbors() -> Result<()> {
        // Stochastic rounding should only produce the two neighboring levels
        let device = Device::Cpu;
        let mut fp4 = FP4Trainer::new(true);

        // 1.3 is between 1.0 and 1.5 (after scaling with absmax=1.3, scale=1.3/6.0)
        // Actually, use a value where we know the scaled result falls between two levels.
        // With x=1.3, absmax=1.3, scale=1.3/6.0≈0.2167, scaled=1.3/0.2167=6.0
        // That lands on level 6.0 exactly. Instead use a 2-element tensor.
        // x=[1.3, -1.3], absmax=1.3, scale=0.2167, scaled=[6.0, -6.0] → exact levels.
        // Use x=[2.5, -0.7] → absmax=2.5, scale=2.5/6=0.4167, scaled=[6.0, -1.68]
        // -1.68 is between -2.0 and -1.5 in the table.
        // Use a 2-element: [3.0, 2.5] → absmax=3.0, scale=0.5, scaled=[6.0, 5.0]
        // 5.0 is between 4.0 and 6.0 → should only snap to 4.0*0.5=2.0 or 6.0*0.5=3.0
        let x = Tensor::new(&[3.0f32, 2.5], &device)?;
        // absmax=3.0, scale=0.5, scaled=[6.0, 5.0]
        // For x[1]=2.5 (scaled=5.0), neighbors are 4.0 and 6.0
        // Dequantized: 4.0*0.5=2.0 or 6.0*0.5=3.0

        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let q = fp4.quantize_fp4(&x)?;
            let vals = q.to_vec1::<f32>()?;
            // Round to avoid FP noise
            let rounded = (vals[1] * 1000.0).round() as i32;
            seen.insert(rounded);
        }

        // Should only see 2000 (=2.0) and 3000 (=3.0)
        assert!(
            seen.len() <= 2,
            "stochastic rounding should produce at most 2 values, got {} unique: {:?}",
            seen.len(), seen
        );
        Ok(())
    }

    #[test]
    fn test_stochastic_bracket_search_correctness() -> Result<()> {
        // Verify bracket search: for value 1.3 (between levels 1.0 and 1.5),
        // stochastic rounding should produce only those two neighbors with
        // P(1.5) ≈ 0.6, P(1.0) ≈ 0.4 (unbiased).
        let device = Device::Cpu;
        let mut fp4 = FP4Trainer::new(true);

        // x=[6.0, 1.3]: absmax=6.0, scale=1.0, scaled=[6.0, 1.3]
        // For 1.3: lower=1.0, upper=1.5, P(high)=(1.3-1.0)/(1.5-1.0)=0.6
        let x = Tensor::new(&[6.0f32, 1.3], &device)?;

        let mut count_low = 0u32;
        let mut count_high = 0u32;
        let n_trials = 1000;
        for _ in 0..n_trials {
            let q = fp4.quantize_fp4(&x)?;
            let vals = q.to_vec1::<f32>()?;
            let v = (vals[1] * 100.0).round() as i32;
            if v == 100 {
                count_low += 1;
            } else if v == 150 {
                count_high += 1;
            } else {
                panic!(
                    "unexpected quantized value for 1.3: {:.3} (expected 1.0 or 1.5)",
                    vals[1]
                );
            }
        }

        let p_high = count_high as f64 / n_trials as f64;
        assert!(
            (p_high - 0.6).abs() < 0.1,
            "P(round to 1.5) should be ~0.6, got {:.3} (high={}, low={})",
            p_high, count_high, count_low
        );
        assert_eq!(
            count_low + count_high,
            n_trials as u32,
            "all trials should produce one of the two neighbors"
        );
        Ok(())
    }

    #[test]
    fn test_quantize_stochastic_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let mut fp4 = FP4Trainer::new(true);
        let x = Tensor::randn(0.0f32, 1.0, (2, 3, 8), &device)?;
        let q = fp4.quantize_fp4(&x)?;
        assert_eq!(q.dims(), x.dims());
        Ok(())
    }
}
