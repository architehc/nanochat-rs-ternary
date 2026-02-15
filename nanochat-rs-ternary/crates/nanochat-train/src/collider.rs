//! Collider: Cross-layer Activation Sparsity for Token Filtering
//!
//! Based on: "Cross-layer Activation Sparsity for Token Filtering" (arXiv:2502.00340)
//!
//! Collider achieves 35% faster backprop by:
//! 1. Computing per-token importance scores based on loss
//! 2. Filtering out low-importance tokens during backward pass
//! 3. Transforming sparse GEMMs to dense operations

use candle_core::{DType, Result, Tensor, D};
use std::cmp::Ordering;

/// Token filtering via cross-layer activation sparsity.
///
/// Filters tokens with low importance scores to speed up backward pass.
/// Importance is computed from per-token cross-entropy loss.
pub struct Collider {
    /// Importance threshold (0-1): tokens below this are filtered
    threshold: f64,

    /// Target sparsity ratio (0-1): fraction of tokens to filter
    sparsity_target: f64,

    /// Apply filtering during backward pass.
    filter_backward: bool,

    /// Transform sparse GEMMs to dense (gather important tokens).
    transform_gemm: bool,
}

impl Collider {
    /// Create new Collider with default settings.
    ///
    /// # Arguments
    /// * `threshold` - Importance threshold (0-1), default 0.3
    /// * `sparsity_target` - Target fraction to filter (0-1), default 0.35
    ///
    /// # Example
    /// ```no_run
    /// use nanochat_train::collider::Collider;
    ///
    /// let collider = Collider::new(0.3, 0.35);
    /// // Will filter ~35% of tokens with importance < 0.3
    /// ```
    pub fn new(threshold: f64, sparsity_target: f64) -> Self {
        Self {
            threshold,
            sparsity_target,
            filter_backward: true,
            transform_gemm: true,
        }
    }

    /// Compute per-token importance scores from loss.
    ///
    /// # Arguments
    /// * `logits` - Model predictions [batch, seq_len, vocab_size]
    /// * `targets` - Target token IDs [batch, seq_len]
    ///
    /// # Returns
    /// Normalized importance scores [batch, seq_len] in range [0, 1]
    pub fn compute_importance(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Compute per-token cross-entropy loss
        let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;
        let losses = self.per_token_cross_entropy(&log_probs, targets)?;

        // Normalize to [0, 1] using min-max scaling
        let min_loss = losses.min_keepdim(1)?;
        let max_loss = losses.max_keepdim(1)?;
        let range = ((&max_loss - &min_loss)? + 1e-8)?;

        let normalized = losses.broadcast_sub(&min_loss)?.broadcast_div(&range)?;

        Ok(normalized)
    }

    /// Compute per-token cross-entropy loss (not reduced).
    fn per_token_cross_entropy(&self, log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _vocab_size) = log_probs.dims3()?;
        let (target_batch, target_seq) = targets.dims2()?;
        if batch_size != target_batch || seq_len != target_seq {
            return Err(candle_core::Error::Msg(format!(
                "Target shape mismatch: logits=[{}, {}, _], targets=[{}, {}]",
                batch_size, seq_len, target_batch, target_seq
            )));
        }

        // Vectorized path: gather target log-probabilities in one op on device.
        // log_probs: [B, S, V], target_ids: [B, S, 1] -> selected: [B, S, 1].
        let target_ids = targets.to_dtype(DType::U32)?.unsqueeze(2)?;
        let selected = log_probs.gather(&target_ids, D::Minus1)?;

        selected.squeeze(2)?.neg()
    }

    /// Create binary mask for important tokens.
    ///
    /// # Arguments
    /// * `importance` - Importance scores [batch, seq_len]
    ///
    /// # Returns
    /// Binary mask [batch, seq_len]: 1 for kept tokens, 0 for filtered
    pub fn create_mask(&self, importance: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = importance.dims2()?;
        let importance_rows = importance.to_vec2::<f32>()?;
        let mut mask_flat = Vec::with_capacity(batch * seq_len);

        for row in &importance_rows {
            let keep_target = (((1.0 - self.sparsity_target.clamp(0.0, 1.0)) * row.len() as f64)
                .round() as usize)
                .clamp(1, row.len());

            let mut sorted = row.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let dyn_idx = row.len().saturating_sub(keep_target);
            let dynamic_threshold = sorted[dyn_idx] as f64;
            let effective_threshold = self.threshold.max(dynamic_threshold);

            for &value in row {
                mask_flat.push(if (value as f64) >= effective_threshold {
                    1.0f32
                } else {
                    0.0f32
                });
            }
        }

        Tensor::from_vec(mask_flat, (batch, seq_len), importance.device())?.to_dtype(DType::F32)
    }

    /// Compute flat token indices for kept tokens.
    ///
    /// Returned indices refer to flattened `[batch, seq]` ordering.
    pub fn kept_token_indices(&self, mask: &Tensor) -> Result<Tensor> {
        // Single host transfer for the full mask to avoid per-token sync overhead.
        let mask_vals = mask.flatten_all()?.to_vec1::<f32>()?;
        let mut kept = Vec::with_capacity(mask_vals.len());
        for (idx, &v) in mask_vals.iter().enumerate() {
            if v >= 0.5 {
                kept.push(idx as u32);
            }
        }

        // Keep at least one token to avoid empty-tensor reductions.
        if kept.is_empty() {
            if let Some((best_idx, _)) = mask_vals
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
            {
                kept.push(best_idx as u32);
            } else {
                kept.push(0);
            }
        }

        let kept_len = kept.len();
        Tensor::from_vec(kept, kept_len, mask.device())
    }

    /// Compact hidden activations using kept token indices.
    ///
    /// Input hidden shape: `[batch, seq, hidden_dim]`.
    /// Output hidden shape: `[n_kept, hidden_dim]`.
    pub fn compact_hidden(&self, hidden: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, hidden_dim) = hidden.dims3()?;
        let indices = self.kept_token_indices(mask)?;
        let hidden_flat = hidden.reshape((batch * seq_len, hidden_dim))?;
        let compact = hidden_flat.index_select(&indices, 0)?;
        Ok((compact, indices))
    }

    /// Compact targets `[batch, seq]` to `[n_kept]` using flattened kept indices.
    pub fn compact_targets(&self, targets: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = targets.dims2()?;
        let flat = targets.reshape(batch * seq_len)?;
        flat.index_select(indices, 0)
    }

    /// Transform sparse token path into a compact dense GEMM input.
    ///
    /// Returns:
    /// - compact hidden activations `[n_kept, hidden_dim]`
    /// - kept token indices over flattened `[batch, seq]`
    pub fn transform_sparse_gemms(
        &self,
        hidden: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.compact_hidden(hidden, mask)
    }

    /// Prepare sparse-backward inputs by filtering/compacting hidden + targets.
    ///
    /// This is the main Collider "sparse backward" entrypoint used by training.
    /// When `transform_gemm` is enabled, low-importance rows are removed so the LM
    /// head/loss path runs on a smaller dense matrix.
    pub fn sparse_backward(
        &self,
        hidden: &Tensor,
        targets: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if !self.filter_backward {
            let (batch, seq_len, hidden_dim) = hidden.dims3()?;
            let hidden_flat = hidden.reshape((batch * seq_len, hidden_dim))?;
            let targets_flat = targets.reshape(batch * seq_len)?;
            return Ok((hidden_flat, targets_flat));
        }

        if self.transform_gemm {
            let (hidden_kept, indices) = self.transform_sparse_gemms(hidden, mask)?;
            let targets_kept = self.compact_targets(targets, &indices)?;
            Ok((hidden_kept, targets_kept))
        } else {
            let filtered = self.filter_activations(hidden, mask)?;
            let (batch, seq_len, hidden_dim) = filtered.dims3()?;
            let hidden_flat = filtered.reshape((batch * seq_len, hidden_dim))?;
            let targets_flat = targets.reshape(batch * seq_len)?;
            Ok((hidden_flat, targets_flat))
        }
    }

    /// Apply token filtering to activations.
    ///
    /// Zeros out activations for low-importance tokens.
    pub fn filter_activations(&self, activations: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Expand mask to match activation dimensions
        // activations: [batch, seq_len, hidden_dim]
        // mask: [batch, seq_len]
        let (batch, seq, hidden) = activations.dims3()?;
        let expanded_mask = mask.unsqueeze(2)?.broadcast_as((batch, seq, hidden))?;

        // Zero out filtered tokens
        activations.mul(&expanded_mask)
    }

    /// Get filtering statistics.
    pub fn stats(&self, mask: &Tensor) -> Result<ColliderStats> {
        let total_elements = mask.elem_count();

        // Count kept tokens (mask value = 1)
        let mask_sum = mask.sum_all()?.to_scalar::<f32>()?;
        let kept_tokens = mask_sum as usize;
        let filtered_tokens = total_elements - kept_tokens;

        Ok(ColliderStats {
            total_tokens: total_elements,
            kept_tokens,
            filtered_tokens,
            sparsity_ratio: filtered_tokens as f64 / total_elements as f64,
        })
    }

    /// Set threshold dynamically.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get current threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get target sparsity.
    pub fn sparsity_target(&self) -> f64 {
        self.sparsity_target
    }
}

impl Default for Collider {
    fn default() -> Self {
        Self::new(0.3, 0.35) // Filter 35% of low-importance tokens
    }
}

/// Collider filtering statistics.
#[derive(Debug, Clone)]
pub struct ColliderStats {
    pub total_tokens: usize,
    pub kept_tokens: usize,
    pub filtered_tokens: usize,
    pub sparsity_ratio: f64,
}

impl ColliderStats {
    /// Get speedup estimate from filtering.
    ///
    /// Collider paper reports 35% backprop speedup with 35% sparsity.
    pub fn estimated_speedup(&self) -> f64 {
        // Approximate: speedup ≈ 1 / (1 - sparsity_ratio)
        // With 35% filtering: 1 / 0.65 ≈ 1.54× faster
        if self.sparsity_ratio > 0.0 {
            1.0 / (1.0 - self.sparsity_ratio * 0.7) // 70% of theoretical max
        } else {
            1.0
        }
    }

    /// Check if sparsity is within healthy range.
    pub fn is_healthy(&self) -> bool {
        // Healthy range: 20-50% sparsity
        (0.2..=0.5).contains(&self.sparsity_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_collider_creation() {
        let collider = Collider::new(0.3, 0.35);
        assert_eq!(collider.threshold(), 0.3);
        assert_eq!(collider.sparsity_target(), 0.35);
    }

    #[test]
    fn test_collider_default() {
        let collider = Collider::default();
        assert_eq!(collider.threshold(), 0.3);
        assert_eq!(collider.sparsity_target(), 0.35);
    }

    #[test]
    fn test_create_mask() {
        let device = Device::Cpu;
        let collider = Collider::new(0.5, 0.35);

        // Importance scores [0.2, 0.6, 0.4, 0.8]
        let importance = Tensor::new(&[[0.2f32, 0.6, 0.4, 0.8]], &device).unwrap();
        let mask = collider.create_mask(&importance).unwrap();

        let mask_data = mask.to_vec2::<f32>().unwrap();
        // Threshold 0.5: only 0.6 and 0.8 should be kept
        assert_eq!(mask_data[0][0], 0.0); // 0.2 < 0.5
        assert_eq!(mask_data[0][1], 1.0); // 0.6 > 0.5
        assert_eq!(mask_data[0][2], 0.0); // 0.4 < 0.5
        assert_eq!(mask_data[0][3], 1.0); // 0.8 > 0.5
    }

    #[test]
    fn test_create_mask_respects_sparsity_target() {
        let device = Device::Cpu;
        let collider = Collider::new(0.0, 0.5); // keep top 50%

        let importance = Tensor::new(&[[0.1f32, 0.2, 0.3, 0.4]], &device).unwrap();
        let mask = collider.create_mask(&importance).unwrap();
        let mask_data = mask.to_vec2::<f32>().unwrap();

        // Keep top-2 (0.3, 0.4), filter lower half.
        assert_eq!(mask_data[0], vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_filter_activations() {
        let device = Device::Cpu;
        let collider = Collider::new(0.5, 0.35);

        // Activations [batch=1, seq=4, hidden=2]
        let activations = Tensor::ones(&[1, 4, 2], DType::F32, &device).unwrap();

        // Mask [batch=1, seq=4]
        let mask = Tensor::new(&[[1.0f32, 0.0, 1.0, 0.0]], &device).unwrap();

        let filtered = collider.filter_activations(&activations, &mask).unwrap();
        let result = filtered.to_vec3::<f32>().unwrap();

        // Check that filtered tokens are zeroed
        assert_eq!(result[0][0], [1.0, 1.0]); // kept
        assert_eq!(result[0][1], [0.0, 0.0]); // filtered
        assert_eq!(result[0][2], [1.0, 1.0]); // kept
        assert_eq!(result[0][3], [0.0, 0.0]); // filtered
    }

    #[test]
    fn test_kept_token_indices() {
        let device = Device::Cpu;
        let collider = Collider::new(0.5, 0.35);
        let mask = Tensor::new(&[[1.0f32, 0.0, 1.0], [0.0, 1.0, 0.0]], &device).unwrap();
        let indices = collider.kept_token_indices(&mask).unwrap();
        assert_eq!(indices.to_vec1::<u32>().unwrap(), vec![0, 2, 4]);
    }

    #[test]
    fn test_compact_hidden_and_targets() {
        let device = Device::Cpu;
        let collider = Collider::new(0.5, 0.35);

        // hidden: [batch=1, seq=4, hidden=2]
        let hidden = Tensor::new(
            &[[[1.0f32, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
            &device,
        )
        .unwrap();
        let targets = Tensor::new(&[[11u32, 22u32, 33u32, 44u32]], &device).unwrap();
        let mask = Tensor::new(&[[1.0f32, 0.0, 1.0, 0.0]], &device).unwrap();

        let (hidden_compact, idx) = collider.compact_hidden(&hidden, &mask).unwrap();
        let targets_compact = collider.compact_targets(&targets, &idx).unwrap();

        assert_eq!(
            hidden_compact.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 10.0], vec![3.0, 30.0]]
        );
        assert_eq!(targets_compact.to_vec1::<u32>().unwrap(), vec![11, 33]);
    }

    #[test]
    fn test_sparse_backward_compacts_rows() {
        let device = Device::Cpu;
        let collider = Collider::new(0.5, 0.35);

        let hidden = Tensor::new(
            &[[[1.0f32, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
            &device,
        )
        .unwrap();
        let targets = Tensor::new(&[[11u32, 22u32, 33u32, 44u32]], &device).unwrap();
        let mask = Tensor::new(&[[1.0f32, 0.0, 1.0, 0.0]], &device).unwrap();

        let (hidden_kept, targets_kept) =
            collider.sparse_backward(&hidden, &targets, &mask).unwrap();
        assert_eq!(
            hidden_kept.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 10.0], vec![3.0, 30.0]]
        );
        assert_eq!(targets_kept.to_vec1::<u32>().unwrap(), vec![11, 33]);
    }

    #[test]
    fn test_stats() {
        let device = Device::Cpu;
        let collider = Collider::new(0.5, 0.35);

        // Mask: [1, 0, 1, 0] = 50% filtered
        let mask = Tensor::new(&[[1.0f32, 0.0, 1.0, 0.0]], &device).unwrap();
        let stats = collider.stats(&mask).unwrap();

        assert_eq!(stats.total_tokens, 4);
        assert_eq!(stats.kept_tokens, 2);
        assert_eq!(stats.filtered_tokens, 2);
        assert!((stats.sparsity_ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_estimated_speedup() {
        let stats = ColliderStats {
            total_tokens: 100,
            kept_tokens: 65,
            filtered_tokens: 35,
            sparsity_ratio: 0.35,
        };

        // 35% sparsity should give ~1.3-1.4× speedup
        let speedup = stats.estimated_speedup();
        assert!(speedup > 1.2 && speedup < 1.5);
    }

    #[test]
    fn test_healthy_sparsity() {
        let good_stats = ColliderStats {
            total_tokens: 100,
            kept_tokens: 70,
            filtered_tokens: 30,
            sparsity_ratio: 0.3,
        };
        assert!(good_stats.is_healthy());

        let too_sparse = ColliderStats {
            total_tokens: 100,
            kept_tokens: 40,
            filtered_tokens: 60,
            sparsity_ratio: 0.6,
        };
        assert!(!too_sparse.is_healthy());
    }
}
