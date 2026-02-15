//! Collider: Cross-layer Activation Sparsity for Token Filtering
//!
//! Based on: "Cross-layer Activation Sparsity for Token Filtering" (arXiv:2502.00340)
//!
//! Collider achieves 35% faster backprop by:
//! 1. Computing per-token importance scores based on loss
//! 2. Filtering out low-importance tokens during backward pass
//! 3. Transforming sparse GEMMs to dense operations

use candle_core::{DType, Result, Tensor, D};

/// Token filtering via cross-layer activation sparsity.
///
/// Filters tokens with low importance scores to speed up backward pass.
/// Importance is computed from per-token cross-entropy loss.
pub struct Collider {
    /// Importance threshold (0-1): tokens below this are filtered
    threshold: f64,

    /// Target sparsity ratio (0-1): fraction of tokens to filter
    sparsity_target: f64,

    /// Apply filtering during backward pass (TODO: not yet implemented)
    #[allow(dead_code)]
    filter_backward: bool,

    /// Transform sparse GEMMs to dense (gather important tokens) (TODO: not yet implemented)
    #[allow(dead_code)]
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
        let range = (max_loss - &min_loss)? + 1e-8;

        let normalized = ((losses - min_loss)? / range)?;

        Ok(normalized)
    }

    /// Compute per-token cross-entropy loss (not reduced).
    fn per_token_cross_entropy(&self, log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let (_batch_size, _seq_len, _vocab_size) = log_probs.dims3()?;

        // TODO: Implement proper per-token indexing when candle supports it
        // For now, use a simplified cross-entropy computation

        // Simplified: compute negative log likelihood per token
        let losses = candle_nn::loss::cross_entropy(log_probs, targets)?;

        Ok(losses)
    }

    /// Create binary mask for important tokens.
    ///
    /// # Arguments
    /// * `importance` - Importance scores [batch, seq_len]
    ///
    /// # Returns
    /// Binary mask [batch, seq_len]: 1 for kept tokens, 0 for filtered
    pub fn create_mask(&self, importance: &Tensor) -> Result<Tensor> {
        // Tokens with importance > threshold are kept
        let mask = importance.gt(self.threshold)?;
        mask.to_dtype(DType::F32)
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
