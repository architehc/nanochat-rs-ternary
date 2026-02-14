//! Multi-Token Prediction (MTP) and Collider Token Filtering
//! Based on: "What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?" (arXiv:2204.05832)
//! and Collider: "Cross-layer Activation Sparsity for Token Filtering" (arXiv:2502.00340)

use candle_core::{Result, Tensor, DType};
use candle_nn::{Linear, Module};
use std::collections::HashMap;

/// Multi-Token Prediction for denser training signals
pub struct MultiTokenPrediction {
    /// Number of future tokens to predict
    n_future_tokens: usize,

    /// Independent output heads for each future position
    output_heads: Vec<Linear>,

    /// Loss weights for each position (decay for distant predictions)
    loss_weights: Vec<f64>,

    /// Use shared or independent representations
    shared_representation: bool,
}

impl MultiTokenPrediction {
    /// Create new MTP module
    pub fn new(vb: VarBuilder, dim: usize, vocab_size: usize, n_future: usize) -> Result<Self> {
        let mut heads = Vec::new();
        let mut weights = Vec::new();

        for i in 0..n_future {
            let head = candle_nn::linear(dim, vocab_size, vb.pp(format!("mtp_head_{}", i)))?;
            heads.push(head);

            // Geometric decay for distant predictions
            let weight = 0.5_f64.powi(i as i32);
            weights.push(weight);
        }

        Ok(Self {
            n_future_tokens: n_future,
            output_heads: heads,
            loss_weights: weights,
            shared_representation: true,
        })
    }

    /// Forward pass predicting multiple future tokens
    pub fn forward(&self, hidden: &Tensor) -> Result<Vec<Tensor>> {
        let mut logits = Vec::new();

        for head in &self.output_heads {
            let logit = head.forward(hidden)?;
            logits.push(logit);
        }

        Ok(logits)
    }

    /// Compute MTP loss with weighted auxiliary losses
    pub fn compute_loss(
        &self,
        predictions: &[Tensor],
        targets: &[Tensor],
    ) -> Result<MTPLoss> {
        let mut total_loss = 0.0;
        let mut primary_loss = 0.0;
        let mut aux_loss = 0.0;

        for (i, (pred, target)) in predictions.iter().zip(targets.iter()).enumerate() {
            let loss = candle_nn::losses::cross_entropy(
                &candle_nn::ops::log_softmax(pred, D::Minus1)?,
                target,
            )?;

            let loss_val = loss.to_scalar::<f64>()?;
            let weighted_loss = loss_val * self.loss_weights[i];

            if i == 0 {
                primary_loss = loss_val;
            } else {
                aux_loss += weighted_loss;
            }

            total_loss += weighted_loss;
        }

        Ok(MTPLoss {
            total: total_loss,
            primary: primary_loss,
            auxiliary: aux_loss,
            n_tokens: predictions.len(),
        })
    }

    /// Generate with multi-token lookahead (inference)
    pub fn generate_lookahead(
        &self,
        model: &dyn Module,
        prompt: &Tensor,
        max_tokens: usize,
    ) -> Result<Tensor> {
        let mut generated = prompt.clone();

        for _ in 0..max_tokens / self.n_future_tokens {
            // Get hidden representation
            let hidden = model.forward(&generated)?;

            // Predict multiple tokens at once
            let predictions = self.forward(&hidden)?;

            // Sample from each prediction head
            for pred in &predictions {
                let probs = candle_nn::ops::softmax(pred, D::Minus1)?;
                let next_token = self.sample_from_probs(&probs)?;
                generated = Tensor::cat(&[&generated, &next_token], 1)?;
            }
        }

        Ok(generated)
    }

    fn sample_from_probs(&self, probs: &Tensor) -> Result<Tensor> {
        // Greedy sampling for simplicity
        // Could use temperature, top-k, top-p sampling
        probs.argmax(D::Minus1)
    }
}

/// MTP Loss structure
#[derive(Debug)]
pub struct MTPLoss {
    pub total: f64,
    pub primary: f64,
    pub auxiliary: f64,
    pub n_tokens: usize,
}

/// Collider: Cross-layer activation sparsity for token filtering
pub struct Collider {
    /// Token importance threshold (0-1)
    threshold: f64,

    /// Target sparsity ratio
    sparsity_target: f64,

    /// Layer-wise token masks
    masks: Vec<Option<Tensor>>,

    /// Apply filtering during backward pass
    filter_backward: bool,

    /// Transform sparse GEMMs to dense
    transform_gemm: bool,
}

impl Collider {
    /// Create new Collider
    pub fn new(threshold: f64, sparsity_target: f64) -> Self {
        Self {
            threshold,
            sparsity_target,
            masks: Vec::new(),
            filter_backward: true,
            transform_gemm: true,
        }
    }

    /// Compute token importance scores based on loss
    pub fn compute_importance(
        &self,
        logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor> {
        // Compute per-token cross-entropy loss
        let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;

        // Gather log probs for target tokens
        let batch_size = logits.dim(0)?;
        let seq_len = logits.dim(1)?;

        // Compute loss per token
        let losses = self.per_token_cross_entropy(&log_probs, targets)?;

        // Normalize to [0, 1] using min-max scaling
        let min_loss = losses.min_keepdim(1)?;
        let max_loss = losses.max_keepdim(1)?;
        let range = max_loss.sub(&min_loss)?.add(1e-8)?;

        let normalized = losses.sub(&min_loss)?.div(&range)?;

        Ok(normalized)
    }

    /// Per-token cross-entropy loss
    fn per_token_cross_entropy(&self, log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let vocab_size = log_probs.dim(2)?;

        // Flatten for gathering
        let flat_log_probs = log_probs.reshape(((), vocab_size))?;
        let flat_targets = targets.flatten_all()?;

        // Gather log probs for targets
        let gathered = flat_log_probs.gather(&flat_targets, 1)?;

        // Negate for loss
        let losses = gathered.neg()?;

        // Reshape back
        losses.reshape(targets.shape())
    }

    /// Create binary mask for important tokens
    pub fn create_mask(&self, importance: &Tensor) -> Result<Tensor> {
        // Tokens with importance > threshold are kept
        importance.gt(self.threshold)
    }

    /// Apply token filtering during forward pass (for training)
    pub fn filter_forward(&self, activations: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Expand mask to match activation dimensions
        let expanded_mask = mask.expand(activations.shape())?;

        // Zero out unimportant tokens
        activations.broadcast_mul(&expanded_mask)
    }

    /// Apply token filtering during backward pass
    pub fn filter_backward(&self, gradients: &mut Gradients, mask: &Tensor) -> Result<()> {
        if !self.filter_backward {
            return Ok(());
        }

        // Apply mask to all gradient tensors
        for (name, grad) in gradients.iter_mut() {
            // Only filter token-related gradients
            if self.should_filter(name) {
                let expanded_mask = mask.expand(grad.shape())?;
                *grad = grad.broadcast_mul(&expanded_mask)?;
            }
        }

        Ok(())
    }

    /// Check if gradient should be filtered
    fn should_filter(&self, name: &str) -> bool {
        // Filter embeddings, attention, and output layers
        name.contains("embed") 
            || name.contains("attn") 
            || name.contains("token")
            || name.contains("lm_head")
    }

    /// Transform sparse GEMM to dense GEMM with reduced dimensions
    pub fn transform_sparse_gemm(&self, a: &Tensor, b: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        if !self.transform_gemm {
            return Ok((a.clone(), b.clone()));
        }

        // Count kept tokens
        let kept_indices: Vec<usize> = mask
            .to_vec1::<u8>()?
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 1)
            .map(|(i, _)| i)
            .collect();

        if kept_indices.is_empty() {
            // Keep all if none selected
            return Ok((a.clone(), b.clone()));
        }

        // Index select to get dense matrices
        let a_dense = a.index_select(&Tensor::new(kept_indices.clone(), a.device())?, 0)?;
        let b_dense = b.index_select(&Tensor::new(kept_indices, b.device())?, 0)?;

        Ok((a_dense, b_dense))
    }

    /// Get filtering statistics
    pub fn stats(&self, mask: &Tensor) -> Result<ColliderStats> {
        let total_tokens = mask.elem_count();
        let kept_tokens = mask.to_vec1::<u8>()?.iter().filter(|&&v| v == 1).count();
        let filtered_tokens = total_tokens - kept_tokens;

        Ok(ColliderStats {
            total_tokens,
            kept_tokens,
            filtered_tokens,
            sparsity_ratio: filtered_tokens as f64 / total_tokens as f64,
        })
    }
}

/// Collider statistics
#[derive(Debug)]
pub struct ColliderStats {
    pub total_tokens: usize,
    pub kept_tokens: usize,
    pub filtered_tokens: usize,
    pub sparsity_ratio: f64,
}

/// Gradients container with filtering support
pub struct Gradients {
    grads: HashMap<String, Tensor>,
}

impl Gradients {
    pub fn new(grads: HashMap<String, Tensor>) -> Self {
        Self { grads }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.grads.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut Tensor)> {
        self.grads.iter_mut()
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.grads.get(name)
    }

    pub fn insert(&mut self, name: String, tensor: Tensor) {
        self.grads.insert(name, tensor);
    }
}

/// Integrated training step with MTP + Collider
pub struct EfficientTrainingStep {
    mtp: MultiTokenPrediction,
    collider: Collider,
    use_mtp: bool,
    use_collider: bool,
}

impl EfficientTrainingStep {
    pub fn new(mtp: MultiTokenPrediction, collider: Collider) -> Self {
        Self {
            mtp,
            collider,
            use_mtp: true,
            use_collider: true,
        }
    }

    /// Execute efficient training step
    pub fn execute(
        &mut self,
        model: &mut dyn Module,
        input: &Tensor,
        targets: &[Tensor],
    ) -> Result<TrainingOutput> {
        // Forward pass
        let hidden = model.forward(input)?;

        // Multi-token prediction
        let predictions = if self.use_mtp {
            self.mtp.forward(&hidden)?
        } else {
            vec![hidden.matmul(&model.lm_head_weights()?)?]
        };

        // Compute MTP loss
        let mtp_loss = self.mtp.compute_loss(&predictions, targets)?;

        // Compute token importance for filtering
        let importance = if self.use_collider {
            self.collider.compute_importance(&predictions[0], &targets[0])?
        } else {
            Tensor::ones(predictions[0].shape(), DType::F32, predictions[0].device())?
        };

        // Create filtering mask
        let mask = if self.use_collider {
            self.collider.create_mask(&importance)?
        } else {
            Tensor::ones(importance.shape(), DType::F32, importance.device())?
        };

        // Backward pass
        let loss_tensor = Tensor::new(mtp_loss.total, input.device())?;
        let mut grads = loss_tensor.backward()?;

        // Apply token filtering
        if self.use_collider {
            self.collider.filter_backward(&mut grads, &mask)?;
        }

        // Get statistics
        let collider_stats = if self.use_collider {
            Some(self.collider.stats(&mask)?)
        } else {
            None
        };

        Ok(TrainingOutput {
            loss: mtp_loss,
            gradients: grads,
            collider_stats,
        })
    }
}

/// Training step output
pub struct TrainingOutput {
    pub loss: MTPLoss,
    pub gradients: Gradients,
    pub collider_stats: Option<ColliderStats>,
}

use candle_core::D;
use candle_nn::VarBuilder;
