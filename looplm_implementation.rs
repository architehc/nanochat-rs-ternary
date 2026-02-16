//! LoopLM (Looped Language Model) Implementation for nanochat-rs-ternary
//! Based on "Scaling Latent Reasoning via Looped Language Models" (arXiv:2510.25741)

use candle_core::{Result, Tensor, DType, D, Device};
use candle_nn::{Linear, Module, VarBuilder};
use serde::Deserialize;

/// LoopLM Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct LoopLMConfig {
    /// Number of recurrent steps (R=4 for Ouro models)
    pub n_loops: usize,

    /// Hidden dimension
    pub dim: usize,

    /// Number of layers (shared across loops)
    pub n_layers: usize,

    /// Number of attention heads
    pub n_heads: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Entropy regularization weight for depth allocation
    pub entropy_weight: f64,

    /// Temperature for exit gate softmax
    pub exit_temperature: f64,
}

impl Default for LoopLMConfig {
    fn default() -> Self {
        Self {
            n_loops: 4,
            dim: 2048,
            n_layers: 24,
            n_heads: 16,
            vocab_size: 50257,
            max_seq_len: 8192,
            entropy_weight: 0.05,
            exit_temperature: 1.0,
        }
    }
}

/// Exit gate for adaptive depth allocation
pub struct ExitGate {
    linear: Linear,
    temperature: f64,
}

impl ExitGate {
    pub fn new(vb: VarBuilder, dim: usize, temperature: f64) -> Result<Self> {
        let linear = candle_nn::linear(dim, 1, vb.pp("linear"))?;
        Ok(Self { linear, temperature })
    }

    /// Compute exit probability for current loop.
    /// Returns a Tensor (not a scalar f64) to preserve gradient tracking
    /// for the entropy regularization loss term.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        // Global average pooling over sequence
        let pooled = hidden.mean(1)?;  // [batch, dim]
        let logits = self.linear.forward(&pooled)?;  // [batch, 1]
        let prob = candle_nn::ops::sigmoid(&logits)?;
        // Return mean probability across batch as a scalar tensor (keeps gradient)
        prob.mean_all()
    }

    /// Extract scalar value for inference-time early exit decisions.
    /// This detaches from the computation graph.
    pub fn forward_scalar(&self, hidden: &Tensor) -> Result<f64> {
        let prob_tensor = self.forward(hidden)?;
        Ok(prob_tensor.to_scalar::<f32>()? as f64)
    }
}

/// LoopLM Transformer Block (shared across recurrent steps)
pub struct SharedTransformerBlock {
    /// Self-attention
    attn: MultiHeadAttention,

    /// Feed-forward network
    ffn: FeedForward,

    /// Layer norms
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl SharedTransformerBlock {
    pub fn new(vb: VarBuilder, config: &LoopLMConfig) -> Result<Self> {
        let attn = MultiHeadAttention::new(vb.pp("attn"), config)?;
        let ffn = FeedForward::new(vb.pp("ffn"), config)?;
        let norm1 = LayerNorm::new(vb.pp("norm1"), config.dim)?;
        let norm2 = LayerNorm::new(vb.pp("norm2"), config.dim)?;

        Ok(Self { attn, ffn, norm1, norm2 })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm architecture
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&normed, mask)?;
        let x = x.add(&attn_out)?;

        let normed = self.norm2.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        let x = x.add(&ffn_out)?;

        Ok(x)
    }
}

/// Looped Language Model
pub struct LoopLM {
    /// Token embeddings
    token_embed: candle_nn::Embedding,

    /// Position embeddings
    pos_embed: candle_nn::Embedding,

    /// Shared transformer blocks (recurrently applied)
    shared_blocks: Vec<SharedTransformerBlock>,

    /// Exit gate for adaptive depth
    exit_gate: ExitGate,

    /// Output head
    lm_head: Linear,

    /// Configuration
    config: LoopLMConfig,
}

impl LoopLM {
    pub fn new(vb: VarBuilder, config: LoopLMConfig) -> Result<Self> {
        let token_embed = candle_nn::embedding(
            config.vocab_size,
            config.dim,
            vb.pp("token_embed"),
        )?;

        let pos_embed = candle_nn::embedding(
            config.max_seq_len,
            config.dim,
            vb.pp("pos_embed"),
        )?;

        // Create shared transformer blocks
        let mut shared_blocks = Vec::new();
        for i in 0..config.n_layers {
            let block = SharedTransformerBlock::new(
                vb.pp(format!("shared_block_{}", i)),
                &config,
            )?;
            shared_blocks.push(block);
        }

        let exit_gate = ExitGate::new(
            vb.pp("exit_gate"),
            config.dim,
            config.exit_temperature,
        )?;

        let lm_head = candle_nn::linear(config.dim, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            token_embed,
            pos_embed,
            shared_blocks,
            exit_gate,
            lm_head,
            config,
        })
    }

    /// Forward pass with iterative computation (training)
    /// Returns logits and exit probability Tensors (gradient-tracked for entropy regularization).
    pub fn forward_train(&self, input_ids: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Embeddings
        let token_emb = self.token_embed.forward(input_ids)?;
        let positions = Tensor::arange(0, seq_len as i64, input_ids.device())?
            .reshape((1, seq_len))?
            .broadcast_as((batch_size, seq_len))?;
        let pos_emb = self.pos_embed.forward(&positions)?;
        let mut hidden = token_emb.add(&pos_emb)?;

        // Create causal mask
        let mask = self.create_causal_mask(seq_len, hidden.device())?;

        // Track exit probabilities as Tensors (preserves gradient tracking)
        let mut exit_probs = Vec::new();

        // Apply shared blocks recurrently
        for _loop_idx in 0..self.config.n_loops {
            for block in &self.shared_blocks {
                hidden = block.forward(&hidden, Some(&mask))?;
            }

            // Compute exit probability (returns Tensor to keep gradient)
            let exit_prob = self.exit_gate.forward(&hidden)?;
            exit_probs.push(exit_prob);
        }

        // Final output
        let logits = self.lm_head.forward(&hidden)?;

        Ok((logits, exit_probs))
    }

    /// Forward pass with adaptive depth (inference)
    pub fn forward_inference(
        &self,
        input_ids: &Tensor,
        max_loops: Option<usize>,
        exit_threshold: f64,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Embeddings
        let token_emb = self.token_embed.forward(input_ids)?;
        let positions = Tensor::arange(0, seq_len as i64, input_ids.device())?
            .reshape((1, seq_len))?
            .broadcast_as((batch_size, seq_len))?;
        let pos_emb = self.pos_embed.forward(&positions)?;
        let mut hidden = token_emb.add(&pos_emb)?;

        let mask = self.create_causal_mask(seq_len, hidden.device())?;
        let max_loops = max_loops.unwrap_or(self.config.n_loops);

        // Apply shared blocks with adaptive depth
        for loop_idx in 0..max_loops {
            for block in &self.shared_blocks {
                hidden = block.forward(&hidden, Some(&mask))?;
            }

            // Check early exit condition (use scalar extraction for inference decisions)
            let exit_prob = self.exit_gate.forward_scalar(&hidden)?;
            if exit_prob > exit_threshold {
                tracing::info!("Early exit at loop {}", loop_idx);
                break;
            }
        }

        // Final output
        let logits = self.lm_head.forward(&hidden)?;
        Ok(logits)
    }

    /// Compute LoopLM training loss with entropy regularization.
    /// exit_probs are Tensors (not f64) so gradients flow through the entropy term.
    pub fn compute_loss(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        exit_probs: &[Tensor],
    ) -> Result<LoopLMLoss> {
        // Standard cross-entropy loss
        let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;
        let nll_loss_tensor = candle_nn::losses::cross_entropy(&log_probs, targets)?;

        // Entropy regularization for exit probabilities (as Tensors for gradient flow)
        // Stack exit probs into a single tensor and normalize to form a distribution
        let exit_stack = Tensor::stack(exit_probs, 0)?; // [n_loops]
        let exit_dist = candle_nn::ops::softmax(&exit_stack, 0)?; // normalize to distribution

        // Compute entropy: H = -sum(p * log(p + eps))
        let eps_tensor = Tensor::new(1e-8f32, exit_dist.device())?;
        let log_exit = (exit_dist.broadcast_add(&eps_tensor))?.log()?;
        let entropy_tensor = (exit_dist.broadcast_mul(&log_exit))?.neg()?.sum_all()?;

        // Uniform entropy (scalar, no gradient needed)
        let uniform_entropy = compute_uniform_entropy(exit_probs.len());

        // Entropy penalty: push toward uniform
        let uniform_tensor = Tensor::new(uniform_entropy as f32, entropy_tensor.device())?;
        let entropy_penalty_tensor = (uniform_tensor.sub(&entropy_tensor))?.abs()?;

        // Total loss = NLL + weight * entropy_penalty (all as Tensors)
        let weight_tensor = Tensor::new(self.config.entropy_weight as f32, nll_loss_tensor.device())?;
        let total_loss_tensor = nll_loss_tensor.add(&(weight_tensor.broadcast_mul(&entropy_penalty_tensor))?)?;

        // Extract scalar values for logging
        let total = total_loss_tensor.to_scalar::<f32>()? as f64;
        let nll = nll_loss_tensor.to_scalar::<f32>()? as f64;
        let entropy_penalty = entropy_penalty_tensor.to_scalar::<f32>()? as f64;
        let exit_entropy = entropy_tensor.to_scalar::<f32>()? as f64;

        Ok(LoopLMLoss {
            total,
            nll,
            entropy_penalty,
            exit_entropy,
            // Keep the tensor for backward pass
            total_loss_tensor,
        })
    }

    fn create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask = Tensor::ones((seq_len, seq_len), DType::F32, device)?;
        let mask = mask.triu(1)?;  // Upper triangular (excluding diagonal)
        let mask = mask * f32::NEG_INFINITY;
        Ok(mask)
    }
}

/// LoopLM Loss structure
#[derive(Debug)]
pub struct LoopLMLoss {
    pub total: f64,
    pub nll: f64,
    pub entropy_penalty: f64,
    pub exit_entropy: f64,
    /// The total loss as a Tensor for calling .backward() with gradient flow
    /// through the exit gate entropy regularization term.
    pub total_loss_tensor: Tensor,
}

/// Compute entropy of exit probability distribution
fn compute_entropy(probs: &[f64]) -> f64 {
    probs.iter()
        .map(|&p| {
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

/// Compute entropy of uniform distribution
fn compute_uniform_entropy(n: usize) -> f64 {
    let p = 1.0 / n as f64;
    - (0..n).map(|_| p * p.log2()).sum::<f64>()
}

/// Multi-head attention (simplified)
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder, config: &LoopLMConfig) -> Result<Self> {
        let head_dim = config.dim / config.n_heads;

        Ok(Self {
            q_proj: candle_nn::linear(config.dim, config.dim, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(config.dim, config.dim, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(config.dim, config.dim, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear(config.dim, config.dim, vb.pp("o_proj"))?,
            n_heads: config.n_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q.reshape((batch, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let scores = q.matmul(&k.transpose(2, 3)?)? * scale;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };

        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2)?.reshape((batch, seq_len, dim))?;
        self.o_proj.forward(&attn_output)
    }
}

/// Feed-forward network
pub struct FeedForward {
    up_proj: Linear,
    down_proj: Linear,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, config: &LoopLMConfig) -> Result<Self> {
        let hidden_dim = config.dim * 4;  // Standard expansion

        Ok(Self {
            up_proj: candle_nn::linear(config.dim, hidden_dim, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear(hidden_dim, config.dim, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.up_proj.forward(x)?;
        let x = candle_nn::ops::gelu(&x)?;
        self.down_proj.forward(&x)
    }
}

/// Layer normalization
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let weight = vb.get((dim,), "weight")?;
        let bias = vb.get((dim,), "bias")?;

        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.var_keepdim(D::Minus1)?;
        let x = ((x - &mean)? / (var + self.eps).sqrt()?)?;
        x.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

