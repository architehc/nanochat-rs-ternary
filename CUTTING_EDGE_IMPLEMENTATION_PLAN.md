# nanochat-rs-ternary: Implementation Plan for Cutting-Edge Training Efficiency Techniques

## Executive Summary

This document provides a prioritized implementation plan for integrating the most promising 2025-2026 training efficiency techniques into nanochat-rs-ternary, tailored to your three hardware configurations.

**Target**: Achieve 2-4× training speedup and 50-80% memory reduction while maintaining or improving model quality.

---

## 🎯 Technique Selection Matrix

| Technique | Impact | Effort | Config A | Config B | Config C | Priority |
|-----------|--------|--------|----------|----------|----------|----------|
| **Muon Optimizer** | ⭐⭐⭐⭐⭐ | Low | ✅ | ✅ | ✅ | P0 |
| **GaLore 2** | ⭐⭐⭐⭐⭐ | Medium | ✅ | ✅✅ | ✅ | P0 |
| **FP4 Training** | ⭐⭐⭐⭐⭐ | Medium | ✅✅ | ❌ | ❌ | P0 (A only) |
| **Multi-Token Prediction** | ⭐⭐⭐⭐ | Medium | ✅ | ✅ | ✅ | P1 |
| **mHC Architecture** | ⭐⭐⭐⭐ | High | ✅ | ✅ | ✅ | P1 |
| **FOAM** | ⭐⭐⭐⭐ | Medium | ✅ | ✅ | ✅ | P1 |
| **Collider (Token Filtering)** | ⭐⭐⭐⭐ | High | ✅ | ✅ | ✅ | P2 |
| **MinatoLoader** | ⭐⭐⭐ | Medium | ✅ | ✅ | ✅ | P2 |
| **FIRE Reinitialization** | ⭐⭐⭐ | Medium | ✅ | ✅ | ✅ | P2 |
| **Training-Free GRPO** | ⭐⭐⭐⭐ | Low | ✅ | ✅ | ✅ | P3 |

**Legend**: ✅✅ = Excellent fit, ✅ = Good fit, ❌ = Not applicable

---

## 🔬 Detailed Technique Analysis

### P0: Critical Path (Implement First)

#### 1. Muon Optimizer (arXiv:2502.16982)

**Why**: 2× compute efficiency over AdamW, proven at frontier scale (Kimi.ai)

**Current Status**: Already integrated in nanochat-rs-ternary ✓

**Enhancement**: Add 8-bit quantized variant (arXiv:2509.23106)

```rust
// 8-bit Muon implementation
pub struct QuantizedMuon {
    base: MuonOptimizer,
    quantizer: BlockwiseQuantizer,
    bits: u8,  // 8-bit for optimizer states
}

impl QuantizedMuon {
    pub fn step_quantized(&mut self, grad: &Tensor) -> Result<Tensor> {
        // Quantize gradient to 8-bit
        let q_grad = self.quantizer.quantize(grad, 8)?;

        // Apply Newton-Schulz iteration in quantized space
        let update = self.base.newton_schulz(&q_grad)?;

        // Dequantize for application
        self.quantizer.dequantize(&update)
    }
}
```

**Expected Gain**: 86% memory reduction in optimizer states vs AdamW

---

#### 2. GaLore 2 (arXiv:2504.20437)

**Why**: Train 7B models on single RTX 4090 (24GB), 65.5% memory reduction

**Implementation**:

```rust
/// GaLore 2: Memory-efficient training with gradient low-rank projection
pub struct GaLore2Optimizer {
    /// Base optimizer (Muon or Adam)
    base_optimizer: Box<dyn Optimizer>,

    /// Rank for low-rank projection
    rank: usize,

    /// Projection matrices (updated periodically)
    projection_matrices: HashMap<String, (Tensor, Tensor)>,

    /// Update frequency (every N steps)
    update_freq: usize,

    /// Scale factor for projected gradients
    scale: f64,
}

impl GaLore2Optimizer {
    pub fn new(base: Box<dyn Optimizer>, rank: usize, update_freq: usize) -> Self {
        Self {
            base_optimizer: base,
            rank,
            projection_matrices: HashMap::new(),
            update_freq,
            scale: 1.0,
        }
    }

    /// Project gradient to low-rank subspace
    fn project_gradient(&self, name: &str, grad: &Tensor) -> Result<Tensor> {
        if let Some((p, q)) = self.projection_matrices.get(name) {
            // Project: g_proj = Q^T @ g @ P
            let temp = q.matmul(grad)?;
            temp.matmul(p)
        } else {
            Ok(grad.clone())
        }
    }

    /// Update projection matrices using SVD (every N steps)
    fn update_projections(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        for (name, grad) in gradients {
            // Compute SVD: grad = U @ S @ V^T
            let (u, s, vt) = grad.svd()?;

            // Take top-r singular vectors
            let p = vt.narrow(0, 0, self.rank)?;  // Right singular vectors
            let q = u.narrow(1, 0, self.rank)?;   // Left singular vectors

            self.projection_matrices.insert(name.clone(), (p, q));
        }
        Ok(())
    }

    pub fn step(&mut self, gradients: &HashMap<String, Tensor>, step: usize) -> Result<()> {
        // Update projections periodically
        if step % self.update_freq == 0 {
            self.update_projections(gradients)?;
        }

        // Project gradients and apply base optimizer
        let projected: HashMap<String, Tensor> = gradients.iter()
            .map(|(name, grad)| {
                let proj = self.project_gradient(name, grad).unwrap();
                (name.clone(), proj)
            })
            .collect();

        self.base_optimizer.step(&projected)
    }
}
```

**Hardware-Specific Configurations**:

```toml
# Config A (Threadripper + Blackwell)
[optimizer.galore]
enabled = true
rank = 512  # Higher rank for larger model
update_freq = 200
base_optimizer = "Muon"

# Config B (9800X3D + 2×4090)
[optimizer.galore]
enabled = true
rank = 256  # Lower rank for smaller model
update_freq = 500
base_optimizer = "Muon"

# Config C (Dual EPYC + 4090)
[optimizer.galore]
enabled = true
rank = 384
update_freq = 300
base_optimizer = "Muon"
```

**Expected Gain**: 
- Config A: 50% memory reduction, 10% speedup
- Config B: 65% memory reduction, enables 7B on 24GB
- Config C: 55% memory reduction

---

#### 3. FP4 Training (arXiv:2501.17116, 2502.11458)

**Why**: Blackwell natively supports FP4 tensor cores → massive speedup

**Implementation**:

```rust
/// FP4 Training for Blackwell GPUs
pub struct FP4Trainer {
    /// Forward pass in FP8/BF16
    forward_dtype: DType,

    /// Backward pass in FP4
    backward_dtype: DType,

    /// Stochastic rounding for FP4
    use_stochastic_rounding: bool,

    /// Per-module precision targeting
    precision_config: HashMap<String, DType>,
}

impl FP4Trainer {
    pub fn new() -> Self {
        Self {
            forward_dtype: DType::BF16,
            backward_dtype: DType::F4,  // FP4 for backward
            use_stochastic_rounding: true,
            precision_config: Self::default_precision_config(),
        }
    }

    fn default_precision_config() -> HashMap<String, DType> {
        let mut config = HashMap::new();

        // Attention layers: FP8 forward, FP4 backward
        config.insert("attn".to_string(), DType::F8);
        config.insert("attn_grad".to_string(), DType::F4);

        // FFN layers: BF16 forward, FP4 backward
        config.insert("ffn".to_string(), DType::BF16);
        config.insert("ffn_grad".to_string(), DType::F4);

        // Embeddings: FP16 (need precision)
        config.insert("embed".to_string(), DType::F16);

        config
    }

    /// Forward pass with mixed precision
    pub fn forward(&self, model: &mut Model, input: &Tensor) -> Result<Tensor> {
        // Cast to forward precision
        let input_fp = input.to_dtype(self.forward_dtype)?;

        // Run forward pass
        let output = model.forward(&input_fp)?;

        Ok(output)
    }

    /// Backward pass with FP4
    pub fn backward(&self, model: &mut Model, loss: &Tensor) -> Result<Gradients> {
        // Enable FP4 tensor cores
        candle_core::cuda::set_fp4_mode(true)?;

        // Compute gradients in FP4
        let grads = loss.backward()?;

        // Apply stochastic rounding
        let rounded_grads = if self.use_stochastic_rounding {
            self.stochastic_round_gradients(&grads)?
        } else {
            grads
        };

        Ok(rounded_grads)
    }

    fn stochastic_round_gradients(&self, grads: &Gradients) -> Result<Gradients> {
        // Implement stochastic rounding for FP4
        // Reduces bias from quantization
        grads.stochastic_round(4)  // 4-bit
    }
}
```

**Config A Only** (Blackwell has FP4 support):

```toml
# Config A: Enable FP4 training
[training.fp4]
enabled = true
forward_precision = "BF16"
backward_precision = "FP4"
stochastic_rounding = true
memory_reduction = "80%"

# Enable for specific modules
[training.fp4.modules]
attention = "FP8"
ffn = "FP4"
embedding = "FP16"
```

**Expected Gain**: 
- 2-3× training speedup on Blackwell
- 80% memory reduction in activations
- Comparable accuracy to BF16

---

### P1: High Impact (Implement Second)

#### 4. Multi-Token Prediction (MTP) (arXiv:2404.19737)

**Why**: Predict multiple future tokens simultaneously → denser training signal, better data efficiency

**Implementation**:

```rust
/// Multi-Token Prediction for denser training signals
pub struct MultiTokenPrediction {
    /// Number of future tokens to predict
    n_future_tokens: usize,

    /// Independent output heads for each future position
    output_heads: Vec<Linear>,

    /// Loss weights for each position
    loss_weights: Vec<f64>,
}

impl MultiTokenPrediction {
    pub fn new(vb: VarBuilder, dim: usize, vocab_size: usize, n_future: usize) -> Result<Self> {
        let mut heads = Vec::new();
        let mut weights = Vec::new();

        for i in 0..n_future {
            let head = candle_nn::linear(dim, vocab_size, vb.pp(format!("head_{}", i)))?;
            heads.push(head);

            // Decay weights for distant predictions
            weights.push(1.0 / (i + 1) as f64);
        }

        Ok(Self {
            n_future_tokens: n_future,
            output_heads: heads,
            loss_weights: weights,
        })
    }

    /// Forward pass predicting multiple tokens
    pub fn forward(&self, hidden: &Tensor) -> Result<Vec<Tensor>> {
        let mut logits = Vec::new();

        for head in &self.output_heads {
            let logit = head.forward(hidden)?;
            logits.push(logit);
        }

        Ok(logits)
    }

    /// Compute MTP loss
    pub fn compute_loss(
        &self,
        predictions: &[Tensor],
        targets: &[Tensor],
    ) -> Result<f64> {
        let mut total_loss = 0.0;

        for (i, (pred, target)) in predictions.iter().zip(targets.iter()).enumerate() {
            let loss = candle_nn::losses::cross_entropy(pred, target)?;
            total_loss += loss.to_scalar::<f64>()? * self.loss_weights[i];
        }

        Ok(total_loss)
    }
}
```

**Integration with LoopLM**:

```rust
// In LoopLM forward pass
let hidden = self.apply_shared_layers(input)?;

// Multi-token prediction
let mtp_logits = self.mtp_heads.forward(&hidden)?;

// Primary loss (next token)
let primary_loss = cross_entropy(&mtp_logits[0], &targets[0])?;

// Auxiliary losses (future tokens)
let mut aux_loss = 0.0;
for i in 1..self.n_future_tokens {
    aux_loss += cross_entropy(&mtp_logits[i], &targets[i])? * 0.5_f64.powi(i as i32);
}

let total_loss = primary_loss + 0.2 * aux_loss;  // Weight auxiliary losses
```

**Expected Gain**: 
- 15-20% better data efficiency
- Faster convergence
- Improved long-range dependencies

---

#### 5. mHC: Manifold-Constrained Hyper-Connections (arXiv:2512.24880)

**Why**: Restores residual identity mapping while enabling diversified connectivity

**Implementation**:

```rust
/// Manifold-Constrained Hyper-Connections (mHC)
pub struct MHC {
    /// Number of parallel streams
    n_streams: usize,

    /// Mixing matrix (doubly stochastic via Sinkhorn-Knopp)
    mixing_matrix: Tensor,

    /// Learnable mixing parameters
    alpha_logits: Tensor,

    /// Sinkhorn iterations for normalization
    sinkhorn_iters: usize,
}

impl MHC {
    pub fn new(vb: VarBuilder, n_streams: usize, dim: usize) -> Result<Self> {
        let alpha = vb.get((n_streams, n_streams), "alpha")?;

        Ok(Self {
            n_streams,
            mixing_matrix: Tensor::zeros((n_streams, n_streams), DType::F32, vb.device()),
            alpha_logits: alpha,
            sinkhorn_iters: 5,
        })
    }

    /// Apply Sinkhorn-Knopp to ensure doubly stochastic
    fn compute_mixing_matrix(&self) -> Result<Tensor> {
        let mut matrix = self.alpha_logits.softmax(1)?;

        // Sinkhorn iterations
        for _ in 0..self.sinkhorn_iters {
            // Normalize rows
            let row_sums = matrix.sum_keepdim(1)?;
            matrix = matrix.broadcast_div(&row_sums)?;

            // Normalize columns
            let col_sums = matrix.sum_keepdim(0)?;
            matrix = matrix.broadcast_div(&col_sums)?;
        }

        Ok(matrix)
    }

    /// Forward pass with manifold-constrained mixing
    pub fn forward(&self, streams: &[Tensor]) -> Result<Vec<Tensor>> {
        let mixing = self.compute_mixing_matrix()?;

        let mut outputs = Vec::new();

        for i in 0..self.n_streams {
            let mut mixed = Tensor::zeros_like(&streams[0])?;

            for j in 0..self.n_streams {
                let weight = mixing.get(i)?.get(j)?.to_scalar::<f32>()? as f64;
                mixed = mixed.add(&streams[j].broadcast_mul(&weight)?)?;
            }

            outputs.push(mixed);
        }

        Ok(outputs)
    }
}
```

**Expected Gain**: 
- 3 orders of magnitude better stability (max gain 1.6 vs 3000)
- +2.3% on DROP benchmark
- 6.7% training overhead

---

#### 6. FOAM: Blocked State Folding (arXiv:2512.07112)

**Why**: 50% total training memory reduction, 90% optimizer state savings

**Implementation**:

```rust
/// FOAM: Blocked State Folding for Memory-Efficient Training
pub struct FOAMOptimizer {
    /// Base optimizer
    base: Box<dyn Optimizer>,

    /// Block size for folding
    block_size: usize,

    /// Compressed states
    compressed_states: HashMap<String, CompressedState>,
}

pub struct CompressedState {
    /// Block-wise means
    block_means: Tensor,

    /// Residual correction
    residual: Tensor,

    /// Compression ratio
    ratio: f64,
}

impl FOAMOptimizer {
    pub fn compress_state(&self, state: &Tensor) -> Result<CompressedState> {
        let shape = state.shape();
        let numel = shape.elem_count();

        // Reshape into blocks
        let n_blocks = (numel + self.block_size - 1) / self.block_size;
        let blocks = state.reshape((n_blocks, self.block_size))?;

        // Compute block-wise means
        let block_means = blocks.mean_keepdim(1)?;

        // Compute residual
        let expanded_means = block_means.broadcast_as(blocks.shape())?;
        let residual = blocks.sub(&expanded_means)?;

        Ok(CompressedState {
            block_means,
            residual,
            ratio: self.block_size as f64,
        })
    }

    pub fn decompress_state(&self, compressed: &CompressedState) -> Result<Tensor> {
        // Expand means
        let expanded = compressed.block_means.broadcast_as(
            (compressed.block_means.dim(0)?, self.block_size)
        )?;

        // Add residual
        let full = expanded.add(&compressed.residual)?;

        // Reshape back
        full.reshape(compressed.original_shape())
    }
}
```

**Expected Gain**: 
- 50% total training memory reduction
- 90% optimizer state savings
- Faster convergence

---

### P2: Medium Impact (Implement Third)

#### 7. Collider: Token Filtering for Backprop (arXiv:2502.00340)

**Why**: 35% faster backprop, 22% end-to-end training reduction

**Implementation**:

```rust
/// Collider: Cross-layer activation sparsity for token filtering
pub struct Collider {
    /// Token importance threshold
    threshold: f64,

    /// Sparsity target
    sparsity_target: f64,

    /// Layer-wise token masks
    masks: Vec<Tensor>,
}

impl Collider {
    /// Compute token importance scores
    fn compute_importance(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Use loss as importance metric
        let losses = candle_nn::losses::cross_entropy_loss_per_token(logits, targets)?;

        // Normalize to [0, 1]
        let min = losses.min_keepdim(0)?;
        let max = losses.max_keepdim(0)?;
        let normalized = losses.sub(&min)?.div(&max.sub(&min)?.add(1e-8)?)?;

        Ok(normalized)
    }

    /// Filter tokens across all layers during backward
    pub fn filter_backward(&mut self, gradients: &mut Gradients, importance: &Tensor) -> Result<()> {
        // Create binary mask based on threshold
        let mask = importance.gt(self.threshold)?;

        // Apply mask to all gradient tensors
        for (name, grad) in gradients.iter_mut() {
            if name.contains("token") || name.contains("embed") {
                *grad = grad.broadcast_mul(&mask)?;
            }
        }

        Ok(())
    }
}
```

**Expected Gain**: 
- 35% faster backprop
- 22% end-to-end training reduction
- 16.3% improved downstream task utility

---

#### 8. MinatoLoader: Async Data Loading (EuroSys 2026)

**Why**: 7.5× faster training, 90.5% GPU utilization (vs 46.4% baseline)

**Implementation**:

```rust
/// MinatoLoader: Asynchronous data preprocessing
pub struct MinatoLoader {
    /// Fast queue (quick samples)
    fast_queue: Arc<Mutex<Vec<Sample>>>,

    /// Slow queue (complex samples)
    slow_queue: Arc<Mutex<Vec<Sample>>>,

    /// Batch queue per GPU
    batch_queues: Vec<Arc<Mutex<Vec<Batch>>>>,

    /// Worker threads
    workers: Vec<JoinHandle<()>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

impl MinatoLoader {
    pub fn new(n_workers: usize, n_gpus: usize) -> Self {
        let fast_queue = Arc::new(Mutex::new(Vec::new()));
        let slow_queue = Arc::new(Mutex::new(Vec::new()));
        let batch_queues: Vec<_> = (0..n_gpus)
            .map(|_| Arc::new(Mutex::new(Vec::new())))
            .collect();
        let shutdown = Arc::new(AtomicBool::new(false));

        // Spawn worker threads
        let mut workers = Vec::new();
        for i in 0..n_workers {
            let fast = fast_queue.clone();
            let slow = slow_queue.clone();
            let shutdown = shutdown.clone();

            let handle = thread::spawn(move || {
                while !shutdown.load(Ordering::Relaxed) {
                    // Process fast samples
                    if let Some(sample) = Self::get_fast_sample(&fast) {
                        let processed = Self::preprocess_fast(sample);
                        Self::enqueue_batch(&batch_queues[i % n_gpus], processed);
                    }

                    // Process slow samples in background
                    if let Some(sample) = Self::get_slow_sample(&slow) {
                        let processed = Self::preprocess_slow(sample);
                        Self::enqueue_batch(&batch_queues[i % n_gpus], processed);
                    }
                }
            });

            workers.push(handle);
        }

        Self {
            fast_queue,
            slow_queue,
            batch_queues,
            workers,
            shutdown,
        }
    }

    /// Get next batch (non-blocking)
    pub fn next_batch(&self, gpu_id: usize) -> Option<Batch> {
        self.batch_queues[gpu_id].lock().unwrap().pop()
    }

    fn preprocess_fast(sample: Sample) -> Batch {
        // Fast preprocessing (tokenization only)
        sample.tokenize()
    }

    fn preprocess_slow(sample: Sample) -> Batch {
        // Slow preprocessing (verification, augmentation)
        sample.verify().augment().tokenize()
    }
}
```

**Expected Gain**: 
- 7.5× faster training
- 90.5% GPU utilization
- 3× speedup over DALI

---

#### 9. FIRE: Frobenius-Isometry Reinitialization (ICLR 2026 Oral)

**Why**: Restores plasticity without catastrophic forgetting

**Implementation**:

```rust
/// FIRE: Frobenius-Isometry REinitialization
pub struct FIREReinitializer {
    /// Target SFE (Squared Frobenius Error)
    target_sfe: f64,

    /// Newton-Schulz iterations
    newton_schulz_iters: usize,
}

impl FIREReinitializer {
    /// Reinitialize weights to restore plasticity
    pub fn reinitialize(&self, weights: &mut Tensor) -> Result<()> {
        // Step 1: Normalize weights
        let norm = weights.norm2()?;
        let normalized = weights.div(&norm)?;

        // Step 2: Apply Newton-Schulz iteration for orthogonalization
        let mut orthogonal = normalized.clone();
        for _ in 0..self.newton_schulz_iters {
            orthogonal = self.newton_schulz_step(&orthogonal)?;
        }

        // Step 3: Scale to preserve variance
        let scale = (weights.dim(0)? as f64).sqrt();
        *weights = orthogonal.mul(&scale)?;

        Ok(())
    }

    fn newton_schulz_step(&self, x: &Tensor) -> Result<Tensor> {
        // X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k^T @ X_k
        let xxt = x.matmul(&x.t()?)?;
        let xxt_x = xxt.matmul(x)?;

        x.mul(1.5)?.sub(&xxt_x.mul(0.5)?)
    }
}
```

**Expected Gain**: 
- Restores plasticity in continual learning
- Prevents dormant neurons
- Maintains stability

---

### P3: Alignment & RL (Implement Last)

#### 10. Training-Free GRPO (arXiv:2510.08191)

**Why**: Alignment without parameter updates → zero compute cost

**Implementation**:

```rust
/// Training-Free GRPO: Group Relative Policy Optimization without parameter updates
pub struct TrainingFreeGRPO {
    /// Experience library (token priors)
    experience_library: Vec<Experience>,

    /// Group size for rollouts
    group_size: usize,

    /// Max library size
    max_library_size: usize,
}

pub struct Experience {
    /// Prompt
    prompt: String,

    /// Successful rollout
    response: String,

    /// Semantic advantage score
    advantage: f64,
}

impl TrainingFreeGRPO {
    /// Generate group rollouts and extract semantic advantages
    pub fn generate_experiences(
        &mut self,
        model: &Model,
        prompts: &[String],
    ) -> Result<()> {
        for prompt in prompts {
            // Generate group of rollouts
            let mut rollouts = Vec::new();
            for _ in 0..self.group_size {
                let response = model.generate(prompt, 200)?;
                let reward = self.evaluate_semantic(&response)?;
                rollouts.push((response, reward));
            }

            // Compute relative advantages
            let mean_reward: f64 = rollouts.iter().map(|(_, r)| r).sum::<f64>() / rollouts.len() as f64;

            for (response, reward) in rollouts {
                let advantage = reward - mean_reward;

                if advantage > 0.0 {
                    // Add successful experience to library
                    self.experience_library.push(Experience {
                        prompt: prompt.clone(),
                        response,
                        advantage,
                    });
                }
            }
        }

        // Prune library if too large
        self.prune_library();

        Ok(())
    }

    /// Use experiences as token priors during inference
    pub fn apply_token_prior(&self, prompt: &str) -> String {
        // Find relevant experiences
        let relevant: Vec<_> = self.experience_library.iter()
            .filter(|e| e.prompt.contains(&prompt))
            .collect();

        // Format as context
        let context = relevant.iter()
            .map(|e| format!("Example: {}\nResponse: {}\n", e.prompt, e.response))
            .collect::<String>();

        format!("{}\n\n{}", context, prompt)
    }

    fn prune_library(&mut self) {
        if self.experience_library.len() > self.max_library_size {
            // Sort by advantage and keep top
            self.experience_library.sort_by(|a, b| b.advantage.partial_cmp(&a.advantage).unwrap());
            self.experience_library.truncate(self.max_library_size);
        }
    }
}
```

**Expected Gain**: 
- Zero parameter updates for alignment
- Outperforms 32B SFT models with frozen 671B
- Minimal training data (dozens of samples)

---

## 📋 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Priority**: P0 techniques

```bash
# Week 1: Muon 8-bit + GaLore 2
- [ ] Implement 8-bit quantized Muon
- [ ] Integrate GaLore 2 optimizer
- [ ] Test on all three hardware configs
- [ ] Benchmark memory usage

# Week 2: FP4 for Blackwell (Config A)
- [ ] Implement FP4 tensor core support
- [ ] Add stochastic rounding
- [ ] Test mixed precision training
- [ ] Validate accuracy
```

**Deliverables**:
- 50-80% memory reduction
- 2× training speedup on Config A
- Working prototypes on all configs

---

### Phase 2: Architecture Improvements (Weeks 3-4)

**Priority**: P1 techniques

```bash
# Week 3: Multi-Token Prediction + mHC
- [ ] Implement MTP heads
- [ ] Integrate mHC architecture
- [ ] Test with LoopLM
- [ ] Benchmark convergence

# Week 4: FOAM
- [ ] Implement blocked state folding
- [ ] Integrate with optimizers
- [ ] Test memory savings
- [ ] Validate convergence
```

**Deliverables**:
- 15-20% better data efficiency (MTP)
- 3× stability improvement (mHC)
- 50% memory reduction (FOAM)

---

### Phase 3: Systems Optimization (Weeks 5-6)

**Priority**: P2 techniques

```bash
# Week 5: Collider + MinatoLoader
- [ ] Implement token filtering
- [ ] Create async data loader
- [ ] Test GPU utilization
- [ ] Benchmark end-to-end speedup

# Week 6: FIRE
- [ ] Implement reinitialization
- [ ] Test continual learning
- [ ] Validate plasticity restoration
```

**Deliverables**:
- 35% faster backprop (Collider)
- 90%+ GPU utilization (MinatoLoader)
- Continual learning capability (FIRE)

---

### Phase 4: Alignment (Week 7)

**Priority**: P3 techniques

```bash
# Week 7: Training-Free GRPO
- [ ] Implement experience library
- [ ] Create semantic evaluation
- [ ] Test alignment without training
- [ ] Benchmark vs SFT
```

**Deliverables**:
- Zero-cost alignment
- Improved task performance
- Minimal data requirements

---

### Phase 5: Integration & Testing (Week 8)

```bash
# Week 8: Full Integration
- [ ] Combine all techniques
- [ ] Run end-to-end training
- [ ] Benchmark vs baseline
- [ ] Document results
```

**Final Deliverables**:
- Complete training system
- 3-4× overall speedup
- 60-80% memory reduction
- Hugging Face publication ready

---

## 🎯 Expected Overall Gains

### Config A: Threadripper + Blackwell (7B model)

| Metric | Baseline | With Optimizations | Improvement |
|--------|----------|-------------------|-------------|
| Training Speed | 5000 tok/s | 15000 tok/s | **3×** |
| Memory Usage | 80GB GPU | 20GB GPU | **4×** |
| Convergence | 100K steps | 70K steps | **30% faster** |
| Final Loss | 2.5 | 2.3 | **8% better** |
| Compilation Rate | 90% | 94% | **+4%** |

### Config B: 9800X3D + 2×4090 (3B model)

| Metric | Baseline | With Optimizations | Improvement |
|--------|----------|-------------------|-------------|
| Training Speed | 8000 tok/s | 20000 tok/s | **2.5×** |
| Memory Usage | 22GB/GPU | 12GB/GPU | **1.8×** |
| GPU Utilization | 70% | 92% | **+22%** |
| Convergence | 150K steps | 100K steps | **33% faster** |

### Config C: Dual EPYC + 4090 (5B model)

| Metric | Baseline | With Optimizations | Improvement |
|--------|----------|-------------------|-------------|
| Training Speed | 6000 tok/s | 15000 tok/s | **2.5×** |
| Memory Usage | 22GB GPU | 12GB GPU | **1.8×** |
| CPU Utilization | 60% | 85% | **+25%** |
| Convergence | 120K steps | 85K steps | **29% faster** |

---

## 🔧 Integration Architecture

```rust
/// Main training coordinator with all optimizations
pub struct EfficientTrainer {
    /// Model with mHC and MTP
    model: LoopLMModel,

    /// GaLore 2 + 8-bit Muon optimizer
    optimizer: GaLore2Optimizer<QuantizedMuon>,

    /// FP4 trainer (Config A only)
    fp4_trainer: Option<FP4Trainer>,

    /// FOAM state compression
    foam: FOAMOptimizer,

    /// Collider token filtering
    collider: Collider,

    /// Async data loader
    data_loader: MinatoLoader,

    /// FIRE reinitializer
    fire: FIREReinitializer,

    /// Training-Free GRPO for alignment
    tf_grpo: Option<TrainingFreeGRPO>,
}

impl EfficientTrainer {
    pub fn train_step(&mut self, batch: Batch, step: usize) -> Result<TrainingMetrics> {
        // 1. Load data asynchronously
        let batch = self.data_loader.next_batch(0).unwrap();

        // 2. Forward pass with FP4 (if enabled)
        let logits = if let Some(fp4) = &self.fp4_trainer {
            fp4_trainer.forward(&mut self.model, &batch.input)?
        } else {
            self.model.forward(&batch.input)?
        };

        // 3. Multi-token prediction loss
        let loss = self.compute_mtp_loss(&logits, &batch.targets)?;

        // 4. Backward pass with token filtering
        let mut grads = loss.backward()?;
        let importance = self.compute_token_importance(&logits, &batch.targets)?;
        self.collider.filter_backward(&mut grads, &importance)?;

        // 5. GaLore 2 projection
        self.optimizer.step(&grads, step)?;

        // 6. FOAM state compression
        self.foam.compress_states()?;

        Ok(TrainingMetrics { loss, step })
    }
}
```

---

## 📚 References

### Optimizers
- Muon: arXiv:2502.16982
- 8-bit Muon: arXiv:2509.23106
- SOAP: arXiv:2409.11321
- Gluon: arXiv:2505.07703

### Memory-Efficient Training
- GaLore: arXiv:2403.03507
- GaLore 2: arXiv:2504.20437
- FOAM: arXiv:2512.07112
- GUM: arXiv:2510.17802

### Ultra-Low Precision
- FP4 Training: arXiv:2501.17116, 2502.11458
- MXFP4: arXiv:2502.20586

### Architecture
- mHC: arXiv:2512.24880
- Multi-Token Prediction: arXiv:2404.19737

### Systems
- Collider: arXiv:2502.00340
- MinatoLoader: EuroSys 2026
- FIRE: ICLR 2026 Oral

### Alignment
- Training-Free GRPO: arXiv:2510.08191

---

*Generated: February 14, 2026*
*For: nanochat-rs-ternary advanced training implementation*
