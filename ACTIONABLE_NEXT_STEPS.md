# nanochat-rs-ternary: Actionable Next Steps

## 🎯 Priority Matrix

| Priority | Task | Effort | Impact | Status |
|----------|------|--------|--------|--------|
| P0 | Test GaLore 2 + 8-bit Muon | Low | High | 🔴 Not Started |
| P0 | Add TensorBoard logging | Low | Medium | 🔴 Not Started |
| P1 | Implement Multi-Token Prediction | Medium | High | 🔴 Not Started |
| P1 | Add Token Filtering | Medium | High | 🔴 Not Started |
| P1 | Async Data Loader | Medium | High | 🔴 Not Started |
| P2 | FP4 for Blackwell | Medium | Very High | 🔴 Not Started |
| P2 | mHC Improvements | Medium | Medium | 🔴 Not Started |
| P3 | Training-Free GRPO | Low | High | 🔴 Not Started |

---

## 🔴 P0: Critical (Do This Week)

### Task 1: Test GaLore 2 + 8-bit Muon Integration

**Why**: This combination should give 50-80% memory reduction

**Steps**:
```bash
# 1. Create test configuration
cat > configs/test_galore_muon.toml << 'EOF'
[optimizer]
type = "GaLore2Muon"
base_optimizer = "MuonQuantized"
galore_rank = 256
galore_update_freq = 500
muon_bits = 8
muon_block_size = 128
EOF

# 2. Run memory benchmark
cargo run --release --example benchmark_memory --   --config configs/test_galore_muon.toml   --model-size 3b   --batch-size 4

# 3. Compare with baseline
cargo run --release --example benchmark_memory --   --config configs/baseline.toml   --model-size 3b   --batch-size 4
```

**Expected Result**: 50-65% memory reduction

**Success Criteria**:
- [ ] Memory usage reduced by >50%
- [ ] Training converges normally
- [ ] No numerical instability
- [ ] All tests pass

---

### Task 2: Add TensorBoard Logging

**Why**: Better training visualization and debugging

**Implementation**:
```rust
// In crates/nanochat-train/src/logging.rs

use tboard::SummaryWriter;

pub struct TensorBoardLogger {
    writer: SummaryWriter,
}

impl TensorBoardLogger {
    pub fn new(log_dir: &str) -> Self {
        Self {
            writer: SummaryWriter::new(log_dir),
        }
    }

    pub fn log_step(&mut self, step: usize, metrics: &TrainingMetrics) {
        self.writer.add_scalar("loss/total", metrics.loss, step);
        self.writer.add_scalar("loss/ce", metrics.ce_loss, step);
        self.writer.add_scalar("loss/entropy", metrics.entropy, step);
        self.writer.add_scalar("training/lr", metrics.learning_rate, step);
        self.writer.add_scalar("training/grad_norm", metrics.grad_norm, step);
        self.writer.add_scalar("throughput/tokens_per_sec", metrics.tokens_per_sec, step);

        if let Some(gain) = metrics.mhc_gain {
            self.writer.add_scalar("mhc/composite_gain", gain, step);
        }
    }
}
```

**Add to Cargo.toml**:
```toml
[dependencies]
tboard = "0.4"
```

**Usage**:
```bash
# Start training with TensorBoard
cargo run --release --example train --   --tensorboard-dir runs/experiment_1

# View in TensorBoard
tensorboard --logdir runs/
```

---

## 🟡 P1: High Priority (Next 2 Weeks)

### Task 3: Multi-Token Prediction

**Why**: 15-20% better data efficiency

**Implementation**:
```rust
// In crates/nanochat-model/src/mtp.rs

pub struct MultiTokenPrediction {
    n_future_tokens: usize,
    output_heads: Vec<Linear>,
    loss_weights: Vec<f64>,
}

impl MultiTokenPrediction {
    pub fn new(vb: VarBuilder, dim: usize, vocab_size: usize, n_future: usize) -> Result<Self> {
        let mut heads = Vec::new();
        let mut weights = Vec::new();

        for i in 0..n_future {
            let head = candle_nn::linear(dim, vocab_size, vb.pp(format!("head_{}", i)))?;
            heads.push(head);
            weights.push(0.5_f64.powi(i as i32));
        }

        Ok(Self {
            n_future_tokens: n_future,
            output_heads: heads,
            loss_weights: weights,
        })
    }

    pub fn compute_loss(&self, predictions: &[Tensor], targets: &[Tensor]) -> Result<f64> {
        let mut total_loss = 0.0;

        for (i, (pred, target)) in predictions.iter().zip(targets.iter()).enumerate() {
            let loss = candle_nn::losses::cross_entropy(
                &candle_nn::ops::log_softmax(pred, D::Minus1)?,
                target,
            )?;
            total_loss += loss.to_scalar::<f64>()? * self.loss_weights[i];
        }

        Ok(total_loss)
    }
}
```

**Integration**:
```rust
// In training loop
let mtp_logits = model.mtp_heads.forward(&hidden)?;
let mtp_loss = model.mtp_heads.compute_loss(&mtp_logits, &future_targets)?;
let total_loss = primary_loss + 0.2 * mtp_loss;
```

---

### Task 4: Token Filtering (Collider-style)

**Why**: 35% faster backprop

**Implementation**:
```rust
// In crates/nanochat-train/src/token_filter.rs

pub struct TokenFilter {
    threshold: f64,
}

impl TokenFilter {
    pub fn compute_importance(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;
        let losses = candle_nn::losses::cross_entropy_loss_per_token(&log_probs, targets)?;

        // Normalize to [0, 1]
        let min = losses.min_keepdim(1)?;
        let max = losses.max_keepdim(1)?;
        let normalized = losses.sub(&min)?.div(&max.sub(&min)?.add(1e-8)?)?;

        Ok(normalized)
    }

    pub fn filter_backward(&self, grads: &mut Gradients, importance: &Tensor) -> Result<()> {
        let mask = importance.gt(self.threshold)?;

        for (name, grad) in grads.iter_mut() {
            if self.should_filter(name) {
                let expanded_mask = mask.expand(grad.shape())?;
                *grad = grad.broadcast_mul(&expanded_mask)?;
            }
        }

        Ok(())
    }
}
```

---

### Task 5: Async Data Loader

**Why**: 90%+ GPU utilization

**Implementation**:
```rust
// In crates/nanochat-data/src/async_loader.rs

pub struct AsyncDataLoader {
    fast_queue: Arc<Mutex<Vec<Batch>>>,
    slow_queue: Arc<Mutex<Vec<Batch>>>,
    batch_queues: Vec<Arc<Mutex<Vec<Batch>>>>,
    workers: Vec<JoinHandle<()>>,
}

impl AsyncDataLoader {
    pub fn new(n_workers: usize, n_gpus: usize) -> Self {
        // Spawn worker threads for preprocessing
        // Separate fast and slow paths
    }

    pub fn next_batch(&self, gpu_id: usize) -> Option<Batch> {
        // Prioritize fast batches
        self.batch_queues[gpu_id].lock().unwrap().pop()
    }
}
```

---

## 🟢 P2: Medium Priority (Next Month)

### Task 6: FP4 Training for Blackwell (Config A)

**Why**: 2-3× speedup on Blackwell GPUs

**Implementation**:
```rust
// In crates/nanochat-train/src/fp4.rs

pub struct FP4Trainer {
    forward_dtype: DType,   // BF16
    backward_dtype: DType,  // FP4
    stochastic_rounding: bool,
}

impl FP4Trainer {
    pub fn enable_blackwell_fp4(&self) -> Result<()> {
        // Enable FP4 tensor cores
        unsafe {
            cuda::enable_fp4_tensor_cores();
        }
        Ok(())
    }

    pub fn quantize_fp4(&self, tensor: &Tensor) -> Result<Tensor> {
        // E2M1 FP4 format
        // 16 values: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
        tensor.stochastic_quantize(4)
    }
}
```

**Configuration** (Config A only):
```toml
[training.fp4]
enabled = true
forward_precision = "BF16"
backward_precision = "FP4"
stochastic_rounding = true
```

---

### Task 7: mHC Improvements

**Why**: 3× stability improvement

**Implementation**:
```rust
// In crates/mhc-lite/src/mhc.rs

impl MhcLite {
    fn compute_mixing_matrix(&self) -> Result<Tensor> {
        let mut matrix = self.alpha.softmax(1)?;

        // Sinkhorn-Knopp iterations for doubly stochastic
        for _ in 0..5 {
            // Normalize rows
            let row_sums = matrix.sum_keepdim(1)?;
            matrix = matrix.broadcast_div(&row_sums)?;

            // Normalize columns
            let col_sums = matrix.sum_keepdim(0)?;
            matrix = matrix.broadcast_div(&col_sums)?;
        }

        Ok(matrix)
    }
}
```

---

## 🔵 P3: Low Priority (When Time Permits)

### Task 8: Training-Free GRPO

**Why**: Zero-cost alignment

**Implementation**:
```rust
// In crates/nanochat-rl/src/training_free_grpo.rs

pub struct TrainingFreeGRPO {
    experience_library: Vec<Experience>,
    group_size: usize,
}

impl TrainingFreeGRPO {
    pub fn generate_experiences(&mut self, model: &Model, prompts: &[String]) {
        for prompt in prompts {
            let rollouts: Vec<_> = (0..self.group_size)
                .map(|_| model.generate(prompt, 200))
                .collect();

            let rewards: Vec<_> = rollouts.iter()
                .map(|r| self.evaluate_semantic(r))
                .collect();

            let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;

            for (rollout, reward) in rollouts.iter().zip(rewards.iter()) {
                if *reward > mean_reward {
                    self.experience_library.push(Experience {
                        prompt: prompt.clone(),
                        rollout: rollout.clone(),
                        advantage: reward - mean_reward,
                    });
                }
            }
        }
    }
}
```

---

## 📋 Weekly Sprint Plan

### Week 1: P0 Critical
```bash
# Day 1-2: Test GaLore 2 + 8-bit Muon
cargo test --test optimizer_tests
cargo run --example benchmark_memory

# Day 3-4: Add TensorBoard
cargo add tboard
# Implement TensorBoardLogger

# Day 5: Fix unwrap() calls
# Replace with proper error handling

# Day 6-7: Create benchmark suite
# Document results
```

### Week 2: P1 High Priority
```bash
# Day 1-3: Multi-Token Prediction
# Implement MTP heads
# Integrate with training loop

# Day 4-5: Token Filtering
# Implement Collider-style filtering
# Test with training

# Day 6-7: Async Data Loader
# Implement MinatoLoader-style loader
# Test GPU utilization
```

### Week 3: P2 Medium Priority
```bash
# Day 1-4: FP4 Training (Config A)
# Implement FP4 quantization
# Test on Blackwell

# Day 5-7: mHC Improvements
# Add Sinkhorn iterations
# Test stability
```

### Week 4: P3 + Polish
```bash
# Day 1-2: Training-Free GRPO
# Implement experience library

# Day 3-4: Documentation
# Update README
# Add troubleshooting guide

# Day 5-7: Final Testing
# Run full benchmarks
# Prepare Hugging Face publication
```

---

## 🎯 Success Criteria

### Week 1
- [ ] GaLore 2 + 8-bit Muon tested
- [ ] Memory reduction >50% confirmed
- [ ] TensorBoard logging working
- [ ] All unwrap() calls fixed

### Week 2
- [ ] MTP implemented and tested
- [ ] Token filtering working
- [ ] Async loader implemented
- [ ] GPU utilization >90%

### Week 3
- [ ] FP4 training on Blackwell
- [ ] 2× speedup confirmed
- [ ] mHC improvements tested
- [ ] Stability improved

### Week 4
- [ ] Training-Free GRPO working
- [ ] Documentation complete
- [ ] All benchmarks passing
- [ ] Ready for Hugging Face

---

## 📚 Resources

### Papers
- GaLore 2: arXiv:2504.20437
- 8-bit Muon: arXiv:2509.23106
- Multi-Token Prediction: arXiv:2404.19737
- Collider: arXiv:2502.00340
- FP4 Training: arXiv:2501.17116
- mHC: arXiv:2512.24880
- Training-Free GRPO: arXiv:2510.08191

### Documentation
- TensorBoard: https://www.tensorflow.org/tensorboard
- Candle: https://github.com/huggingface/candle
- Rust Tracing: https://docs.rs/tracing

---

*Generated: February 15, 2026*
*Next Review: February 22, 2026*
