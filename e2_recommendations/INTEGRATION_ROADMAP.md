# nanochat-rs-ternary: Complete Integration Roadmap

## Executive Summary

This document provides a step-by-step integration plan for incorporating cutting-edge 2025-2026 training efficiency techniques into nanochat-rs-ternary, tailored to your three hardware configurations.

**Target Outcomes**:
- **3-4× training speedup**
- **60-80% memory reduction**
- **Superior model quality** (compilation success >94%, HumanEval-Rust >72%)
- **Hugging Face publication ready** in 8 weeks

---

## 🎯 Priority Integration Matrix

| Technique | Config A | Config B | Config C | Impact | Timeline |
|-----------|----------|----------|----------|--------|----------|
| **8-bit Muon** | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ | Week 1 |
| **GaLore 2** | ✅ | ✅✅ | ✅ | ⭐⭐⭐⭐⭐ | Week 1-2 |
| **FP4 Training** | ✅✅ | ❌ | ❌ | ⭐⭐⭐⭐⭐ | Week 2 |
| **Multi-Token Prediction** | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | Week 3 |
| **mHC Architecture** | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | Week 3-4 |
| **FOAM** | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | Week 4 |
| **Collider** | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | Week 5 |
| **MinatoLoader** | ✅ | ✅ | ✅ | ⭐⭐⭐ | Week 5 |
| **FIRE** | ✅ | ✅ | ✅ | ⭐⭐⭐ | Week 6 |
| **Training-Free GRPO** | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | Week 7 |

**Legend**: ✅✅ = Critical, ✅ = Important, ❌ = Not applicable

---

## 📅 Week-by-Week Implementation Plan

### Week 1: Optimizer Upgrades (P0)

**Goal**: Implement 8-bit Muon + GaLore 2

#### Day 1-2: 8-bit Quantized Muon
```rust
// In crates/nanochat-train/src/optimizers/quantized_muon.rs

pub struct QuantizedMuon {
    base: MuonOptimizer,
    quantizer: BlockwiseQuantizer,
    bits: u8,  // 8-bit for optimizer states
}

impl QuantizedMuon {
    pub fn step_quantized(&mut self, grad: &Tensor) -> Result<Tensor> {
        // 1. Quantize gradient to 8-bit blocks
        let q_grad = self.quantizer.quantize(grad, 8)?;

        // 2. Apply Newton-Schulz in quantized space
        let update = self.base.newton_schulz(&q_grad)?;

        // 3. Dequantize for weight update
        self.quantizer.dequantize(&update)
    }
}
```

**Expected Gain**: 86% memory reduction in optimizer states

#### Day 3-5: GaLore 2 Integration
```rust
// In crates/nanochat-train/src/optimizers/galore2.rs

pub struct GaLore2Optimizer<OPT> {
    base: OPT,
    rank: usize,
    update_freq: usize,
    projections: HashMap<String, ProjectionPair>,
}

impl<OPT: Optimizer> GaLore2Optimizer<OPT> {
    pub fn step(&mut self, grads: &Gradients, step: usize) -> Result<()> {
        // Update projections every N steps
        if step % self.update_freq == 0 {
            self.update_projections(grads)?;
        }

        // Project gradients to low-rank subspace
        let projected = self.project_gradients(grads)?;

        // Apply base optimizer
        self.base.step(&projected)
    }
}
```

**Hardware-Specific Ranks**:
- Config A (7B): rank=512
- Config B (3B): rank=256
- Config C (5B): rank=384

**Expected Gain**: 50-65% memory reduction, enables 7B on 24GB GPU

#### Day 6-7: Testing & Benchmarking
```bash
# Test on all three configs
cargo test --workspace --test optimizer_tests

# Benchmark memory usage
cargo run --release --example benchmark_memory --   --config configs/config_a.toml   --optimizer galore2_quantized_muon
```

**Deliverables**:
- [ ] 8-bit Muon implementation
- [ ] GaLore 2 integration
- [ ] Memory benchmarks showing 50-80% reduction
- [ ] All tests passing

---

### Week 2: FP4 Training for Blackwell (P0)

**Goal**: Implement FP4 mixed-precision training for Config A

#### Day 1-2: FP4 Tensor Core Support
```rust
// In crates/nanochat-train/src/fp4_trainer.rs

pub struct FP4Trainer {
    forward_dtype: DType,  // BF16
    backward_dtype: DType, // FP4
    stochastic_rounding: bool,
    fp4_table: [f32; 16],  // E2M1 format
}

impl FP4Trainer {
    pub fn enable_blackwell_fp4(&self) -> Result<()> {
        // Enable FP4 tensor cores on Blackwell
        unsafe {
            cudaDeviceSetFp4Mode(1);
        }
        Ok(())
    }

    pub fn quantize_fp4(&self, tensor: &Tensor) -> Result<Tensor> {
        // Stochastic rounding to FP4
        tensor.stochastic_quantize(4, &self.fp4_table)
    }
}
```

#### Day 3-4: Mixed Precision Configuration
```toml
# configs/config_a_fp4.toml
[training.fp4]
enabled = true
forward_precision = "BF16"
backward_precision = "FP4"
stochastic_rounding = true

[training.fp4.modules]
embedding = "FP16"      # Keep high precision
attention = "FP8"       # FP8 for attention
ffn = "FP4"             # FP4 for FFN (largest)
norm = "FP32"           # Full precision for stability
```

#### Day 5-7: Integration & Testing
```bash
# Test FP4 training
cargo run --release --example train_fp4 --   --config configs/config_a_fp4.toml   --model-size 7b   --batch-size 32

# Benchmark vs BF16
cargo run --release --example benchmark_fp4 --   --compare bf16,fp4
```

**Expected Gain**: 2-3× speedup, 80% memory reduction (Config A only)

**Deliverables**:
- [ ] FP4 tensor core support
- [ ] Mixed precision training
- [ ] Speed benchmarks showing 2-3× improvement
- [ ] Accuracy validation (within 1% of BF16)

---

### Week 3: Multi-Token Prediction + mHC (P1)

**Goal**: Implement MTP and mHC architecture

#### Day 1-3: Multi-Token Prediction
```rust
// In crates/nanochat-model/src/mtp.rs

pub struct MultiTokenPrediction {
    n_future_tokens: usize,
    output_heads: Vec<Linear>,
    loss_weights: Vec<f64>,  // [1.0, 0.5, 0.25, 0.125]
}

impl MultiTokenPrediction {
    pub fn compute_loss(&self, predictions: &[Tensor], targets: &[Tensor]) -> Result<f64> {
        let mut total_loss = 0.0;

        for (i, (pred, target)) in predictions.iter().zip(targets.iter()).enumerate() {
            let loss = cross_entropy(pred, target)?;
            total_loss += loss * self.loss_weights[i];
        }

        Ok(total_loss)
    }
}
```

**Integration with LoopLM**:
```rust
// In LoopLM forward pass
let hidden = self.apply_shared_layers(input)?;

// Primary prediction (next token)
let primary_logits = self.lm_head.forward(&hidden)?;

// Auxiliary predictions (future tokens)
let aux_logits = self.mtp_heads.forward(&hidden)?;

// Combined loss
let loss = cross_entropy(&primary_logits, &targets[0])? + 
           0.2 * self.mtp_loss(&aux_logits, &targets[1..]?);
```

**Expected Gain**: 15-20% better data efficiency

#### Day 4-7: mHC Architecture
```rust
// In crates/mhc-lite/src/mhc_advanced.rs

pub struct ManifoldConstrainedHC {
    n_streams: usize,
    alpha_logits: Tensor,
    sinkhorn_iters: usize,
}

impl ManifoldConstrainedHC {
    pub fn compute_mixing_matrix(&self) -> Result<Tensor> {
        let mut matrix = self.alpha_logits.softmax(1)?;

        // Sinkhorn-Knopp for doubly stochastic
        for _ in 0..self.sinkhorn_iters {
            matrix = matrix.normalize_rows()?;
            matrix = matrix.normalize_cols()?;
        }

        Ok(matrix)
    }
}
```

**Expected Gain**: 3× stability improvement, +2.3% on DROP

**Deliverables**:
- [ ] MTP implementation
- [ ] mHC integration
- [ ] Convergence benchmarks
- [ ] Stability analysis

---

### Week 4: FOAM (P1)

**Goal**: Implement blocked state folding

```rust
// In crates/nanochat-train/src/foam.rs

pub struct FOAMOptimizer<OPT> {
    base: OPT,
    block_size: usize,
    compressed_states: HashMap<String, CompressedState>,
}

pub struct CompressedState {
    block_means: Tensor,  // 1 value per block
    residual: Tensor,     // Sparse residual
}

impl<OPT: Optimizer> FOAMOptimizer<OPT> {
    pub fn compress(&self, state: &Tensor) -> Result<CompressedState> {
        let blocks = state.reshape((n_blocks, block_size))?;
        let block_means = blocks.mean_keepdim(1)?;
        let residual = blocks.sub(&block_means.broadcast_as(blocks.shape())?)?;

        Ok(CompressedState { block_means, residual })
    }
}
```

**Expected Gain**: 50% total training memory reduction

**Deliverables**:
- [ ] FOAM implementation
- [ ] Memory benchmarks
- [ ] Convergence validation

---

### Week 5: Systems Optimization (P2)

#### Day 1-3: Collider Token Filtering
```rust
// In crates/nanochat-train/src/collider.rs

pub struct Collider {
    threshold: f64,
    filter_backward: bool,
}

impl Collider {
    pub fn filter_backward(&self, grads: &mut Gradients, mask: &Tensor) -> Result<()> {
        for (name, grad) in grads.iter_mut() {
            if self.should_filter(name) {
                *grad = grad.broadcast_mul(mask)?;
            }
        }
        Ok(())
    }
}
```

**Expected Gain**: 35% faster backprop, 22% end-to-end speedup

#### Day 4-5: MinatoLoader
```rust
// In crates/nanochat-data/src/minato_loader.rs

pub struct MinatoLoader {
    fast_queue: Arc<Mutex<Vec<Sample>>>,
    slow_queue: Arc<Mutex<Vec<Sample>>>,
    batch_queues: Vec<Arc<Mutex<Vec<Batch>>>>,
}

impl MinatoLoader {
    pub fn next_batch(&self, gpu_id: usize) -> Option<Batch> {
        self.batch_queues[gpu_id].lock().unwrap().pop()
    }
}
```

**Expected Gain**: 90.5% GPU utilization (vs 46.4% baseline)

**Deliverables**:
- [ ] Collider implementation
- [ ] MinatoLoader integration
- [ ] GPU utilization benchmarks

---

### Week 6: FIRE + Continual Learning (P2)

```rust
// In crates/nanochat-train/src/fire.rs

pub struct FIRE {
    newton_schulz_iters: usize,
}

impl FIRE {
    pub fn reinitialize(&self, weights: &mut Tensor) -> Result<()> {
        // Newton-Schulz iteration for orthogonalization
        let mut orthogonal = weights.normalize()?;
        for _ in 0..self.newton_schulz_iters {
            orthogonal = self.newton_schulz_step(&orthogonal)?;
        }
        *weights = orthogonal * weights.dim(0)?.sqrt();
        Ok(())
    }
}
```

**Expected Gain**: Restores plasticity, prevents dormant neurons

**Deliverables**:
- [ ] FIRE implementation
- [ ] Continual learning tests
- [ ] Plasticity benchmarks

---

### Week 7: Training-Free GRPO (P3)

```rust
// In crates/nanochat-rl/src/training_free_grpo.rs

pub struct TrainingFreeGRPO {
    experience_library: Vec<Experience>,
    group_size: usize,
}

impl TrainingFreeGRPO {
    pub fn generate_experiences(&mut self, model: &Model, prompts: &[String]) -> Result<()> {
        for prompt in prompts {
            // Generate group rollouts
            let rollouts = (0..self.group_size)
                .map(|_| model.generate(prompt, 200))
                .collect::<Result<Vec<_>>>()?;

            // Compute semantic advantages
            let mean_reward = rollouts.iter().map(|r| self.evaluate(r)).sum::<f64>() / rollouts.len() as f64;

            // Add successful experiences to library
            for rollout in rollouts {
                if self.evaluate(&rollout) > mean_reward {
                    self.experience_library.push(Experience {
                        prompt: prompt.clone(),
                        rollout,
                    });
                }
            }
        }
        Ok(())
    }
}
```

**Expected Gain**: Zero-cost alignment, outperforms 32B SFT models

**Deliverables**:
- [ ] Training-Free GRPO implementation
- [ ] Alignment benchmarks
- [ ] Comparison with SFT

---

### Week 8: Integration & Final Testing

#### Day 1-3: Full System Integration
```rust
// Main training coordinator
pub struct EfficientTrainer {
    model: LoopLMModel,           // With mHC + MTP
    optimizer: GaLore2<QuantizedMuon>,  // 8-bit + low-rank
    fp4_trainer: Option<FP4Trainer>,    // Config A only
    foam: FOAMOptimizer,
    collider: Collider,
    data_loader: MinatoLoader,
    fire: FIRE,
    tf_grpo: Option<TrainingFreeGRPO>,
}
```

#### Day 4-5: End-to-End Testing
```bash
# Full training run on all configs
./train_config_a.sh  # 7B model
./train_config_b.sh  # 3B model
./train_config_c.sh  # 5B model

# Benchmark vs baseline
./benchmark_all.sh
```

#### Day 6-7: Documentation & Publication Prep
- [ ] Model cards
- [ ] Benchmark results
- [ ] Hugging Face repository
- [ ] Technical report

---

## 📊 Expected Final Results

### Config A: Threadripper 3995WX + 96GB Blackwell (7B model)

| Metric | Baseline | With All Optimizations | Improvement |
|--------|----------|----------------------|-------------|
| **Training Speed** | 5K tok/s | 20K tok/s | **4×** |
| **GPU Memory** | 80GB | 16GB | **5×** |
| **Total Steps** | 100K | 60K | **40% faster** |
| **Final Loss** | 2.5 | 2.2 | **12% better** |
| **Compile Rate** | 90% | 95% | **+5%** |
| **HumanEval** | 68% | 75% | **+7%** |
| **Training Time** | 7 days | 2.5 days | **2.8× faster** |

### Config B: 9800X3D + 2× RTX 4090 (3B model)

| Metric | Baseline | With All Optimizations | Improvement |
|--------|----------|----------------------|-------------|
| **Training Speed** | 8K tok/s | 24K tok/s | **3×** |
| **GPU Memory** | 22GB/GPU | 10GB/GPU | **2.2×** |
| **GPU Utilization** | 70% | 93% | **+23%** |
| **Total Steps** | 150K | 90K | **40% faster** |
| **Training Time** | 5 days | 1.8 days | **2.8× faster** |

### Config C: Dual EPYC 56-core + RTX 4090 (5B model)

| Metric | Baseline | With All Optimizations | Improvement |
|--------|----------|----------------------|-------------|
| **Training Speed** | 6K tok/s | 18K tok/s | **3×** |
| **GPU Memory** | 22GB | 10GB | **2.2×** |
| **CPU Utilization** | 60% | 88% | **+28%** |
| **Total Steps** | 120K | 72K | **40% faster** |
| **Training Time** | 6 days | 2.2 days | **2.7× faster** |

---

## 🏆 Hugging Face Publication Checklist

### Model Repository Structure
```
nanochat-rs-ternary-7b/
├── README.md                 # Model card
├── MODEL_CARD.md            # Detailed specs
├── config.json              # Model configuration
├── model.gguf               # Quantized weights
├── tokenizer.json           # Tokenizer
├── generation_config.json   # Generation settings
├── evaluation/
│   ├── humaneval_rust.json
│   ├── compilation_rate.json
│   ├── semantic_correctness.json
│   └── performance_benchmarks.json
├── examples/
│   ├── basic_usage.rs
│   ├── semantic_verification.rs
│   └── compiler_integration.rs
└── training/
    ├── config.toml
    ├── training_log.txt
    └── reproducibility.md
```

### Publication Steps
1. [ ] Upload model weights (GGUF format)
2. [ ] Create model card with benchmarks
3. [ ] Add example code
4. [ ] Write technical report
5. [ ] Announce on social media
6. [ ] Submit to paperswithcode

---

## 📚 References

### Optimizers
- Muon: arXiv:2502.16982
- 8-bit Muon: arXiv:2509.23106
- GaLore: arXiv:2403.03507
- GaLore 2: arXiv:2504.20437
- FOAM: arXiv:2512.07112

### Low-Precision Training
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

## 🤝 Contributing

We welcome contributions! Priority areas:
- Additional semantic checks
- More efficient kernels
- Better data preprocessing
- Extended benchmarks

---

*Generated: February 14, 2026*
*For: nanochat-rs-ternary advanced training implementation*
