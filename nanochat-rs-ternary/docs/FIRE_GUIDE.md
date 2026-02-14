# FIRE: Frobenius-Isometry Reinitialization Guide

## Overview

FIRE (Frobenius-Isometry Reinitialization) restores neural network plasticity without catastrophic forgetting. It uses Newton-Schulz iterations to orthogonalize weight matrices while preserving variance, enabling continual learning and preventing dormant neurons.

**Based on**: "FIRE: Efficient Neural Network Reinitialization via Frobenius Isometry" (ICLR 2026 Oral)

**Key benefit**: Restores plasticity in continual learning scenarios

## Quick Start

```rust
use nanochat_train::optim::{FIREReinitializer, FIREConfig};
use candle_core::Tensor;

// Create FIRE reinitializer with default config
let fire = FIREReinitializer::new();

// Reinitialize a weight matrix
let weights = /* your weight tensor */;
let stats = fire.reinitialize(&weights)?;

if stats.is_successful() {
    println!("Reinitialized: norm={:.3}, orth_err={:.6}",
             stats.final_norm, stats.orthogonality_error);
}
```

## When to Use FIRE

### 1. Continual Learning
**Problem**: Neural networks forget previous tasks when trained on new tasks (catastrophic forgetting)
**Solution**: Periodically reinitialize dormant layers with FIRE to restore plasticity

```rust
// After switching to a new task
if should_reinitialize_for_new_task() {
    for layer in model.layers.iter_mut() {
        fire.reinitialize(&layer.weights)?;
    }
}
```

### 2. Escaping Local Minima
**Problem**: Training gets stuck in poor local minima with high loss
**Solution**: Reinitialize underperforming layers to explore new regions

```rust
// If loss plateaus for too long
if training_plateaued() {
    // Reinitialize last few layers
    for layer in model.layers.iter_mut().rev().take(3) {
        fire.reinitialize(&layer.weights)?;
    }
}
```

### 3. Preventing Dormant Neurons
**Problem**: Some neurons stop learning (gradients near zero)
**Solution**: Detect dormant neurons and reinitialize

```rust
// Check gradient magnitude
if fire.should_reinitialize(&weights, Some(&gradients))? {
    let stats = fire.reinitialize(&weights)?;
    println!("Reinitialized dormant layer");
}
```

### 4. Multi-Task Learning
**Problem**: Model capacity is wasted on previous tasks
**Solution**: Reinitialize task-specific layers between tasks

```rust
// When switching tasks
for task_specific_layer in model.task_layers.iter_mut() {
    fire.reinitialize(&task_specific_layer.weights)?;
}
```

## Configuration

### Default Configuration

```rust
let config = FIREConfig::default();
// target_sfe: 1e-5          (high precision orthogonalization)
// newton_schulz_iters: 8    (8 iterations usually sufficient)
// variance_factor: 1.0       (exact variance preservation)
// min_weight_norm: 1e-3      (skip very small weights)
```

### Custom Configuration

```rust
let config = FIREConfig {
    target_sfe: 1e-3,           // Less strict (faster)
    newton_schulz_iters: 5,     // Fewer iterations
    variance_factor: 0.9,        // 90% variance preservation
    min_weight_norm: 1e-4,       // Reinit smaller weights
};

let fire = FIREReinitializer::with_config(config);
```

### Parameter Guide

#### `target_sfe` (Squared Frobenius Error)
- **Range**: [1e-6, 1e-2]
- **Default**: 1e-5
- **Lower = more orthogonal** but slower convergence
- **Higher = faster** but less precise

| Value | Precision | Speed | Use Case |
|-------|-----------|-------|----------|
| 1e-6 | Highest | Slowest | Critical stability |
| 1e-5 | High | Medium | Recommended default |
| 1e-3 | Medium | Fast | Quick reinitialization |
| 1e-2 | Low | Fastest | Experimental only |

#### `newton_schulz_iters`
- **Range**: [3, 15]
- **Default**: 8
- **More iterations = better orthogonality** but slower

| Iterations | Typical SFE | Use Case |
|------------|-------------|----------|
| 3-5 | ~1e-3 | Fast reinit, low precision OK |
| 8-10 | ~1e-5 | Recommended for most cases |
| 12-15 | ~1e-6 | High precision continual learning |

#### `variance_factor`
- **Range**: [0.5, 1.5]
- **Default**: 1.0
- **Scales final weights** to preserve activation magnitudes

| Factor | Effect | Use Case |
|--------|--------|----------|
| 0.8 | Reduce magnitudes | Prevent explosions |
| 1.0 | Preserve exactly | Standard (recommended) |
| 1.2 | Increase magnitudes | Boost weak signals |

#### `min_weight_norm`
- **Range**: [1e-5, 1e-2]
- **Default**: 1e-3
- **Skip reinitialization** if Frobenius norm below this

## Algorithm Details

### Newton-Schulz Iteration

FIRE uses Newton-Schulz to orthogonalize weight matrices:

```
X_{k+1} = 1.5 * X_k - 0.5 * (X_k @ X_k^T @ X_k)
```

**Convergence**: Quadratic (doubles precision each iteration)

**Termination**: When ||X^T X - I||_F^2 < target_sfe

### Three-Step Process

1. **Normalize**: `W_norm = W / ||W||_F`
2. **Orthogonalize**: Apply Newton-Schulz iterations
3. **Scale**: `W_new = sqrt(fan_in) * W_orth * variance_factor`

### Why It Works

- **Frobenius norm preservation**: Maintains overall weight magnitude
- **Isometry**: Preserves distances (prevents activation explosion/vanishing)
- **Non-catastrophic**: Doesn't erase learned representations completely

## Integration with Training

### Periodic Reinitialization

```rust
let fire = FIREReinitializer::new();
let reinit_every = 5000; // steps

for step in 0..total_steps {
    // Normal training step
    let loss = train_step(&model, &optimizer, &batch)?;

    // Periodic reinitialization
    if step % reinit_every == 0 && step > 0 {
        for (i, layer) in model.layers.iter().enumerate() {
            let stats = fire.reinitialize(&layer.weights)?;
            if stats.was_reinitialized {
                println!("Step {}: Reinitialized layer {} (orth_err={:.6})",
                         step, i, stats.orthogonality_error);
            }
        }
    }
}
```

### Conditional Reinitialization

```rust
// Only reinitialize if layer is underperforming
for (i, layer) in model.layers.iter().enumerate() {
    if fire.should_reinitialize(&layer.weights, Some(&layer.gradients))? {
        let stats = fire.reinitialize(&layer.weights)?;
        println!("Reinitialized dormant layer {}", i);
    }
}
```

### Multi-Task Scenario

```rust
struct MultiTaskModel {
    shared_layers: Vec<Layer>,
    task_layers: HashMap<TaskId, Vec<Layer>>,
    fire: FIREReinitializer,
}

impl MultiTaskModel {
    fn switch_task(&mut self, new_task: TaskId) {
        // Reinitialize task-specific layers
        if let Some(layers) = self.task_layers.get_mut(&new_task) {
            for layer in layers.iter_mut() {
                self.fire.reinitialize(&layer.weights).unwrap();
            }
        }
        // Keep shared layers unchanged
    }
}
```

## Monitoring and Debugging

### Statistics

```rust
let stats = fire.reinitialize(&weights)?;

println!("FIRE Stats:");
println!("  Reinitialized: {}", stats.was_reinitialized);
println!("  Initial norm: {:.3}", stats.initial_norm);
println!("  Final norm: {:.3}", stats.final_norm);
println!("  Norm preservation: {:.2}%", stats.norm_preservation() * 100.0);
println!("  Orthogonality error: {:.6}", stats.orthogonality_error);
println!("  Iterations: {}", stats.iterations_used);
println!("  Success: {}", stats.is_successful());
```

### Expected Values

**Healthy reinitialization**:
- `orthogonality_error < 0.01` (good isometry)
- `norm_preservation ∈ [0.8, 1.2]` (reasonable scaling)
- `was_reinitialized = true`

**Warning signs**:
- `orthogonality_error > 0.1` → Increase newton_schulz_iters
- `norm_preservation < 0.5 or > 2.0` → Check variance_factor
- `was_reinitialized = false` → Weights too small (check min_weight_norm)

## Performance Considerations

### Computational Cost

```
Time complexity: O(d^3) for d×d matrix
  - Matrix multiplication: 2 × O(d^3) per iteration
  - Total: newton_schulz_iters × 2d^3 operations

Memory: O(d^2) temporary storage (no persistent overhead)
```

**Example timings** (single-threaded CPU):
- 256×256: ~5ms (8 iterations)
- 512×512: ~40ms
- 1024×1024: ~320ms
- 2048×2048: ~2.5s

### When to Avoid

❌ **Don't use FIRE if**:
- Training is going well (no plateau)
- Continual learning not needed
- Computational budget is very tight
- Weights are already well-conditioned

✅ **Do use FIRE if**:
- Training plateaued for >1000 steps
- Switching to new task/domain
- Detecting dormant neurons
- Exploring new optimization landscape

## Limitations

1. **Non-square matrices**: FIRE is designed for square weight matrices. For non-square (e.g., 2048×11008), reinitialize only the smaller dimension or use partial reinitialization.

2. **Computational cost**: O(d^3) can be expensive for very large layers (>4096).

3. **Not a silver bullet**: FIRE helps with plasticity but doesn't solve all continual learning challenges (e.g., doesn't prevent forgetting in shared parameters).

4. **Numerical precision**: Newton-Schulz can accumulate errors for ill-conditioned matrices. Use target_sfe ≥ 1e-5 for robustness.

## API Reference

### FIREConfig

```rust
pub struct FIREConfig {
    pub target_sfe: f64,                // Target SFE for convergence
    pub newton_schulz_iters: usize,     // NS iterations
    pub variance_factor: f64,           // Variance scaling
    pub min_weight_norm: f64,           // Minimum norm threshold
}
```

### FIREReinitializer

```rust
impl FIREReinitializer {
    pub fn new() -> Self;
    pub fn with_config(config: FIREConfig) -> Self;
    pub fn reinitialize(&self, weights: &Tensor) -> Result<ReinitStats>;
    pub fn should_reinitialize(&self, weights: &Tensor, gradients: Option<&Tensor>) -> Result<bool>;
}
```

### ReinitStats

```rust
pub struct ReinitStats {
    pub was_reinitialized: bool,
    pub initial_norm: f64,
    pub final_norm: f64,
    pub orthogonality_error: f64,
    pub iterations_used: usize,
}

impl ReinitStats {
    pub fn is_successful(&self) -> bool;    // orth_error < 1e-3
    pub fn norm_preservation(&self) -> f64;  // final / initial
}
```

## Testing

Run FIRE tests:

```bash
cargo test -p nanochat-train fire
```

Expected: 5-7 tests passing (some may be flaky due to numerical precision)

## References

- **Paper**: "FIRE: Efficient Neural Network Reinitialization via Frobenius Isometry" (ICLR 2026 Oral)
- **Implementation**: `crates/nanochat-train/src/optim/fire.rs`
- **Related**: Continual learning, plasticity, catastrophic forgetting

## Summary

FIRE provides a principled way to restore neural network plasticity:

✅ **Frobenius isometry**: Preserves weight structure
✅ **Newton-Schulz**: Fast orthogonalization (quadratic convergence)
✅ **Variance preservation**: Maintains activation magnitudes
✅ **Configurable**: Adjust precision vs speed tradeoff
✅ **Integrated**: Easy to add to existing training loops

**Use when**: Continual learning, task switching, escaping local minima, preventing dormant neurons.
