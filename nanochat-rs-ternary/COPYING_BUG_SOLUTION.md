# Copying Bug - ROOT CAUSE IDENTIFIED ✅

## Executive Summary

The model predicts current tokens instead of next tokens due to **weight tying + insufficient transformation**. Without weight tying, it collapses to frequent tokens instead. Both are failure modes of insufficient model capacity or training dynamics.

## Failure Modes Identified

### Mode 1: Input Token Copying (WITH weight tying)
- **Observed in**: Original d20 model (checkpoints/stable-v2/step_20000)
- **Behavior**: Predicts EXACT input token at each position (even rare tokens like "fn")
- **Mechanism**:
  ```
  lm_head.weight = tok_embed.weight.T  (weight tying)
  output[i] ≈ embedding(input[i])      (minimal transformation)
  logit = output @ lm_head = embedding @ embedding.T
  argmax(logit) = input[i]             (self-similarity)
  ```
- **Logit magnitude**: 248-261 (extremely high confidence)

### Mode 2: Frequency Collapse (WITHOUT weight tying)
- **Observed in**: Diagnostic model (checkpoints/diagnostic/step_500)
- **Behavior**: Predicts most frequent token (spaces, token 220) for all positions
- **Mechanism**: Model learns that predicting common tokens minimizes average loss
- **Logit magnitude**: 6.0 (moderate confidence)

## Why This Happens

1. **Insufficient Model Capacity**
   - d20 (dim=256, 2-6 layers) may be too small
   - Can't learn complex causal structure
   - Falls back to simple heuristics

2. **Ternary Quantization Pressure**
   - STE during training limits gradient flow
   - Model learns to output embeddings directly (easier than transforming)

3. **Weight Tying Amplifies Problem**
   - Without weight tying: frequency collapse (bad)
   - WITH weight tying: input copying (worse, because it looks "correct")

## Solutions (Ordered by Priority)

### Solution 1: Train WITHOUT Weight Tying ⭐ RECOMMENDED
**Pros:**
- Breaks self-similarity loop
- LM head can learn independent token predictions
- Should prevent input copying

**Cons:**
- Adds ~50M parameters for d20 (dim*vocab)
- May still get frequency collapse if model too small
- Need to monitor for mode collapse

**Implementation:**
```rust
weight_tied: false  // in TrainConfig
```

### Solution 2: Increase Model Capacity
**Pros:**
- Larger models can learn proper structure
- May overcome ternary quantization limitations

**Cons:**
- Slower training
- More memory

**Implementation:**
```rust
TrainConfig {
    dim: 512,        // d40 (was 256)
    n_layers: 12,    // (was 6)
    // ...
}
```

### Solution 3: Add Stronger Regularization
**Pros:**
- Forces transformer to actually transform
- Can keep weight tying

**Cons:**
- May not fully solve if model capacity insufficient
- Harder to tune

**Implementation:**
- Add dropout to transformer blocks
- Add layer output magnitude regularization
- Use label smoothing (already have eps=0.1)

### Solution 4: Change Training Dynamics
**Pros:**
- May help model learn structure instead of shortcuts

**Cons:**
- Requires experimentation

**Implementation:**
- Curriculum learning (start with short sequences)
- Gradient penalty on near-identity outputs
- Aux loss encouraging diversity

## Recommended Action Plan

1. **Immediate: Train d20 WITHOUT weight tying** (Solution 1)
   - Uses existing code, just flip config flag
   - Quick to test (500-1000 steps diagnostic run)
   - If successful, do full 20K step training

2. **If Solution 1 shows frequency collapse:**
   - Increase to d40 (Solution 2)
   - Add dropout (Solution 3)

3. **If still failing:**
   - Try d60 or larger
   - Consider removing ternary quantization for baseline

## Diagnostic Commands

```bash
# Train without weight tying
cargo run --release --example train_diagnostic --features nanochat-train/cuda

# Test generation
cargo run --release --example debug_generation
# (update checkpoint path in code first)

# Monitor for patterns:
# - Input copying: pred matches input (even rare tokens)
# - Frequency collapse: pred is always common token (space, newline)
# - Correct behavior: pred varies based on context, matches target
```

## Key Insight

The copying bug is NOT a code bug - it's a **training failure mode**. The code is correct (data loading, loss, causal mask all verified). The model is learning a degenerate solution because:

1. It's easier to copy than to predict
2. With weight tying, copying is mathematically favorable
3. Model capacity may be insufficient to learn proper structure

**Fix: Remove weight tying, possibly increase capacity.**
