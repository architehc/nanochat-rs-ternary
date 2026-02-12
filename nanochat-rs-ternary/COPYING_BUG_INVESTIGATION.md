# Investigation: Model Predicts Current Tokens Instead of Next Tokens

## Status: ROOT CAUSE NOT YET IDENTIFIED

Despite fixing the mHC identity bypass bug (alpha_logit 5.0 → 0.0), the trained model at `checkpoints/stable-v2/step_20000` still exhibits the copying behavior.

## Confirmed Facts

### ✅ Data Loading is CORRECT
```rust
// TokenFileDataset::get_item in crates/nanochat-train/src/data/dataset.rs
let input = self.tokens[start..end].to_vec();        // [t0, t1, t2, ...]
let target = self.tokens[start + 1..end + 1].to_vec(); // [t1, t2, t3, ...]
```
Targets ARE properly shifted by 1 position.

### ✅ Loss Calculation is CORRECT
```rust
// forward_loss in crates/nanochat-train/src/model.rs
let logits = self.forward(input_ids)?;  // [batch, seq, vocab]
candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)
```
Logits at position i are aligned with target i (which is token i+1).

### ✅ Causal Mask is CORRECT
```rust
// causal_mask in crates/nanochat-train/src/attention.rs
// Position i can attend to positions 0..=i, not i+1 onwards
for i in 0..seq_len {
    for j in (i + 1)..seq_len {
        mask_data[i * seq_len + j] = f32::NEG_INFINITY;
    }
}
```
Standard causal masking.

### ✅ Transformer Layers ARE Learning
Weight magnitudes from checkpoint analysis:
- Attention projections: output_scale = 0.8-1.2 (healthy)
- FFN projections: output_scale = 0.9-1.5 (healthy)
- RMSNorm weights: mean ~1.0, std ~0.1 (expected)

Layers are NOT acting as identity - they produce meaningful transformations.

### ✅ mHC Parameters are CORRECT
All alpha_logit values in checkpoint = 0.0 (confirmed via safetensors inspection).
This gives alpha = sigmoid(0.0) = 0.5 (balanced mixing, not identity).

## The Mystery

Given that:
1. Data is correctly shifted (input[i] → target[i+1])
2. Loss aligns logits[i] with target[i] = token[i+1]
3. Causal mask prevents attending to future
4. Layers produce meaningful outputs
5. mHC is not causing identity bypass

**Why does the model predict current tokens at inference?**

### Observed Behavior
```
Input:  [22184 ("fn"), 1388 ("main"), 3419 ("()"), 1391 ("{")]
Output logits:
  Position 0: predicts 22184 ("fn")    ← should predict 1388!
  Position 1: predicts 1388 ("main")   ← should predict 3419!
  Position 2: predicts 3419 ("()")     ← should predict 1391!
  Position 3: predicts 1391 ("{")      ← should predict next token!
```

Logit magnitudes are HUGE (248-261), suggesting the model is very confident.
This is consistent with weight tying: `logit[token_id] = embedding @ embedding.T = ||embedding||²`

## Hypotheses to Investigate

### Hypothesis 1: Training Loop Bug
- Maybe there's a subtle bug in how `train_step` processes batches?
- Check if gradients are actually flowing to transformer weights
- Verify loss is actually decreasing on CAUSAL prediction (not just reconstruction)

### Hypothesis 2: Inference vs Training Mismatch
- Training uses teacher forcing (correct)
- Maybe inference is somehow different?
- Check if position encodings are applied differently

### Hypothesis 3: Quantization Issue
- Training uses FP32
- Maybe ternary quantization during training (STE) is too aggressive?
- Model learns to minimize loss by copying embeddings through quasi-identity transforms

### Hypothesis 4: Embedding-LM Head Interaction
- With weight tying: `lm_head.weight = tok_embed.weight.T`
- If transformer barely transforms embeddings, output ≈ input embedding
- Then `logit = output @ lm_head = embedding @ embedding.T`
- Max value is at current token's index (self-similarity)

### Hypothesis 5: Optimizer/Learning Rate Issue
- Maybe model converged to a local minimum?
- Loss of 3.29 seems reasonable, but maybe it's a degenerate solution?
- Try training WITHOUT weight tying to break the embedding-lm_head loop?

## Next Steps

1. **Add intermediate activation logging** during training
   - Log attention output magnitudes
   - Log FFN output magnitudes
   - Check if they're growing or shrinking over training

2. **Test model WITHOUT weight tying**
   - Train a small model with separate LM head
   - See if it still exhibits copying behavior

3. **Inspect gradient flow**
   - Check if gradients reach all layers or are vanishing
   - Verify transformer weights are actually updating

4. **Compare loss on shifted vs unshifted targets**
   - Compute loss with targets = inputs (reconstruction)
   - Compute loss with targets = inputs[1:] (causal prediction)
   - See which one the model actually minimizes

5. **Test with larger model**
   - Maybe d20 (256 dim) is too small?
   - Try d40 or d60 to see if behavior changes

## Files to Check

- `crates/nanochat-train/src/train.rs` - train_step implementation
- `crates/nanochat-train/src/model.rs` - forward pass + loss
- `crates/nanochat-train/src/block.rs` - transformer block with mHC
- `examples/train_rust_maxgpu.rs` - training loop

## Timeline

- Feb 12: mHC fix committed (alpha_logit 5.0 → 0.0)
- Feb 12: Training completed (20K steps, loss=3.29)
- Feb 12: Generation test reveals bug persists
- Feb 12: Investigation started (this document)
