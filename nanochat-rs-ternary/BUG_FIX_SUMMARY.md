# Critical Bug Fix: mHC Identity Bypass

## Problem

The model was predicting the **current input token** at each position instead of the **next token**, making it useless for language modeling.

### Symptoms
- Model predicted EXACT input tokens with massive confidence (logits 238-275)
- Example: Input `[22184, 1388, 3419, 1391]` ("fn main() {")
  - Position 0: predicted token 22184 ("fn") - same as input[0]
  - Position 1: predicted token 1388 ("main") - same as input[1]
  - Position 2: predicted token 3419 ("()") - same as input[2]
  - Position 3: predicted token 1391 ("{") - same as input[3]
- All different contexts produced identical predictions (no context dependence)
- Training loss was 3.29 (better than random 10.8) but model learned the wrong task

## Root Cause

The mHC (Multi-stream Hierarchical Connections) `alpha_logit` parameter was initialized to **5.0** instead of **0.0**.

### Technical Details

With `alpha_logit = 5.0`:
- `alpha = sigmoid(5.0) ≈ 0.993`
- Mixing matrix: `H_res = [[0.993, 0.007], [0.007, 0.993]]` (almost identity)
- Input embedding flows through all transformer layers unchanged
- With weight tying: `logit[token_id] = embedding @ embedding[token_id].T = ||embedding||²` → huge!

### Why This Happened

The initialization created a near-identity skip connection that bypassed all transformer computation:

1. **Expand**: `[v, v]` (duplicate input embedding)
2. **Each block**: Streams barely mix due to alpha≈1.0
3. **Collapse**: `(v + v) / 2 = v` (recovers original embedding)
4. **LM head**: Weight-tied, so predicting same token gets massive logit

Even if attention and FFN layers learned something, the near-identity mixing prevented any actual transformation.

## Solution

**Changed `alpha_logit` initialization from 5.0 to 0.0** in `crates/nanochat-train/src/mhc.rs`.

With `alpha_logit = 0.0`:
- `alpha = sigmoid(0.0) = 0.5`
- Mixing matrix: `H_res = [[0.5, 0.5], [0.5, 0.5]]` (balanced mixing)
- Streams properly mix, preventing identity bypass
- Transformer layers must learn to transform representations

## Verification Steps Performed

1. ✅ Created `test_loss_alignment.rs` - confirmed untrained model naturally predicts current tokens
2. ✅ Created `test_context_dependence.rs` - confirmed all contexts produce identical predictions
3. ✅ Created `debug_generation.rs` - revealed exact token copying with massive logits
4. ✅ Created `test_data_shift.rs` - verified data loader correctly shifts targets by 1
5. ✅ Created `test_layer_outputs.rs` - confirmed model predicts exact input tokens
6. ✅ Traced mHC initialization and found alpha_logit=5.0 bug
7. ✅ Fixed initialization to alpha_logit=0.0
8. ✅ Updated tests to expect alpha=0.5
9. ✅ All 64 nanochat-train tests pass

## Impact

**All previous training runs (20,000+ steps) were wasted** - the model learned to copy inputs, not to do language modeling.

The checkpoint at `checkpoints/stable-v2/step_20000` is **not usable** for generation because it has the identity bypass built-in from initialization.

## Next Steps

1. **Delete old checkpoints** (they have alpha≈0.993 from old initialization)
2. **Start fresh training** from random init with alpha=0.5
3. **Monitor generation quality** - model should now predict next tokens, not current tokens
4. **Verify loss convergence** - should still reach ~3.0-4.0 but now learning the right task

## Files Changed

- `crates/nanochat-train/src/mhc.rs` - Changed alpha_logit init 5.0 → 0.0
- `crates/nanochat-train/src/mhc.rs` - Updated tests for alpha=0.5
- Added diagnostic examples:
  - `examples/test_loss_alignment.rs`
  - `examples/test_data_shift.rs`
  - `examples/test_layer_outputs.rs`
  - `examples/test_context_dependence.rs`
  - `examples/debug_generation.rs` (updated)
  - `examples/inspect_mhc_params.rs`

## Commit

```
f116390 Fix critical mHC identity bypass bug in transformer model
```

Pushed to `master` branch.
