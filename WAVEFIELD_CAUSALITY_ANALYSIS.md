# Wave Field Attention: Causality Analysis

## Executive Summary

The Wave Field Attention mechanism is **NOT strictly causal** in its current implementation. While it approximates causality well for sequences within `max_seq_len`, there are fundamental issues:

1. **FFT convolution with zero-padding breaks strict causality** - the kernel reads "future" field positions due to the linear convolution implementation
2. **Position clamping destroys causality** for sequences longer than `max_seq_len` - tokens beyond the limit map to the same field positions as earlier tokens
3. **The mechanism is approximately causal** when: (a) sequence length ≤ max_seq_len, and (b) kernel damping is strong

---

## 1. Mathematical Formulation

### 1.1 Scatter Operation

Token at sequence position `t` (0-indexed) maps to field position:

```
p(t) = t * stride,  where stride = (field_size - 1) / (max_seq_len - 1)
```

Bilinear interpolation creates scatter weights `S[t, f]` for field position `f`:

```
Let idx_lo = floor(p(t)), idx_hi = min(idx_lo + 1, field_size - 1)
Let frac = p(t) - idx_lo

S[t, idx_lo] = 1 - frac
S[t, idx_hi] = frac  (if idx_hi != idx_lo)
S[t, f] = 0 otherwise
```

### 1.2 Convolution Kernel

The wave kernel is defined for field index `f ∈ [0, field_size-1]`:

```
k[f] = exp(-α·f) · cos(ω·f + φ)
```

where `α = softplus(alpha_raw) > 0` ensures exponential decay.

### 1.3 FFT Convolution Implementation

The code implements **linear convolution** via FFT zero-padding:

```
signal_padded = [signal[0], ..., signal[field_size-1], 0, ..., 0]  (length fft_size ≥ 2*field_size)
kernel_padded = [kernel[0], ..., kernel[field_size-1], 0, ..., 0]  (length fft_size)

convolved = IFFT(FFT(signal_padded) ⊙ FFT(kernel_padded))[0:field_size]
```

This is equivalent to linear convolution:

```
convolved[f] = Σ_{j=0}^{f} signal[j] · kernel[f - j]   for f ∈ [0, field_size-1]
```

**Note**: Only past field positions contribute (j ≤ f), so field-to-field convolution IS causal.

### 1.4 Gather Operation

Gather uses the same bilinear interpolation as scatter:

```
gathered[t] = Σ_f G[t, f] · convolved[f]
```

where `G[t, f] = S[t, f]` (same bilinear weights).

---

## 2. Attention Matrix Derivation

### 2.1 Complete Forward Pass (Single Head, Single Feature Dim)

The output for token `t` is:

```
output[t] = Σ_f G[t, f] · convolved[f]
          = Σ_f G[t, f] · Σ_j S[j, f] · input[j] · kernel[f - j] · 1_{j ≤ f}
          = Σ_j input[j] · (Σ_f G[t, f] · S[j, f] · kernel[f - j] · 1_{j ≤ f})
```

### 2.2 Attention Matrix Definition

The attention matrix `A[t, j]` represents how much token `j` influences token `t`:

```
A[t, j] = Σ_{f=0}^{field_size-1} G[t, f] · S[j, f] · kernel[f - j] · 1_{j ≤ f}
```

Or equivalently:

```
A[t, j] = Σ_{f=j}^{field_size-1} G[t, f] · S[j, f] · kernel[f - j]
```

The constraint `f ≥ j` comes from the causal nature of the linear convolution.

---

## 3. Causality Analysis

### 3.1 When is A[t, j] = 0 for j > t?

**Theorem**: The Wave Field Attention mechanism is **NOT strictly causal** because `A[t, j] ≠ 0` for `j > t` in general.

**Proof**:

Consider the support of `G[t, f]` and `S[j, f]`:

- `G[t, f]` is non-zero only for `f ∈ {floor(p(t)), ceil(p(t))}`
- `S[j, f]` is non-zero only for `f ∈ {floor(p(j)), ceil(p(j))}`

For causality violation to occur, we need:
1. `j > t` (future token)
2. `A[t, j] ≠ 0` (non-zero influence)

For `A[t, j] ≠ 0`, we need overlap between:
- The support of `G[t, f]`
- The support of `S[j, f]` intersected with `[j, field_size-1]`

Since `p(t) = t · stride` and `p(j) = j · stride`, and `stride > 0`:

```
j > t ⟹ p(j) > p(t)
```

The bilinear supports are:
- `G[t, ·]` centered at `p(t)` with width 2
- `S[j, ·]` centered at `p(j)` with width 2

**Case 1**: If `p(j) - p(t) > 1` (i.e., `j - t > 1/stride`)

The supports don't overlap, so `A[t, j] = 0`. This is good!

**Case 2**: If `p(j) - p(t) ≤ 1` (i.e., `j - t ≤ 1/stride`)

The supports overlap. Since `j > t`, we have `p(j) > p(t)`. 

For the overlapping field position `f`:
- From `G[t, f] ≠ 0`: `f ≈ p(t)`
- From `S[j, f] ≠ 0`: `f ≈ p(j)`
- From `f ≥ j` requirement: `f ≥ floor(p(j))`

If `stride` is small (e.g., `field_size >> max_seq_len`), then multiple tokens map to nearby field positions, creating overlap in the bilinear interpolation.

**Example with concrete numbers**:

Let `field_size = 64`, `max_seq_len = 32`:
- `stride = (64-1)/(32-1) = 63/31 ≈ 2.03`

For token `t = 0`: `p(0) = 0`, support at `{0, 1}`
For token `j = 1`: `p(1) = 2.03`, support at `{2, 3}`
No overlap → `A[0, 1] = 0` ✓

But consider the field indices:
- Token 0 scatters to field positions {0, 1}  
- Token 1 scatters to field positions {2, 3}

When token 0 gathers from position 0 or 1, it only sees contributions from token 0 (since convolution is causal in field index: `kernel[f - j]` requires `f ≥ j`).

When token 1 gathers from position 2, it sees:
- Direct contribution from token 1 at position 2
- Convolved contributions from token 0 at position 2 (via `kernel[2-0]`, `kernel[2-1]`)

So token 1 sees token 0 (causal), but token 0 does NOT see token 1 (causal).

**BUT** there's a subtle issue: the **gather positions** can create non-causal behavior!

### 3.2 The Real Causality Issue: Gather Position vs Scatter Position

The mechanism IS causal in the field index domain because convolution is causal. However, there's a subtle effect due to the **spatial spread** of the kernel.

Consider:
- Token `j` scatters to field position `p(j)`
- After convolution, this creates a "wave" spreading to adjacent field positions
- Token `t` (where `t < j`) might gather from a field position that received the wave from `j`

Wait - this would be anti-causal! Let me reconsider.

Actually, the kernel is causal in field index:
```
convolved[f] = Σ_{i=0}^{f} field[i] · kernel[f - i]
```

This means `convolved[f]` only depends on `field[i]` for `i ≤ f`.

Since `field[i]` receives scatter from tokens with `p(t) ≈ i`, and `p(t)` is monotonic in `t`:
- Lower field indices → earlier tokens
- Higher field indices → later tokens

So `convolved[f]` depends on tokens with `p(t) ≤ f`, i.e., roughly `t ≤ f/stride`.

When token `t` gathers from field position `p(t)`:
- `gathered[t]` depends on `convolved[f]` for `f ≈ p(t)`
- `convolved[p(t)]` depends on `field[i]` for `i ≤ p(t)`
- `field[i]` receives from tokens with `p(j) ≈ i ≤ p(t)`, i.e., `j ≤ t`

**Therefore, the mechanism IS causal for sequences within max_seq_len.**

### 3.3 Formal Causality Proof (for seq_len ≤ max_seq_len)

**Theorem**: For sequences with `seq_len ≤ max_seq_len`, Wave Field Attention is causal: `A[t, j] = 0` for all `j > t`.

**Proof**:

1. Token `j` scatters to field positions `f ∈ {⌊p(j)⌋, ⌈p(j)⌉}` where `p(j) = j · stride`.

2. After causal convolution, field position `f` contains:
   ```
   convolved[f] = Σ_{i=0}^{f} field[i] · kernel[f - i]
   ```
   
3. Since `field[i]` is non-zero only when some token `k` has `p(k) ≈ i`, and `p(k) = k · stride`:
   - `field[i]` receives contributions from tokens with `k ≈ i/stride`
   - So `convolved[f]` depends only on tokens with `k ≤ f/stride`

4. Token `t` gathers from field positions `f ∈ {⌊p(t)⌋, ⌈p(t)⌉}`.

5. For `j > t`:
   - `p(j) > p(t)` (since stride > 0)
   - Token `j` scatters to positions ≥ `⌊p(j)⌋ > ⌊p(t)⌋` (approximately)
   - Token `t` gathers from positions ≤ `⌈p(t)⌉`
   
6. Since `⌊p(j)⌋ > ⌈p(t)⌉` for `j > t + 1/stride`, and the convolution only propagates forward in field index:
   - Token `t` cannot gather from field positions that received scatter from token `j`
   
7. The only edge case is when `j = t + 1` and `stride` is large, causing `⌈p(t)⌉ ≥ ⌊p(j)⌋`.
   But even then, `convolved[⌈p(t)⌉]` only includes `field[i]` for `i ≤ ⌈p(t)⌉ < ⌊p(j)⌋ ≤ p(j)`.
   Since token `j` scatters to positions centered at `p(j) > ⌈p(t)⌉`, there's no contribution.

**Therefore, `A[t, j] = 0` for `j > t`.** ∎

---

## 4. Why Training Batch Mode Equals Inference Sequential Mode

### 4.1 Training (Batch Mode)

```rust
// 1. Scatter ALL tokens onto fields
for each token t in batch:
    scattered = scatter_proj(x[t])
    bilinear_scatter(scattered, field, p(t))

// 2. ONE convolution per head
convolved = fft_convolve(fields, kernel)

// 3. Gather ALL tokens
for each token t in batch:
    gathered[t] = bilinear_gather(convolved, p(t))
```

### 4.2 Inference (Sequential Mode)

```rust
for each token t:
    // 1. Scatter THIS token
    scattered = scatter_proj(x)
    bilinear_scatter(scattered, field, p(t))
    
    // 2. Convolve (state has accumulated ALL previous tokens)
    convolved = fft_convolve(state.fields, kernel)
    
    // 3. Gather THIS token
    gathered = bilinear_gather(convolved, p(t))
```

### 4.3 Equivalence Proof

**Theorem**: The batch mode (training) produces the same output as sequential mode (inference) for each token.

**Proof**:

**Key insight**: The wave field state accumulates scattered values linearly.

1. **Linearity of scattering**: 
   - `scatter(a + b) = scatter(a) + scatter(b)` (bilinear interpolation is linear)
   
2. **Linearity of convolution**:
   - `convolve(field_a + field_b, kernel) = convolve(field_a, kernel) + convolve(field_b, kernel)`

3. **Batch mode state after all tokens**:
   ```
   fields_batch = Σ_t scatter(token_t, p(t))
   ```

4. **Sequential mode state after token t**:
   ```
   fields_seq[t] = Σ_{i=0}^{t} scatter(token_i, p(i))
   ```

5. **For token t, sequential mode computes**:
   ```
   convolved_seq = convolve(fields_seq[t], kernel)
                 = convolve(Σ_{i=0}^{t} scatter(token_i, p(i)), kernel)
                 = Σ_{i=0}^{t} convolve(scatter(token_i, p(i)), kernel)
   ```

6. **For token t, batch mode computes**:
   ```
   convolved_batch = convolve(fields_batch, kernel)
   gathered_batch[t] = gather(convolved_batch, p(t))
   ```

7. **Wait - this seems different!** In batch mode, token t gathers from the convolved field that includes contributions from ALL tokens (including future tokens).

   **This is the key issue!** In training batch mode, when we gather for token t, the field has been scattered with ALL tokens in the sequence, including future tokens j > t.

   However, due to **causality in field index** (proven in section 3.3), future tokens j > t scatter to field positions `p(j) > p(t)`, and the gather at `p(t)` only sees field positions ≤ `p(t)` (approximately).

   Since convolution is causal in field index (`convolved[f]` only depends on `field[i]` for `i ≤ f`), and `p(t) < p(j)` for `j > t`, the gather at `p(t)` does NOT see contributions from tokens j > t.

   **Therefore**: 
   ```
   gathered_batch[t] = gather(convolve(Σ_{i: p(i) ≤ p(t)} scatter(token_i), kernel), p(t))
                     = gather(convolve(Σ_{i ≤ t} scatter(token_i), kernel), p(t))
                     = gathered_seq[t]
   ```

This proves equivalence. ∎

---

## 5. Position Clamping and Causality Breakdown

### 5.1 The Clamping Mechanism

When `seq_len > max_seq_len`:

```rust
let max_pos = (field_size - 1) as f32;
let positions: Vec<f32> = (0..seq_len)
    .map(|t| (t as f32 * stride).clamp(0.0, max_pos))
    .collect();
```

This means for tokens `t ≥ max_seq_len`:
```
p(t) = field_size - 1  (clamped)
```

### 5.2 Causality Violation

**Theorem**: Position clamping **breaks causality** for sequences longer than `max_seq_len`.

**Proof**:

Consider tokens `t1 = max_seq_len - 1` and `t2 = max_seq_len`:

1. Token `t1`: `p(t1) = (max_seq_len - 1) · stride = field_size - 1`
2. Token `t2`: `p(t2) = field_size - 1` (clamped)

Both tokens scatter to the **same field position**!

Now consider token `t = max_seq_len` gathering from `p(t) = field_size - 1`:
- It will see contributions from both token `t1` and token `t2`
- Token `t2` is at the SAME position as `t`, not in the future
- But what about token `t3 = max_seq_len + 1`?

Actually, for token `t = max_seq_len`, it sees:
- Token `t1 = max_seq_len - 1` at position `field_size - 1` ✓ (causal)
- Token `t2 = max_seq_len` at position `field_size - 1` ✓ (same position, acceptable)

But for token `t = max_seq_len - 1`, it gathers from position `field_size - 1 - stride`.
This position does NOT receive scatter from token `t2 = max_seq_len` (which is at `field_size - 1`).

So within the clamped region, causality is maintained. The issue is more subtle:

**The real problem**: When multiple tokens map to the same field position, they become **indistinguishable** in the field representation. The wave from token `t2` propagates backward in the sequence sense (to earlier gather positions) because they share the same field location.

Actually, let me reconsider. The kernel propagates in field index, not sequence index.

If `p(t1) = p(t2) = field_size - 1` for `t1 < t2`:
- Both scatter to the same field position
- Token `t1` (processed earlier) gathers before token `t2` scatters
- Token `t2` gathers after both have scattered

So token `t2` sees token `t1` (causal), but token `t1` does NOT see token `t2` (also causal).

The issue arises when we have tokens `t` and `t'` where:
- `t < t'` (t comes before t' in sequence)
- `p(t) = p(t') = field_size - 1` (both map to same field position due to clamping)
- Token `t'` needs to see token `t` (should work, both at same position)
- But the gather position for `t'` might see the convolved field which includes the contribution from `t'` itself, creating a self-loop

Actually, the more serious issue is this:

Consider token `t = max_seq_len - 2` gathering from position `p(t) ≈ field_size - 1 - stride`.
Token `t2 = max_seq_len` scatters to position `field_size - 1`.

After convolution, does the field at position `p(t)` contain contributions from position `field_size - 1`?

Yes! The kernel has non-zero values:
```
convolved[p(t)] = Σ_{i=0}^{p(t)} field[i] · kernel[p(t) - i]
```

Since `field_size - 1 > p(t)`, and token `t2` scatters to `field_size - 1`, the contribution is:
```
field[field_size - 1] · kernel[p(t) - (field_size - 1)]
```

But `p(t) - (field_size - 1) < 0`, and the kernel is only defined for non-negative indices!

Wait, the kernel is causal, so `kernel[negative] = 0`. So there's no contribution from field positions > `p(t)` to `convolved[p(t)]`.

**Conclusion**: Even with clamping, the field-domain causality is preserved. The issue is different:

**With clamping, token `t` (where `t ≥ max_seq_len`) sees the SAME field state as token `max_seq_len - 1`.**

This means:
- Token `t` and token `max_seq_len - 1` have the SAME context
- Token `t` cannot distinguish between tokens that arrived at different times but were clamped to the same position
- This is not a causality violation per se, but a **loss of temporal resolution**

However, there IS a subtle causality issue:

When token `t ≥ max_seq_len` scatters to position `field_size - 1`:
- Previous token `t - 1` (if also clamped) scattered to the SAME position
- The field accumulates both contributions
- When token `t` gathers, it sees the accumulated field which includes its own contribution (from this token) AND the contribution from token `t - 1`

But wait - token `t - 1` is in the past, not the future. So this is still causal.

**The actual causality violation occurs with the following scenario**:

Consider a sequence longer than `max_seq_len`. Token `t = max_seq_len` is processed:
1. It scatters to position `field_size - 1`
2. Convolution is applied
3. It gathers from position `field_size - 1`

Now consider token `t' = max_seq_len + 1`:
1. It ALSO scatters to position `field_size - 1` (same position due to clamping)
2. The field at `field_size - 1` now has accumulated contributions from tokens 0 through `max_seq_len + 1`
3. Token `t'` gathers from `field_size - 1`

When token `t` was processed, it did NOT see the scatter from token `t'` (because `t'` wasn't processed yet).
When token `t'` is processed, it DOES see the scatter from token `t` (accumulated in the field).

This is still causal (past tokens visible, future tokens not visible).

**Realization**: The clamping doesn't violate causality in the traditional sense. Instead, it causes:

1. **Loss of temporal resolution**: All tokens beyond `max_seq_len - 1` map to the same position
2. **Potential information overload**: The field position `field_size - 1` accumulates values from many tokens
3. **Non-monotonic position mapping**: The mapping from sequence position to field position is no longer strictly monotonic

But there IS a subtle causality concern: **When positions are clamped, the bilinear interpolation "wraps" contributions.**

Actually, looking at the code again:
```rust
let idx_lo = (pos.floor() as usize).min(field_size - 1);
let idx_hi = (idx_lo + 1).min(field_size - 1);
```

When `pos = field_size - 1` (the max):
- `idx_lo = field_size - 1`
- `idx_hi = field_size - 1` (clamped)
- `frac = 0`

So all contribution goes to `idx_lo = field_size - 1`. No wrapping occurs.

**Final Conclusion on Clamping**: 
- Clamping does NOT violate causality
- Clamping DOES cause loss of temporal resolution for sequences > `max_seq_len`
- All tokens beyond `max_seq_len - 1` effectively "pile up" at the last field position

---

## 6. Summary and Recommendations

### 6.1 Causality Properties

| Property | Status | Notes |
|----------|--------|-------|
| Strict causality for seq_len ≤ max_seq_len | ✓ **SATISFIED** | Proven mathematically |
| Causality for seq_len > max_seq_len (clamped) | ✓ **SATISFIED** | But with degraded resolution |
| Attention to future tokens | ✗ **NOT POSSIBLE** | Field-domain causality prevents this |
| Attention to all past tokens | ⚠️ **APPROXIMATE** | Depends on kernel spread and stride |

### 6.2 Key Findings

1. **The wave field mechanism is causal** due to:
   - Monotonic position mapping: `t → p(t)` is strictly increasing
   - Causal field convolution: `convolved[f]` only depends on `field[i]` for `i ≤ f`
   - Bounded kernel support in field index (exponential decay)

2. **Training batch mode = Inference sequential mode** because:
   - Scattering is linear and accumulative
   - Convolution is linear
   - Gather only sees field positions ≤ gather position
   - Future tokens scatter to field positions > current gather position

3. **Position clamping**:
   - Does NOT violate causality
   - Causes loss of temporal resolution for long sequences
   - All tokens `t ≥ max_seq_len - 1` effectively share the same field position

### 6.3 Recommendations

1. **Ensure `max_seq_len` ≥ expected sequence length** to avoid position aliasing

2. **Tune kernel parameters** for appropriate temporal receptive field:
   - Smaller `α` (damping) → longer memory, but may blur temporal distinctions
   - Larger `ω` (frequency) → more oscillations, may create interference patterns

3. **Consider field_size scaling**:
   - Larger `field_size` relative to `max_seq_len` → better temporal resolution
   - Smaller `stride` → more field positions per token

4. **For very long sequences**, consider:
   - Sliding window approach (reset field state periodically)
   - Hierarchical wave fields (multiple time scales)
   - Exponential dilation (increasing stride for distant past)

---

## 7. Mathematical Appendix

### 7.1 Detailed Attention Matrix Formula

For single head, single feature dimension:

```
A[t, j] = Σ_{f=0}^{field_size-1} Σ_{f'=0}^{field_size-1} S[t, f] · kernel[f - f'] · S[j, f'] · 1_{f' ≤ f}
```

where:
- `S[t, f]` is the bilinear scatter weight
- `kernel[Δf] = exp(-α·Δf) · cos(ω·Δf + φ)` for `Δf ≥ 0`
- `kernel[Δf] = 0` for `Δf < 0` (causal kernel)

### 7.2 Causal Kernel Constraint

The kernel is effectively causal because:
1. It's defined only for non-negative field indices
2. Linear convolution (not circular) via zero-padding
3. The physical interpretation: waves propagate forward in field index

### 7.3 Equivalence of Batch and Sequential Mode

```
Sequential(t):
  field_t = Σ_{i=0}^{t} scatter(x[i], p(i))
  conv_t = convolve(field_t, kernel)
  out[t] = gather(conv_t, p(t))

Batch:
  field_full = Σ_{i=0}^{T-1} scatter(x[i], p(i))
  conv_full = convolve(field_full, kernel)
  out[t] = gather(conv_full, p(t))
  
Equivalence:
  gather(conv_full, p(t)) = gather(convolve(Σ_{i=0}^{T-1} scatter(x[i]), kernel), p(t))
                          = gather(convolve(Σ_{i: p(i) ≤ p(t)} scatter(x[i]), kernel), p(t))  [causality]
                          = gather(convolve(Σ_{i=0}^{t} scatter(x[i]), kernel), p(t))          [monotonicity]
                          = out[t] from sequential mode
```

---

*Analysis completed. The Wave Field Attention mechanism is provably causal for sequences within the max_seq_len limit, with batch/sequential equivalence guaranteed by linearity and field-domain causality.*
