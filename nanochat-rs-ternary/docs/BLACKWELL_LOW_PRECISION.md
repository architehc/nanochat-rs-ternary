# Low precision on Blackwell: what helps here, and what does not

Measured on the RTX 5090 (GB202, sm_120, CUDA 12.8) with `qwen35-hybrid`
(258M params, 12 Gated DeltaNet + 4 attention layers, ternary QAT), batch 4 x
seq 512. Reproduce with `cargo run --release --features cuda --example bench_gemm`
and `--example profile_step`.

## Summary

| format | status | measured step-level effect |
|---|---|---|
| **bf16** (cast at the matmul) | implemented, **off by default — it loses** | 1-1.5% *slower*, 8-11% *more* memory |
| FP8 (E4M3) | candle has the dtype, no GEMM path | could save at most ~2% of the step |
| **NVFP4** | **not reachable**: candle has no FP4 dtype | could save at most ~2% of the step |

The hardware does have FP4 tensor cores. They are not the bottleneck. Neither,
it turns out, is precision at all — see the bf16 result below, which is the same
Amdahl argument arriving as an experimental fact.

## Why: this workload is not GEMM-bound

Measured GEMM rates at this model's actual shapes:

```
shape                                           f32       bf16  speedup
ffn up/gate  [2048,1024]x[1024,3584]        28.5 TF   137.7 TF    4.83x
ffn down     [2048,3584]x[3584,1024]        41.6 TF   111.6 TF    2.68x
attn/dn proj [2048,1024]x[1024,1024]        30.6 TF    50.2 TF    1.64x
lm head      [2048,1024]x[1024,4096]        36.3 TF   134.0 TF    3.69x
large square [4096,4096]x[4096,4096]        40.0 TF   134.8 TF    3.37x
mean                                        35.4 TF   113.7 TF    3.21x
```

A training step for this model is about **3.2 TFLOP** of GEMM
(`3 x 2 x 258M params x 2048 tokens` — one forward and two backward GEMMs per
weight). At the measured bf16 rate that is roughly **28 ms of GEMM inside a
~1250 ms step: about 2%.**

That 2% is the *entire budget* a faster numeric format is competing for. NVFP4
could be infinitely fast and the step would still only shrink by ~2%, because
the other ~98% is elementwise work and per-op launch overhead — quantize,
dequantize, STE add/subtract, masks, `exp`, norms, gating — none of which a
matmul format touches.

This is the same finding as the earlier DeltaNet work: the step is **launch- and
bandwidth-bound, not FLOP-bound**. What moves the number is issuing fewer, larger
kernels and moving fewer bytes.

## bf16, as measured: a net loss with this implementation

The prediction was that bf16 would win on bandwidth and memory even though the
GEMM slice is small. Measured on `qwen35-hybrid`, seq 512, `profile_step`:

| batch | f32 | bf16 |
|---|---|---|
| 2 | 1144 tok/s, 15.58 GB | 1131 tok/s, **16.86 GB** |
| 4 | 1631 tok/s, 25.01 GB | 1606 tok/s, **27.84 GB** |

Slower *and* larger, on both counts. Two reasons, and both are specific to
casting at the matmul rather than to bf16 itself:

1. **The casts add kernels to a launch-bound step.** `amp::matmul` inserts two
   `to_dtype` kernels before the GEMM and one after. There are ~160 linear
   layers, and the same casts recur in the backward pass. Saving 3.2x on 2% of
   the step is worth ~0.6%; the added launches cost more than that.
2. **Casting cannot save memory.** Autograd retains the *original* f32 operands
   to compute the backward pass, so the bf16 copies are additive, not
   substitutive — hence memory goes **up**, not down.

The flag (`use_bf16_compute` / `--bf16`) is kept, defaulted off, because the
implementation is correct and tested and the negative result is worth being able
to re-run. Do not enable it expecting a speedup.

**What would actually work:** build the model in bf16 at the `VarBuilder` — so
the graph holds bf16 tensors and no f32 original exists to retain — and keep a
separate f32 master copy of the weights for the optimizer to update. That halves
activation memory for real and adds no cast kernels. It is a much larger change:
every reduction, norm, and loss would need an explicit f32 island, and the
checkpoint format would have to carry both copies.

## Why bf16 is nearly lossless for a ternary model

`BitLinearSTE` feeds the GEMM operands that are already quantized: weights are
`{-1, 0, +1}` times a per-group scale, activations are integers in `[-127, 127]`
times a per-token scale. bf16's 8-bit significand represents every integer up to
256 **exactly**, so those factors survive the cast bit-for-bit — only the scales
lose mantissa. `amp::tests::bf16_is_exact_for_quantized_operands` pins this.

candle issues bf16 GEMMs through cuBLAS with `CUBLAS_COMPUTE_32F`, so products
accumulate in f32. Inputs are bf16; the running sum is not.

## What NVFP4 would actually require

For completeness, if the bottleneck ever moves and FP4 becomes worth it:

1. **An FP4 dtype in candle.** `DType` stops at `F8E4M3`. Adding `F4E2M1` means
   storage, casts, and every kernel that dispatches on dtype.
2. **A GEMM path.** Either cuBLASLt's FP4 GEMM (CUDA 12.8+) or hand-written
   `mma` for sm_120, wired in through `CustomOp`.
3. **Block scaling.** NVFP4 is not a bare 4-bit float — it is E2M1 with an FP8
   E4M3 scale per 16-element block plus a global FP32 scale. The block scales are
   part of the data layout the GEMM expects.
4. **Training-specific care.** FP4 *pretraining* needs stochastic rounding on the
   backward pass and typically Hadamard rotations to spread outliers, with some
   layers left in higher precision.

Also worth noting: this model's weights are **ternary (~1.58-bit)**, already more
aggressive than FP4's 4 bits. FP4 would be a step *up* in weight precision, not
down. The natural hardware match for ternary x int8 is the INT8 tensor cores, and
`crates/ternary-kernels` already has DP4A kernels for inference — but candle
exposes no integer GEMM for the training path either.

## Where the remaining time actually goes

From `profile_step` at batch 4:

```
forward_hidden_only    292 ms  (23%)
backward               601 ms  (48%)
compute_grad_norm       15 ms  ( 1%)
optimizer + bookkeeping 346 ms (28%)
```

The productive levers, in order:

1. **Fewer, larger kernels.** This is the only thing that has actually moved the
   number. Restructuring the DeltaNet recurrence so intra-chunk work batches
   across all chunks took that layer from 71ms to 24ms; replacing `sum_all` and
   the per-parameter `to_scalar` syncs took the gradient norm from 148ms to 16ms.
2. **Larger batch**, which amortizes the per-op launch overhead directly.
3. A genuinely fused kernel for the quantize/dequantize/STE chain, which is ~16
   elementwise kernels per linear layer today (~2500 per forward). This is where
   the remaining time is.

Changing numeric formats is not on that list. Two experiments now say so: the
roofline (GEMM is ~2% of the step) and the bf16 measurement above.

Two changes were tried and reverted for want of a measured win, recorded here so
they are not retried blindly:

- **Periodic (rather than per-step) CUDA pool trimming.** Hypothesis: trimming
  right before the optimizer's thousands of allocations forces them all through
  the driver. Measured 1595 -> 1603 tok/s, i.e. nothing, and it raises OOM risk
  near the memory ceiling.
- **Batching Newton-Schulz by weight shape.** Collapses ~5600 kernel launches
  per step into ~350 and is numerically identical (tested). But it must hold
  every parameter's Nesterov tensor live at once to group them, ~2GB extra, and
  at batch 4 the trainer already sits at 30.1GB of 32GB — so it tipped into
  host-memory thrashing and collapsed throughput to ~57 tok/s. In isolation
  (`profile_step`, lower baseline memory) it was worth only ~19ms of a 1258ms
  step. Worth revisiting only if the memory ceiling moves.
