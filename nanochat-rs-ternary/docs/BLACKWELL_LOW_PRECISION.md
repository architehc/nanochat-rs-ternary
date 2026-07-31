# Low precision on Blackwell: what helps here, and what does not

Measured on the RTX 5090 (GB202, sm_120, CUDA 12.8) with `qwen35-hybrid`
(258M params, 12 Gated DeltaNet + 4 attention layers, ternary QAT), batch 4 x
seq 512. Reproduce with `cargo run --release --features cuda --example bench_gemm`
and `--example profile_step`.

## Summary

| format | status | step-level effect |
|---|---|---|
| **bf16** | **implemented** (`use_bf16_compute`) | GEMMs 3.2x faster, matmul operands halve in memory |
| FP8 (E4M3) | candle has the dtype, no GEMM path | would save at most ~2% of the step |
| **NVFP4** | **not reachable**: candle has no FP4 dtype | would save at most ~2% of the step |

The hardware does have FP4 tensor cores. They are not the bottleneck.

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

## Why bf16 is still worth it

bf16 helps for reasons that are mostly *not* the 3.2x GEMM rate:

- **Bandwidth.** The elementwise ops that dominate the step are memory-bound.
  Halving operand width halves their traffic.
- **Memory.** candle's autograd holds both matmul operands for the backward pass.
  In bf16 that is half the bytes, which buys batch size — and larger batches
  amortize the per-op launch overhead that actually dominates.
- **Accuracy cost is near zero here.** See below.

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

1. Fewer/larger kernels in the DeltaNet recurrence (done once; more is possible).
2. bf16 to cut bandwidth and free memory for larger batches (this change).
3. Larger batch, which directly amortizes launch overhead.

A faster GEMM format is not on that list until the first three are exhausted.
