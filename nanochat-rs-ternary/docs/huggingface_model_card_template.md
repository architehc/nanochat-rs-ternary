---
language: en
license: mit
library_name: nanochat
tags:
  - rust
  - ternary-quantization
  - code-generation
---

# nanochat Model

Ternary-quantized nanochat model exported as GGUF + mHC weights.

## Files

- `model.gguf`: model weights in GGUF format
- `model.mhc`: mHC residual parameters

## Inference (Rust)

```bash
cargo run --release -p nanochat-serve -- \
  --model model.gguf \
  --mhc model.mhc \
  --tokenizer models/gpt2-tokenizer.json \
  --port 8000
```

## Training Profile

Document model architecture and training profile here:

- config:
- total steps:
- tokens:
- optimizer:
- precision:

## Evaluation

Add benchmark results (e.g. HumanEval-Rust, compilation success, latency).
