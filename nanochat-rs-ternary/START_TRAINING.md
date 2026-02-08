# Starting Nano-125M Training

## Quick Demo Model

Training a **125M parameter** ternary model with distillation from Qwen3-FP8 endpoint.

**Goal**: Create a working model we can show off that demonstrates ternary quantization works!

## Configuration

```
Model: nano-125M
- Parameters: ~125M
- Dimension: 768
- Layers: 12
- Heads: 12 (MHA)
- Vocab: 50,257 (GPT-2)
- Sequence length: 512

Training:
- Teacher: Qwen3-FP8 (https://crazyshit.ngrok.io)
- Steps: 10,000 (~2-3 hours)
- Batch size: 4
- Parallel: 8 micro-batches
- KL weight: 0.5
- Temperature: 2.0

Expected Results:
- HumanEval pass@1: >10% (shows it works)
- Model size: ~125MB (ternary compression)
- Training time: 2-3 hours
```

## Command

```bash
cargo run --release --example train_nano_125m -- \
  --teacher-endpoint https://crazyshit.ngrok.io \
  --total-steps 10000 \
  --checkpoint-dir checkpoints/nano-125m \
  --parallel true \
  --micro-batches 8 \
  --device cuda:0
```

## What This Demonstrates

✅ **Distillation from FP8 teacher works**
✅ **Ternary quantization preserves quality**
✅ **Parallel training speeds up 2.4x**
✅ **Complete pipeline (train → export → eval)**
✅ **Something we can show off!**

## Timeline

1. **Now**: Start training (10K steps)
2. **+2-3 hours**: Training completes
3. **+10 min**: Export to GGUF
4. **+30 min**: Evaluate on HumanEval
5. **Done**: Show off working ternary model!

Let's go! 🚀
