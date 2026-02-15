# nanochat-rs Training Scripts

Production-ready training scripts for different model sizes and use cases.

## Quick Start

### End-to-End Validation (2 minutes)
```bash
./scripts/validate_e2e.sh
```

Validates the entire pipeline:
- ✅ Training (1000 steps)
- ✅ GGUF export
- ✅ Inference server
- ✅ API endpoints
- ✅ Generation quality

### Production Training

#### Nano 125M - Development & Testing
```bash
./scripts/train_nano_125m.sh
```
- **Duration**: ~2 hours
- **GPU Memory**: 4GB
- **Steps**: 50K
- **Use Case**: Quick iteration, debugging, CI/CD

#### Small 560M (d20) - Production Code Generation
```bash
./scripts/train_small_560m.sh
```
- **Duration**: ~8 hours
- **GPU Memory**: 16GB
- **Steps**: 100K
- **Features**: Hybrid attention (20% DeltaNet)
- **Use Case**: Rust code completion, edge deployment

#### Medium 3B MoE - High-Quality Codegen
```bash
./scripts/train_medium_3b.sh
```
- **Duration**: ~24 hours
- **GPU Memory**: 48GB
- **Steps**: 200K
- **Features**: 8 experts (2 active), 8-bit Muon, mHC N=4
- **Use Case**: Production Rust codegen, competitive with GPT-3.5

### Custom Production Run (8 hours)
```bash
./scripts/train_production_8h.sh
```
- Fully configured 8-hour training job
- TensorBoard monitoring
- Automatic checkpointing
- Production-grade setup

## Features Enabled in All Scripts

### E3 Optimizations
- ✅ Multi-Token Prediction (MTP)
- ✅ Async data loader
- ✅ TensorBoard logging
- ✅ Gradient clipping
- ✅ WSD learning rate schedule

### Hybrid Optimizer
- **Linear layers**: Muon (2× faster than AdamW)
- **Other params**: Lion (better generalization)
- **Memory**: 8-bit quantization available (86% reduction)

### mHC Routing
- **Nano/Small**: N=2 (simple, 1 param per layer)
- **Medium**: N=4 (full BvN, 24 permutations)
- **Overhead**: <0.001% of compute

## Monitoring

All scripts start TensorBoard automatically on `http://localhost:6006`:

```bash
# View metrics
tensorboard --logdir runs/

# Or manually
tensorboard --logdir runs/experiment_name/tensorboard
```

Metrics tracked:
- Loss (total, cross-entropy, entropy)
- Learning rate
- Gradient norm
- Throughput (tokens/sec)
- mHC composite gain

## Model Configurations

Located in `configs/models/`:

| Config | Params | Hidden | Layers | Heads | Context | Features |
|--------|--------|--------|--------|-------|---------|----------|
| nano_125m.toml | 125M | 768 | 12 | 12 | 2K | GQA |
| small_560m.toml | 560M | 1024 | 20 | 16 | 4K | Hybrid attn |
| medium_3b.toml | 3B | 2048 | 28 | 32 | 8K | MoE + Hybrid |

All models use:
- ✅ Ternary quantization (1.58-bit)
- ✅ GQA (4-8 KV heads)
- ✅ SwiGLU FFN
- ✅ RMSNorm
- ✅ Weight tying

## Export to GGUF

After training:

```bash
cargo run --release -p nanochat-train -- export \
    --checkpoint runs/experiment/checkpoint_N.safetensors \
    --output models/nanochat.gguf \
    --mhc-output models/nanochat.mhc
```

## Inference

Start server:

```bash
cargo run --release -p nanochat-serve -- \
    --model models/nanochat.gguf \
    --mhc models/nanochat.mhc \
    --port 8000
```

Test generation:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nanochat",
        "messages": [{"role": "user", "content": "Write a Rust function"}],
        "max_tokens": 200
    }'
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Enable `gradient_checkpointing = true`
- Use 8-bit optimizer: `use_8bit_muon = true`

### Slow Training
- Check GPU utilization: `nvidia-smi dmon`
- Increase `num_workers` for data loading
- Increase `prefetch_batches`

### NaN Loss
- Reduce learning rate
- Increase `warmup_steps`
- Check entropy weight (should be 0.01-0.05)

### Low Quality
- Train longer (check loss convergence)
- Increase model size
- Validate dataset quality
- Check mHC composite gain (should be ≤ 1.0)

## Performance Benchmarks

Expected throughput on NVIDIA RTX PRO 6000:

| Model | Tokens/sec | Steps/min | Memory |
|-------|-----------|-----------|--------|
| 125M | ~50K | ~600 | 4GB |
| 560M | ~25K | ~400 | 16GB |
| 3B MoE | ~10K | ~200 | 48GB |

## Production Checklist

Before deploying:

- [ ] Run E2E validation: `./scripts/validate_e2e.sh`
- [ ] Train for sufficient steps (check loss plateau)
- [ ] Validate generation quality manually
- [ ] Check mHC composite gain ≤ 1.0
- [ ] Export to GGUF successfully
- [ ] Test inference API
- [ ] Benchmark throughput and latency
- [ ] Monitor GPU memory usage

## Next Steps

1. Run validation: `./scripts/validate_e2e.sh`
2. Choose model size based on use case
3. Run training script
4. Monitor via TensorBoard
5. Export to GGUF when complete
6. Deploy inference server

For questions or issues, check the main README.md or open an issue.
