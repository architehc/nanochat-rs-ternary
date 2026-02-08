# Training a Useful Nanochat Model

## Model Selection

Given our infrastructure and the Qwen3 FP8 endpoint, we have options:

### Option 1: Nano-1B (Recommended for Demo)
- **Size**: ~1B parameters
- **Architecture**: Standard transformer (no MoE for simplicity)
- **Training time**: ~6-8 hours (50K steps)
- **Use case**: General purpose, good quality-to-size ratio
- **Memory**: ~4GB model + ~8GB training

### Option 2: Nano-125M (Fast Demo)
- **Size**: ~125M parameters  
- **Architecture**: Smaller, faster iteration
- **Training time**: ~2-3 hours (50K steps)
- **Use case**: Proof of concept, quick testing
- **Memory**: ~500MB model + ~2GB training

### Option 3: Qwen3-Hybrid (Production)
- **Size**: ~3B active (80B total with MoE)
- **Architecture**: 512 experts, top-10 routing
- **Training time**: ~18-24 hours (100K steps)
- **Use case**: Production deployment
- **Memory**: ~10GB model + ~30GB training

## Recommendation: Start with Nano-1B ✅

**Why**:
- Large enough to be actually useful
- Small enough to train in reasonable time
- Shows full pipeline (distillation, quantization, evaluation)
- Can generate coherent code on HumanEval

## Training Configuration

```rust
TrainConfig {
    dim: 1024,
    n_layers: 24,
    n_heads: 16,
    n_kv_heads: 4,  // GQA
    ffn_mult: 2.667,
    vocab_size: 50257,  // GPT-2 tokenizer
    max_seq_len: 2048,
    batch_size: 4,
    learning_rate: 0.02,  // Muon
    total_steps: 50000,
    warmup_steps: 2000,
}
```

## Data Strategy

**Option A: Synthetic Code Generation** (Easy, start immediately)
- Generate Python functions with patterns
- Math operations, string manipulation, list processing
- Good for testing infrastructure

**Option B: The Stack** (Better quality, requires download)
- ~6TB of code from GitHub
- Python subset: ~50GB
- High quality, real-world code

**Option C: Continue from Qwen3** (Best quality, requires checkpoint)
- Start from Qwen3-Coder checkpoint
- Apply ternary quantization
- Fine-tune with distillation

## Proposed Plan

### Phase 1: Quick Demo (2-3 hours) ⭐ START HERE
1. Use **nano-125M** config
2. **Distillation** from `https://crazyshit.ngrok.io`
3. **Synthetic data** (code patterns)
4. Train **10K steps** (~2 hours)
5. Evaluate on HumanEval
6. Target: >10% pass@1 (proof it works)

### Phase 2: Useful Model (6-8 hours)
1. Use **nano-1B** config
2. **Distillation** from Qwen3 endpoint
3. **The Stack Python** subset (or synthetic)
4. Train **50K steps** (~6 hours)
5. Evaluate on HumanEval
6. Target: >30% pass@1 (actually useful)

### Phase 3: Production (18-24 hours)
1. Use **Qwen3-hybrid** config
2. Full pipeline with selective ternarization
3. Train **100K steps**
4. Target: >70% pass@1

## Next Steps

1. Choose configuration (recommend nano-125M for quick demo)
2. Prepare training data
3. Start distillation training
4. Monitor progress
5. Evaluate on HumanEval
6. Compare ternary vs teacher quality

Ready to start?
