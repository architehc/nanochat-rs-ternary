# Production-Grade Model Training Guide

Complete pipeline for training a production-ready Rust code generation model with compiler-guided reinforcement learning.

## 🎯 Training Plan

### Phase 1: Extended Supervised Pre-training
**Goal:** Achieve strong base capabilities on Rust code patterns

- **Start:** step_2000 (6-hour base model)
- **Target:** step_15000 (13,000 additional steps)
- **Duration:** ~12-18 hours (estimate)
- **Strategy:** Automatic restart on OOM, checkpoint every 500 steps
- **Data:** 4.2M tokens from Tokio, Serde, Clap

**Expected outcomes:**
- Strong understanding of Rust syntax
- Common patterns (structs, traits, impls, error handling)
- Function signatures and type usage
- Module structure and imports

### Phase 2: RL Fine-tuning with Compiler Feedback
**Goal:** Optimize for compilable, idiomatic, high-quality code

- **Start:** Best supervised checkpoint (step_15000)
- **Iterations:** 1000 RL iterations
- **Duration:** ~6-8 hours (estimate)
- **Strategy:** GRPO with rustc + AST analysis

**Optimization targets:**
- ✅ Compilation success (primary)
- ✅ Idiomatic Rust patterns (iterators, pattern matching)
- ✅ Proper error handling (Result, Option, ?)
- ✅ Code quality (documentation, no panics)
- ✅ Reasonable complexity (low nesting, simple logic)

### Phase 3: Evaluation
**Goal:** Measure model capabilities

- Compilation success rate on held-out data
- Code quality metrics (AST analysis)
- Manual inspection of generated samples

### Phase 4: Export to GGUF
**Goal:** Production deployment format

- Export to GGUF for efficient inference
- Quantize to Q1_58 ternary format (68% compression)
- Deploy with nanochat-serve

## 🚀 Quick Start

### 1. Start Training

```bash
# In tmux or screen (recommended for long runs)
./scripts/train_production.sh
```

This will run all phases automatically:
- Phase 1: Supervised training to 15K steps
- Phase 2: RL fine-tuning for 1000 iterations
- Phase 3: Evaluation (manual)
- Phase 4: Export instructions

### 2. Monitor Progress

```bash
# In another terminal
./scripts/monitor_production.sh

# Or watch live logs
tail -f production_training_*.log

# Or watch GPU
watch -n 1 nvidia-smi
```

### 3. Check Intermediate Results

```bash
# List checkpoints
ls -lh checkpoints/production-supervised/step_*/

# Check latest
ls -lh checkpoints/production-supervised/step_* | tail -1

# Test generation (when ready)
cargo run --example generate_code
```

## 📊 Expected Checkpoints

```
checkpoints/
├── rust-6hour/
│   └── step_2000/           # Starting point (69MB)
├── production-supervised/
│   ├── step_2500/           # +500 steps
│   ├── step_3000/
│   ├── ...
│   └── step_15000/          # Final supervised (69MB each)
├── production-rl/
│   ├── rl-iter-100/         # RL checkpoints
│   ├── rl-iter-200/
│   ├── ...
│   └── rl-iter-1000/        # Final RL-tuned
└── production-final/
    └── model.gguf            # Exported for deployment
```

**Total storage estimate:** ~2-3GB for all checkpoints

## ⚙️ Configuration

### Hardware Requirements
- **GPU:** NVIDIA with 20GB+ VRAM (RTX 4090, A6000, etc.)
- **CPU:** 8+ cores recommended
- **RAM:** 32GB+ system memory
- **Disk:** 5GB+ free space

### Current Settings (in train_production.sh)
```bash
BATCH_SIZE=2              # Fits in 24GB VRAM
TARGET_STEPS=15000        # 13K additional steps
CHECKPOINT_INTERVAL=500   # Save every 500 steps
RL_ITERATIONS=1000        # RL fine-tuning iterations
DEVICE="cuda:0"           # Primary GPU
```

### Adjusting for Different Hardware

**If you have more VRAM (32GB+):**
```bash
BATCH_SIZE=4              # Faster training
```

**If you have less VRAM (16GB):**
```bash
BATCH_SIZE=1              # May need smaller model
# Consider using d20 config instead of nano-125M
```

**For longer/shorter training:**
```bash
TARGET_STEPS=20000        # More steps = better quality
TARGET_STEPS=10000        # Faster, less capable
```

## 📈 Progress Tracking

### Supervised Training Metrics
- **Loss:** Should decrease from ~176 → ~8-10
- **Perplexity:** Lower is better
- **Steps/second:** ~0.5-1.5 steps/sec typical
- **ETA:** Calculated automatically

### RL Training Metrics
- **Average Reward:** Should stabilize around 25-30
- **Compilation Rate:** Target 95-100%
- **Parse Rate:** Target 100%
- **Reward Std Dev:** Lower = more consistent

## 🔍 Monitoring Examples

### Check current phase
```bash
tail production_training_*.log | grep "PHASE"
```

### Watch loss values
```bash
tail -f production_training_*.log | grep "loss="
```

### GPU utilization
```bash
nvidia-smi dmon -s u
```

### Checkpoint progress
```bash
ls -lht checkpoints/production-supervised/step_* | head -5
```

## 🛠️ Troubleshooting

### Training crashes immediately
- Check CUDA is available: `nvidia-smi`
- Verify data exists: `ls -lh data/rust_tokens.bin`
- Check VRAM: May need smaller batch size

### OOM errors persist
- This is expected! Script auto-restarts
- Each run trains ~600-1100 steps before OOM
- Progress saved in checkpoints

### Training seems stuck
- Check GPU utilization: Should be 90-100%
- Check process is running: `pgrep -a train_rust_maxgpu`
- May be between restart cycles (3-5 sec pause)

### Loss not decreasing
- Early steps (< 500): Loss is volatile
- Check learning rate: May need adjustment
- Verify data quality: `wc -c data/rust_tokens.bin`

### Checkpoints not saving
- Check disk space: `df -h`
- Verify permissions: `ls -ld checkpoints/`
- Check checkpoint interval reached

## 📊 Quality Benchmarks

### Compilation Success Rate
- **Baseline (pre-RL):** ~60-80%
- **Target (post-RL):** 95-100%
- **Measured by:** rustc --error-format=json

### Code Quality Metrics
- **Idiomatic patterns:** 70%+ using iterators/closures
- **Error handling:** 80%+ using Result/Option
- **Documentation:** 50%+ functions with docs
- **Complexity:** Avg cyclomatic < 5

### Manual Evaluation
- Code compiles without modification
- Solves the specified task correctly
- Uses appropriate data structures
- Follows Rust idioms and conventions
- Has reasonable error handling

## 🎓 Training Tips

### 1. Use tmux/screen
Long training runs should use persistent sessions:
```bash
tmux new -s training
./scripts/train_production.sh
# Ctrl+B, D to detach
# tmux attach -t training to reattach
```

### 2. Monitor regularly
Check progress every few hours:
```bash
./scripts/monitor_production.sh
```

### 3. Save intermediate checkpoints
Don't wait for completion - test intermediate steps:
```bash
# Test at step_5000, step_10000, etc.
cargo run --example generate_code -- --checkpoint checkpoints/production-supervised/step_5000
```

### 4. Clean old checkpoints
Save disk space by removing early checkpoints:
```bash
# Keep every 1000th checkpoint
rm -rf checkpoints/production-supervised/step_{2500,3000,3500,4000,4500}
```

### 5. Log everything
All output is logged to `production_training_*.log`:
```bash
# Search for errors
grep -i error production_training_*.log

# Find best loss
grep "loss=" production_training_*.log | sort -t= -k2 -n | head
```

## 🚢 Deployment

After training completes:

### 1. Test the model
```bash
cargo run --example generate_code -- --checkpoint checkpoints/production-supervised/step_15000
```

### 2. Run benchmarks
```bash
cargo run -p nanochat-eval -- --checkpoint checkpoints/production-supervised/step_15000
```

### 3. Export to GGUF
```bash
cargo run --example export_gguf -- \
    --checkpoint checkpoints/production-supervised/step_15000 \
    --output checkpoints/production-final/model.gguf
```

### 4. Start inference server
```bash
cargo run --release -p nanochat-serve -- \
    --model checkpoints/production-final/model.gguf \
    --port 8080
```

### 5. Test API
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Rust function to reverse a string"}],
    "max_tokens": 256
  }'
```

## 📚 References

- **Base training:** scripts/train_rust_maxgpu.rs
- **RL training:** crates/nanochat-rl/
- **Evaluation:** crates/nanochat-eval/
- **Inference:** crates/nanochat-serve/
- **Export:** examples/export_gguf.rs

## 🎯 Success Criteria

A production-grade model should achieve:

✅ **Compilation:** 95%+ success rate on diverse prompts
✅ **Quality:** High reward scores (>25) in RL evaluation
✅ **Idiomaticity:** Uses iterators, pattern matching, proper error handling
✅ **Correctness:** Solves specified tasks accurately
✅ **Documentation:** Generated code is understandable
✅ **Performance:** Fast inference (< 100ms per generation)
✅ **Size:** Reasonable model size (< 100MB in GGUF)

---

**Ready to train?** Run `./scripts/train_production.sh` and monitor with `./scripts/monitor_production.sh`!
