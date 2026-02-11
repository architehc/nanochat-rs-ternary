# Training Data Summary

## Current Dataset

### Small Dataset (Currently Training)
- **File**: `data/rust_tokens.bin`
- **Size**: 17MB
- **Tokens**: 4.2M
- **Source**: clap, serde, tokio (3 repositories)
- **Status**: ✅ In use for current training run

### Large Dataset (Ready for Future Use)
- **File**: `data/rust_tokens_large.bin`
- **Size**: 259MB
- **Tokens**: 68.0M (16.2x larger!)
- **Source**: 13 repositories, 11,042 files
- **Status**: ✅ Ready to use

## Repositories Included

| Repository | Files | Size | Description |
|------------|-------|------|-------------|
| rust-lang | 35,990 | 441M | The Rust compiler itself |
| servo | 1,384 | 1.2G | Browser engine |
| cargo | 1,345 | 37M | Package manager |
| tikv | 1,305 | 53M | Distributed KV store |
| diesel | 833 | 16M | ORM framework |
| tokio | 763 | 9.5M | Async runtime |
| clap | 328 | 6.3M | CLI parsing |
| actix-web | 311 | 5.1M | Web framework |
| regex | 225 | 12M | Regular expressions |
| serde | 208 | 3.1M | Serialization |
| rayon | 191 | 2.6M | Data parallelism |
| crossbeam | 115 | 2.7M | Concurrency primitives |
| hyper | 92 | 2.2M | HTTP library |
| **Total** | **43,090** | **1.8G** | **Production Rust code** |

## Dataset Comparison

| Metric | Small (Current) | Large (New) | Growth |
|--------|----------------|-------------|---------|
| Tokens | 4.2M | 68.0M | **16.2x** |
| Size | 17MB | 259MB | **15.2x** |
| Files | 3 repos | 13 repos | **4.3x** |
| Diversity | Basic | High | ⭐⭐⭐⭐⭐ |

## Quality Improvements Expected

With 16x more training data:
- **Better generalization**: More patterns learned
- **Higher compilation rate**: More correct examples
- **Richer vocabulary**: More library APIs seen
- **Fewer hallucinations**: More grounded in real code
- **Better async/concurrency**: tokio, rayon, crossbeam included
- **Improved error handling**: Result/Option patterns from std
- **Real-world patterns**: Production code from rust-lang/rust

## Usage

### For Next Training Run

```bash
# Backup current data
mv data/rust_tokens.bin data/rust_tokens_small.bin

# Use large dataset
cp data/rust_tokens_large.bin data/rust_tokens.bin

# Train with more steps (data is 16x larger)
bash train_stable_v2.sh --total-steps 50000
```

### Dataset Statistics

```bash
# Check token count
ls -lh data/*.bin

# Inspect distribution
cargo run --example inspect_data
```

## Notes

- **Current training** (step ~60/1000) uses **small dataset**
  - Let it finish for baseline comparison
  - ETA: ~6 more hours

- **Large dataset** ready for next run
  - Needs longer training (50K+ steps recommended)
  - Will take ~3-4 days on CPU
  - GPU highly recommended for large dataset

## Credits

Data sourced from top open-source Rust projects:
- Rust Foundation (rust-lang/rust, cargo)
- Mozilla (servo)
- PingCAP (tikv)
- Tokio project
- And many more amazing Rust projects ❤️

All code remains under original licenses.
