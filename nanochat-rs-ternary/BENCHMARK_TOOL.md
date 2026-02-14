# Ternary Model Inference Benchmark Tool

## Overview

Complete benchmarking infrastructure for measuring ternary model inference performance across different configurations, prompt lengths, and quantization strategies.

## Architecture

### Core Components

1. **bench.rs** (nanochat-model crate)
   - Data structures for benchmark results
   - Hardware detection utilities
   - Timer and measurement helpers

2. **benchmark_inference.rs** (examples/)
   - CLI tool for running benchmarks
   - Synthetic workload simulation
   - Results reporting and JSON export

## Features

### Measurement Capabilities

- **Prefill Performance**
  - Throughput (tokens/sec)
  - Latency (ms/token)
  - Batch size support
  - Multiple prompt lengths

- **Decode Performance**
  - Autoregressive token generation
  - Per-token latency
  - Throughput measurement

- **Memory Estimation**
  - Model weights (ternary compression factor)
  - Activation memory
  - KV cache memory
  - Peak usage estimates

- **Hardware Detection**
  - CPU model and core count
  - System memory (GB)
  - GPU info (placeholder for future)

### Configuration Support

All model configs are supported:
- `d20` (20M params, debug/test)
- `nano_125m` (125M params)
- `nano_1b` (1B params)
- `qwen3_coder_80b` (80B params with MoE)

## Usage

### Basic Example

**NOTE:** The `benchmark_inference` example is experimental and not yet exposed. Use `cargo bench` for performance benchmarks instead.

```bash
# Use Criterion benchmarks instead:
cargo bench --workspace

# Or benchmark specific components:
cargo bench -p ternary-kernels  # GEMV kernels
cargo bench mhc_overhead        # mHC overhead
```

### Full Options (Experimental - not yet available)

```bash
# cargo run --release --example benchmark_inference -- \
#   --model hybrid.gguf \              # Model file (currently placeholder)
#   --mhc hybrid.mhc \                 # mHC weights (optional)
#   --config qwen3_coder_80b \         # Model configuration
#   --prompt-lengths 128,512,2048 \    # Comma-separated prompt lengths
  --num-tokens 100 \                 # Decode tokens to generate
  --batch-size 1 \                   # Prefill batch size
  --warmup 3 \                       # Warmup iterations
  --iterations 10 \                  # Benchmark iterations
  --output results.json \            # JSON output file
  --device cpu                       # Device (cpu, cuda:0, etc.)
```

## Sample Output

```
═══════════════════════════════════════════════════════════
  Ternary Model Inference Benchmark
═══════════════════════════════════════════════════════════

Configuration:
  Model: qwen3_coder_80b
  Parameters: 80.5B
  Device: cpu
  Model file: hybrid.gguf

Hardware:
  CPU: AMD EPYC 7663 56-Core Processor (224 cores)
  System Memory: 944.7GB

Benchmark Plan:
  Prompt lengths: [128, 512, 2048]
  Decode tokens: 100
  Batch size: 1
  Warmup iterations: 3
  Benchmark iterations: 10

═══════════════════════════════════════════════════════════
Benchmarking prompt length: 2048 tokens
═══════════════════════════════════════════════════════════

Warming up (3 iterations)...
...
Benchmarking prefill (10 iterations)...
  Throughput: 23256.4 tokens/sec
  Latency: 0.04 ms/token
  Total time: 0.09s

Benchmarking decode (10 iterations)...
  Throughput: 23223.1 tokens/sec
  Latency: 0.04 ms/token
  Total time: 0.00s

Memory Estimate:
  Model weights: 20.13GB
  Activations: 32.00KB
  KV cache: 24.00MB
  Peak: 20.18GB

═══════════════════════════════════════════════════════════
  Inference Benchmark Results
═══════════════════════════════════════════════════════════

Model: qwen3_coder_80b
Quantization: Q1_58 (Ternary)

[... detailed report ...]
```

## JSON Export Format

```json
{
  "config": "qwen3_coder_80b",
  "model_file": "hybrid.gguf",
  "device": "cpu",
  "results": [
    {
      "model_name": "qwen3_coder_80b",
      "quant_type": "Q1_58 (Ternary)",
      "prefill": {
        "batch_size": 1,
        "seq_len": 2048,
        "total_time": 0.088,
        "tokens_per_sec": 23256.4,
        "ms_per_token": 0.043,
        "gpu_util": null
      },
      "decode": {
        "batch_size": 1,
        "num_tokens": 100,
        "total_time": 0.004,
        "tokens_per_sec": 23223.1,
        "ms_per_token": 0.043,
        "gpu_util": null
      },
      "memory": {
        "model_weights": 21610127360,
        "activations": 32768,
        "kv_cache": 25165824,
        "peak_memory": 21635325952,
        "bandwidth_gbs": null
      },
      "hardware": {
        "cpu_model": "AMD EPYC 7663 56-Core Processor",
        "cpu_cores": 224,
        "gpu_model": null,
        "gpu_memory_gb": null,
        "system_memory_gb": 944.7
      }
    }
  ]
}
```

## Implementation Details

### Parameter Counting

The `param_count_estimate()` method accurately counts:
- Token embeddings (vocab_size × dim)
- Per-layer attention projections (Q, K, V, O)
- FFN layers:
  - Dense: gate + up + down projections
  - MoE: n_experts × expert_dim × 3 + router + shared expert
- Norms (RMSNorm: 1 param per dim, 2 per layer)
- LM head (unless weight-tied with embeddings)

### Memory Estimation

**Model Weights**:
- Ternary quantization: ~0.25 bytes per parameter
- Includes 2-bit weights + FP32 scales (group_size=128)

**Activations**:
- FP32: 4 bytes per element
- Estimate: hidden_dim × batch_size × intermediate_factor
- Conservative 8× multiplier for SwiGLU intermediates

**KV Cache**:
- FP16: 2 bytes per element
- Formula: 2 × n_layers × n_kv_heads × seq_len × head_dim × 2

### Synthetic Benchmarking

Until model loading is integrated, the tool simulates compute:
- Estimates FLOPs: ~2 FLOPs per param per token
- Assumes ~1 TFLOP/s throughput for ternary GEMV
- Busy-wait loop to simulate actual runtime
- Provides realistic timing baselines

## Future Enhancements

### Phase 1 (Integration Ready)
- [ ] Load actual GGUF models
- [ ] Real ternary GEMV calls via ternary-kernels
- [ ] Actual inference passes (not simulated)

### Phase 2 (Profiling)
- [ ] Per-layer timing breakdown
- [ ] GPU utilization monitoring
- [ ] Memory bandwidth measurement
- [ ] NUMA-aware profiling

### Phase 3 (Analysis)
- [ ] Compare multiple quantization levels
- [ ] FP32 vs FP16 vs FP8 vs Ternary comparisons
- [ ] Quality-performance tradeoff analysis
- [ ] Perplexity measurement integration

### Phase 4 (Production)
- [ ] Continuous benchmarking CI
- [ ] Historical trend tracking
- [ ] Regression detection
- [ ] Performance target validation

## Comparison with Other Tools

| Feature | This Tool | llama.cpp bench | candle bench |
|---------|-----------|-----------------|--------------|
| Ternary-specific | ✅ Yes | ❌ No | ❌ No |
| MoE support | ✅ Yes | ⚠️ Partial | ⚠️ Partial |
| Memory estimation | ✅ Yes | ⚠️ Basic | ⚠️ Basic |
| JSON export | ✅ Yes | ❌ No | ❌ No |
| Per-layer profiling | 🚧 Planned | ✅ Yes | ✅ Yes |
| Real-time monitoring | 🚧 Planned | ✅ Yes | ❌ No |

## Contributing

When adding new benchmark capabilities:
1. Add data structure to `bench.rs`
2. Update `to_json()` method for export
3. Add measurement logic to CLI tool
4. Update this documentation

## References

- Model config definitions: `crates/nanochat-model/src/config.rs`
- Benchmark structures: `crates/nanochat-model/src/bench.rs`
- CLI tool: `examples/benchmark_inference.rs`
- Task tracking: Issue #5 in project plan
