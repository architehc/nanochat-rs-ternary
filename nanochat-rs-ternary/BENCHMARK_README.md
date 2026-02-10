# Nanochat Model Benchmarking System

## Overview

Comprehensive evaluation framework for measuring Rust code generation model quality across multiple dimensions:

- **Compilation Success Rate**: Percentage of generated code that compiles
- **Code Quality Metrics**: Lines of code, function count, cyclomatic complexity
- **Performance Metrics**: Generation speed (tokens/sec), latency
- **Detailed Analysis**: Per-prompt breakdowns, common error patterns, sample outputs

## Usage

### Basic Benchmark Run

```bash
cargo run --release -p nanochat-eval --example benchmark_model -- \
  --checkpoint checkpoints/stable-v2/step_10000 \
  --n-samples 100 \
  --output results.json
```

### Full Configuration

```bash
cargo run --release -p nanochat-eval --example benchmark_model -- \
  --checkpoint <checkpoint_path>  # Model checkpoint to evaluate
  --n-samples <N>                 # Total samples (default: 100)
  --temperature <T>               # Sampling temperature (default: 0.8)
  --max-tokens <N>                # Max tokens per sample (default: 200)
  --output <file.json>            # Output file (default: benchmark_results.json)
  --device <cpu|cuda:0>           # Compute device (default: cuda:0)
```

## Test Prompts

The benchmark includes 16 diverse Rust patterns:

1. **Basic Functions**: `factorial`, `is_palindrome`, `reverse_string`
2. **Data Structures**: `struct Point`, `enum Color`, `impl Point`
3. **Collections**: `find_duplicates`, `merge_sorted`
4. **Error Handling**: `Result<T, E>`, `Option<T>`
5. **Iterators**: `sum_squares`, `filter_even`
6. **Async**: `async fn fetch_data`
7. **Traits**: `trait Parser`, `impl ToString`
8. **Macros**: `macro_rules! vec_of_strings`

## Output Format

### JSON Structure

```json
{
  "checkpoint": "checkpoints/stable-v2/step_10000",
  "timestamp": "2026-02-09T23:30:00Z",
  "total_samples": 100,
  "compile_success_rate": 45.0,
  "compile_errors": ["error[E0308]: mismatched types", ...],
  "avg_lines": 12.3,
  "avg_functions": 1.2,
  "avg_complexity": 3.5,
  "avg_generation_time_ms": 150.0,
  "tokens_per_second": 25.5,
  "samples_by_prompt": [
    {
      "prompt": "fn factorial(n: u64) -> u64 {",
      "n_samples": 6,
      "compile_success": 4,
      "avg_lines": 8.5,
      "common_errors": [...],
      "sample_outputs": [...]
    }
  ]
}
```

### Console Output

```
═══════════════════════════════════════════════════════════
  Benchmark Results
═══════════════════════════════════════════════════════════

Compilation:
  Success: 45/100 (45.0%)
  Failures: 55

Code Quality:
  Avg lines per sample: 12.3
  Avg functions per sample: 1.2
  Avg complexity per sample: 3.5

Performance:
  Total time: 15.0s
  Avg generation time: 150ms
  Samples per second: 6.67
  Tokens per second: 25.5
```

## Metrics Explained

### Compilation Success Rate

- Uses `rustc` to compile each generated sample
- Captures full compiler output (errors, warnings, notes)
- Groups common error patterns
- **Target**: >80% for production models

### Code Quality

- **Lines**: Non-empty lines (excludes blank lines)
- **Functions**: Counts `fn` declarations in modules, impls, and traits (via `syn` AST parser)
- **Complexity**: Simplified cyclomatic complexity (counts decision points: `if`, `while`, `for`, `match`, `&&`, `||`, `?`)

### Performance

- **Generation Time**: Wall-clock time per sample (includes model inference)
- **Tokens per Second**: Generated tokens / total generation time
- **Samples per Second**: Total samples / benchmark elapsed time

## Implementation Details

### Dependencies

- **nanochat-train**: Model loading and inference
- **nanochat-rl**: Compiler feedback via `rustc --error-format=json`
- **tokenizers**: GPT-2 tokenizer for encoding/decoding
- **syn**: Rust AST parsing for function counting
- **candle**: ML framework (CPU/CUDA)
- **clap**: CLI argument parsing
- **serde/serde_json**: JSON serialization
- **chrono**: Timestamps

### AST Parsing

Uses the `syn` crate to parse generated code and extract structural information:

```rust
fn count_functions(code: &str) -> usize {
    let parsed = syn::parse_file(code);
    if let Ok(file) = parsed {
        // Count top-level functions
        // Count methods in impl blocks
        // Count trait methods
    }
}
```

### Complexity Calculation

Simplified cyclomatic complexity based on keyword counting:

```rust
fn calculate_complexity(code: &str) -> usize {
    let keywords = ["if ", "while ", "for ", "match ", "&&", "||", "?"];
    let mut complexity = 1; // Base complexity
    for keyword in &keywords {
        complexity += code.matches(keyword).count();
    }
    complexity
}
```

## Current Status (2026-02-09)

### Step 9000 Benchmark Results

- **Training**: 14800/10000 steps (went beyond target)
- **Loss**: ~1.5 (good convergence)
- **Compilation Success**: 0% ⚠️ **CRITICAL ISSUE**
- **Generation**: Model collapsed, generating repeated `{` tokens
- **Avg Output**: 1 line, ~100 tokens (all `{`)

### Known Issues

1. **Model Collapse**: Despite low loss, model generates degenerate output
2. **Root Cause**: Likely training data imbalance or optimization instability
3. **Next Steps**:
   - Investigate token distribution in training data
   - Add entropy regularization to prevent collapse
   - Try different learning rates / schedules
   - Consider adding diversity penalties

## Tracking Progress Over Time

Run benchmarks at regular intervals (e.g., every 1000 steps) and compare:

```bash
# Step 5000
cargo run --release -p nanochat-eval --example benchmark_model -- \
  --checkpoint checkpoints/stable-v2/step_5000 \
  --n-samples 100 --output results_5k.json

# Step 10000
cargo run --release -p nanochat-eval --example benchmark_model -- \
  --checkpoint checkpoints/stable-v2/step_10000 \
  --n-samples 100 --output results_10k.json

# Compare
jq '.compile_success_rate' results_5k.json results_10k.json
```

## Future Enhancements

- [ ] Add more diverse test prompts (generics, lifetimes, unsafe)
- [ ] Measure compilation warnings separately
- [ ] Add test execution (compile + run tests)
- [ ] Add semantic similarity metrics (compare to reference implementations)
- [ ] Add human evaluation rubric (code style, idiomaticity)
- [ ] Add diversity metrics (unique AST structures, vocabulary richness)
- [ ] Add pass@k evaluation (multiple samples, success if any compile)
- [ ] Add benchmark against HumanEval-Rust or similar datasets

## Related Files

- `crates/nanochat-eval/examples/benchmark_model.rs`: Main benchmark implementation
- `crates/nanochat-rl/src/compiler.rs`: Rustc integration for compilation checks
- `scripts/train_pipeline_accelerated.sh`: End-to-end training + evaluation pipeline
