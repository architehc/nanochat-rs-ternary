# Code Generation Quality Evaluation Framework

Complete evaluation infrastructure for measuring code generation quality on HumanEval and MBPP benchmarks.

## Overview

The `nanochat-eval` crate provides a comprehensive framework for:
- Loading HumanEval and MBPP datasets
- Querying models for code completions
- Sandboxed Python code execution
- Pass@k metric calculation
- Model comparison and reporting

## Architecture

### Components

1. **datasets.rs** - Dataset loaders
   - `HumanEvalDataset`: 164 programming problems
   - `MBPPDataset`: 974 programming problems
   - Unified `CodeProblem` interface

2. **executor.rs** - Sandboxed execution
   - `CodeExecutor`: Runs generated code with tests
   - Timeout handling (default: 5s)
   - Error classification (syntax, assertion, timeout, etc.)
   - Async execution with tokio

3. **metrics.rs** - Pass@k calculation
   - `PassAtK`: Unbiased estimator for pass@1, pass@10, pass@100
   - `EvalMetrics`: Complete evaluation results
   - Error breakdown and statistics

4. **eval.rs** - Main evaluation harness
   - `Evaluator`: Coordinates dataset → model → execution → metrics
   - Progress bars with indicatif
   - Parallel execution support
   - HTTP API client for model querying

## Usage

### Basic Evaluation

```bash
# Download HumanEval dataset first
wget https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
gunzip HumanEval.jsonl.gz

# Start your model server
cargo run --release -p nanochat-serve -- \
  --model hybrid.gguf \
  --mhc hybrid.mhc \
  --port 8080

# Run evaluation
cargo run --release --example evaluate_codegen -- \
  --dataset humaneval \
  --data-path HumanEval.jsonl \
  --model-endpoint http://localhost:8080/v1/completions \
  --model-name qwen3-ternary \
  --num-samples 10 \
  --output results.json
```

### With Baseline Comparison

```bash
# Evaluate ternary model vs FP8 baseline
cargo run --release --example evaluate_codegen -- \
  --dataset humaneval \
  --data-path HumanEval.jsonl \
  --model-endpoint http://localhost:8080/v1/completions \
  --model-name qwen3-ternary \
  --baseline-endpoint https://crazyshit.ngrok.io/v1/completions \
  --baseline-name qwen3-fp8 \
  --num-samples 10 \
  --output comparison.json
```

### MBPP Evaluation

```bash
# Download MBPP dataset
wget https://github.com/google-research/google-research/raw/master/mbpp/mbpp.json

# Run evaluation
cargo run --release --example evaluate_codegen -- \
  --dataset mbpp \
  --data-path mbpp.json \
  --model-endpoint http://localhost:8080/v1/completions \
  --num-samples 10
```

### Options

```
--dataset <DATASET>              Dataset to evaluate (humaneval, mbpp) [default: humaneval]
--data-path <PATH>               Path to dataset file
--model-endpoint <URL>           Model endpoint [default: http://localhost:8080/v1/completions]
--model-name <NAME>              Model identifier [default: nanochat-ternary]
--num-samples <N>                Samples per problem [default: 10]
--temperature <TEMP>             Sampling temperature [default: 0.8]
--max-tokens <N>                 Max tokens to generate [default: 512]
--execution-timeout <SECS>       Code execution timeout [default: 5]
--max-problems <N>               Limit number of problems (for testing)
--output <FILE>                  Save results as JSON
--baseline-endpoint <URL>        Baseline model for comparison
--baseline-name <NAME>           Baseline model name
```

## Sample Output

```
═══════════════════════════════════════════════════════════
  Code Generation Quality Evaluation
═══════════════════════════════════════════════════════════

Loading dataset: humaneval from HumanEval.jsonl
  Loaded 164 HumanEval problems

Evaluating qwen3-ternary model...
[00:05:23] ████████████████████████████████████████ 164/164 Evaluation complete

═══════════════════════════════════════════════════════════
  Code Generation Evaluation Results
═══════════════════════════════════════════════════════════

Model: qwen3-ternary
Dataset: HumanEval

Pass@k Scores:
  pass@1:  75.61% (124/164)
  pass@10: 89.02% (146/164)

Performance:
  Problems solved: 124/164
  Avg execution time: 45.2ms

Error Breakdown:
  AssertionError: 28 (17.1%)
  SyntaxError: 8 (4.9%)
  Timeout: 4 (2.4%)

═══════════════════════════════════════════════════════════
```

## Metrics

### Pass@k Calculation

The pass@k metric estimates the probability that at least one of k generated samples passes all test cases.

**Formula** (unbiased estimator):
```
pass@k = E[1 - C(n-c, k) / C(n, k)]
```
where:
- n = total samples generated per problem
- c = samples that passed tests
- C(n, k) = n choose k (binomial coefficient)

**Interpretation**:
- **pass@1**: Single attempt success rate (greedy or sampled)
- **pass@10**: Success rate when generating 10 samples
- **pass@100**: Success rate when generating 100 samples

Higher pass@k indicates better code generation quality.

### HumanEval Benchmark

- **164 problems**: Hand-crafted Python programming challenges
- **Difficulty**: Medium to hard (LeetCode-style)
- **Coverage**: Algorithms, data structures, string manipulation, math
- **Standard**: Industry benchmark for code generation models

**Typical Scores**:
- GPT-4: ~90% pass@1
- GPT-3.5: ~48% pass@1
- CodeGen-16B: ~29% pass@1
- Target for ternary model: >70% pass@1 (within 10% of FP8 baseline)

### MBPP Benchmark

- **974 problems**: Mostly Basic Python Programming
- **Difficulty**: Easy to medium
- **Coverage**: Basic Python operations, simple algorithms
- **Use case**: Measuring fundamental coding ability

**Typical Scores**:
- GPT-4: ~85% pass@1
- GPT-3.5: ~52% pass@1
- Target for ternary model: >60% pass@1

## Testing

The framework includes comprehensive tests:

```bash
cargo test -p nanochat-eval
```

**Test Coverage**:
- `test_extract_function_name`: Function name parsing
- `test_humaneval_parse`: JSON parsing
- `test_binomial_coeff`: Combinatorial calculations
- `test_pass_at_k_*`: Metric calculation edge cases
- `test_successful_execution`: Code execution happy path
- `test_failed_assertion`: Test failure handling
- `test_syntax_error`: Syntax error detection
- `test_timeout`: Timeout handling

All 10 tests pass ✅

## JSON Export Format

```json
{
  "reports": [
    {
      "model_name": "qwen3-ternary",
      "dataset_name": "HumanEval",
      "pass_at_1": {
        "k": 1,
        "score": 0.7561,
        "num_problems": 164,
        "num_solved": 124
      },
      "pass_at_10": {
        "k": 10,
        "score": 0.8902,
        "num_problems": 164,
        "num_solved": 146
      },
      "total_problems": 164,
      "problems_solved": 124,
      "avg_execution_time_ms": 45.2,
      "error_counts": {
        "AssertionError": 28,
        "SyntaxError": 8,
        "Timeout": 4
      },
      "per_problem_results": {
        "HumanEval/0": {
          "task_id": "HumanEval/0",
          "num_passed": 8,
          "num_total": 10,
          "avg_time_ms": 42.5,
          "error_types": ["AssertionError", "AssertionError"]
        }
      }
    }
  ],
  "timestamp": "2026-02-08T10:30:00Z"
}
```

## Implementation Details

### Sandboxed Execution

- Spawns subprocess with `python3 -c`
- 5-second timeout (configurable)
- Captures stdout/stderr
- Classifies errors by type
- No persistent state between runs

**Security Note**: Current implementation uses subprocess isolation only. For production, consider additional sandboxing (Docker, firejail, etc.).

### Error Classification

| Error Type | Detection | Example |
|------------|-----------|---------|
| SyntaxError | `SyntaxError` in stderr | Missing colon, invalid syntax |
| AssertionError | `AssertionError` in stderr | Test case failure |
| ImportError | `ImportError`/`ModuleNotFoundError` | Missing dependencies |
| Timeout | Process exceeds timeout | Infinite loop, long runtime |
| RuntimeError | Any other stderr output | Exceptions, crashes |

### Model API Contract

The evaluator expects an OpenAI-compatible completions endpoint:

**Request**:
```json
{
  "prompt": "def has_close_elements(...):\n",
  "max_tokens": 512,
  "temperature": 0.8,
  "stop": ["\nclass ", "\ndef ", "\n#"]
}
```

**Response**:
```json
{
  "choices": [
    {
      "text": "    for i in range(len(numbers)):\n        ..."
    }
  ]
}
```

## Future Enhancements

### Phase 1 - Additional Datasets
- [ ] CodeContests (competitive programming)
- [ ] LeetCode problems
- [ ] Apps (application-level code generation)

### Phase 2 - Advanced Metrics
- [ ] Code quality scoring (readability, efficiency)
- [ ] Security vulnerability detection
- [ ] Test coverage analysis
- [ ] Execution speed comparison

### Phase 3 - Analysis Tools
- [ ] Per-category breakdown (algorithms, data structures, etc.)
- [ ] Difficulty correlation
- [ ] Error pattern analysis
- [ ] Learning curves (vs training steps)

### Phase 4 - Production Features
- [ ] Distributed evaluation (multiple GPUs/nodes)
- [ ] Caching and incremental evaluation
- [ ] Web dashboard for results visualization
- [ ] CI/CD integration

## Related Work

- **HumanEval**: Chen et al., "Evaluating Large Language Models Trained on Code" (2021)
- **MBPP**: Austin et al., "Program Synthesis with Large Language Models" (2021)
- **Pass@k metric**: Kulal et al., "SPoC: Search-based Pseudocode to Code" (2019)

## References

- HumanEval dataset: https://github.com/openai/human-eval
- MBPP dataset: https://github.com/google-research/google-research/tree/master/mbpp
- Evaluation harness: `crates/nanochat-eval/`
- CLI tool: `examples/evaluate_codegen.rs`
