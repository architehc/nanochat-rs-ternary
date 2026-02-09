# nanochat-rl: Reinforcement Learning for Rust Code Generation

Compiler-guided reinforcement learning system for training Rust code generation models using GRPO/GSPO algorithm.

## Overview

This crate implements a complete RL pipeline that combines:

1. **Compiler Feedback** - Integrates with `rustc` to evaluate generated code
2. **AST Analysis** - Deep code quality metrics using the `syn` crate
3. **Reward Function** - Combines compilation success, AST quality, and idiomatic patterns
4. **GRPO Algorithm** - Group Relative Policy Optimization for stable training
5. **Optional Qwen3 Integration** - External code evaluation for additional signal

## Architecture

```
┌─────────────┐
│   Prompt    │
│  "Write a   │
│  function..." │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Generate N Samples │
│   (Policy Model)    │
└──────┬──────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Evaluate Each Sample:               │
│  1. Compile with rustc               │
│  2. Parse AST (syn)                  │
│  3. Compute quality metrics          │
│  4. Optional: Qwen3 evaluation       │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Compute Rewards     │
│  (Compiler + AST +   │
│   Qwen3)             │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Normalize Within    │
│  Group (GRPO)        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Update Policy       │
│  (Gradient Descent)  │
└──────────────────────┘
```

## Components

### 1. Compiler Feedback (`compiler.rs`)

Compiles generated code and extracts structured feedback:

```rust
use nanochat_rl::CompilerFeedback;

let compiler = CompilerFeedback::new()?;
let result = compiler.compile(code)?;

println!("Success: {}", result.success);
println!("Errors: {}", result.n_errors);
println!("Warnings: {}", result.n_warnings);
```

### 2. AST Analysis (`ast_analysis.rs`)

Parses code and extracts deep quality metrics:

```rust
use nanochat_rl::analyze_ast;

let metrics = analyze_ast(code)?;

println!("Cyclomatic complexity: {}", metrics.complexity.cyclomatic);
println!("Uses Result: {}", metrics.quality.uses_result);
println!("Uses iterators: {}", metrics.idioms.uses_iterators);
```

**Metrics collected:**
- **Complexity**: cyclomatic complexity, nesting depth, LOC
- **Structure**: functions, structs, enums, traits, impls
- **Quality**: Result/Option usage, documentation, error handling, panics, unsafe
- **Idioms**: iterators, pattern matching, closures, method chaining

### 3. Reward Function (`reward.rs`)

Combines all signals into a single reward:

```rust
use nanochat_rl::{compute_reward, RewardConfig};

let config = RewardConfig::default();
let reward = compute_reward(&compile_result, &ast_metrics, &config);
```

**Reward components:**
- ✅ Compilation success: +10.0
- ✅ No errors: +5.0
- ✅ No warnings: +1.0
- ✅ Parseable AST: +8.0
- ✅ Good structure: +0-2.0
- ✅ Quality patterns: +0-3.0
- ✅ Idiomatic Rust: +0-2.0
- ❌ High complexity: -0.5 per excess unit
- ❌ Panics (unwrap/expect): -2.0 each
- ❌ Unsafe code: -1.0

### 4. GRPO Algorithm (`grpo.rs`)

Group-based policy optimization with relative rewards:

```rust
use nanochat_rl::{GrpoTrainer, GrpoConfig, GrpoBatch};

let config = GrpoConfig::default();
let trainer = GrpoTrainer::new(config);

// Create batch and normalize rewards within groups
let mut batch = GrpoBatch::new(prompts, n_samples);
batch.normalize_rewards();

// Compute loss
let loss = trainer.compute_loss(&log_probs, &relative_rewards, None, &entropy);
```

**Benefits of GRPO:**
- More stable than absolute rewards
- Naturally normalized (zero mean, unit variance)
- Encourages diversity through group comparison
- Less sensitive to reward scale

### 5. Qwen3 Integration (`qwen.rs`)

Optional external evaluation using Qwen3 Coder:

```rust
use nanochat_rl::QwenClient;

let client = QwenClient::new(endpoint, api_key);
let eval = client.evaluate_code(code, context).await?;

println!("Quality: {}/10", eval.quality_score);
println!("Correctness: {}/10", eval.correctness_score);
println!("Idiomatic: {}/10", eval.idiomaticity_score);
```

## Usage

### Basic Training Loop

```rust
use nanochat_rl::{RLConfig, RLTrainer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut config = RLConfig::default();
    config.base_checkpoint = "checkpoints/rust-6hour/step_2000".to_string();
    config.n_iterations = 100;
    config.n_samples = 4;
    config.batch_size = 2;
    config.device = "cuda:0".to_string();

    let mut trainer = RLTrainer::new(config)?;
    trainer.train().await?;

    Ok(())
}
```

### Command Line

```bash
# Basic training
cargo run --example train_rl -- \
    --checkpoint checkpoints/rust-6hour/step_2000 \
    --iterations 100 \
    --n-samples 4 \
    --batch-size 2

# With Qwen3 endpoint
cargo run --example train_rl -- \
    --checkpoint checkpoints/rust-6hour/step_2000 \
    --iterations 100 \
    --qwen-endpoint https://api.together.ai/v1/chat/completions

# Custom hyperparameters
cargo run --example train_rl -- \
    --checkpoint checkpoints/rust-6hour/step_2000 \
    --iterations 1000 \
    --lr 5e-6 \
    --kl-coef 0.05
```

## Configuration

### RLConfig

```rust
pub struct RLConfig {
    pub base_checkpoint: String,      // Starting model checkpoint
    pub grpo: GrpoConfig,             // GRPO algorithm config
    pub reward: RewardConfig,         // Reward function weights
    pub n_samples: usize,             // Samples per prompt (default: 4)
    pub max_tokens: usize,            // Max generation length (default: 256)
    pub temperature: f64,             // Sampling temperature (default: 0.8)
    pub n_iterations: usize,          // Training iterations (default: 1000)
    pub batch_size: usize,            // Prompts per iteration (default: 2)
    pub device: String,               // "cpu" or "cuda:0" (default: "cuda:0")
    pub qwen_endpoint: Option<String>, // Optional Qwen3 URL
}
```

### GrpoConfig

```rust
pub struct GrpoConfig {
    pub learning_rate: f64,           // Policy LR (default: 1e-5)
    pub kl_coef: f64,                 // KL penalty (default: 0.1)
    pub clip_ratio: Option<f64>,      // PPO clipping (default: 0.2)
    pub entropy_coef: f64,            // Exploration bonus (default: 0.01)
    pub value_coef: f64,              // Value loss weight (default: 0.5)
    pub max_grad_norm: f64,           // Gradient clipping (default: 1.0)
    pub n_epochs: usize,              // Optimization epochs (default: 4)
    pub mini_batch_size: usize,       // Mini-batch size (default: 4)
}
```

### RewardConfig

```rust
pub struct RewardConfig {
    pub w_compile_success: f64,       // Weight for compilation (default: 10.0)
    pub w_no_errors: f64,             // Weight for no errors (default: 5.0)
    pub w_no_warnings: f64,           // Weight for no warnings (default: 1.0)
    pub w_parseable: f64,             // Weight for valid syntax (default: 8.0)
    pub w_structure: f64,             // Weight for structure (default: 2.0)
    pub w_quality: f64,               // Weight for quality (default: 3.0)
    pub w_idioms: f64,                // Weight for idioms (default: 2.0)
    pub w_complexity: f64,            // Complexity penalty (default: -0.5)
    pub panic_penalty: f64,           // Panic penalty (default: -2.0)
    pub unsafe_penalty: f64,          // Unsafe penalty (default: -1.0)
    pub doc_bonus: f64,               // Doc bonus (default: 1.0)
    pub max_complexity: f64,          // Max cyclomatic (default: 10.0)
}
```

## Testing

```bash
# Run all tests
cargo test -p nanochat-rl

# Run specific test module
cargo test -p nanochat-rl compiler::tests
cargo test -p nanochat-rl ast_analysis::tests
cargo test -p nanochat-rl reward::tests
cargo test -p nanochat-rl grpo::tests

# Run with output
cargo test -p nanochat-rl -- --nocapture
```

## Monitoring

Training logs are written to `rl_training.log`:

```csv
iteration,avg_reward,reward_std,compile_success_rate,parse_success_rate
1,12.5432,3.2145,0.75,0.875
2,14.2341,2.8976,0.8125,0.9375
...
```

Use the monitoring script:

```bash
# Watch training progress
watch -n 5 tail -20 rl_training.log

# Plot rewards
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('rl_training.log')
plt.plot(df['iteration'], df['avg_reward'])
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('RL Training Progress')
plt.show()
"
```

## Integration with Base Model

This crate is designed to work with the `nanochat-model` and `nanochat-train` crates. The typical workflow:

1. **Pre-train** a base model on Rust code (supervised learning)
   ```bash
   cargo run --example train_rust_maxgpu --features nanochat-train/cuda
   ```

2. **Export** to checkpoint
   ```bash
   # Checkpoint saved at: checkpoints/rust-6hour/step_2000
   ```

3. **RL training** with compiler feedback
   ```bash
   cargo run --example train_rl -- --checkpoint checkpoints/rust-6hour/step_2000
   ```

4. **Evaluate** on benchmarks
   ```bash
   cargo run --example eval_rust_codegen
   ```

## Future Work

- [ ] Integrate with actual model inference (currently uses placeholders)
- [ ] Implement policy gradient updates (currently computes loss only)
- [ ] Add PPO-style value function critic
- [ ] Support multi-GPU training
- [ ] Add more diverse coding prompts
- [ ] Integrate with HumanEval-Rust benchmark
- [ ] Add curriculum learning (start easy, increase difficulty)
- [ ] Implement reward shaping based on test case passing

## References

- **GRPO**: "Group Relative Policy Optimization" - stabilizes RL training via group comparisons
- **BitNet b1.58**: Ternary quantization for efficient inference
- **Compiler Feedback**: Using rustc for ground-truth evaluation
- **AST Analysis**: syn crate for deep Rust code understanding

## License

Same as parent project (nanochat-rs-ternary)
