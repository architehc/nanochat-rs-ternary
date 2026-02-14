# Training-Free GRPO Guide

## Overview

Training-Free GRPO (Group Relative Policy Optimization) enables reinforcement learning **without backpropagation**. It learns from an experience library of successful rollouts, using semantic advantages to guide generation without gradient computation.

**Key innovation**: Zero-cost continual learning from sparse reward signals

**Based on**: "Training-Free Group Relative Policy Optimization" (2024)

## Core Concept

Traditional RL:
```
Generate → Compute Reward → Backprop Gradients → Update Weights
```

Training-Free GRPO:
```
Generate Group → Compute Advantages → Store Good Experiences → Sample from Library
```

**No backprop = No gradient computation = No memory overhead**

## Quick Start

```rust
use nanochat_rl::{TrainingFreeGRPO, GRPOConfig, Experience};

// Create GRPO experience library
let mut grpo = TrainingFreeGRPO::new();

// Generate group of rollouts for a prompt
let prompts = vec!["Write a function to".to_string(); 8];
let responses = /* generate from model */;
let rewards = /* evaluate each response */;

// Extract successful experiences (above mean reward)
grpo.add_rollout_group(
    prompts.into_iter().zip(responses).collect(),
    rewards,
);

// Sample from experience library for future generation
if let Some(exp) = grpo.sample_experience() {
    println!("Successful pattern: {} -> {}", exp.prompt, exp.response);
}

// Save library for continual learning
grpo.save_to_file("experiences.json")?;
```

## Architecture

### Experience

Each experience stores a successful rollout:

```rust
pub struct Experience {
    pub prompt: String,        // Input prompt
    pub response: String,      // Generated response
    pub advantage: f64,        // reward - group_mean
    pub timestamp: u64,        // For aging/pruning
    pub metadata: HashMap,     // Optional (task ID, domain, etc.)
}
```

### Algorithm

For each prompt:

1. **Generate Group**: Create N rollouts (typically 8)
2. **Evaluate**: Compute reward for each rollout
3. **Compute Advantages**: `advantage = reward - mean(group_rewards)`
4. **Store Successful**: Keep experiences with `advantage > threshold`
5. **Prune**: Remove old/low-value experiences if library full

### Advantage-Weighted Sampling

When sampling from library:

```
weight(exp) = exp(advantage / temperature)
P(exp) = weight(exp) / sum(all weights)
```

**Higher advantage → Higher probability of being sampled**

## Configuration

### Default Configuration

```rust
let config = GRPOConfig::default();
// group_size: 8                   (8 rollouts per prompt)
// max_library_size: 10000         (10K experiences max)
// max_experience_age: 604800      (1 week)
// min_advantage_threshold: 0.0    (only positive advantage)
// sampling_temperature: 1.0       (neutral)
// use_advantage_weighting: true   (weight by advantage)
```

### Custom Configuration

```rust
let config = GRPOConfig {
    group_size: 16,                  // Larger groups = better advantage estimates
    max_library_size: 50000,         // More capacity
    max_experience_age: 86400 * 30,  // Keep for 30 days
    min_advantage_threshold: 0.1,    // Only store clearly good examples
    sampling_temperature: 0.5,       // Sharper distribution (favor best)
    use_advantage_weighting: true,
};

let grpo = TrainingFreeGRPO::with_config(config);
```

### Parameter Guide

#### `group_size`
- **Range**: [4, 32]
- **Default**: 8
- **Larger = better advantage estimates** but more generation cost

| Size | Advantage Precision | Cost | Use Case |
|------|---------------------|------|----------|
| 4 | Low | Fastest | Quick experiments |
| 8 | Medium | Balanced | Recommended default |
| 16 | High | Expensive | High-quality learning |
| 32 | Very High | Very expensive | Critical domains |

#### `max_library_size`
- **Range**: [1000, 1000000]
- **Default**: 10000
- **Larger = more diverse experiences** but more memory

**Memory usage**: ~1KB per experience
- 10K experiences ≈ 10 MB
- 100K experiences ≈ 100 MB
- 1M experiences ≈ 1 GB

#### `max_experience_age`
- **Range**: [3600, 2592000] (1 hour to 30 days)
- **Default**: 604800 (7 days)
- **Prevents stale experiences** from outdated model versions

#### `min_advantage_threshold`
- **Range**: [-1.0, 1.0]
- **Default**: 0.0
- **Only store experiences above this advantage**

| Threshold | Selectivity | Library Growth |
|-----------|-------------|----------------|
| -0.5 | Very loose | Fast (most rollouts) |
| 0.0 | Balanced | Medium (above mean) |
| 0.5 | Strict | Slow (only top) |
| 1.0 | Very strict | Very slow (exceptional) |

#### `sampling_temperature`
- **Range**: [0.1, 2.0]
- **Default**: 1.0
- **Controls sampling distribution sharpness**

| Temperature | Distribution | Effect |
|-------------|--------------|--------|
| 0.1 | Very sharp | Always pick best |
| 0.5 | Sharp | Favor high advantage |
| 1.0 | Neutral | Proportional to advantage |
| 2.0 | Flat | More uniform sampling |

## Use Cases

### 1. Code Generation with Compiler Feedback

```rust
use nanochat_rl::{TrainingFreeGRPO, CompilerFeedback};

let mut grpo = TrainingFreeGRPO::new();

// Training loop
for prompt in coding_prompts {
    // Generate group of code samples
    let samples: Vec<String> = (0..8)
        .map(|_| model.generate(&prompt))
        .collect();

    // Evaluate with compiler
    let rewards: Vec<f64> = samples.iter()
        .map(|code| {
            let result = CompilerFeedback::compile(code);
            if result.success { 1.0 } else { 0.0 }
        })
        .collect();

    // Store successful compilations
    let rollouts: Vec<_> = samples.into_iter()
        .map(|s| (prompt.clone(), s))
        .collect();

    grpo.add_rollout_group(rollouts, rewards);

    println!("Library size: {}", grpo.library_size());
}

// Later: sample successful patterns for few-shot prompts
if let Some(exp) = grpo.sample_experience() {
    let few_shot_prompt = format!(
        "Example:\n{}\n\nNow write: {}",
        exp.response, new_prompt
    );
}
```

### 2. Continual Learning Across Tasks

```rust
let mut grpo = TrainingFreeGRPO::new();

// Task 1: String manipulation
for prompt in string_tasks {
    // ... generate and evaluate ...
    grpo.add_rollout_group(rollouts, rewards);
}

// Task 2: File I/O (library retains string patterns)
for prompt in file_io_tasks {
    // ... generate and evaluate ...
    grpo.add_rollout_group(rollouts, rewards);
}

// Task 3: Can leverage patterns from both previous tasks
let task3_examples = grpo.get_experiences_for_prompt("Read file");
```

### 3. Multi-Domain Training

```rust
let mut grpo = TrainingFreeGRPO::new();

for (domain, prompts) in domains {
    for prompt in prompts {
        let rollouts = generate_group(&model, &prompt, 8);
        let rewards = evaluate_domain_specific(&domain, &rollouts);

        let mut rollout_pairs: Vec<_> = rollouts.into_iter()
            .map(|r| (prompt.clone(), r))
            .collect();

        grpo.add_rollout_group(rollout_pairs, rewards);

        // Tag experiences with domain metadata
        if let Some(last_exp) = grpo.experience_library.last_mut() {
            last_exp.metadata.insert("domain".to_string(), domain.clone());
        }
    }
}

// Sample domain-specific experiences
let web_exps: Vec<_> = grpo.experience_library.iter()
    .filter(|e| e.metadata.get("domain") == Some(&"web".to_string()))
    .collect();
```

### 4. Active Learning / Uncertainty Sampling

```rust
// Generate diverse rollouts
let rollouts = generate_with_high_temperature(&model, prompt, 16);
let rewards = evaluate(rollouts);

grpo.add_rollout_group(
    rollouts.into_iter().map(|r| (prompt.clone(), r)).collect(),
    rewards,
);

// Check if we need more data for this prompt
let prompt_exps = grpo.get_experiences_for_prompt(&prompt);
if prompt_exps.len() < 5 {
    println!("Need more examples for prompt: {}", prompt);
    // Generate more rollouts...
}
```

## Integration with Model

### During Generation

Use experiences to guide generation via:

1. **Few-shot prompting**: Prepend successful examples
2. **Logit biasing**: Boost tokens from successful rollouts
3. **Constrained decoding**: Only generate patterns seen in library

**Example: Few-shot prompting**

```rust
fn generate_with_experience_library(
    model: &Model,
    prompt: &str,
    grpo: &TrainingFreeGRPO,
    n_examples: usize,
) -> String {
    // Sample successful experiences
    let examples: Vec<_> = (0..n_examples)
        .filter_map(|_| grpo.sample_experience())
        .collect();

    // Build few-shot prompt
    let mut full_prompt = String::new();
    for exp in examples {
        full_prompt.push_str(&format!("Q: {}\nA: {}\n\n", exp.prompt, exp.response));
    }
    full_prompt.push_str(&format!("Q: {}\nA:", prompt));

    // Generate with examples as context
    model.generate(&full_prompt)
}
```

## Monitoring

### Statistics

```rust
let stats = grpo.stats();
println!("{}", stats.summary());

// Output:
// GRPO Stats:
// Total rollouts: 8000
// Stored: 3200 (40.0% of rollouts)
// Pruned: 500
// Current library: 2700 experiences
// Advantage: avg=0.234, max=2.145, min=0.001
// Storage efficiency: 40.0%
// Retention rate: 84.4%
```

### Metrics to Track

**Storage Efficiency**: `stored / total_rollouts`
- **Low (<20%)**: Very selective (may miss good examples)
- **Medium (30-50%)**: Balanced (recommended)
- **High (>70%)**: Too permissive (storing mediocre examples)

**Retention Rate**: `current_size / total_stored`
- **Low (<50%)**: Aggressive pruning (losing history)
- **Medium (60-80%)**: Healthy turnover
- **High (>90%)**: Library not at capacity OR old experiences still valuable

**Average Advantage**:
- **Low (<0.1)**: Experiences barely above mean (weak signal)
- **Medium (0.2-0.5)**: Good separation (healthy)
- **High (>1.0)**: Exceptional experiences (very selective)

## Performance

### Memory Usage

```
Per experience: ~1 KB (prompt + response + metadata)
10K library: ~10 MB
100K library: ~100 MB
```

**Negligible compared to model weights** (125M params = 500 MB)

### Computational Cost

**Generation**: O(group_size × model_inference)
- Same as standard RL, no backprop overhead

**Library operations**:
- `add_rollout_group`: O(group_size) amortized
- `sample_experience`: O(library_size) worst case, O(1) amortized
- `prune_library`: O(library_size × log(library_size)) when triggered

**Compared to gradient-based RL**:
- **No gradient computation**: Save ~2× forward pass time
- **No optimizer step**: Save memory for Adam states
- **No weight updates**: Stateless, easier to distribute

## Limitations

1. **No direct policy updates**: Library guides generation but doesn't update model weights
2. **Memory grows with diversity**: Need to prune eventually
3. **Requires good reward function**: Garbage in, garbage out
4. **Cold start**: Empty library initially, needs bootstrapping

## Combining with Gradient RL

Training-Free GRPO complements gradient-based methods:

```rust
// Phase 1: Bootstrap with gradient RL
for epoch in 0..10 {
    train_with_grpo_gradients(&model, &optimizer);
}

// Phase 2: Continual learning with Training-Free GRPO
let mut grpo = TrainingFreeGRPO::new();
for new_task in tasks {
    // Generate experiences (no backprop)
    generate_and_store(&model, &new_task, &mut grpo);

    // Use library for few-shot guidance
    let examples = grpo.sample_multiple(5);
    fine_tune_with_examples(&model, examples);
}
```

## Best Practices

✅ **Do**:
- Use larger group_size (16-32) for critical domains
- Prune regularly to keep library fresh
- Tag experiences with metadata (task, domain, timestamp)
- Monitor storage efficiency (aim for 30-50%)
- Save library periodically for continual learning

❌ **Don't**:
- Use tiny group_size (<4) - advantage estimates too noisy
- Set min_advantage_threshold negative - stores below-mean examples
- Keep experiences forever - old patterns may be outdated
- Ignore sampling temperature - affects diversity

## API Reference

### TrainingFreeGRPO

```rust
impl TrainingFreeGRPO {
    pub fn new() -> Self;
    pub fn with_config(config: GRPOConfig) -> Self;

    pub fn add_rollout_group(
        &mut self,
        rollouts: Vec<(String, String)>,  // (prompt, response)
        rewards: Vec<f64>,
    );

    pub fn sample_experience(&self) -> Option<&Experience>;
    pub fn get_experiences_for_prompt(&self, prompt: &str) -> Vec<&Experience>;

    pub fn stats(&self) -> GRPOStats;
    pub fn library_size(&self) -> usize;
    pub fn is_at_capacity(&self) -> bool;

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()>;
    pub fn load_from_file(&mut self, path: &str) -> std::io::Result<()>;

    pub fn clear(&mut self);
}
```

### GRPOConfig

```rust
pub struct GRPOConfig {
    pub group_size: usize,                // Rollouts per prompt
    pub max_library_size: usize,          // Max experiences
    pub max_experience_age: u64,          // Max age (seconds)
    pub min_advantage_threshold: f64,     // Min advantage to store
    pub sampling_temperature: f64,        // Sampling sharpness
    pub use_advantage_weighting: bool,    // Weight by advantage
}
```

### Experience

```rust
pub struct Experience {
    pub prompt: String,
    pub response: String,
    pub advantage: f64,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

impl Experience {
    pub fn age_seconds(&self) -> u64;
    pub fn is_stale(&self, max_age: u64) -> bool;
}
```

### GRPOStats

```rust
pub struct GRPOStats {
    pub total_rollouts: usize,
    pub total_stored: usize,
    pub total_pruned: usize,
    pub current_library_size: usize,
    pub avg_advantage: f64,
    pub max_advantage: f64,
    pub min_advantage: f64,
}

impl GRPOStats {
    pub fn storage_efficiency(&self) -> f64;
    pub fn retention_rate(&self) -> f64;
    pub fn summary(&self) -> String;
}
```

## Testing

Run Training-Free GRPO tests:

```bash
cargo test -p nanochat-rl training_free_grpo
```

Expected: 7 tests passing

## References

- **Paper**: "Training-Free Group Relative Policy Optimization" (2024)
- **Implementation**: `crates/nanochat-rl/src/training_free_grpo.rs`
- **Related**: GRPO, experience replay, continual learning

## Summary

Training-Free GRPO enables RL without backpropagation:

✅ **Zero gradient computation**: No backprop overhead
✅ **Continual learning**: Accumulate knowledge across tasks
✅ **Memory efficient**: ~1KB per experience
✅ **Stateless**: No optimizer state to manage
✅ **Composable**: Works with any reward function
✅ **Persistent**: Save/load experience library

**Use when**: Continual learning, sparse rewards, no gradient access, multi-domain training.

**Expected benefit**: Enables RL fine-tuning without training infrastructure overhead.
