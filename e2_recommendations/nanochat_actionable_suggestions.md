# nanochat-rs-ternary: Actionable Improvement Suggestions

## Quick Wins (Can implement today)

### 1. Add CI/CD Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ master, main ]
  pull_request:
    branches: [ master, main ]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-action@stable
      with:
        toolchain: stable

    - name: Cache cargo dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

    - name: Check formatting
      run: cargo fmt --all -- --check
      working-directory: ./nanochat-rs-ternary

    - name: Run clippy
      run: cargo clippy --workspace -- -D warnings
      working-directory: ./nanochat-rs-ternary

    - name: Run tests
      run: cargo test --workspace --verbose
      working-directory: ./nanochat-rs-ternary
      timeout-minutes: 30

    - name: Build release
      run: cargo build --release --workspace
      working-directory: ./nanochat-rs-ternary
```

---

### 2. Add rustfmt Configuration

Create `nanochat-rs-ternary/rustfmt.toml`:

```toml
edition = "2021"
max_width = 100
tab_spaces = 4
use_small_heuristics = "Default"
reorder_imports = true
reorder_modules = true
remove_nested_parens = true
merge_derives = true
use_try_shorthand = true
use_field_init_shorthand = true
format_code_in_doc_comments = true
wrap_comments = true
format_strings = true
```

---

### 3. Add Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/nanochat-rs-ternary"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "rust"
```

---

## Medium-term Improvements (1-2 weeks)

### 4. Add Structured Logging

Add to `crates/nanochat-train/Cargo.toml`:

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
```

Create `crates/nanochat-train/src/logging.rs`:

```rust
use tracing::{info, warn, error, debug, span, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_logging() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();
}

pub struct TrainingMetrics {
    pub loss: f64,
    pub ce_loss: f64,
    pub entropy: f64,
    pub learning_rate: f64,
    pub grad_norm: f64,
    pub tokens_per_sec: f64,
}

pub fn log_training_step(step: usize, metrics: &TrainingMetrics) {
    let span = span!(Level::INFO, "training_step", step = step);
    let _enter = span.enter();

    info!(
        loss = metrics.loss,
        ce_loss = metrics.ce_loss,
        entropy = metrics.entropy,
        lr = metrics.learning_rate,
        grad_norm = metrics.grad_norm,
        tokens_per_sec = metrics.tokens_per_sec,
        "Training step completed"
    );

    if metrics.entropy < 5.0 {
        warn!(
            entropy = metrics.entropy,
            "Low entropy detected - model may be collapsing"
        );
    }

    if metrics.grad_norm > 10.0 {
        warn!(
            grad_norm = metrics.grad_norm,
            "High gradient norm detected - potential instability"
        );
    }
}
```

---

### 5. Add Metrics Collection

Add to `crates/nanochat-serve/Cargo.toml`:

```toml
[dependencies]
prometheus = "0.13"
lazy_static = "1.4"
```

Create `crates/nanochat-serve/src/metrics.rs`:

```rust
use prometheus::{Counter, Gauge, Histogram, Registry};
use lazy_static::lazy_static;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    pub static ref INFERENCE_REQUESTS: Counter = Counter::new(
        "nanochat_inference_requests_total",
        "Total number of inference requests"
    ).unwrap();

    pub static ref INFERENCE_LATENCY: Histogram = Histogram::with_opts(
        prometheus::HistogramOpts::new(
            "nanochat_inference_latency_seconds",
            "Inference latency in seconds"
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
    ).unwrap();

    pub static ref ACTIVE_REQUESTS: Gauge = Gauge::new(
        "nanochat_active_requests",
        "Number of currently active requests"
    ).unwrap();

    pub static ref MODEL_LOAD_TIME: Gauge = Gauge::new(
        "nanochat_model_load_time_seconds",
        "Time to load model in seconds"
    ).unwrap();
}

pub fn register_metrics() {
    REGISTRY.register(Box::new(INFERENCE_REQUESTS.clone())).unwrap();
    REGISTRY.register(Box::new(INFERENCE_LATENCY.clone())).unwrap();
    REGISTRY.register(Box::new(ACTIVE_REQUESTS.clone())).unwrap();
    REGISTRY.register(Box::new(MODEL_LOAD_TIME.clone())).unwrap();
}
```

---

### 6. Add Configuration Management

Create `crates/nanochat-core/src/config.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub group_size: usize,
    pub mhc_n_streams: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dim: 768,
            n_layers: 12,
            n_heads: 12,
            vocab_size: 50257,
            max_seq_len: 512,
            group_size: 128,
            mhc_n_streams: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub total_steps: usize,
    pub warmup_steps: usize,
    pub learning_rate: f64,
    pub muon_lr: f64,
    pub lion_lr: f64,
    pub entropy_weight: f64,
    pub grad_clip: f64,
    pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 256,
            total_steps: 10000,
            warmup_steps: 1000,
            learning_rate: 0.001,
            muon_lr: 0.02,
            lion_lr: 1e-4,
            entropy_weight: 0.01,
            grad_clip: 5.0,
            checkpoint_interval: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub log_level: String,
    pub data_path: String,
    pub checkpoint_dir: String,
}

impl AppConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
```

---

### 7. Add Error Handling Improvements

Create `crates/nanochat-core/src/error.rs`:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NanoChatError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Training diverged: loss={loss}, step={step}")]
    TrainingDiverged { loss: f64, step: usize },

    #[error("Model collapse detected: entropy={entropy}")]
    ModelCollapse { entropy: f64 },

    #[error("CUDA out of memory: requested={requested}GB, available={available}GB")]
    CudaOutOfMemory { requested: f64, available: f64 },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, NanoChatError>;
```

---

## Long-term Improvements (1-2 months)

### 8. Add Python Bindings

Create `bindings/python/Cargo.toml`:

```toml
[package]
name = "nanochat-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "nanochat_py"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
nanochat-model = { path = "../../nanochat-rs-ternary/crates/nanochat-model" }
nanochat-serve = { path = "../../nanochat-rs-ternary/crates/nanochat-serve" }
```

Create `bindings/python/src/lib.rs`:

```rust
use pyo3::prelude::*;

#[pyclass]
struct PyModel {
    inner: nanochat_model::Model,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let model = nanochat_model::Model::load(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: model })
    }

    fn generate(&self, prompt: &str, max_tokens: usize) -> PyResult<String> {
        self.inner.generate(prompt, max_tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn generate_batch(&self, prompts: Vec<String>, max_tokens: usize) -> PyResult<Vec<String>> {
        prompts.iter()
            .map(|p| self.generate(p, max_tokens))
            .collect()
    }
}

#[pymodule]
fn nanochat_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    Ok(())
}
```

---

### 9. Add Model Zoo

Create `nanochat/model_zoo.py`:

```python
"""Model zoo for easy model downloading and caching."""

import os
from pathlib import Path
from typing import Optional
import hashlib
import requests
from tqdm import tqdm

MODELS = {
    "nano-125m-rust": {
        "url": "https://huggingface.co/architehc/nano-125m-rust/resolve/main/model.gguf",
        "config": "nano-125m",
        "sha256": "abc123...",  # Add actual hash
        "description": "125M parameter ternary model for Rust code generation",
        "size_mb": 45,
    },
    "nano-7b-rust": {
        "url": "https://huggingface.co/architehc/nano-7b-rust/resolve/main/model.gguf",
        "config": "7b",
        "sha256": "def456...",  # Add actual hash
        "description": "7B parameter ternary model for Rust code generation",
        "size_mb": 1800,
    },
}

class ModelZoo:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/nanochat")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self):
        """List all available models."""
        for name, info in MODELS.items():
            print(f"{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size_mb']} MB")
            print(f"  Config: {info['config']}")
            print()

    def download(self, name: str, force: bool = False) -> Path:
        """Download a model to cache."""
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")

        model_info = MODELS[name]
        model_path = self.cache_dir / f"{name}.gguf"

        if model_path.exists() and not force:
            print(f"Model {name} already cached at {model_path}")
            return model_path

        print(f"Downloading {name} ({model_info['size_mb']} MB)...")
        response = requests.get(model_info["url"], stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(model_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        # Verify checksum
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        if sha256.hexdigest() != model_info["sha256"]:
            model_path.unlink()
            raise ValueError("Downloaded model has incorrect checksum")

        print(f"Model downloaded to {model_path}")
        return model_path

    def load(self, name: str):
        """Download if needed and load model."""
        from nanochat_py import PyModel

        model_path = self.download(name)
        return PyModel(str(model_path))

def load_model(name: str):
    """Load a model from the zoo."""
    zoo = ModelZoo()
    return zoo.load(name)
```

---

### 10. Add Interactive CLI

Create `nanochat-cli/src/main.rs`:

```rust
use clap::{Parser, Subcommand};
use colored::Colorize;
use rustyline::DefaultEditor;

#[derive(Parser)]
#[command(name = "nanochat")]
#[command(about = "Ternary quantized code generation model")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Chat {
        #[arg(short, long)]
        model: String,
        #[arg(short, long, default_value = "200")]
        max_tokens: usize,
    },
    Generate {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value = "200")]
        max_tokens: usize,
    },
    Benchmark {
        #[arg(short, long)]
        model: String,
        #[arg(short, long, default_value = "100")]
        n_samples: usize,
    },
    Download {
        name: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { model, max_tokens } => {
            interactive_chat(&model, max_tokens)?;
        }
        Commands::Generate { model, prompt, max_tokens } => {
            generate_code(&model, &prompt, max_tokens)?;
        }
        Commands::Benchmark { model, n_samples } => {
            run_benchmark(&model, n_samples)?;
        }
        Commands::Download { name } => {
            download_model(&name)?;
        }
    }

    Ok(())
}

fn interactive_chat(model_path: &str, max_tokens: usize) -> rustyline::Result<()> {
    println!("{}", "Loading model...".yellow());
    let model = load_model(model_path)?;
    println!("{}", "Model loaded! Type 'exit' to quit.\n".green());

    let mut rl = DefaultEditor::new()?;

    loop {
        let prompt = rl.readline(">> ")?;

        if prompt.trim() == "exit" {
            break;
        }

        match model.generate(&prompt, max_tokens) {
            Ok(output) => {
                println!("{}", "--- Generated Code ---".cyan());
                println!("{}", output);
                println!("{}", "----------------------\n".cyan());
            }
            Err(e) => {
                eprintln!("{} {}", "Error:".red(), e);
            }
        }
    }

    Ok(())
}
```

---

## Testing Improvements

### 11. Add Property-Based Tests

Add to `crates/mhc-lite/tests/property_tests.rs`:

```rust
use proptest::prelude::*;
use mhc_lite::{MhcLiteN2, MhcLiteN4};

proptest! {
    #[test]
    fn n2_mixing_matrix_is_doubly_stochastic(alpha in 0.0f32..1.0f32) {
        let mhc = MhcLiteN2::new(alpha);
        let matrix = mhc.mixing_matrix();

        for i in 0..2 {
            let row_sum: f32 = (0..2).map(|j| matrix[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }

        for j in 0..2 {
            let col_sum: f32 = (0..2).map(|i| matrix[[i, j]]).sum();
            assert!((col_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn composite_gain_is_bounded(alpha in 0.0f32..1.0f32) {
        let mhc = MhcLiteN2::new(alpha);
        let gain = mhc.composite_gain();
        assert!(gain <= 1.0 + 1e-6);
        assert!(gain >= 0.0);
    }
}
```

---

### 12. Add Fuzz Testing

Create `fuzz/Cargo.toml`:

```toml
[package]
name = "nanochat-fuzz"
version = "0.0.0"
publish = false
edition = "2021"

[package.metadata]
cargo-fuzz = true

[dependencies]
libfuzzer-sys = "0.4"
nanochat-core = { path = "../nanochat-rs-ternary/crates/ternary-core" }

[[bin]]
name = "fuzz_ternary_packing"
path = "fuzz_targets/fuzz_ternary_packing.rs"
```

---

## Performance Optimizations

### 13. Add NUMA-Aware Allocation

Create `crates/ternary-core/src/numa.rs`:

```rust
pub struct NumaAllocator;

impl NumaAllocator {
    #[cfg(target_os = "linux")]
    pub fn alloc_on_node(size: usize, node: i32) -> *mut u8 {
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return std::ptr::null_mut();
            }

            ptr as *mut u8
        }
    }

    #[cfg(target_os = "linux")]
    pub fn current_node() -> i32 {
        unsafe {
            let mut node = 0;
            let mut cpu = 0;
            libc::getcpu(&mut cpu, &mut node);
            node
        }
    }
}
```

---

### 14. Add Kernel Auto-Tuning

Create `crates/ternary-kernels/src/autotune.rs`:

```rust
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelChoice {
    Avx512,
    Avx2,
    Scalar,
}

pub struct KernelAutotuner {
    cache: HashMap<Shape, KernelChoice>,
}

impl KernelAutotuner {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn select_kernel(&mut self, shape: Shape) -> KernelChoice {
        if let Some(&choice) = self.cache.get(&shape) {
            return choice;
        }

        // Benchmark and select best kernel
        let choice = self.benchmark_kernels(shape);
        self.cache.insert(shape, choice);
        choice
    }

    fn benchmark_kernels(&self, shape: Shape) -> KernelChoice {
        // Implementation here
        KernelChoice::Scalar
    }
}
```

---

## Summary

| Priority | Suggestion | Impact | Effort |
|----------|------------|--------|--------|
| P0 | CI/CD Pipeline | High | Low |
| P0 | rustfmt Config | Medium | Low |
| P1 | Structured Logging | High | Low |
| P1 | Metrics Collection | High | Medium |
| P1 | Configuration Mgmt | Medium | Low |
| P2 | Python Bindings | High | Medium |
| P2 | Model Zoo | High | Low |
| P2 | Property Tests | Medium | Low |
| P3 | NUMA Allocation | High | Medium |
| P3 | Kernel Auto-tune | High | High |

---

*Generated: February 14, 2026*
*For: nanochat-rs-ternary repository*
