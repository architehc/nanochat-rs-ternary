# nanochat-py

Python bindings for nanochat ternary quantized models.

## Installation

```bash
# Development install
pip install maturin
maturin develop

# Production build
maturin build --release
pip install target/wheels/*.whl
```

## Usage

```python
from nanochat_py import PyModel

# Load model
model = PyModel.load("model.gguf", "model.mhc")

# Generate from token IDs
prompt = [1, 2, 3, 4, 5]
output = model.generate(prompt, max_tokens=100)

# Batch generation
prompts = [[1, 2], [3, 4, 5]]
outputs = model.generate_batch(prompts, max_tokens=50)

# Get model config
config = model.get_config()
print(config)

# Verify mHC matrices
valid = model.verify_mhc()
print(f"mHC matrices valid: {valid}")
```

## Features

- **Fast inference**: Ternary quantization for 1.58-bit weights
- **mHC residuals**: Doubly stochastic residual connections
- **Batch processing**: Generate multiple sequences in parallel
- **Low memory**: Efficient packed representation

## Requirements

- Python >= 3.8
- maturin (for building)

## License

Same as nanochat-rs-ternary
