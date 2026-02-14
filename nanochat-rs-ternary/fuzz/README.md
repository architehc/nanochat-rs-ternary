# Fuzz Testing for nanochat-rs-ternary

Fuzzing infrastructure using `cargo-fuzz` and `libfuzzer`.

## Setup

Install cargo-fuzz:

```bash
cargo install cargo-fuzz
```

## Running Fuzz Tests

### Fuzz ternary packing/unpacking

```bash
cargo fuzz run fuzz_ternary_packing
```

### Fuzz GGUF parsing

```bash
cargo fuzz run fuzz_gguf_parsing
```

### Run with corpus

```bash
# Build corpus
mkdir -p fuzz/corpus/fuzz_ternary_packing
# Add seed inputs if desired

# Run with corpus
cargo fuzz run fuzz_ternary_packing -- -max_total_time=60
```

## What We're Testing

### fuzz_ternary_packing
- Memory safety: no panics, no out-of-bounds
- Roundtrip correctness: pack → unpack preserves values
- Edge cases: NaN, infinity, all zeros, all max values
- Ternary constraint: values in {-1, 0, +1} after quantization

### fuzz_gguf_parsing
- No panics on malformed input
- No out-of-bounds reads
- Graceful error handling for invalid files
- Robustness to crafted/corrupted files

## Interpreting Results

- **No crashes**: Code is memory-safe
- **Assertions pass**: Invariants hold
- **Crash found**: Bug detected! See crash-* files for reproducer

## Continuous Fuzzing

For production, integrate with:
- OSS-Fuzz (Google's continuous fuzzing service)
- Local CI with `cargo fuzz run -- -max_total_time=300`

## Corpus Management

Successful inputs are stored in `fuzz/corpus/*/`.

To minimize corpus:
```bash
cargo fuzz cmin fuzz_ternary_packing
```

To merge corpora:
```bash
cargo fuzz cmin --merge fuzz_ternary_packing
```
