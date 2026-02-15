# Getting Started

1. Build:
   `cargo build --release`
2. Validate:
   `./scripts/validate_e2e.sh`
3. Train (quick):
   `./scripts/train_nano_125m.sh`
4. Serve:
   `cargo run --release -p nanochat-serve -- --model <gguf> --mhc <mhc> --tokenizer models/gpt2-tokenizer.json`
