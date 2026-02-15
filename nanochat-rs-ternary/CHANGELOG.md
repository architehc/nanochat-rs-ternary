# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-02-15

### Fixed
- Implemented Collider gradient masking in `crates/nanochat-train/src/train.rs`.
  - Added token-level gradient gating via a detach-based logits mask.
  - Removed the previous Collider TODO stub in the training step.
- Removed panic-prone runtime `unwrap()` usage in `crates/nanochat-serve/src/server.rs`.
  - Added safe UNIX timestamp helper.
  - Replaced engine mutex lock unwraps with explicit error handling.
  - Replaced SSE JSON serialization unwraps with fallible handling and error accounting.

### Notes
- Test-only `unwrap()` calls remain under `#[cfg(test)]` in server tests.
