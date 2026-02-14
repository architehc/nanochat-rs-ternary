//! Fuzz target for GGUF file parsing.
//!
//! Tests for:
//! - No panics on malformed input
//! - No out-of-bounds reads
//! - Graceful error handling

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tempfile::NamedTempFile;
use ternary_core::gguf::GgufFile;

fuzz_target!(|data: &[u8]| {
    // Write fuzzer input to a temporary file
    let mut temp_file = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return, // Skip if can't create temp file
    };

    if temp_file.write_all(data).is_err() {
        return; // Skip if can't write
    }

    let path = temp_file.path();

    // Try to parse the GGUF file (should not panic)
    let _result = GgufFile::open(path);

    // We don't care about the result (Ok or Err), just that it doesn't panic
    // Valid GGUF files will parse successfully
    // Invalid files will return Err, which is expected behavior
});
