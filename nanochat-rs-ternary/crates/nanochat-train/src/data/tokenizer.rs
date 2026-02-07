//! GPT-2 tokenizer wrapper using HuggingFace tokenizers crate.

use std::path::Path;

/// GPT-2 BPE tokenizer (50257 vocab).
pub struct Gpt2Tokenizer {
    inner: tokenizers::Tokenizer,
    pub vocab_size: usize,
}

impl Gpt2Tokenizer {
    /// Load from a local tokenizer.json file.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer from {:?}: {}", path, e))?;
        let vocab_size = inner.get_vocab_size(true);
        Ok(Self { inner, vocab_size })
    }

    /// Load from raw bytes (e.g. embedded tokenizer.json).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| format!("Failed to load tokenizer from bytes: {}", e))?;
        let vocab_size = inner.get_vocab_size(true);
        Ok(Self { inner, vocab_size })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| format!("Encoding error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.inner.decode(ids, true)
            .map_err(|e| format!("Decoding error: {}", e).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_tokenizer_from_file() {
        // Create a minimal tokenizer JSON for testing
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        // We can't easily create a full GPT-2 tokenizer in test,
        // so just verify the API compiles and from_file returns an error for invalid JSON
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{{}}").unwrap();
        let result = Gpt2Tokenizer::from_file(&path);
        // Empty JSON won't be a valid tokenizer, but the API should work
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_from_bytes() {
        // Invalid bytes should return error, not panic
        let result = Gpt2Tokenizer::from_bytes(b"not json");
        assert!(result.is_err());
    }
}
