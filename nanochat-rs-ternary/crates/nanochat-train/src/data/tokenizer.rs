//! Tokenizer wrapper and BPE training via HuggingFace tokenizers crate.

use std::path::Path;

/// BPE tokenizer wrapper.
pub struct NanochatTokenizer {
    inner: tokenizers::Tokenizer,
    pub vocab_size: usize,
}

impl NanochatTokenizer {
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
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| format!("Encoding error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.inner
            .decode(ids, true)
            .map_err(|e| format!("Decoding error: {}", e).into())
    }
}

/// Train a byte-level BPE tokenizer on a text file and save tokenizer.json + tokens.bin.
///
/// Returns (vocab_size, total_tokens).
pub fn prepare_data(
    text_path: &Path,
    vocab_size: usize,
    output_dir: &Path,
) -> Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    use tokenizers::models::bpe::{BpeTrainer, BPE};
    use tokenizers::AddedToken;

    println!(
        "Training BPE tokenizer on {:?} (target vocab_size={})",
        text_path, vocab_size
    );

    // Create byte-level BPE tokenizer
    let mut tokenizer = tokenizers::Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(
            false, // add_prefix_space
            true,  // trim_offsets
            true,  // use_regex
        ),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));

    // Configure trainer
    let special_tokens = vec![
        AddedToken::from("<pad>", true),
        AddedToken::from("<eos>", true),
    ];
    let mut trainer: tokenizers::models::TrainerWrapper = BpeTrainer::builder()
        .vocab_size(vocab_size)
        .special_tokens(special_tokens)
        .min_frequency(2)
        .show_progress(true)
        .build()
        .into();

    // Train on the text file (read lines as iterator)
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(text_path)?;
    let reader = BufReader::new(file);
    let lines = reader.lines().map(|l| l.unwrap_or_default());
    tokenizer
        .train(&mut trainer, lines)
        .map_err(|e| format!("Training failed: {}", e))?;

    // Save tokenizer.json
    std::fs::create_dir_all(output_dir)?;
    let tok_path = output_dir.join("tokenizer.json");
    tokenizer
        .save(&tok_path, false)
        .map_err(|e| format!("Failed to save tokenizer: {}", e))?;

    // Encode all text to token IDs
    let text = std::fs::read_to_string(text_path)?;
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| format!("Encoding failed: {}", e))?;
    let ids = encoding.get_ids();

    // Save as tokens.bin (flat u32 LE)
    let bin_path = output_dir.join("tokens.bin");
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    std::fs::write(&bin_path, &bytes)?;

    let final_vocab = tokenizer.get_vocab_size(true);
    let total_tokens = ids.len();

    println!("Tokenizer trained successfully:");
    println!("  Vocab size: {}", final_vocab);
    println!("  Total tokens: {}", total_tokens);
    println!(
        "  Compression: {:.1}x (bytes/token)",
        text.len() as f64 / total_tokens as f64
    );
    println!("  Saved: {:?}", tok_path);
    println!("  Saved: {:?}", bin_path);

    Ok((final_vocab, total_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_tokenizer_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{{}}").unwrap();
        let result = NanochatTokenizer::from_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_from_bytes() {
        let result = NanochatTokenizer::from_bytes(b"not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_data_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let text_path = dir.path().join("input.txt");
        let out_dir = dir.path().join("output");

        // Write sample text
        let sample = "Hello world! This is a test of the tokenizer training pipeline.\n\
                       The quick brown fox jumps over the lazy dog.\n\
                       Pack my box with five dozen liquor jugs.\n"
            .repeat(100); // Repeat to give BPE enough data for merges
        std::fs::write(&text_path, &sample).unwrap();

        // Train tokenizer
        let (vocab_size, total_tokens) = prepare_data(&text_path, 300, &out_dir).unwrap();
        assert!(vocab_size > 0 && vocab_size <= 300);
        assert!(total_tokens > 0);

        // Verify tokenizer.json loads
        let tok = NanochatTokenizer::from_file(&out_dir.join("tokenizer.json")).unwrap();
        assert_eq!(tok.vocab_size, vocab_size);

        // Verify tokens.bin matches encoding
        let bin = std::fs::read(out_dir.join("tokens.bin")).unwrap();
        let loaded_tokens: Vec<u32> = bin
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(loaded_tokens.len(), total_tokens);

        // Verify decode roundtrip on a short string
        let ids = tok.encode("Hello world").unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, "Hello world");
    }
}
