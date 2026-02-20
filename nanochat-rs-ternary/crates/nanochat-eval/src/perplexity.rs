//! Perplexity evaluation using the inference model.
//!
//! Loads a GGUF + mHC model and computes perplexity on held-out token data.
//! Runs on CPU using the inference model (not the training model).

use std::io;
use std::path::Path;

/// Result of a perplexity evaluation run.
#[derive(Debug, Clone)]
pub struct PerplexityResult {
    /// Total number of tokens evaluated.
    pub n_tokens: usize,
    /// Average cross-entropy loss (nats).
    pub avg_loss: f64,
    /// Perplexity = exp(avg_loss).
    pub perplexity: f64,
    /// Bits per byte = avg_loss / ln(2) * (tokens / bytes).
    /// Approximated as avg_loss / ln(2) (assumes ~1 token per ~4 bytes for BPE).
    pub bits_per_byte: f64,
    /// Number of chunks processed.
    pub n_chunks: usize,
}

/// Evaluate perplexity of a model on a token file.
///
/// # Arguments
/// * `gguf_path` - Path to GGUF model file
/// * `mhc_path` - Path to mHC binary parameters file
/// * `tokens_path` - Path to binary token file (u32 little-endian)
/// * `seq_len` - Context window size for each chunk
/// * `max_tokens` - Maximum number of tokens to evaluate (0 = all)
/// * `stride` - Stride between chunks (0 = use seq_len, i.e. no overlap)
pub fn evaluate_perplexity(
    gguf_path: &str,
    mhc_path: &str,
    tokens_path: &str,
    seq_len: usize,
    max_tokens: usize,
    stride: usize,
) -> io::Result<PerplexityResult> {
    // Load model
    eprintln!("Loading model from {} + {}...", gguf_path, mhc_path);
    let mut model = nanochat_model::model::NanochatModel::from_gguf(gguf_path, mhc_path)?;
    let vocab_size = model.config.vocab_size;
    eprintln!(
        "Model loaded: dim={}, layers={}, vocab={}",
        model.config.dim, model.config.n_layers, vocab_size
    );

    // Load tokens
    let tokens = load_tokens(tokens_path)?;
    let n_total = if max_tokens > 0 {
        max_tokens.min(tokens.len())
    } else {
        tokens.len()
    };
    eprintln!("Evaluating on {} tokens ({} total in file)", n_total, tokens.len());

    let effective_stride = if stride > 0 { stride } else { seq_len };

    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;
    let mut n_chunks = 0usize;

    // Process in chunks
    let mut offset = 0;
    while offset + seq_len < n_total {
        let chunk_end = (offset + seq_len).min(n_total);
        let chunk = &tokens[offset..chunk_end];

        // Reset caches for each independent chunk
        model.reset_caches();

        // Feed tokens one at a time, computing CE loss for each prediction
        for i in 0..chunk.len() - 1 {
            let logits = model.forward_token(chunk[i], i);

            // Compute cross-entropy: -log(softmax(logits)[target])
            let target = chunk[i + 1] as usize;
            let ce = cross_entropy_from_logits(&logits, target, vocab_size);
            total_loss += ce as f64;
            total_tokens += 1;
        }

        n_chunks += 1;
        if n_chunks.is_multiple_of(10) {
            let running_avg = total_loss / total_tokens as f64;
            let running_ppl = running_avg.exp();
            eprintln!(
                "  chunk {}: tokens={}, avg_loss={:.4}, ppl={:.1}",
                n_chunks, total_tokens, running_avg, running_ppl
            );
        }

        offset += effective_stride;
    }

    if total_tokens == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No tokens to evaluate (seq_len too large or empty token file)",
        ));
    }

    let avg_loss = total_loss / total_tokens as f64;
    let perplexity = avg_loss.exp();
    // Approximate bits per byte: CE in nats / ln(2), scaled by avg tokens/bytes ratio
    // For GPT-2 BPE, ~1 token ≈ ~3.7 bytes on average
    let bits_per_byte = avg_loss / 2.0f64.ln() / 3.7;

    let result = PerplexityResult {
        n_tokens: total_tokens,
        avg_loss,
        perplexity,
        bits_per_byte,
        n_chunks,
    };

    eprintln!("\n=== Perplexity Results ===");
    eprintln!("  Tokens evaluated: {}", result.n_tokens);
    eprintln!("  Chunks: {}", result.n_chunks);
    eprintln!("  Avg CE loss: {:.4}", result.avg_loss);
    eprintln!("  Perplexity: {:.2}", result.perplexity);
    eprintln!("  Bits/byte: {:.4}", result.bits_per_byte);

    Ok(result)
}

/// Load binary token file (u32 little-endian).
fn load_tokens(path: &str) -> io::Result<Vec<u32>> {
    let data = std::fs::read(path)?;
    if data.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Token file size {} is not a multiple of 4 bytes",
                data.len()
            ),
        ));
    }
    let tokens: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(tokens)
}

/// Compute cross-entropy loss from raw logits for a single target.
///
/// Uses log-sum-exp trick for numerical stability.
fn cross_entropy_from_logits(logits: &[f32], target: usize, vocab_size: usize) -> f32 {
    debug_assert!(target < vocab_size);
    debug_assert!(logits.len() >= vocab_size);

    // Find max for numerical stability
    let max_logit = logits[..vocab_size]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // log_sum_exp = max + log(sum(exp(logit - max)))
    let sum_exp: f64 = logits[..vocab_size]
        .iter()
        .map(|&l| ((l - max_logit) as f64).exp())
        .sum();
    let log_sum_exp = max_logit as f64 + sum_exp.ln();

    // CE = -log_softmax(logits)[target] = -(logits[target] - log_sum_exp)
    let ce = -(logits[target] as f64 - log_sum_exp);
    ce as f32
}

/// Write perplexity results to a CSV file (appending).
pub fn append_eval_csv(
    csv_path: &Path,
    step: usize,
    result: &PerplexityResult,
) -> io::Result<()> {
    use std::io::Write;

    let needs_header = !csv_path.exists();

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)?;

    if needs_header {
        writeln!(
            file,
            "step,avg_loss,perplexity,bits_per_byte,n_tokens,n_chunks,timestamp"
        )?;
    }

    let timestamp = chrono::Utc::now().to_rfc3339();
    writeln!(
        file,
        "{},{:.6},{:.2},{:.4},{},{},{}",
        step,
        result.avg_loss,
        result.perplexity,
        result.bits_per_byte,
        result.n_tokens,
        result.n_chunks,
        timestamp,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_uniform() {
        // Uniform logits → CE = ln(vocab_size)
        let vocab = 100;
        let logits = vec![0.0f32; vocab];
        let ce = cross_entropy_from_logits(&logits, 0, vocab);
        let expected = (vocab as f32).ln();
        assert!(
            (ce - expected).abs() < 1e-5,
            "uniform CE: got {}, expected {}",
            ce,
            expected
        );
    }

    #[test]
    fn test_cross_entropy_peaked() {
        // Strong signal on correct target → low CE
        let vocab = 10;
        let mut logits = vec![0.0f32; vocab];
        logits[3] = 100.0; // Very confident on token 3
        let ce = cross_entropy_from_logits(&logits, 3, vocab);
        assert!(ce < 0.01, "peaked CE should be near 0: got {}", ce);
    }

    #[test]
    fn test_cross_entropy_wrong_target() {
        // Strong signal on wrong target → high CE
        let vocab = 10;
        let mut logits = vec![0.0f32; vocab];
        logits[3] = 100.0;
        let ce = cross_entropy_from_logits(&logits, 0, vocab);
        assert!(ce > 90.0, "wrong target CE should be high: got {}", ce);
    }

    #[test]
    fn test_load_tokens_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");

        // Write test tokens
        let tokens: Vec<u32> = vec![1, 42, 1000, 0, u32::MAX];
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).unwrap();

        let loaded = load_tokens(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded, tokens);
    }

    #[test]
    fn test_load_tokens_invalid_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        std::fs::write(&path, &[1, 2, 3]).unwrap(); // 3 bytes, not multiple of 4
        assert!(load_tokens(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn test_append_eval_csv() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("eval.csv");

        let result = PerplexityResult {
            n_tokens: 1000,
            avg_loss: 5.5,
            perplexity: 244.7,
            bits_per_byte: 2.14,
            n_chunks: 10,
        };

        // First write creates header
        append_eval_csv(&csv_path, 100, &result).unwrap();
        let content = std::fs::read_to_string(&csv_path).unwrap();
        assert!(content.starts_with("step,avg_loss,"));
        assert!(content.contains("100,5.500000,244.70"));

        // Second write appends without header
        append_eval_csv(&csv_path, 200, &result).unwrap();
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 data rows
    }
}
