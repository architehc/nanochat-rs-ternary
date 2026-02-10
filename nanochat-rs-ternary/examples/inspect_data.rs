//! Inspect training data to diagnose model collapse
//!
//! Usage: cargo run --example inspect_data

use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use tokenizers::Tokenizer;
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Training Data Inspection");
    println!("═══════════════════════════════════════════════════════════\n");

    // Load tokenizer
    let tokenizer = Tokenizer::from_file("models/gpt2-tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    println!("✓ Tokenizer loaded\n");

    // Read binary data
    let mut file = File::open("data/rust_tokens.bin")?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Convert to u32 tokens (4-byte encoding)
    let tokens: Vec<u32> = buffer
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("Data Statistics:");
    println!("  Total tokens: {}", tokens.len());
    println!("  File size: {} bytes\n", buffer.len());

    // Count token frequencies
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for &token in &tokens {
        *freq.entry(token).or_insert(0) += 1;
    }

    println!("Token Distribution:");
    println!("  Unique tokens: {}", freq.len());

    // Find top 20 most frequent tokens
    let mut freq_vec: Vec<_> = freq.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));

    println!("\nTop 20 Most Frequent Tokens:");
    for (i, (&token, &count)) in freq_vec.iter().take(20).enumerate() {
        let text = tokenizer.id_to_token(token)
            .unwrap_or_else(|| format!("<UNK:{}>", token));
        let pct = count as f64 / tokens.len() as f64 * 100.0;
        println!("  {:2}. Token {:5} ({}): {:8} ({:.2}%)",
                 i + 1, token, text, count, pct);
    }

    // Check for token 1391 specifically (the collapsed token)
    println!("\n─────────────────────────────────────────────────────────");
    let token_1391_count = freq.get(&1391).copied().unwrap_or(0);
    let token_1391_text = tokenizer.id_to_token(1391)
        .unwrap_or_else(|| "<UNK:1391>".to_string());
    println!("Token 1391 Analysis:");
    println!("  Text: '{}'", token_1391_text);
    println!("  Count: {}", token_1391_count);
    println!("  Percentage: {:.2}%", token_1391_count as f64 / tokens.len() as f64 * 100.0);

    // Sample a few sequences
    println!("\n─────────────────────────────────────────────────────────");
    println!("Sample Sequences (first 5 sequences of 20 tokens):\n");

    for seq_idx in 0..5 {
        let start = seq_idx * 20;
        if start + 20 > tokens.len() {
            break;
        }

        let sequence = &tokens[start..start + 20];
        let token_ids: Vec<u32> = sequence.to_vec();
        let text = tokenizer.decode(&token_ids, false)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

        println!("Sequence {}:", seq_idx + 1);
        println!("  Tokens: {:?}", &sequence[..10]);
        println!("  Text: {}\n", text.chars().take(100).collect::<String>());
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("Analysis Complete!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
