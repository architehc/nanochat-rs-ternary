//! Pure cross-entropy of a checkpoint on token data.
//!
//! The training log's `loss` column is NOT cross-entropy: with label smoothing
//! eps and entropy weight lambda it reports
//!
//! ```text
//! (1-eps)*CE + eps*uniform - lambda*H(p)
//! ```
//!
//! so perplexity cannot be read off the training log. This computes plain CE
//! (via `forward_loss`) over the *tail* of the token file — the region least
//! likely to have been visited by shuffled training windows. Not a true
//! held-out split (the repo has none), so treat the number as slightly
//! optimistic.
//!
//! ```bash
//! cargo run --release --features cuda --example eval_ce -- \
//!   --checkpoint checkpoints/qwen35_hybrid_seq512/final \
//!   --data data/rust_v4_4k/tokens.bin
//! ```

#[cfg(feature = "cuda")]
mod imp {

use candle_core::{Device, Tensor};
use nanochat_train::train::Trainer;

pub fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let ckpt = get("--checkpoint").expect("--checkpoint required");
    let data = get("--data").expect("--data required");
    let seq_len: usize = get("--seq-len").and_then(|s| s.parse().ok()).unwrap_or(512);
    let n_batches: usize = get("--batches").and_then(|s| s.parse().ok()).unwrap_or(50);
    let batch: usize = get("--batch-size").and_then(|s| s.parse().ok()).unwrap_or(4);

    let device = Device::new_cuda(0)?;
    let trainer = Trainer::from_checkpoint(&ckpt, device.clone())?;
    let vocab = trainer.config.vocab_size;

    // Raw u32 token stream.
    let bytes = std::fs::read(&data)?;
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let need = n_batches * batch * (seq_len + 1);
    anyhow::ensure!(tokens.len() > need, "token file too small");
    // Evaluate on the tail of the file.
    let tail = &tokens[tokens.len() - need..];
    println!(
        "checkpoint {ckpt}\n{} tokens total, evaluating on final {} ({} batches x {} x {})",
        tokens.len(), need, n_batches, batch, seq_len
    );

    let mut total_ce = 0f64;
    for b in 0..n_batches {
        let mut inp = Vec::with_capacity(batch * seq_len);
        let mut tgt = Vec::with_capacity(batch * seq_len);
        for s in 0..batch {
            let off = (b * batch + s) * (seq_len + 1);
            inp.extend_from_slice(&tail[off..off + seq_len]);
            tgt.extend_from_slice(&tail[off + 1..off + seq_len + 1]);
        }
        let input = Tensor::from_vec(inp, (batch, seq_len), &device)?;
        let target = Tensor::from_vec(tgt, (batch, seq_len), &device)?;
        let ce = trainer.model.forward_loss(&input, &target)?.to_scalar::<f32>()? as f64;
        total_ce += ce;
        if (b + 1) % 10 == 0 {
            println!("  [{:>3}/{n_batches}] running CE {:.4}", b + 1, total_ce / (b + 1) as f64);
        }
    }
    let ce = total_ce / n_batches as f64;
    println!("\npure cross-entropy : {ce:.4} nats");
    println!("perplexity         : {:.2}", ce.exp());
    println!("random baseline    : {:.4} nats (ln {vocab})", (vocab as f64).ln());
    println!("bits per token     : {:.3}", ce / std::f64::consts::LN_2);
    Ok(())
}

}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    imp::run()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example requires --features cuda");
}
