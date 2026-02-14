//! Test if data loader properly shifts targets

use anyhow::Result;
use nanochat_train::data::dataset::{Dataset, TokenFileDataset};

fn main() -> Result<()> {
    println!("Testing data loader target shifting...\n");

    // Create a simple test dataset
    let test_tokens = vec![10u32, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let seq_len = 4;

    let dataset = TokenFileDataset::new(test_tokens.clone(), seq_len);

    println!("Test tokens: {:?}", test_tokens);
    println!("Sequence length: {}\n", seq_len);

    // Check first few samples
    for idx in 0..3 {
        let (input, target) = dataset.get_item(idx);
        println!("Sample {}:", idx);
        println!("  Input:  {:?}", input);
        println!("  Target: {:?}", target);

        // Verify shift
        let mut correct = true;
        for i in 0..input.len() {
            if target[i] != input[i] + 10 {
                correct = false;
                println!(
                    "  ❌ ERROR: target[{}] = {}, expected {}",
                    i,
                    target[i],
                    input[i] + 10
                );
            }
        }

        if correct {
            println!("  ✓ Targets correctly shifted by 1 position");
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("If targets match input + 10, the shift is working correctly.");
    println!("The model should learn: input[i] -> predict target[i] = input[i+1]");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
