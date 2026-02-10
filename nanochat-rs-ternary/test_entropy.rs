use candle_core::{Tensor, Device, Result};

fn main() -> Result<()> {
    let device = Device::Cpu;
    
    // Test tensor: 3 tokens x 5 vocab
    let logits = Tensor::randn(0.0f32, 1.0, (3, 5), &device)?;
    println!("Logits shape: {:?}", logits.dims());
    
    // Softmax over vocab (dim 1)
    let probs = candle_nn::ops::softmax(&logits, 1)?;
    let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;
    
    // Element-wise multiply
    let prod = (&probs * &log_probs)?;
    println!("Product shape: {:?}", prod.dims());
    
    // Sum over vocab dimension (dim 1)
    let entropy_per_token = prod.sum(1)?.neg()?;
    println!("Entropy per token shape: {:?}", entropy_per_token.dims());
    println!("Entropy per token: {:?}", entropy_per_token.to_vec1::<f32>()?);
    
    // Mean across tokens
    let avg_entropy = entropy_per_token.mean_all()?;
    println!("Average entropy: {}", avg_entropy.to_scalar::<f32>()?);
    
    Ok(())
}
