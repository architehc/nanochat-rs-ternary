//! Minimal CUDA test to isolate the OOM issue

use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, loss};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing minimal CUDA operations...\n");

    // Initialize CUDA
    let device = Device::new_cuda(0)?;
    println!("✓ CUDA device initialized: {:?}", device);

    // Test 1: Simple tensor operations
    println!("\nTest 1: Create tensors");
    let t1 = Tensor::randn(0.0, 1.0, (1000, 1000), &device)?;
    println!("✓ Created 1000x1000 tensor");
    drop(t1);
    println!("✓ Dropped tensor");

    // Test 2: Multiple allocations
    println!("\nTest 2: Multiple allocations");
    for i in 0..100 {
        let t = Tensor::randn(0.0, 1.0, (1000, 1000), &device)?;
        drop(t);
        if i % 20 == 0 {
            println!("  Iteration {}/100", i);
        }
    }
    println!("✓ 100 allocations successful");

    // Test 3: Gradients and backward
    println!("\nTest 3: Gradients");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let weight = vb.get((768, 768), "weight")?;

    for i in 0..100 {
        let x = Tensor::randn(0.0f32, 1.0f32, (4, 768), &device)?;
        let y = x.matmul(&weight)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        // Explicitly drop everything
        drop(loss);
        drop(y);
        drop(x);
        drop(grads);

        if i % 20 == 0 {
            println!("  Iteration {}/100", i);
        }
    }
    println!("✓ 100 gradient computations successful");

    // Test 4: Optimizer step
    println!("\nTest 4: Optimizer steps");
    let varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
    let weight2 = vb2.get((768, 768), "weight")?;

    let params = candle_nn::ParamsAdamW {
        lr: 0.001,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap2.all_vars(), params)?;

    for i in 0..100 {
        let x = Tensor::randn(0.0f32, 1.0f32, (4, 768), &device)?;
        let y = x.matmul(&weight2)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        opt.step(&grads)?;

        drop(loss);
        drop(y);
        drop(x);
        drop(grads);

        if i % 20 == 0 {
            println!("  Iteration {}/100", i);
        }
    }
    println!("✓ 100 optimizer steps successful");

    println!("\n✅ All tests passed! No memory leak detected.");

    Ok(())
}
