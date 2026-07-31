//! Fast full-tensor reductions.
//!
//! Candle's [`Tensor::sum_all`] reduces every axis in a single kernel, and on
//! CUDA that path is dramatically slower than reducing one axis at a time.
//! Measured on an RTX 5090, f32:
//!
//! ```text
//!   tensor            sum_all     sum(1) then sum(0)
//!   [3584, 1024]       1.7 ms          0.1 ms
//!   [4096, 4096]       7.3 ms          0.2 ms
//! ```
//!
//! 1.7 ms for 14 MB is ~8 GB/s against a card that streams ~1.5 TB/s, so
//! `sum_all` is not bandwidth-bound — it is simply a bad kernel choice. Both the
//! gradient-norm computation and Muon's Newton-Schulz normalization call this
//! once per parameter per step, where the difference is worth ~200 ms/step on a
//! 258M-parameter model.

use candle_core::{Result, Tensor};

/// Sum every element of `t`, reducing one axis at a time.
///
/// Equivalent to `t.sum_all()` up to floating-point summation order, which
/// differs because the reduction is staged rather than done in one pass. Callers
/// here use the result for gradient clipping and normalization, where that
/// difference is far below the noise floor.
pub fn sum_all_fast(t: &Tensor) -> Result<Tensor> {
    let mut out = t.clone();
    // Reduce trailing axes first: the last axis is contiguous, so each stage is
    // a coalesced read.
    while !out.dims().is_empty() {
        let last = out.dims().len() - 1;
        out = out.sum(last)?;
    }
    Ok(out)
}

/// `sum(t^2)` over every element, as a scalar tensor.
pub fn sum_squares(t: &Tensor) -> Result<Tensor> {
    sum_all_fast(&t.sqr()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn matches_sum_all() -> Result<()> {
        let dev = Device::Cpu;
        for shape in [vec![7usize], vec![4, 5], vec![2, 3, 4], vec![2, 3, 4, 5]] {
            let t = Tensor::randn(0f32, 1.0, shape.clone(), &dev)?;
            let got = sum_all_fast(&t)?.to_scalar::<f32>()?;
            let want = t.sum_all()?.to_scalar::<f32>()?;
            assert!(
                (got - want).abs() < 1e-4 * want.abs().max(1.0),
                "shape {shape:?}: {got} vs {want}"
            );
        }
        Ok(())
    }

    #[test]
    fn sum_squares_is_squared_norm() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::new(&[3f32, 4.0], &dev)?;
        assert!((sum_squares(&t)?.to_scalar::<f32>()? - 25.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn handles_scalar_and_zero_sized() -> Result<()> {
        let dev = Device::Cpu;
        let scalar = Tensor::new(2f32, &dev)?;
        assert!((sum_all_fast(&scalar)?.to_scalar::<f32>()? - 2.0).abs() < 1e-6);

        let empty = Tensor::zeros((0,), DType::F32, &dev)?;
        assert!(sum_all_fast(&empty)?.to_scalar::<f32>()?.abs() < 1e-6);
        Ok(())
    }
}
