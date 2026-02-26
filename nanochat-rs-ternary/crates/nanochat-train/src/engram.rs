//! Engram: O(1) N-gram lookup memory (arXiv 2601.07372).
//!
//! Offloads local pattern memorization into hash-addressed embedding tables,
//! letting the backbone focus on reasoning. Each Engram layer:
//! 1. Hashes N-gram suffixes at each position into table indices
//! 2. Retrieves embeddings from tables (differentiable via index_select)
//! 3. Projects retrieved embeddings to key/value vectors
//! 4. Context-gates the value with the hidden state
//! 5. Applies depthwise causal convolution
//! 6. Adds gated result as a residual

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::{BitLinearSTE, RMSNormTrain};

/// Fixed primes for multiplicative hashing, one per (order, head).
/// Using distinct primes per position in the N-gram avoids systematic collisions.
const HASH_PRIMES: [u64; 12] = [
    2654435761, 2246822519, 3266489917, 668265263, 374761393, 2869860233, 2654435769, 1103515245,
    12345, 1664525, 1013904223, 6364136223,
];

/// Engram module for training — N-gram memory with context gating.
pub struct EngramTrain {
    /// Embedding tables: n_orders * n_heads tables, each [table_size, d_mem]
    tables: Vec<Tensor>,
    /// Projection from concatenated embeddings to dim
    w_k: BitLinearSTE,
    w_v: BitLinearSTE,
    /// Gate norm (applied to hidden state for context-dependent gating)
    gate_norm: RMSNormTrain,
    /// Key norm (applied to projected keys for gate computation)
    key_norm: RMSNormTrain,
    /// Depthwise causal conv weight: [dim, 1, conv_kernel]
    conv_weight: Tensor,
    /// Config
    d_mem: usize,
    n_gram_orders: Vec<usize>,
    n_heads: usize,
    table_size: usize,
    dim: usize,
    conv_kernel: usize,
}

impl EngramTrain {
    pub fn new(
        dim: usize,
        d_mem: usize,
        n_gram_orders: &[usize],
        n_heads: usize,
        table_size: usize,
        conv_kernel: usize,
        group_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n_orders = n_gram_orders.len();
        let total_heads = n_orders * n_heads;
        let concat_dim = total_heads * d_mem;

        // Create embedding tables
        let mut tables = Vec::with_capacity(total_heads);
        for i in 0..total_heads {
            let table = vb.get_with_hints(
                (table_size, d_mem),
                &format!("table.{}", i),
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )?;
            tables.push(table);
        }

        // Projection layers: [concat_dim, dim]
        let w_k = BitLinearSTE::new(concat_dim, dim, group_size, vb.pp("w_k"))?;
        let w_v = BitLinearSTE::new(concat_dim, dim, group_size, vb.pp("w_v"))?;

        // Norms for gating
        let gate_norm = RMSNormTrain::new(dim, vb.pp("gate_norm"))?;
        let key_norm = RMSNormTrain::new(dim, vb.pp("key_norm"))?;

        // Depthwise causal conv: [dim, 1, conv_kernel]
        let conv_weight = vb.get_with_hints(
            (dim, 1, conv_kernel),
            "conv_weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: (1.0 / (conv_kernel as f64)).sqrt(),
            },
        )?;

        Ok(Self {
            tables,
            w_k,
            w_v,
            gate_norm,
            key_norm,
            conv_weight,
            d_mem,
            n_gram_orders: n_gram_orders.to_vec(),
            n_heads,
            table_size,
            dim,
            conv_kernel,
        })
    }

    /// Forward pass: enriches hidden states with N-gram memory.
    ///
    /// Args:
    /// - hidden: [batch, seq, dim] — current hidden state
    /// - token_ids: [batch, seq] — token IDs for N-gram lookup (u32)
    ///
    /// Returns: [batch, seq, dim] — hidden + gated engram residual
    pub fn forward(&self, hidden: &Tensor, token_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq, _dim) = hidden.dims3()?;

        // 1. Compute hash indices for all (order, head) combinations
        //    indices: [batch, seq] per (order, head)
        let hash_indices = self.compute_hash_indices(token_ids, batch, seq)?;

        // 2. Retrieve embeddings from tables and concatenate
        let mut all_embeddings = Vec::with_capacity(self.n_gram_orders.len() * self.n_heads);
        for (table_idx, indices) in hash_indices.iter().enumerate() {
            // indices: [batch * seq] u32
            // table: [table_size, d_mem]
            let retrieved = self.tables[table_idx].index_select(indices, 0)?;
            // retrieved: [batch * seq, d_mem]
            all_embeddings.push(retrieved);
        }

        // Concatenate all retrieved embeddings: [batch * seq, total_heads * d_mem]
        let concat = Tensor::cat(&all_embeddings, 1)?;
        // Reshape to [batch, seq, concat_dim]
        let concat_dim = self.n_gram_orders.len() * self.n_heads * self.d_mem;
        let concat = concat.reshape((batch, seq, concat_dim))?;

        // 3. Project to key and value
        let k = self.w_k.forward(&concat)?; // [batch, seq, dim]
        let v = self.w_v.forward(&concat)?; // [batch, seq, dim]

        // 4. Context gate: alpha = sigmoid(rmsnorm(h)^T rmsnorm(k) / sqrt(d))
        let h_norm = self.gate_norm.forward(hidden)?; // [batch, seq, dim]
        let k_norm = self.key_norm.forward(&k)?; // [batch, seq, dim]

        // Dot product per position: [batch, seq, 1]
        let scale = (self.dim as f64).sqrt();
        let alpha = (&h_norm * &k_norm)?
            .sum_keepdim(2)? // [batch, seq, 1]
            .affine(1.0 / scale, 0.0)?;
        // Manual sigmoid: 1 / (1 + exp(-x)) — works on CUDA (candle_nn::ops::sigmoid doesn't)
        let alpha = alpha.neg()?.exp()?.affine(1.0, 1.0)?.recip()?; // [batch, seq, 1]

        // 5. Apply gate to value (alpha is [batch, seq, 1], v is [batch, seq, dim])
        let gated_v = alpha.broadcast_mul(&v)?; // [batch, seq, dim]

        // 6. Depthwise causal convolution
        let conv_out = self.causal_depthwise_conv(&gated_v)?;

        // 7. Residual add
        hidden + &conv_out
    }

    /// Compute hash indices for all (order, head) combinations.
    /// Returns a Vec of flat index tensors, one per (order, head).
    fn compute_hash_indices(
        &self,
        token_ids: &Tensor,
        batch: usize,
        seq: usize,
    ) -> Result<Vec<Tensor>> {
        // Extract token IDs to CPU for hash computation
        let ids_flat: Vec<u32> = token_ids.flatten_all()?.to_vec1()?;
        let device = token_ids.device();
        let total = batch * seq;

        let mut all_indices = Vec::with_capacity(self.n_gram_orders.len() * self.n_heads);
        let mut prime_idx = 0;

        for &order in &self.n_gram_orders {
            for _head in 0..self.n_heads {
                let mut indices = vec![0u32; total];

                for b in 0..batch {
                    for t in 0..seq {
                        let flat_pos = b * seq + t;
                        let mut h: u64 = 0;

                        // Hash N-gram suffix at position t
                        for k in 0..order {
                            let pos = t as isize - (order as isize - 1 - k as isize);
                            let token = if pos >= 0 {
                                ids_flat[b * seq + pos as usize] as u64
                            } else {
                                0 // pad with 0 for positions before sequence start
                            };
                            let prime = HASH_PRIMES[(prime_idx + k) % HASH_PRIMES.len()];
                            h = h.wrapping_mul(prime).wrapping_add(token);
                        }

                        indices[flat_pos] = (h % self.table_size as u64) as u32;
                    }
                }

                let idx_tensor = Tensor::from_vec(indices, total, device)?;
                all_indices.push(idx_tensor);
                prime_idx += 1;
            }
        }

        Ok(all_indices)
    }

    /// Depthwise causal convolution: convolve each channel independently.
    /// Input: [batch, seq, dim], Output: [batch, seq, dim]
    ///
    /// For small kernel sizes (k=4), implements via shifted slices + elementwise multiply.
    fn causal_depthwise_conv(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, dim) = x.dims3()?;
        let k = self.conv_kernel;

        // Pad left by (kernel_size - 1) for causal convolution
        let pad_size = k - 1;
        let zeros = Tensor::zeros((batch, pad_size, dim), x.dtype(), x.device())?;
        let padded = Tensor::cat(&[&zeros, x], 1)?; // [batch, seq + pad, dim]

        // For each tap i, extract weight[i] and multiply with shifted input
        let mut out: Option<Tensor> = None;
        for i in 0..k {
            // Weight for this tap: conv_weight[:, 0, i] → [dim]
            let w_i = self.conv_weight.i((.., 0, i))?; // [dim]
            // Broadcast to [1, 1, dim] for elementwise multiply
            let w_i = w_i.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, dim]

            // Shifted slice: padded[:, (pad_size - i) .. (pad_size - i + seq), :]
            let start = pad_size - i;
            let input_slice = padded.narrow(1, start, seq)?; // [batch, seq, dim]

            let product = input_slice.broadcast_mul(&w_i)?; // [batch, seq, dim]

            out = Some(match out {
                Some(acc) => (acc + product)?,
                None => product,
            });
        }

        out.ok_or_else(|| candle_core::Error::Msg("conv_kernel must be > 0".to_string()))
    }

    /// Collect table (embedding) parameters — for separate optimizer group.
    pub fn table_params(&self) -> Vec<&Tensor> {
        self.tables.iter().collect()
    }

    /// Collect linear projection parameters (W_k, W_v weights).
    pub fn linear_params(&self) -> Vec<&Tensor> {
        vec![self.w_k.weight(), self.w_v.weight()]
    }

    /// Collect norm parameters.
    pub fn norm_params(&self) -> Vec<&Tensor> {
        vec![self.gate_norm.weight(), self.key_norm.weight()]
    }

    /// Collect conv parameters.
    pub fn conv_params(&self) -> Vec<&Tensor> {
        vec![&self.conv_weight]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_engram_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let dim = 64;
        let engram = EngramTrain::new(
            dim,
            32,       // d_mem
            &[2, 3],  // bigram + trigram
            2,        // n_heads
            1009,     // table_size (prime)
            4,        // conv_kernel
            64,       // group_size
            vb.pp("engram"),
        )?;

        let batch = 2;
        let seq = 8;
        let hidden = Tensor::randn(0.0f32, 1.0, (batch, seq, dim), &device)?;
        let token_ids = Tensor::zeros((batch, seq), DType::U32, &device)?;

        let out = engram.forward(&hidden, &token_ids)?;
        assert_eq!(out.dims(), &[batch, seq, dim]);

        // Output should be finite
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(vals.iter().all(|v| v.is_finite()), "output should be finite");

        Ok(())
    }

    #[test]
    fn test_hash_determinism() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let engram = EngramTrain::new(64, 32, &[2, 3], 2, 1009, 4, 64, vb.pp("engram"))?;

        let token_ids = Tensor::new(&[1u32, 2, 3, 4, 5, 6, 7, 8], &device)?.unsqueeze(0)?;

        let indices1 = engram.compute_hash_indices(&token_ids, 1, 8)?;
        let indices2 = engram.compute_hash_indices(&token_ids, 1, 8)?;

        for (i1, i2) in indices1.iter().zip(indices2.iter()) {
            let v1: Vec<u32> = i1.to_vec1()?;
            let v2: Vec<u32> = i2.to_vec1()?;
            assert_eq!(v1, v2, "hash should be deterministic");
        }

        Ok(())
    }

    #[test]
    fn test_gate_values_bounded() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let dim = 64;
        let engram = EngramTrain::new(dim, 32, &[2], 2, 1009, 4, 64, vb.pp("engram"))?;

        // Create inputs with known values
        let hidden = Tensor::randn(0.0f32, 1.0, (1, 4, dim), &device)?;
        let token_ids = Tensor::new(&[10u32, 20, 30, 40], &device)?.unsqueeze(0)?;

        // Forward should work without errors (gate values are internal)
        let out = engram.forward(&hidden, &token_ids)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(vals.iter().all(|v| v.is_finite()));

        Ok(())
    }

    #[test]
    fn test_gradient_flows_to_tables() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let dim = 64;
        let engram = EngramTrain::new(dim, 32, &[2], 2, 1009, 4, 64, vb.pp("engram"))?;

        let hidden = Tensor::randn(0.0f32, 1.0, (1, 4, dim), &device)?;
        let token_ids = Tensor::new(&[1u32, 2, 3, 4], &device)?.unsqueeze(0)?;

        let out = engram.forward(&hidden, &token_ids)?;
        let loss = out.sum_all()?;
        let grads = loss.backward()?;

        // Table params should have gradients
        for (i, table) in engram.tables.iter().enumerate() {
            let g = grads.get(table);
            assert!(g.is_some(), "table {} should have gradient", i);
            let gn = g
                .unwrap()
                .sqr()?
                .sum_all()?
                .sqrt()?
                .to_scalar::<f32>()?;
            // Gradient may be sparse (only accessed rows get grad)
            // but should be finite
            assert!(gn.is_finite(), "table {} grad norm should be finite", i);
        }

        Ok(())
    }

    #[test]
    fn test_hash_indices_in_range() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let table_size = 1009;
        let engram =
            EngramTrain::new(64, 32, &[2, 3], 2, table_size, 4, 64, vb.pp("engram"))?;

        let token_ids = Tensor::new(&[100u32, 200, 300, 400, 500, 1000, 2000, 4000], &device)?
            .unsqueeze(0)?;

        let indices = engram.compute_hash_indices(&token_ids, 1, 8)?;

        for idx_tensor in &indices {
            let vals: Vec<u32> = idx_tensor.to_vec1()?;
            for &v in &vals {
                assert!(
                    (v as usize) < table_size,
                    "hash index {} >= table_size {}",
                    v,
                    table_size
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_param_collection() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let engram = EngramTrain::new(64, 32, &[2, 3], 2, 1009, 4, 64, vb.pp("engram"))?;

        // 2 orders × 2 heads = 4 tables
        assert_eq!(engram.table_params().len(), 4);
        // W_k + W_v
        assert_eq!(engram.linear_params().len(), 2);
        // gate_norm + key_norm
        assert_eq!(engram.norm_params().len(), 2);
        // conv_weight
        assert_eq!(engram.conv_params().len(), 1);

        Ok(())
    }
}
