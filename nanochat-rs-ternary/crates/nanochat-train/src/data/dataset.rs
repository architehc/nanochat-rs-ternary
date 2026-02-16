//! Dataset trait and implementations for training data.

use candle_core::{Device, Result, Tensor};
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// A dataset returns (input_ids, target_ids) pairs.
pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get_item(&self, idx: usize) -> (Vec<u32>, Vec<u32>);
}

/// Generates synthetic token sequences for training loop validation.
pub struct SyntheticDataset {
    data: Vec<Vec<u32>>,
    seq_len: usize,
}

impl SyntheticDataset {
    pub fn new(vocab_size: u32, seq_len: usize, num_samples: usize, seed: u64) -> Self {
        assert!(seq_len > 0, "seq_len must be > 0");
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let seq = match i % 3 {
                0 => {
                    // Repeated token
                    let tok = (i as u32 % (vocab_size - 1)) + 1;
                    vec![tok; seq_len + 1]
                }
                1 => {
                    // Sequential (modular)
                    let start = i as u32 % vocab_size;
                    (0..=seq_len as u32)
                        .map(|j| (start + j) % vocab_size)
                        .collect()
                }
                _ => {
                    // Random
                    use rand::Rng;
                    (0..=seq_len)
                        .map(|_| rng.gen_range(0..vocab_size))
                        .collect()
                }
            };
            data.push(seq);
        }

        Self { data, seq_len }
    }
}

impl Dataset for SyntheticDataset {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get_item(&self, idx: usize) -> (Vec<u32>, Vec<u32>) {
        let seq = &self.data[idx];
        let input = seq[..self.seq_len].to_vec();
        let target = seq[1..=self.seq_len].to_vec();
        (input, target)
    }
}

/// Pre-tokenized dataset loaded from a flat binary file of u32 tokens.
pub struct TokenFileDataset {
    tokens: Vec<u32>,
    seq_len: usize,
    n_chunks: usize,
}

impl TokenFileDataset {
    pub fn new(tokens: Vec<u32>, seq_len: usize) -> Self {
        assert!(seq_len > 0, "seq_len must be > 0");
        let n_chunks = tokens.len().saturating_sub(1) / seq_len;
        Self {
            tokens,
            seq_len,
            n_chunks,
        }
    }

    /// Load from a binary file of little-endian u32 values.
    pub fn from_binary_file(path: &std::path::Path, seq_len: usize) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        if data.len() % 4 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "token file {} has {} bytes (not a multiple of 4)",
                    path.display(),
                    data.len()
                ),
            ));
        }
        let tokens: Vec<u32> = data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Self::new(tokens, seq_len))
    }
}

impl Dataset for TokenFileDataset {
    fn len(&self) -> usize {
        self.n_chunks
    }

    fn get_item(&self, idx: usize) -> (Vec<u32>, Vec<u32>) {
        let start = idx * self.seq_len;
        let end = start + self.seq_len;
        let input = self.tokens[start..end].to_vec();
        let target = self.tokens[start + 1..end + 1].to_vec();
        (input, target)
    }
}

/// Batched data iterator.
pub struct DataLoader<'a> {
    dataset: &'a dyn Dataset,
    batch_size: usize,
    indices: Vec<usize>,
    pos: usize,
    device: Device,
}

impl<'a> DataLoader<'a> {
    pub fn new(
        dataset: &'a dyn Dataset,
        batch_size: usize,
        shuffle: bool,
        seed: u64,
        device: &Device,
    ) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }
        Self {
            dataset,
            batch_size,
            indices,
            pos: 0,
            device: device.clone(),
        }
    }

    pub fn reset(&mut self, shuffle: bool, seed: u64) {
        self.pos = 0;
        if shuffle {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            self.indices.shuffle(&mut rng);
        }
    }

    pub fn n_batches(&self) -> usize {
        self.dataset.len().div_ceil(self.batch_size)
    }
}

impl Iterator for DataLoader<'_> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.pos..end];
        self.pos = end;

        let result = (|| {
            let mut inputs = Vec::new();
            let mut targets = Vec::new();

            for &idx in batch_indices {
                let (inp, tgt) = self.dataset.get_item(idx);
                inputs.push(inp);
                targets.push(tgt);
            }

            let seq_len = inputs[0].len();
            let batch_size = inputs.len();

            let input_flat: Vec<u32> = inputs.into_iter().flatten().collect();
            let target_flat: Vec<u32> = targets.into_iter().flatten().collect();

            let input_tensor = Tensor::from_vec(input_flat, (batch_size, seq_len), &self.device)?;
            let target_tensor = Tensor::from_vec(target_flat, (batch_size, seq_len), &self.device)?;

            Ok((input_tensor, target_tensor))
        })();

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset_shapes() {
        let ds = SyntheticDataset::new(100, 16, 50, 42);
        assert_eq!(ds.len(), 50);

        let (input, target) = ds.get_item(0);
        assert_eq!(input.len(), 16);
        assert_eq!(target.len(), 16);
    }

    #[test]
    fn test_dataloader_iterates_all() -> Result<()> {
        let ds = SyntheticDataset::new(100, 8, 20, 42);
        let device = Device::Cpu;
        let loader = DataLoader::new(&ds, 8, false, 42, &device);

        let mut count = 0;
        let mut total_samples = 0;
        for batch in loader {
            let (inp, tgt) = batch?;
            count += 1;
            total_samples += inp.dim(0)?;
            assert_eq!(inp.dim(1)?, 8);
            assert_eq!(tgt.dim(1)?, 8);
        }

        assert_eq!(count, 3); // ceil(20/8) = 3
        assert_eq!(total_samples, 20);
        Ok(())
    }

    #[test]
    fn test_dataloader_batch_sizes() -> Result<()> {
        let ds = SyntheticDataset::new(100, 8, 10, 42);
        let device = Device::Cpu;
        let loader = DataLoader::new(&ds, 4, false, 42, &device);

        let batches: Vec<_> = loader.collect::<Result<Vec<_>>>()?;
        assert_eq!(batches.len(), 3); // ceil(10/4) = 3
        assert_eq!(batches[0].0.dim(0)?, 4);
        assert_eq!(batches[1].0.dim(0)?, 4);
        assert_eq!(batches[2].0.dim(0)?, 2); // last batch is smaller
        Ok(())
    }
}
