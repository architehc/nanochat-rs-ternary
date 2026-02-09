//! Synthetic code patterns that resemble real Python code structures.
//!
//! This generates more realistic training data than pure random tokens.

use super::Dataset;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::Rng;

/// Python code templates with realistic token patterns
pub struct CodePatternsDataset {
    samples: Vec<Vec<u32>>,
    seq_len: usize,
}

impl CodePatternsDataset {
    pub fn new(vocab_size: u32, seq_len: usize, num_samples: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut samples = Vec::with_capacity(num_samples);

        // Common Python keywords/patterns (mapped to token IDs)
        // These are approximate - in reality you'd use actual tokenizer output
        let patterns = vec![
            // "def function_name():"
            vec![2934, 2163, 62, 2393, 2599, 3419, 25],
            // "import module"
            vec![11748, 8265],
            // "if condition:"
            vec![361, 4006, 25],
            // "for i in range(n):"
            vec![1640, 1312, 287, 2837, 7, 77, 2599, 25],
            // "return result"
            vec![7783, 1255],
            // "class ClassName:"
            vec![4871, 30004, 25],
            // "print(variable)"
            vec![4798, 7, 45286, 8],
            // "= value"
            vec![28, 1988],
            // "[item for item in list]"
            vec![58, 9186, 329, 2378, 287, 1351, 60],
            // "try: ... except:"
            vec![28311, 25, 2644, 2845, 25],
        ];

        for i in 0..num_samples {
            let mut seq = Vec::with_capacity(seq_len + 1);

            match i % 5 {
                0 => {
                    // Function definition pattern
                    let pattern = &patterns[0]; // "def ..."
                    for _ in 0..=(seq_len / pattern.len()) {
                        seq.extend_from_slice(pattern);
                    }
                    seq.truncate(seq_len + 1);
                }
                1 => {
                    // Import + function pattern
                    let p1 = &patterns[1]; // "import ..."
                    let p2 = &patterns[0]; // "def ..."
                    seq.extend_from_slice(p1);
                    for _ in 0..=(seq_len / p2.len()) {
                        seq.extend_from_slice(p2);
                    }
                    seq.truncate(seq_len + 1);
                }
                2 => {
                    // Control flow
                    let p1 = &patterns[2]; // "if ..."
                    let p2 = &patterns[4]; // "return ..."
                    for _ in 0..=(seq_len / (p1.len() + p2.len())) {
                        seq.extend_from_slice(p1);
                        seq.extend_from_slice(p2);
                    }
                    seq.truncate(seq_len + 1);
                }
                3 => {
                    // Loop pattern
                    let p1 = &patterns[3]; // "for ..."
                    let p2 = &patterns[6]; // "print(...)"
                    for _ in 0..=(seq_len / (p1.len() + p2.len())) {
                        seq.extend_from_slice(p1);
                        seq.extend_from_slice(p2);
                    }
                    seq.truncate(seq_len + 1);
                }
                _ => {
                    // Mix of random patterns
                    while seq.len() < seq_len + 1 {
                        let pattern = patterns.choose(&mut rng).unwrap();
                        seq.extend_from_slice(pattern);
                    }
                    seq.truncate(seq_len + 1);
                }
            }

            // Add some variation with random tokens (10% of tokens)
            for j in 0..seq.len() {
                if rng.gen::<f32>() < 0.1 {
                    seq[j] = rng.gen_range(0..vocab_size);
                }
            }

            samples.push(seq);
        }

        Self { samples, seq_len }
    }
}

impl Dataset for CodePatternsDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get_item(&self, idx: usize) -> (Vec<u32>, Vec<u32>) {
        let seq = &self.samples[idx];
        let input = seq[..self.seq_len].to_vec();
        let target = seq[1..=self.seq_len].to_vec();
        (input, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_patterns_dataset() {
        let ds = CodePatternsDataset::new(50257, 128, 100, 42);
        assert_eq!(ds.len(), 100);

        let (input, target) = ds.get_item(0);
        assert_eq!(input.len(), 128);
        assert_eq!(target.len(), 128);

        // Verify tokens are in vocab range
        assert!(input.iter().all(|&t| t < 50257));
    }
}
