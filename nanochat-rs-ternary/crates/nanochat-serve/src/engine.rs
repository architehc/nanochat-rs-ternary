//! Inference engine: KV-cache management, sampling, text generation.

use nanochat_model::config::ModelConfig;
use nanochat_model::model::NanochatModel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            max_tokens: 256,
            seed: None,
        }
    }
}

/// Token generated during streaming.
pub struct GeneratedToken {
    pub token_id: u32,
    pub finish_reason: Option<String>,
}

/// Inference engine wrapping the model with generation logic.
pub struct InferenceEngine {
    pub model: NanochatModel,
    pub eot_token: u32,
}

impl InferenceEngine {
    pub fn new(model: NanochatModel) -> Self {
        Self { model, eot_token: 50256 } // GPT-2 <|endoftext|>
    }

    /// Create engine with random weights for testing.
    pub fn new_random(config: ModelConfig) -> Self {
        Self {
            model: NanochatModel::new_random(config),
            eot_token: 50256,
        }
    }

    /// Generate tokens autoregressively.
    ///
    /// prompt_ids: input token sequence
    /// params: sampling parameters
    ///
    /// Returns: generated token IDs (not including prompt).
    pub fn generate(&mut self, prompt_ids: &[u32], params: &SamplingParams) -> Vec<u32> {
        let mut tokens = Vec::new();
        self.generate_streaming(prompt_ids, params, |tok| {
            tokens.push(tok.token_id);
            tok.finish_reason.is_none()
        });
        tokens
    }

    /// Generate tokens one at a time, calling `on_token` for each.
    /// Returns when generation is complete or `on_token` returns false.
    pub fn generate_streaming<F>(
        &mut self,
        prompt_ids: &[u32],
        params: &SamplingParams,
        mut on_token: F,
    ) where
        F: FnMut(GeneratedToken) -> bool,
    {
        self.model.reset_caches();

        let mut rng: Box<dyn RngCore> = match params.seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(StdRng::from_entropy()),
        };

        // Prefill: process all prompt tokens
        let mut logits = vec![];
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = self.model.forward_token(tid, pos);
        }

        // Decode: generate one token at a time
        let mut pos = prompt_ids.len();

        for _ in 0..params.max_tokens {
            if logits.is_empty() {
                break;
            }

            let next_token = sample_token(&logits, params, &mut *rng);

            let is_eot = next_token == self.eot_token;
            let at_limit = pos + 1 >= self.model.config.max_seq_len;

            let finish_reason = if is_eot {
                Some("stop".to_string())
            } else if at_limit {
                Some("length".to_string())
            } else {
                None
            };

            let should_continue = on_token(GeneratedToken {
                token_id: next_token,
                finish_reason: finish_reason.clone(),
            });

            if !should_continue || is_eot || at_limit {
                break;
            }

            logits = self.model.forward_token(next_token, pos);
            pos += 1;
        }
    }
}

// Trait alias for RNG used in sampling
use rand::RngCore;

/// Sample a token from logits given sampling parameters.
pub fn sample_token(logits: &[f32], params: &SamplingParams, rng: &mut dyn RngCore) -> u32 {
    if params.temperature < 1e-6 {
        return argmax(logits);
    }

    // Apply temperature
    let mut probs: Vec<f32> = logits.iter().map(|&l| l / params.temperature).collect();

    // Top-k filtering
    if params.top_k > 0 && params.top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let threshold = probs[indices[params.top_k - 1]];
        for p in probs.iter_mut() {
            if *p < threshold {
                *p = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax
    let max_val = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        *p = (*p - max_val).exp();
        sum += *p;
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // Top-p (nucleus) filtering
    if params.top_p < 1.0 {
        let mut sorted: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cumsum = 0.0;
        let mut cutoff_idx = sorted.len();
        for (i, &(_, p)) in sorted.iter().enumerate() {
            cumsum += p;
            if cumsum > params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        let keep: std::collections::HashSet<usize> =
            sorted[..cutoff_idx].iter().map(|&(idx, _)| idx).collect();
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *p = 0.0;
            }
        }
        // Re-normalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in probs.iter_mut() {
                *p *= inv;
            }
        }
    }

    // Categorical sampling
    categorical_sample(&probs, rng)
}

/// Argmax over a slice.
fn argmax(x: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Sample from a categorical distribution using the inverse CDF method.
fn categorical_sample(probs: &[f32], rng: &mut dyn RngCore) -> u32 {
    let u: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i as u32;
        }
    }
    // Fallback: return last non-zero probability token
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[1.0, 0.0, 0.0]), 0);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(sample_token(&logits, &params, &mut rng), 2);
    }

    #[test]
    fn test_sample_with_temperature() {
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let params = SamplingParams {
            temperature: 0.01, // very low temp -> nearly greedy
            top_k: 0,
            top_p: 1.0,
            max_tokens: 10,
            seed: Some(42),
        };
        let mut rng = StdRng::seed_from_u64(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert_eq!(token, 2); // Should pick max with very low temp
    }

    #[test]
    fn test_categorical_sample_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let t = categorical_sample(&probs, &mut rng);
            counts[t as usize] += 1;
        }
        // Each should be roughly 250 ± 50
        for &c in &counts {
            assert!(c > 150 && c < 350, "count {} out of range", c);
        }
    }

    #[test]
    fn test_categorical_sample_deterministic_seed() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng1 = StdRng::seed_from_u64(123);
        let mut rng2 = StdRng::seed_from_u64(123);
        for _ in 0..100 {
            assert_eq!(
                categorical_sample(&probs, &mut rng1),
                categorical_sample(&probs, &mut rng2),
            );
        }
    }

    #[test]
    fn test_sampling_params_default() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_k, 50);
        assert!((p.top_p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_engine_generate() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
        };
        let mut engine = InferenceEngine::new_random(config);
        engine.eot_token = 0; // Use 0 as EOT for small vocab

        let prompt = vec![1u32, 5, 10];
        let params = SamplingParams {
            temperature: 0.0, // greedy for determinism
            max_tokens: 5,
            ..Default::default()
        };

        let output = engine.generate(&prompt, &params);
        assert!(!output.is_empty(), "generate produced no tokens");
        assert!(output.len() <= 5, "exceeded max_tokens");

        for &t in &output {
            assert!((t as usize) < 256, "invalid token: {}", t);
        }
    }

    #[test]
    fn test_sample_with_top_k() {
        let logits = vec![1.0, 2.0, 10.0, 3.0, 0.5, 0.1, 0.2, 0.3];
        let params = SamplingParams {
            temperature: 0.01, // near-greedy
            top_k: 3,
            top_p: 1.0,
            max_tokens: 10,
            seed: None,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert_eq!(token, 2); // Should pick max
    }

    #[test]
    fn test_sample_with_top_p() {
        let logits = vec![10.0, 0.0, -10.0, -10.0, -10.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            max_tokens: 10,
            seed: None,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert_eq!(token, 0); // Token 0 has overwhelming probability
    }

    #[test]
    fn test_sample_with_top_k_and_top_p() {
        let logits = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0];
        let params = SamplingParams {
            temperature: 0.5,
            top_k: 4,
            top_p: 0.8,
            max_tokens: 10,
            seed: None,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert!(token < 4, "token {} should be in top-4", token);
    }

    #[test]
    fn test_engine_new() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
        };
        let model = NanochatModel::new_random(config);
        let engine = InferenceEngine::new(model);
        assert_eq!(engine.model.config.dim, 128);
    }

    #[test]
    fn test_engine_empty_prompt() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
        };
        let mut engine = InferenceEngine::new_random(config);

        let params = SamplingParams::default();
        let output = engine.generate(&[], &params);
        assert!(output.is_empty());
    }

    #[test]
    fn test_generate_streaming() {
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 256,
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
        };
        let mut engine = InferenceEngine::new_random(config);
        engine.eot_token = 0;

        let prompt = vec![1u32, 5];
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 3,
            ..Default::default()
        };

        let mut tokens = Vec::new();
        let mut had_finish = false;
        engine.generate_streaming(&prompt, &params, |tok| {
            tokens.push(tok.token_id);
            if tok.finish_reason.is_some() {
                had_finish = true;
            }
            tok.finish_reason.is_none()
        });

        assert!(!tokens.is_empty());
    }
}
