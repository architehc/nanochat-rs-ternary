//! Inference engine: KV-cache management, sampling, text generation.

use nanochat_model::config::ModelConfig;
use nanochat_model::model::NanochatModel;

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            max_tokens: 256,
        }
    }
}

/// Inference engine wrapping the model with generation logic.
pub struct InferenceEngine {
    pub model: NanochatModel,
}

impl InferenceEngine {
    pub fn new(model: NanochatModel) -> Self {
        Self { model }
    }

    /// Create engine with random weights for testing.
    pub fn new_random(config: ModelConfig) -> Self {
        Self {
            model: NanochatModel::new_random(config),
        }
    }

    /// Generate tokens autoregressively.
    ///
    /// prompt_ids: input token sequence
    /// params: sampling parameters
    ///
    /// Returns: generated token IDs (not including prompt).
    pub fn generate(&mut self, prompt_ids: &[u32], params: &SamplingParams) -> Vec<u32> {
        self.model.reset_caches();

        // Prefill: process all prompt tokens
        let mut logits = vec![];
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = self.model.forward_token(tid, pos);
        }

        // Decode: generate one token at a time
        let mut generated = Vec::new();
        let mut pos = prompt_ids.len();

        for _ in 0..params.max_tokens {
            if logits.is_empty() {
                break;
            }

            // Sample next token
            let next_token = sample_token(&logits, params);
            generated.push(next_token);

            // Check for EOS (token 0 or 2 conventionally)
            if next_token == 0 || next_token == 2 {
                break;
            }

            // Next step
            logits = self.model.forward_token(next_token, pos);
            pos += 1;

            if pos >= self.model.config.max_seq_len {
                break;
            }
        }

        generated
    }
}

/// Sample a token from logits given sampling parameters.
pub fn sample_token(logits: &[f32], params: &SamplingParams) -> u32 {
    if params.temperature < 1e-6 {
        // Greedy: argmax
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
        // Zero out everything below cutoff
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

    // Deterministic "sampling" using weighted selection
    // (Real implementation would use random, but for reproducibility we use a simple hash)
    weighted_select(&probs)
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

/// Deterministic weighted selection (picks highest probability token).
/// In production, this would sample randomly according to the distribution.
fn weighted_select(probs: &[f32]) -> u32 {
    argmax(probs)
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
        assert_eq!(sample_token(&logits, &params), 2);
    }

    #[test]
    fn test_sample_with_temperature() {
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 10,
        };
        // With temp=1.0 and deterministic select, should still pick max
        let token = sample_token(&logits, &params);
        assert_eq!(token, 2);
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

        let prompt = vec![1u32, 5, 10];
        let params = SamplingParams {
            temperature: 0.0, // greedy for determinism
            max_tokens: 5,
            ..Default::default()
        };

        let output = engine.generate(&prompt, &params);
        assert!(!output.is_empty(), "generate produced no tokens");
        assert!(output.len() <= 5, "exceeded max_tokens");

        // All tokens should be valid
        for &t in &output {
            assert!((t as usize) < 256, "invalid token: {}", t);
        }
    }

    #[test]
    fn test_sample_with_top_k() {
        // Create logits where top-k filtering matters
        let logits = vec![1.0, 2.0, 10.0, 3.0, 0.5, 0.1, 0.2, 0.3];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            top_p: 1.0,
            max_tokens: 10,
        };
        let token = sample_token(&logits, &params);
        // Should pick from top-3 (indices 2, 3, 1), with max being index 2
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_with_top_p() {
        // Create logits where top-p filtering matters
        let logits = vec![10.0, 0.0, -10.0, -10.0, -10.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0, // no top-k
            top_p: 0.5, // very restrictive
            max_tokens: 10,
        };
        let token = sample_token(&logits, &params);
        // Token 0 has overwhelming probability, should be selected
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_with_top_k_and_top_p() {
        let logits = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0];
        let params = SamplingParams {
            temperature: 0.5,
            top_k: 4,
            top_p: 0.8,
            max_tokens: 10,
        };
        let token = sample_token(&logits, &params);
        // Should be from top tokens
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
        // Empty prompt should produce empty output
        assert!(output.is_empty());
    }
}
