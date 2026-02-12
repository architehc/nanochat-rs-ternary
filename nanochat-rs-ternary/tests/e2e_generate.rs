//! End-to-End Generation Test — full model forward pass sanity.
//!
//! Creates a model with random weights (d20 config), runs forward passes,
//! and verifies output validity.

use nanochat_model::config::ModelConfig;
use nanochat_model::model::NanochatModel;
use nanochat_serve::engine::{InferenceEngine, SamplingParams};

fn make_test_config() -> ModelConfig {
    // Use d20 config and modify for smaller test
    let mut config = ModelConfig::d20();
    config.dim = 128;
    config.n_layers = 4;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 256;
    config.max_seq_len = 64;
    config
}

// ───────────────────── Forward Pass Tests ─────────────────────

#[test]
fn e2e_single_token_produces_valid_logits() {
    let config = make_test_config();
    let mut model = NanochatModel::new_random(config.clone());

    let logits = model.forward_token(1, 0);
    assert_eq!(logits.len(), config.vocab_size);

    // All logits should be finite
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit[{}] = {} not finite", i, l);
    }
}

#[test]
fn e2e_sequence_forward_produces_valid_logits() {
    let config = make_test_config();
    let mut model = NanochatModel::new_random(config.clone());

    let tokens = vec![1u32, 10, 50, 100, 200];
    let logits = model.forward_sequence(&tokens);
    assert_eq!(logits.len(), config.vocab_size);

    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit[{}] = {} not finite", i, l);
    }
}

#[test]
fn e2e_autoregressive_consistency() {
    let config = make_test_config();

    // Forward all tokens at once (sequence prefill)
    let mut model_a = NanochatModel::new_random(config.clone());
    let tokens = vec![1u32, 5, 10, 20];
    let logits_prefill = model_a.forward_sequence(&tokens);

    // Forward tokens one at a time (autoregressive)
    let mut model_b = NanochatModel::new_random(config.clone());
    let mut logits_ar = vec![];
    for (pos, &tid) in tokens.iter().enumerate() {
        logits_ar = model_b.forward_token(tid, pos);
    }

    // Both should produce the same final logits
    assert_eq!(logits_prefill.len(), logits_ar.len());
    for i in 0..logits_prefill.len() {
        let diff = (logits_prefill[i] - logits_ar[i]).abs();
        assert!(
            diff < 1e-3,
            "logit[{}]: prefill={}, ar={}, diff={}",
            i, logits_prefill[i], logits_ar[i], diff
        );
    }
}

#[test]
fn e2e_mhc_always_doubly_stochastic_in_model() {
    let config = make_test_config();
    let model = NanochatModel::new_random(config);
    model.verify_mhc().expect("mHC verification failed on fresh model");
}

#[test]
fn e2e_param_count_nonzero() {
    let config = make_test_config();
    let model = NanochatModel::new_random(config);
    let counts = model.param_count();
    assert!(counts.total > 0, "total params should be > 0");
    assert!(counts.embed > 0, "embed params should be > 0");
    assert!(counts.attn > 0, "attn params should be > 0");
    assert!(counts.ffn > 0, "ffn params should be > 0");
    assert!(counts.mhc > 0, "mhc params should be > 0");
}

// ───────────────────── Generation Tests ─────────────────────

#[test]
fn e2e_generate_greedy() {
    let config = make_test_config();
    let mut engine = InferenceEngine::new_random(config);

    let params = SamplingParams {
        temperature: 0.0, // greedy
        top_k: 0,
        top_p: 1.0,
        max_tokens: 10,
        seed: None,
    };

    let output = engine.generate(&[1, 5, 10], &params);
    assert!(!output.is_empty(), "greedy generation produced no tokens");
    assert!(output.len() <= 10, "exceeded max_tokens");

    for &t in &output {
        assert!((t as usize) < 256, "invalid token: {}", t);
    }
}

#[test]
fn e2e_generate_with_temperature() {
    let config = make_test_config();
    let mut engine = InferenceEngine::new_random(config);

    let params = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        max_tokens: 10,
        seed: Some(42),
    };

    let output = engine.generate(&[1, 2, 3], &params);
    assert!(output.len() <= 10);
    for &t in &output {
        assert!((t as usize) < 256, "invalid token: {}", t);
    }
}

#[test]
fn e2e_generate_deterministic() {
    let config = make_test_config();

    // Same model, same input, same params → same output
    let mut engine_a = InferenceEngine::new_random(config.clone());
    let mut engine_b = InferenceEngine::new_random(config);

    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 5,
        seed: None,
    };

    let output_a = engine_a.generate(&[1, 10], &params);
    let output_b = engine_b.generate(&[1, 10], &params);

    assert_eq!(output_a, output_b, "deterministic generation should be reproducible");
}

#[test]
fn e2e_generate_empty_prompt() {
    let config = make_test_config();
    let mut engine = InferenceEngine::new_random(config);

    let params = SamplingParams::default();
    let output = engine.generate(&[], &params);
    assert!(output.is_empty(), "empty prompt should produce empty output");
}

#[test]
fn e2e_generate_single_token_prompt() {
    let config = make_test_config();
    let mut engine = InferenceEngine::new_random(config);

    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 3,
        ..Default::default()
    };

    let output = engine.generate(&[42], &params);
    assert!(output.len() <= 3);
}

// ───────────────────── Shape Stress Tests ─────────────────────

#[test]
fn e2e_d20_config_forward() {
    let config = ModelConfig::d20();
    let mut model = NanochatModel::new_random(config.clone());

    let logits = model.forward_token(1, 0);
    assert_eq!(logits.len(), config.vocab_size);
    for &l in &logits {
        assert!(l.is_finite());
    }
}

#[test]
fn e2e_reset_caches_allows_reuse() {
    let config = make_test_config();
    let mut model = NanochatModel::new_random(config.clone());

    // First run
    let logits1 = model.forward_sequence(&[1, 2, 3]);
    assert_eq!(logits1.len(), config.vocab_size);

    // Reset and run again
    model.reset_caches();
    let logits2 = model.forward_sequence(&[1, 2, 3]);
    assert_eq!(logits2.len(), config.vocab_size);

    // Should produce identical results after reset
    for i in 0..logits1.len() {
        assert!(
            (logits1[i] - logits2[i]).abs() < 1e-5,
            "logit[{}] differs after reset: {} vs {}",
            i, logits1[i], logits2[i]
        );
    }
}
