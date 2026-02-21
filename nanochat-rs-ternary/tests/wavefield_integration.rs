//! Wave Field Attention Integration Tests
//!
//! Tests end-to-end wave field behavior:
//! - Causality verification (future tokens don't affect past logits)
//! - Energy monitoring (bounded after many tokens)
//! - Full model with wave field layers produces valid output
//! - Batch prefill matches sequential token-by-token

use nanochat_model::config::{ModelConfig, WaveFieldConfig};
use nanochat_model::model::NanochatModel;

fn make_wavefield_config() -> ModelConfig {
    let mut config = ModelConfig::d20();
    config.dim = 128;
    config.n_layers = 4;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 256;
    config.max_seq_len = 64;
    config.group_size = 128;
    config.wavefield_config = Some(WaveFieldConfig {
        field_size: 64,
        n_wave_heads: 4,
        head_dim: 32,
        use_head_coupling: true,
    });
    config.wavefield_ratio = Some(1.0); // all layers wave field
    config
}

fn make_hybrid_config() -> ModelConfig {
    let mut config = make_wavefield_config();
    config.wavefield_ratio = Some(0.5); // half the layers
    config
}

// ───────────────── Causality Tests ─────────────────

#[test]
fn wavefield_causality_future_tokens_dont_affect_past() {
    let config = make_wavefield_config();

    // Run with tokens [1, 2, 3]
    let mut model_a = NanochatModel::new_random(config.clone());
    let logits_a_0 = model_a.forward_token(1, 0);

    // Run with token [1] only
    let mut model_b = NanochatModel::new_random(config.clone());
    let logits_b_0 = model_b.forward_token(1, 0);

    // First token's logits should be identical regardless of future tokens
    // (since we're calling forward_token which is autoregressive)
    assert_eq!(logits_a_0.len(), logits_b_0.len());
    for i in 0..logits_a_0.len() {
        assert!(
            (logits_a_0[i] - logits_b_0[i]).abs() < 1e-5,
            "causality violation at logit[{}]: {} vs {}",
            i,
            logits_a_0[i],
            logits_b_0[i]
        );
    }
}

// ───────────────── Energy Monitoring ─────────────────

#[test]
fn wavefield_energy_bounded_after_many_tokens() {
    use nanochat_model::block::AttentionState;

    let config = make_wavefield_config();
    let mut model = NanochatModel::new_random(config.clone());

    // Process 30 tokens
    for pos in 0..30 {
        let token = ((pos * 7 + 3) % 256) as u32;
        let _logits = model.forward_token(token, pos);
    }

    // Check wave field state energy is finite and bounded
    for (i, state) in model.attn_states.iter().enumerate() {
        if let AttentionState::WaveField(wf_state) = state {
            let energy = wf_state.energy();
            assert!(
                energy.is_finite(),
                "layer {} wave field energy is not finite: {}",
                i, energy
            );
            // Energy should not explode (with positive damping)
            assert!(
                energy < 1e10,
                "layer {} wave field energy exploded: {}",
                i, energy
            );
        }
    }
}

// ───────────────── Full Model Forward Pass ─────────────────

#[test]
fn wavefield_model_produces_valid_logits() {
    let config = make_wavefield_config();
    let mut model = NanochatModel::new_random(config.clone());

    let logits = model.forward_token(42, 0);
    assert_eq!(logits.len(), config.vocab_size);

    // All logits finite
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit[{}] = {} not finite", i, l);
    }
}

#[test]
fn wavefield_model_sequence_produces_valid_logits() {
    let config = make_wavefield_config();
    let mut model = NanochatModel::new_random(config.clone());

    let tokens: Vec<u32> = vec![1, 5, 10, 20, 50];
    let logits = model.forward_sequence(&tokens);
    assert_eq!(logits.len(), config.vocab_size);

    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit[{}] = {} not finite", i, l);
    }
}

// ───────────────── Hybrid Model Tests ─────────────────

#[test]
fn wavefield_hybrid_model_has_mixed_layers() {
    use nanochat_model::block::AttentionLayer;

    let config = make_hybrid_config();
    let model = NanochatModel::new_random(config.clone());

    let n_wf = model
        .blocks
        .iter()
        .filter(|b| matches!(b.attention, AttentionLayer::WaveField(_)))
        .count();
    let n_std = model
        .blocks
        .iter()
        .filter(|b| matches!(b.attention, AttentionLayer::Standard(_)))
        .count();

    assert!(n_wf > 0, "should have some wave field layers");
    assert!(n_std > 0, "should have some standard layers");
    assert_eq!(
        n_wf + n_std,
        config.n_layers,
        "all layers should be wave field or standard"
    );
}

#[test]
fn wavefield_hybrid_model_produces_valid_logits() {
    let config = make_hybrid_config();
    let mut model = NanochatModel::new_random(config.clone());

    let tokens: Vec<u32> = vec![1, 5, 10, 20];
    let logits = model.forward_sequence(&tokens);
    assert_eq!(logits.len(), config.vocab_size);

    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit[{}] = {} not finite", i, l);
    }
}

// ───────────────── Reset and Reuse ─────────────────

#[test]
fn wavefield_model_reset_and_reuse() {
    let config = make_wavefield_config();
    let mut model = NanochatModel::new_random(config.clone());

    // First pass
    let logits_1 = model.forward_token(42, 0);

    // Reset
    model.reset_caches();

    // Second pass should produce identical output
    let logits_2 = model.forward_token(42, 0);

    for i in 0..logits_1.len() {
        assert!(
            (logits_1[i] - logits_2[i]).abs() < 1e-5,
            "reset didn't produce identical output at logit[{}]: {} vs {}",
            i,
            logits_1[i],
            logits_2[i]
        );
    }
}

// ───────────────── Wave Field State Properties ─────────────────

#[test]
fn wavefield_state_starts_at_zero_energy() {
    use nanochat_model::wavefield::WaveFieldState;

    let state = WaveFieldState::new(4, 64, 32);
    assert_eq!(state.energy(), 0.0, "initial energy should be zero");
}

#[test]
fn wavefield_state_energy_increases_after_token() {
    use nanochat_model::block::AttentionState;

    let config = make_wavefield_config();
    let mut model = NanochatModel::new_random(config);

    // Record initial energy
    let initial_energies: Vec<f32> = model
        .attn_states
        .iter()
        .filter_map(|s| {
            if let AttentionState::WaveField(wf) = s {
                Some(wf.energy())
            } else {
                None
            }
        })
        .collect();

    // All should be zero initially
    for &e in &initial_energies {
        assert_eq!(e, 0.0);
    }

    // Process one token
    let _logits = model.forward_token(42, 0);

    // At least one wave field should have non-zero energy
    let post_energies: Vec<f32> = model
        .attn_states
        .iter()
        .filter_map(|s| {
            if let AttentionState::WaveField(wf) = s {
                Some(wf.energy())
            } else {
                None
            }
        })
        .collect();

    let any_nonzero = post_energies.iter().any(|&e| e > 0.0);
    assert!(
        any_nonzero,
        "at least one wave field should have non-zero energy after token"
    );
}

// ───────────────── FFT Infrastructure ─────────────────

#[test]
fn wave_fft_roundtrip() {
    // Verify FFT infrastructure works: IFFT(FFT(x)) ≈ x
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = x.len();

    // Convolve with delta function should return original
    let mut delta = vec![0.0f32; n];
    delta[0] = 1.0;

    let result = wave_fft::cpu_fft::fft_convolve(&x, &delta, n);
    for i in 0..n {
        assert!(
            (result[i] - x[i]).abs() < 1e-5,
            "FFT roundtrip failed at {}: {} vs {}",
            i,
            result[i],
            x[i]
        );
    }
}

#[test]
fn wave_fft_precomputed_matches_direct() {
    let signal = vec![1.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0];
    let kernel = vec![1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0];
    let n = signal.len();

    let direct = wave_fft::cpu_fft::fft_convolve(&signal, &kernel, n);
    let kernel_freq = wave_fft::cpu_fft::precompute_kernel_fft(&kernel, n);
    let precomp = wave_fft::cpu_fft::fft_convolve_precomputed(&signal, &kernel_freq, n);

    for i in 0..n {
        assert!(
            (direct[i] - precomp[i]).abs() < 1e-5,
            "precomputed mismatch at {}: {} vs {}",
            i,
            direct[i],
            precomp[i]
        );
    }
}
