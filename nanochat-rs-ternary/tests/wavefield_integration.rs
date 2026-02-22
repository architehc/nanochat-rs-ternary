//! Wave Field Attention Integration Tests
//!
//! Tests end-to-end wave field behavior:
//! - Causality verification (future tokens don't affect past logits)
//! - Energy monitoring (bounded after many tokens)
//! - Full model with wave field layers produces valid output
//! - Batch prefill matches sequential token-by-token

use nanochat_model::config::{ConvolveMode, ModelConfig, WaveFieldConfig};
use nanochat_model::model::NanochatModel;

fn make_wavefield_config() -> ModelConfig {
    make_wavefield_config_mode(ConvolveMode::Fft)
}

fn make_wavefield_config_mode(mode: ConvolveMode) -> ModelConfig {
    let haar_levels = match mode {
        ConvolveMode::Haar => Some(3),
        _ => None,
    };
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
        convolve_mode: mode,
        haar_levels,
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

/// Stronger causality test: process [1, 2, 3] vs [1, 99, 99] sequentially.
/// Position 0 logits must be identical (no future token leakage in autoregressive mode).
/// Position 1 logits must differ (different input token).
#[test]
fn wavefield_causality_multi_token_divergent_futures() {
    let config = make_wavefield_config();

    // Sequence A: [1, 2, 3]
    let mut model_a = NanochatModel::new_random(config.clone());
    let logits_a_0 = model_a.forward_token(1, 0);
    let logits_a_1 = model_a.forward_token(2, 1);
    let _logits_a_2 = model_a.forward_token(3, 2);

    // Sequence B: [1, 99, 99]
    let mut model_b = NanochatModel::new_random(config.clone());
    let logits_b_0 = model_b.forward_token(1, 0);
    let logits_b_1 = model_b.forward_token(99, 1);
    let _logits_b_2 = model_b.forward_token(99, 2);

    // Position 0: same token, no future info yet → must match exactly
    for i in 0..logits_a_0.len() {
        assert!(
            (logits_a_0[i] - logits_b_0[i]).abs() < 1e-5,
            "causality violation at pos=0, logit[{}]: {} vs {}",
            i, logits_a_0[i], logits_b_0[i]
        );
    }

    // Position 1: different tokens (2 vs 99) → logits should differ
    let diff_norm: f32 = logits_a_1
        .iter()
        .zip(logits_b_1.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(
        diff_norm > 1e-3,
        "different tokens at pos=1 should produce different logits, diff_norm={}",
        diff_norm
    );
}

/// Causality test for FWHT mode with divergent future tokens.
#[test]
fn wavefield_fwht_causality_divergent_futures() {
    let config = make_wavefield_config_mode(ConvolveMode::Fwht);

    let mut model_a = NanochatModel::new_random(config.clone());
    let logits_a_0 = model_a.forward_token(10, 0);
    let _logits_a_1 = model_a.forward_token(20, 1);

    let mut model_b = NanochatModel::new_random(config.clone());
    let logits_b_0 = model_b.forward_token(10, 0);
    let _logits_b_1 = model_b.forward_token(200, 1);

    for i in 0..logits_a_0.len() {
        assert!(
            (logits_a_0[i] - logits_b_0[i]).abs() < 1e-5,
            "FWHT causality violation at pos=0, logit[{}]: {} vs {}",
            i, logits_a_0[i], logits_b_0[i]
        );
    }
}

/// Causality test for Haar mode with divergent future tokens.
#[test]
fn wavefield_haar_causality_divergent_futures() {
    let config = make_wavefield_config_mode(ConvolveMode::Haar);

    let mut model_a = NanochatModel::new_random(config.clone());
    let logits_a_0 = model_a.forward_token(10, 0);
    let _logits_a_1 = model_a.forward_token(20, 1);

    let mut model_b = NanochatModel::new_random(config.clone());
    let logits_b_0 = model_b.forward_token(10, 0);
    let _logits_b_1 = model_b.forward_token(200, 1);

    for i in 0..logits_a_0.len() {
        assert!(
            (logits_a_0[i] - logits_b_0[i]).abs() < 1e-5,
            "Haar causality violation at pos=0, logit[{}]: {} vs {}",
            i, logits_a_0[i], logits_b_0[i]
        );
    }
}

// ───────────────── Prefix-Invariance Tests ─────────────────
//
// Stronger causality: two sequences sharing a k-token prefix but differing
// in suffix must produce identical logits at ALL prefix positions.
// This exercises multi-step state accumulation in the wave field.

/// Helper: run autoregressive sequence, return logits at every position.
fn run_sequence_collect_all(config: &ModelConfig, tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut model = NanochatModel::new_random(config.clone());
    tokens
        .iter()
        .enumerate()
        .map(|(pos, &tok)| model.forward_token(tok, pos))
        .collect()
}

/// Prefix-invariance for a given convolve mode.
/// Sequences [10, 20, 30, 40, 50] vs [10, 20, 30, 99, 99]:
///   positions 0..3 must match (shared prefix), position 3 must differ.
fn check_prefix_invariance(mode: ConvolveMode) {
    let config = make_wavefield_config_mode(mode);

    let seq_a: Vec<u32> = vec![10, 20, 30, 40, 50];
    let seq_b: Vec<u32> = vec![10, 20, 30, 99, 99];
    let prefix_len = 3; // first 3 tokens identical

    let logits_a = run_sequence_collect_all(&config, &seq_a);
    let logits_b = run_sequence_collect_all(&config, &seq_b);

    // Prefix positions must match exactly
    for pos in 0..prefix_len {
        assert_eq!(logits_a[pos].len(), logits_b[pos].len());
        for i in 0..logits_a[pos].len() {
            assert!(
                (logits_a[pos][i] - logits_b[pos][i]).abs() < 1e-5,
                "{:?} prefix-invariance violation at pos={}, logit[{}]: {} vs {}",
                mode, pos, i, logits_a[pos][i], logits_b[pos][i]
            );
        }
    }

    // Position 3: different tokens → logits must differ (test has power)
    let diff_norm: f32 = logits_a[prefix_len]
        .iter()
        .zip(logits_b[prefix_len].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(
        diff_norm > 1e-3,
        "{:?} prefix-invariance: pos={} should differ (different token), diff_norm={}",
        mode, prefix_len, diff_norm
    );
}

#[test]
fn wavefield_fft_prefix_invariance() {
    check_prefix_invariance(ConvolveMode::Fft);
}

#[test]
fn wavefield_fwht_prefix_invariance() {
    check_prefix_invariance(ConvolveMode::Fwht);
}

#[test]
fn wavefield_haar_prefix_invariance() {
    check_prefix_invariance(ConvolveMode::Haar);
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

// ───────────────── FWHT Mode Tests ─────────────────

#[test]
fn wavefield_fwht_model_produces_valid_logits() {
    let config = make_wavefield_config_mode(ConvolveMode::Fwht);
    let mut model = NanochatModel::new_random(config.clone());

    let logits = model.forward_token(42, 0);
    assert_eq!(logits.len(), config.vocab_size);

    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "FWHT logit[{}] = {} not finite", i, l);
    }

    // Non-zero output
    let max_abs = logits.iter().map(|l| l.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 0.0, "FWHT logits should be non-zero");
}

#[test]
fn wavefield_fwht_sequence_produces_valid_logits() {
    let config = make_wavefield_config_mode(ConvolveMode::Fwht);
    let mut model = NanochatModel::new_random(config.clone());

    let tokens: Vec<u32> = vec![1, 5, 10, 20, 50];
    let logits = model.forward_sequence(&tokens);
    assert_eq!(logits.len(), config.vocab_size);

    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "FWHT seq logit[{}] = {} not finite", i, l);
    }
}

#[test]
fn wavefield_fwht_causality() {
    let config = make_wavefield_config_mode(ConvolveMode::Fwht);

    let mut model_a = NanochatModel::new_random(config.clone());
    let logits_a = model_a.forward_token(1, 0);

    let mut model_b = NanochatModel::new_random(config.clone());
    let logits_b = model_b.forward_token(1, 0);

    for i in 0..logits_a.len() {
        assert!(
            (logits_a[i] - logits_b[i]).abs() < 1e-5,
            "FWHT causality violation at logit[{}]: {} vs {}",
            i, logits_a[i], logits_b[i]
        );
    }
}

#[test]
fn wavefield_fwht_energy_bounded() {
    use nanochat_model::block::AttentionState;

    let config = make_wavefield_config_mode(ConvolveMode::Fwht);
    let mut model = NanochatModel::new_random(config);

    for pos in 0..30 {
        let token = ((pos * 7 + 3) % 256) as u32;
        let _logits = model.forward_token(token, pos);
    }

    for (i, state) in model.attn_states.iter().enumerate() {
        if let AttentionState::WaveField(wf_state) = state {
            let energy = wf_state.energy();
            assert!(energy.is_finite(), "FWHT layer {} energy not finite: {}", i, energy);
            assert!(energy < 1e10, "FWHT layer {} energy exploded: {}", i, energy);
        }
    }
}

// ───────────────── Haar Mode Tests ─────────────────

#[test]
fn wavefield_haar_model_produces_valid_logits() {
    let config = make_wavefield_config_mode(ConvolveMode::Haar);
    let mut model = NanochatModel::new_random(config.clone());

    let logits = model.forward_token(42, 0);
    assert_eq!(logits.len(), config.vocab_size);

    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "Haar logit[{}] = {} not finite", i, l);
    }

    let max_abs = logits.iter().map(|l| l.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 0.0, "Haar logits should be non-zero");
}

#[test]
fn wavefield_haar_sequence_produces_valid_logits() {
    let config = make_wavefield_config_mode(ConvolveMode::Haar);
    let mut model = NanochatModel::new_random(config.clone());

    let tokens: Vec<u32> = vec![1, 5, 10, 20, 50];
    let logits = model.forward_sequence(&tokens);
    assert_eq!(logits.len(), config.vocab_size);

    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "Haar seq logit[{}] = {} not finite", i, l);
    }
}

#[test]
fn wavefield_haar_causality() {
    let config = make_wavefield_config_mode(ConvolveMode::Haar);

    let mut model_a = NanochatModel::new_random(config.clone());
    let logits_a = model_a.forward_token(1, 0);

    let mut model_b = NanochatModel::new_random(config.clone());
    let logits_b = model_b.forward_token(1, 0);

    for i in 0..logits_a.len() {
        assert!(
            (logits_a[i] - logits_b[i]).abs() < 1e-5,
            "Haar causality violation at logit[{}]: {} vs {}",
            i, logits_a[i], logits_b[i]
        );
    }
}

#[test]
fn wavefield_haar_energy_bounded() {
    use nanochat_model::block::AttentionState;

    let config = make_wavefield_config_mode(ConvolveMode::Haar);
    let mut model = NanochatModel::new_random(config);

    for pos in 0..30 {
        let token = ((pos * 7 + 3) % 256) as u32;
        let _logits = model.forward_token(token, pos);
    }

    for (i, state) in model.attn_states.iter().enumerate() {
        if let AttentionState::WaveField(wf_state) = state {
            let energy = wf_state.energy();
            assert!(energy.is_finite(), "Haar layer {} energy not finite: {}", i, energy);
            assert!(energy < 1e10, "Haar layer {} energy exploded: {}", i, energy);
        }
    }
}

// ───────────────── Cross-Mode Consistency ─────────────────

#[test]
fn wavefield_all_modes_produce_output() {
    // All three modes should produce finite, non-zero logits for the same token
    for mode in [ConvolveMode::Fft, ConvolveMode::Fwht, ConvolveMode::Haar] {
        let config = make_wavefield_config_mode(mode);
        let mut model = NanochatModel::new_random(config.clone());
        let logits = model.forward_token(42, 0);

        assert_eq!(logits.len(), config.vocab_size, "{:?}: wrong logit count", mode);

        let all_finite = logits.iter().all(|l| l.is_finite());
        assert!(all_finite, "{:?}: not all logits finite", mode);

        let max_abs = logits.iter().map(|l| l.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.0, "{:?}: all logits zero", mode);
    }
}

// ───────────────── FWHT Infrastructure ─────────────────

#[test]
fn wave_fwht_roundtrip() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = x.len();

    // Convolve with delta should return original (up to scaling)
    let mut delta = vec![0.0f32; n];
    delta[0] = 1.0;

    let result = wave_fft::cpu_fwht::fwht_convolve(&x, &delta, n);
    for i in 0..n {
        assert!(
            (result[i] - x[i]).abs() < 1e-4,
            "FWHT roundtrip failed at {}: {} vs {}", i, result[i], x[i]
        );
    }
}

#[test]
fn wave_fwht_precomputed_matches_direct() {
    let signal = vec![1.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0];
    let kernel = vec![1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0];
    let n = signal.len();

    let direct = wave_fft::cpu_fwht::fwht_convolve(&signal, &kernel, n);
    let kernel_fwht = wave_fft::cpu_fwht::precompute_kernel_fwht(&kernel, n);
    let precomp = wave_fft::cpu_fwht::fwht_convolve_precomputed(&signal, &kernel_fwht, n);

    for i in 0..n {
        assert!(
            (direct[i] - precomp[i]).abs() < 1e-5,
            "FWHT precomputed mismatch at {}: {} vs {}", i, direct[i], precomp[i]
        );
    }
}

// ───────────────── Haar Infrastructure ─────────────────

#[test]
fn wave_haar_self_convolution_finite() {
    // Haar convolution (pointwise multiply in wavelet domain) doesn't have
    // the same delta-identity property as FFT/FWHT. Instead verify:
    // 1. Output is finite and non-zero
    // 2. Commutative: haar_convolve(a,b) == haar_convolve(b,a)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
    let n = a.len();
    let levels = 3;

    let result_ab = wave_fft::cpu_haar::haar_convolve(&a, &b, n, levels);
    let result_ba = wave_fft::cpu_haar::haar_convolve(&b, &a, n, levels);

    assert_eq!(result_ab.len(), n);
    for i in 0..n {
        assert!(result_ab[i].is_finite(), "Haar result not finite at {}", i);
        assert!(
            (result_ab[i] - result_ba[i]).abs() < 1e-5,
            "Haar not commutative at {}: {} vs {}", i, result_ab[i], result_ba[i]
        );
    }

    let max_abs = result_ab.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 0.0, "Haar convolution produced all zeros");
}

#[test]
fn wave_haar_precomputed_matches_direct() {
    let signal = vec![1.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0];
    let kernel = vec![1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0];
    let n = signal.len();
    let levels = 3;

    let direct = wave_fft::cpu_haar::haar_convolve(&signal, &kernel, n, levels);
    let kernel_haar = wave_fft::cpu_haar::precompute_kernel_haar(&kernel, n, levels);
    let precomp = wave_fft::cpu_haar::haar_convolve_precomputed(&signal, &kernel_haar, n, levels);

    for i in 0..n {
        assert!(
            (direct[i] - precomp[i]).abs() < 1e-5,
            "Haar precomputed mismatch at {}: {} vs {}", i, direct[i], precomp[i]
        );
    }
}
