//! mHC Property Tests — verify doubly stochastic invariants hold for
//! every possible parameterization of mHC-lite.
//!
//! Key property: BvN parameterization produces EXACT doubly stochastic matrices.
//! The composite gain of any sequence of such matrices is bounded by 1.0.

use mhc_lite::{
    verify_doubly_stochastic, verify_doubly_stochastic_2x2, composite_amax_gain,
    MhcLiteN2, MhcLiteN4,
};

// ───────────────────── N=2 Property Tests ─────────────────────

#[test]
fn n2_identity_is_doubly_stochastic() {
    let mhc = MhcLiteN2::new_identity();
    let h = mhc.h_res();
    verify_doubly_stochastic_2x2(&h, 1e-6).unwrap();
}

#[test]
fn n2_1000_random_logits_all_doubly_stochastic() {
    for seed in 0..1000 {
        let alpha_logit = ((seed as f32) * 0.731).sin() * 10.0;
        let mhc = MhcLiteN2::from_weights(
            alpha_logit,
            [0.0; 2],
            [0.5; 2],
            [0.0; 2],
            [0.5; 2],
        );
        let h = mhc.h_res();
        verify_doubly_stochastic_2x2(&h, 1e-6)
            .unwrap_or_else(|e| panic!("N2 seed {}: {}", seed, e));
    }
}

#[test]
fn n2_extreme_logits_still_doubly_stochastic() {
    for &logit in &[-100.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 100.0] {
        let mhc = MhcLiteN2::from_weights(
            logit,
            [logit; 2],
            [logit * 0.1; 2],
            [logit * -0.5; 2],
            [logit * 0.3; 2],
        );
        let h = mhc.h_res();
        verify_doubly_stochastic_2x2(&h, 1e-5)
            .unwrap_or_else(|e| panic!("N2 logit={}: {}", logit, e));
    }
}

#[test]
fn n2_composite_gain_bounded_64_layers() {
    let mut matrices_2x2 = Vec::new();
    // Build 4x4 embedding of 2x2 matrices for composite_amax_gain
    // (which operates on 4x4 matrices)
    let mut matrices_4x4 = Vec::new();

    for seed in 0..64 {
        let alpha = ((seed as f32) * 1.37).sin() * 5.0;
        let mhc = MhcLiteN2::from_weights(alpha, [0.0; 2], [0.5; 2], [0.0; 2], [0.5; 2]);
        let h = mhc.h_res();
        matrices_2x2.push(h);

        // Embed 2x2 into 4x4 block-diagonal for composite_amax_gain
        let mut m4 = [[0.0f32; 4]; 4];
        m4[0][0] = h[0][0];
        m4[0][1] = h[0][1];
        m4[1][0] = h[1][0];
        m4[1][1] = h[1][1];
        m4[2][2] = 1.0; // identity block for unused dims
        m4[3][3] = 1.0;
        matrices_4x4.push(m4);
    }

    let gain = composite_amax_gain(&matrices_4x4);
    assert!(
        gain <= 1.0 + 1e-4,
        "N2 composite gain {} exceeds bound after 64 layers",
        gain
    );
}

#[test]
fn n2_expand_collapse_preserves_data() {
    let dim = 32;
    let x: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
    let expanded = MhcLiteN2::expand_input(&x, dim);
    assert_eq!(expanded.len(), 2 * dim);
    let collapsed = MhcLiteN2::collapse_output(&expanded, dim);
    assert_eq!(collapsed.len(), dim);

    for i in 0..dim {
        assert!(
            (x[i] - collapsed[i]).abs() < 1e-6,
            "expand/collapse mismatch at {}: {} vs {}",
            i, x[i], collapsed[i]
        );
    }
}

#[test]
fn n2_h_pre_h_post_are_nonnegative() {
    for seed in 0..100 {
        let logit = ((seed as f32) * 2.71).sin() * 10.0;
        let mhc = MhcLiteN2::from_weights(
            logit,
            [logit * 0.3, logit * -0.7],
            [logit * 0.1, logit * -0.2],
            [logit * -0.5, logit * 0.9],
            [logit * 0.4, logit * -0.1],
        );
        let h_pre = mhc.h_pre();
        let h_post = mhc.h_post();

        for i in 0..2 {
            assert!(h_pre[i] >= 0.0, "h_pre[{}] = {} negative at seed {}", i, h_pre[i], seed);
            assert!(h_post[i] >= 0.0, "h_post[{}] = {} negative at seed {}", i, h_post[i], seed);
        }
    }
}

// ───────────────────── N=4 Property Tests ─────────────────────

#[test]
fn n4_identity_is_doubly_stochastic() {
    let mhc = MhcLiteN4::new_identity();
    let h = mhc.h_res();
    verify_doubly_stochastic(&h, 1e-5).unwrap();
}

#[test]
fn n4_1000_random_logits_all_doubly_stochastic() {
    for seed in 0..1000 {
        let mut logits = [0.0f32; 24];
        for i in 0..24 {
            logits[i] = ((seed * 24 + i) as f32 * 0.7).sin() * 5.0;
        }
        let mhc = MhcLiteN4::from_weights(
            logits,
            [0.0; 4],
            [0.5; 4],
            [0.0; 4],
            [0.5; 4],
        );
        let h = mhc.h_res();
        verify_doubly_stochastic(&h, 1e-5)
            .unwrap_or_else(|e| panic!("N4 seed {}: {}", seed, e));
    }
}

#[test]
fn n4_extreme_logits_still_doubly_stochastic() {
    // Large positive logits (one perm dominates)
    let mut logits = [-100.0f32; 24];
    logits[5] = 100.0; // one permutation dominates
    let mhc = MhcLiteN4::from_weights(logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4]);
    verify_doubly_stochastic(&mhc.h_res(), 1e-5).unwrap();

    // Uniform logits (equal mix)
    let logits = [0.0f32; 24];
    let mhc = MhcLiteN4::from_weights(logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4]);
    verify_doubly_stochastic(&mhc.h_res(), 1e-5).unwrap();

    // All large positive (also uniform after softmax)
    let logits = [100.0f32; 24];
    let mhc = MhcLiteN4::from_weights(logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4]);
    verify_doubly_stochastic(&mhc.h_res(), 1e-5).unwrap();
}

#[test]
fn n4_composite_gain_bounded_64_layers() {
    let mut matrices = Vec::new();
    for seed in 0..64 {
        let mut logits = [0.0f32; 24];
        for i in 0..24 {
            logits[i] = ((seed * 24 + i) as f32 * 0.7).sin() * 3.0;
        }
        let mhc = MhcLiteN4::from_weights(logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4]);
        matrices.push(mhc.h_res());
    }

    let gain = composite_amax_gain(&matrices);
    assert!(
        gain <= 1.0 + 1e-4,
        "N4 composite gain {} exceeds bound after 64 layers",
        gain
    );
}

#[test]
fn n4_composite_gain_bounded_256_layers() {
    // Stress test: 256 random layers
    let mut matrices = Vec::new();
    for seed in 0..256 {
        let mut logits = [0.0f32; 24];
        for i in 0..24 {
            logits[i] = ((seed * 24 + i) as f32 * 1.13).cos() * 8.0;
        }
        let mhc = MhcLiteN4::from_weights(logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4]);
        matrices.push(mhc.h_res());
    }

    let gain = composite_amax_gain(&matrices);
    assert!(
        gain <= 1.0 + 1e-3,
        "N4 composite gain {} exceeds bound after 256 layers",
        gain
    );
}

#[test]
fn n4_expand_collapse_preserves_data() {
    let dim = 32;
    let x: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
    let expanded = MhcLiteN4::expand_input(&x, dim);
    assert_eq!(expanded.len(), 4 * dim);
    let collapsed = MhcLiteN4::collapse_output(&expanded, dim);
    assert_eq!(collapsed.len(), dim);

    for i in 0..dim {
        assert!(
            (x[i] - collapsed[i]).abs() < 1e-6,
            "expand/collapse mismatch at {}: {} vs {}",
            i, x[i], collapsed[i]
        );
    }
}

#[test]
fn n4_h_pre_h_post_are_nonnegative() {
    for seed in 0..100 {
        let mut logits = [0.0f32; 24];
        for i in 0..24 {
            logits[i] = ((seed * 24 + i) as f32 * 0.37).sin() * 10.0;
        }
        let pre_logits: [f32; 4] = std::array::from_fn(|i| ((seed * 4 + i) as f32 * 1.7).sin() * 5.0);
        let pre_bias: [f32; 4] = std::array::from_fn(|i| ((seed * 4 + i) as f32 * 2.3).cos() * 3.0);
        let post_logits: [f32; 4] = std::array::from_fn(|i| ((seed * 4 + i) as f32 * 0.9).sin() * 7.0);
        let post_bias: [f32; 4] = std::array::from_fn(|i| ((seed * 4 + i) as f32 * 3.1).cos() * 4.0);

        let mhc = MhcLiteN4::from_weights(logits, pre_logits, pre_bias, post_logits, post_bias);
        let h_pre = mhc.h_pre();
        let h_post = mhc.h_post();

        for i in 0..4 {
            assert!(h_pre[i] >= 0.0, "h_pre[{}] = {} negative at seed {}", i, h_pre[i], seed);
            assert!(h_post[i] >= 0.0, "h_post[{}] = {} negative at seed {}", i, h_post[i], seed);
        }
    }
}

// ───────────────────── Serialization Property Tests ─────────────────────

#[test]
fn n2_serialize_roundtrip_preserves_values() {
    for seed in 0..100 {
        let alpha = ((seed as f32) * 1.37).sin() * 10.0;
        let pre = [((seed as f32) * 0.3).sin(), ((seed as f32) * 0.7).cos()];
        let pre_b = [((seed as f32) * 1.1).sin(), ((seed as f32) * 1.3).cos()];
        let post = [((seed as f32) * 2.1).sin(), ((seed as f32) * 2.3).cos()];
        let post_b = [((seed as f32) * 3.1).sin(), ((seed as f32) * 3.3).cos()];

        let mhc = MhcLiteN2::from_weights(alpha, pre, pre_b, post, post_b);
        let bytes = mhc.to_bytes();
        let mhc2 = MhcLiteN2::from_bytes(&bytes).unwrap();

        assert!((mhc.alpha_logit - mhc2.alpha_logit).abs() < 1e-7);
        for i in 0..2 {
            assert!((mhc.pre_logits[i] - mhc2.pre_logits[i]).abs() < 1e-7);
            assert!((mhc.pre_bias[i] - mhc2.pre_bias[i]).abs() < 1e-7);
            assert!((mhc.post_logits[i] - mhc2.post_logits[i]).abs() < 1e-7);
            assert!((mhc.post_bias[i] - mhc2.post_bias[i]).abs() < 1e-7);
        }
    }
}

#[test]
fn n4_serialize_roundtrip_preserves_values() {
    for seed in 0..100 {
        let mut res_logits = [0.0f32; 24];
        for i in 0..24 {
            res_logits[i] = ((seed * 24 + i) as f32 * 0.7).sin() * 5.0;
        }
        let pre_logits: [f32; 4] = std::array::from_fn(|i| ((seed * 4 + i) as f32).sin());
        let pre_bias: [f32; 4] = std::array::from_fn(|i| ((seed * 4 + i) as f32).cos());
        let post_logits: [f32; 4] = std::array::from_fn(|i| ((seed * 4 + i + 10) as f32).sin());
        let post_bias: [f32; 4] = std::array::from_fn(|i| ((seed * 4 + i + 10) as f32).cos());

        let mhc = MhcLiteN4::from_weights(res_logits, pre_logits, pre_bias, post_logits, post_bias);
        let bytes = mhc.to_bytes();
        let mhc2 = MhcLiteN4::from_bytes(&bytes).unwrap();

        for i in 0..24 {
            assert!((mhc.res_logits[i] - mhc2.res_logits[i]).abs() < 1e-7);
        }
        for i in 0..4 {
            assert!((mhc.pre_logits[i] - mhc2.pre_logits[i]).abs() < 1e-7);
        }
    }
}
