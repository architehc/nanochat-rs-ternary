//! mHC Overhead Benchmark — verify mHC adds <0.001% overhead.
//!
//! Measures the cost of mHC operations (h_res, h_pre, h_post, prepare_input,
//! apply, expand/collapse) vs a single ternary GEMV at production sizes.
//!
//! At 4096×11008, a single GEMV takes ~microseconds. mHC operations on
//! N=4 involve 24 multiplies + 24 adds for h_res (32 FLOPs per layer) vs
//! ~90M FLOPs for the GEMV. That's <0.00004% overhead.

use criterion::{criterion_group, criterion_main, Criterion};

use mhc_lite::{
    composite_amax_gain, verify_doubly_stochastic, verify_doubly_stochastic_2x2, MhcLiteN2,
    MhcLiteN4,
};
use ternary_core::planar::PlanarWeights;
use ternary_kernels::cpu;

fn gen_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let v = (i as u32).wrapping_mul(2654435761) >> 16;
            (v % 200) as f32 / 100.0 - 1.0
        })
        .collect()
}

fn gen_activations(cols: usize) -> Vec<i8> {
    (0..cols)
        .map(|i| (((i * 37 + 13) % 200) as i32 - 100) as i8)
        .collect()
}

fn bench_mhc_n2_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mhc_n2");

    let mhc = MhcLiteN2::from_weights(2.5, [0.3, -0.7], [0.5, 0.5], [-0.2, 0.8], [0.5, 0.5]);
    let dim = 4096;
    let x_single: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let x_expanded = MhcLiteN2::expand_input(&x_single, dim);

    group.bench_function("h_res", |b| {
        b.iter(|| mhc.h_res());
    });

    group.bench_function("h_pre", |b| {
        b.iter(|| mhc.h_pre());
    });

    group.bench_function("h_post", |b| {
        b.iter(|| mhc.h_post());
    });

    group.bench_function("prepare_input", |b| {
        b.iter(|| mhc.prepare_input(&x_expanded, dim));
    });

    group.bench_function("apply", |b| {
        let layer_out = x_single.clone();
        b.iter(|| mhc.apply(&x_expanded, &layer_out, dim));
    });

    group.bench_function("expand_input", |b| {
        b.iter(|| MhcLiteN2::expand_input(&x_single, dim));
    });

    group.bench_function("collapse_output", |b| {
        b.iter(|| MhcLiteN2::collapse_output(&x_expanded, dim));
    });

    group.finish();
}

fn bench_mhc_n4_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mhc_n4");

    let mut logits = [0.0f32; 24];
    #[allow(clippy::needless_range_loop)] // Index needed for deterministic data generation
    for i in 0..24 {
        logits[i] = (i as f32 * 0.7).sin() * 3.0;
    }
    let mhc = MhcLiteN4::from_weights(logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4]);
    let dim = 4096;
    let x_single: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let x_expanded = MhcLiteN4::expand_input(&x_single, dim);

    group.bench_function("h_res", |b| {
        b.iter(|| mhc.h_res());
    });

    group.bench_function("h_pre", |b| {
        b.iter(|| mhc.h_pre());
    });

    group.bench_function("h_post", |b| {
        b.iter(|| mhc.h_post());
    });

    group.bench_function("prepare_input", |b| {
        b.iter(|| mhc.prepare_input(&x_expanded, dim));
    });

    group.bench_function("apply", |b| {
        let layer_out = x_single.clone();
        b.iter(|| mhc.apply(&x_expanded, &layer_out, dim));
    });

    group.bench_function("expand_input", |b| {
        b.iter(|| MhcLiteN4::expand_input(&x_single, dim));
    });

    group.bench_function("collapse_output", |b| {
        b.iter(|| MhcLiteN4::collapse_output(&x_expanded, dim));
    });

    group.finish();
}

fn bench_mhc_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("mhc_verify");

    // N=2 DS check
    let mhc2 = MhcLiteN2::new_identity();
    let h2 = mhc2.h_res();
    group.bench_function("verify_ds_2x2", |b| {
        b.iter(|| verify_doubly_stochastic_2x2(&h2, 1e-6));
    });

    // N=4 DS check
    let mhc4 = MhcLiteN4::new_identity();
    let h4 = mhc4.h_res();
    group.bench_function("verify_ds_4x4", |b| {
        b.iter(|| verify_doubly_stochastic(&h4, 1e-5));
    });

    // Composite gain (64 layers)
    let matrices: Vec<_> = (0..64)
        .map(|seed| {
            let mut logits = [0.0f32; 24];
            #[allow(clippy::needless_range_loop)] // Index needed for deterministic data generation
            for i in 0..24 {
                logits[i] = ((seed * 24 + i) as f32 * 0.7).sin() * 3.0;
            }
            let mhc = MhcLiteN4::from_weights(logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4]);
            mhc.h_res()
        })
        .collect();

    group.bench_function("composite_amax_gain_64", |b| {
        b.iter(|| composite_amax_gain(&matrices));
    });

    group.finish();
}

fn bench_mhc_vs_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("mhc_vs_gemv_overhead");

    // Production-sized GEMV for comparison
    let m = 4096;
    let k = 11008;
    let w = gen_weights(m, k);
    let pw = PlanarWeights::from_row_major(&w, m, k, 128);
    let x = gen_activations(k);
    let mut y = vec![0.0f32; m];
    let act_scale = 1.0 / 127.0;

    group.bench_function("gemv_4096x11008_ffi", |b| {
        b.iter(|| cpu::gemv(&pw, &x, act_scale, &mut y));
    });

    // mHC N=4 full layer overhead (h_res + prepare + apply)
    let dim = 4096;
    let mut logits = [0.0f32; 24];
    #[allow(clippy::needless_range_loop)] // Index needed for deterministic data generation
    for i in 0..24 {
        logits[i] = (i as f32 * 0.7).sin() * 3.0;
    }
    let mhc = MhcLiteN4::from_weights(logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4]);
    let x_single: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let x_expanded = MhcLiteN4::expand_input(&x_single, dim);

    group.bench_function("mhc_n4_full_layer", |b| {
        b.iter(|| {
            let _h = mhc.h_res();
            let _prepared = mhc.prepare_input(&x_expanded, dim);
            let _applied = mhc.apply(&x_expanded, &x_single, dim);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mhc_n2_operations,
    bench_mhc_n4_operations,
    bench_mhc_verification,
    bench_mhc_vs_gemv,
);
criterion_main!(benches);
