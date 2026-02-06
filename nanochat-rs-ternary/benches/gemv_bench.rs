//! GEMV Benchmarks — criterion benchmarks for all kernel paths.
//!
//! Measures GOPS (giga operations per second) at various matrix dimensions.
//! Expected results (single-thread, Zen4 EPYC):
//!   Shape          VPERMW    Scalar-ref
//!   2048²          ~30 GOPS  ~18 GOPS
//!   4096²          ~25 GOPS  ~18 GOPS
//!   4096×11008     ~20 GOPS  ~16 GOPS
//!   11008×4096     ~20 GOPS  ~18 GOPS

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use ternary_core::planar::PlanarWeights;
use ternary_core::verify::gemv_scalar_ref;
use ternary_kernels::cpu;

/// Generate deterministic pseudo-random weights.
fn gen_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let v = (i as u32).wrapping_mul(2654435761) >> 16;
            (v % 200) as f32 / 100.0 - 1.0
        })
        .collect()
}

/// Generate deterministic activations.
fn gen_activations(cols: usize) -> Vec<i8> {
    (0..cols)
        .map(|i| (((i * 37 + 13) % 200) as i32 - 100) as i8)
        .collect()
}

fn bench_gemv_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_scalar_ref");

    let shapes: &[(usize, usize)] = &[
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (4096, 11008),
        (11008, 4096),
    ];

    for &(m, k) in shapes {
        let w = gen_weights(m, k);
        let pw = PlanarWeights::from_row_major(&w, m, k, 128);
        let x = gen_activations(k);
        let mut y = vec![0.0f32; m];
        let act_scale = 1.0 / 127.0;

        // Throughput in operations: 2 * M * K (multiply + accumulate per element)
        let ops = 2 * m * k;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", m, k)),
            &(),
            |b, _| {
                b.iter(|| {
                    gemv_scalar_ref(&pw, &x, act_scale, &mut y);
                });
            },
        );
    }

    group.finish();
}

fn bench_gemv_ffi(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_ffi_dispatch");

    let shapes: &[(usize, usize)] = &[
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (4096, 11008),
        (11008, 4096),
    ];

    for &(m, k) in shapes {
        let w = gen_weights(m, k);
        let pw = PlanarWeights::from_row_major(&w, m, k, 128);
        let x = gen_activations(k);
        let mut y = vec![0.0f32; m];
        let act_scale = 1.0 / 127.0;

        let ops = 2 * m * k;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("ffi", format!("{}x{}", m, k)),
            &(),
            |b, _| {
                b.iter(|| {
                    cpu::gemv(&pw, &x, act_scale, &mut y);
                });
            },
        );
    }

    group.finish();
}

fn bench_gemv_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_scalar_vs_ffi");

    // Focus on the production-critical shapes
    let shapes: &[(usize, usize)] = &[
        (2048, 2048),
        (4096, 4096),
        (4096, 11008),
        (11008, 4096),
    ];

    for &(m, k) in shapes {
        let w = gen_weights(m, k);
        let pw = PlanarWeights::from_row_major(&w, m, k, 128);
        let x = gen_activations(k);
        let mut y = vec![0.0f32; m];
        let act_scale = 1.0 / 127.0;

        let label = format!("{}x{}", m, k);

        group.bench_with_input(
            BenchmarkId::new("scalar", &label),
            &(),
            |b, _| {
                b.iter(|| gemv_scalar_ref(&pw, &x, act_scale, &mut y));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ffi", &label),
            &(),
            |b, _| {
                b.iter(|| cpu::gemv(&pw, &x, act_scale, &mut y));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gemv_scalar, bench_gemv_ffi, bench_gemv_comparison);
criterion_main!(benches);
