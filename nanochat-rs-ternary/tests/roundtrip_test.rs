//! Roundtrip Test — full pipeline validation:
//!   pack → GGUF write → GGUF read → load PlanarWeights → GEMV → verify
//!
//! Also validates mHC binary I/O roundtrip.

use std::path::PathBuf;

use mhc_lite::{
    load_mhc_file, save_mhc_file, verify_doubly_stochastic, verify_doubly_stochastic_2x2,
    MhcLayerParams, MhcLiteN2, MhcLiteN4,
};
use ternary_core::gguf::{GgufFile, GgufValue, GgufWriter, GGUF_TYPE_Q1_58};
use ternary_core::pack::pack_matrix;
use ternary_core::planar::PlanarWeights;
use ternary_core::verify::{gemv_scalar_ref, verify_gemv};
use ternary_kernels::cpu;

fn test_path(name: &str) -> PathBuf {
    PathBuf::from("/tmp/claude-1000/-home-habitat-ternary-clawd/95e7afdf-b472-41a0-a3d5-73532dc4ecb7/scratchpad")
        .join("roundtrip")
        .join(name)
}

/// Generate deterministic weights.
fn gen_weights(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let v = ((i + seed) as u32).wrapping_mul(2654435761) >> 16;
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

// ───────────────────── GGUF Roundtrip ─────────────────────

#[test]
fn roundtrip_single_ternary_tensor() {
    let path = test_path("rt_single.gguf");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let rows = 64;
    let cols = 256;
    let gs = 128;
    let w = gen_weights(rows, cols, 42);
    let pw_orig = PlanarWeights::from_row_major(&w, rows, cols, gs);

    // GGUF write
    let mut writer = GgufWriter::new();
    writer.add_metadata("model.group_size", GgufValue::U32(gs as u32));
    writer.add_metadata(
        "model.name",
        GgufValue::String("roundtrip_test".to_string()),
    );
    writer.add_ternary_tensor("layer.0.weight", &pw_orig);
    writer.write(&path).unwrap();

    // GGUF read
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(gguf.tensors.len(), 1);
    assert_eq!(gguf.tensors[0].name, "layer.0.weight");
    assert_eq!(gguf.tensors[0].dtype, GGUF_TYPE_Q1_58);

    // Load back as PlanarWeights
    let pw_loaded = gguf.load_planar_weights("layer.0.weight", gs).unwrap();
    assert_eq!(pw_loaded.rows, rows);
    assert_eq!(pw_loaded.cols, cols);

    // GEMV on both should produce identical results
    let x = gen_activations(cols);
    let act_scale = 1.0 / 127.0;

    let mut y_orig = vec![0.0f32; rows];
    gemv_scalar_ref(&pw_orig, &x, act_scale, &mut y_orig);

    let mut y_loaded = vec![0.0f32; rows];
    gemv_scalar_ref(&pw_loaded, &x, act_scale, &mut y_loaded);

    for r in 0..rows {
        assert!(
            (y_orig[r] - y_loaded[r]).abs() < 1e-5,
            "row {}: orig={}, loaded={}, diff={}",
            r,
            y_orig[r],
            y_loaded[r],
            (y_orig[r] - y_loaded[r]).abs()
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn roundtrip_multiple_tensors() {
    let path = test_path("rt_multi.gguf");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let gs = 128;
    let shapes = [(128, 256), (256, 512), (64, 128), (512, 128)];

    let mut writer = GgufWriter::new();
    writer.add_metadata("model.group_size", GgufValue::U32(gs as u32));
    writer.add_metadata("model.n_layers", GgufValue::U32(shapes.len() as u32));

    let mut originals = Vec::new();
    for (i, &(rows, cols)) in shapes.iter().enumerate() {
        let w = gen_weights(rows, cols, i * 1000);
        let pw = PlanarWeights::from_row_major(&w, rows, cols, gs);
        writer.add_ternary_tensor(&format!("layer.{}.weight", i), &pw);
        originals.push(pw);
    }
    writer.write(&path).unwrap();

    // Read back
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(gguf.tensors.len(), shapes.len());

    // Verify each tensor
    for (i, &(rows, cols)) in shapes.iter().enumerate() {
        let name = format!("layer.{}.weight", i);
        let pw = gguf.load_planar_weights(&name, gs).unwrap();
        assert_eq!(pw.rows, rows);
        assert_eq!(pw.cols, cols);

        let x = gen_activations(cols);
        let act_scale = 1.0 / 127.0;

        let mut y_orig = vec![0.0f32; rows];
        gemv_scalar_ref(&originals[i], &x, act_scale, &mut y_orig);

        let mut y_loaded = vec![0.0f32; rows];
        gemv_scalar_ref(&pw, &x, act_scale, &mut y_loaded);

        let result = verify_gemv(&pw, &x, act_scale, &y_orig, 1e-5);
        assert_eq!(
            result.fail, 0,
            "tensor {} [{}x{}]: {} mismatches, max_diff={}",
            name, rows, cols, result.fail, result.max_diff
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn roundtrip_gguf_to_ffi_gemv() {
    let path = test_path("rt_ffi.gguf");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let rows = 256;
    let cols = 512;
    let gs = 128;
    let w = gen_weights(rows, cols, 99);
    let pw_orig = PlanarWeights::from_row_major(&w, rows, cols, gs);

    let mut writer = GgufWriter::new();
    writer.add_metadata("model.group_size", GgufValue::U32(gs as u32));
    writer.add_ternary_tensor("w", &pw_orig);
    writer.write(&path).unwrap();

    let gguf = GgufFile::open(&path).unwrap();
    let pw_loaded = gguf.load_planar_weights("w", gs).unwrap();

    let x = gen_activations(cols);
    let act_scale = 1.0 / 127.0;

    // Scalar ref on original
    let mut y_ref = vec![0.0f32; rows];
    gemv_scalar_ref(&pw_orig, &x, act_scale, &mut y_ref);

    // C FFI on loaded (full roundtrip: pack → GGUF → load → FFI GEMV)
    let mut y_ffi = vec![0.0f32; rows];
    cpu::gemv(&pw_loaded, &x, act_scale, &mut y_ffi);

    for r in 0..rows {
        let diff = (y_ref[r] - y_ffi[r]).abs();
        assert!(
            diff < 1e-4,
            "row {}: ref={}, ffi={}, diff={}",
            r,
            y_ref[r],
            y_ffi[r],
            diff
        );
    }

    std::fs::remove_file(&path).ok();
}

// ───────────────────── mHC Binary Roundtrip ─────────────────────

#[test]
fn roundtrip_mhc_n2() {
    let path = test_path("rt_mhc_n2.bin");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let layers: Vec<MhcLayerParams> = (0..8)
        .map(|i| {
            let alpha = ((i as f32) * 1.37).sin() * 5.0;
            MhcLayerParams::N2(MhcLiteN2::from_weights(
                alpha,
                [((i as f32) * 0.3).sin(), ((i as f32) * 0.7).cos()],
                [0.5; 2],
                [((i as f32) * 1.1).sin(), ((i as f32) * 1.3).cos()],
                [0.5; 2],
            ))
        })
        .collect();

    save_mhc_file(&path, 2, &layers).unwrap();
    let (header, loaded) = load_mhc_file(&path).unwrap();

    assert_eq!(header.n_layers, 8);
    assert_eq!(header.n_streams, 2);
    assert_eq!(loaded.len(), 8);

    for (i, (orig, loaded)) in layers.iter().zip(loaded.iter()).enumerate() {
        if let (MhcLayerParams::N2(a), MhcLayerParams::N2(b)) = (orig, loaded) {
            assert!(
                (a.alpha_logit - b.alpha_logit).abs() < 1e-7,
                "layer {} alpha mismatch",
                i
            );
            // Verify loaded parameters still produce DS matrix
            let h = b.h_res();
            verify_doubly_stochastic_2x2(&h, 1e-6).unwrap();
        } else {
            panic!("layer {} type mismatch", i);
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn roundtrip_mhc_n4() {
    let path = test_path("rt_mhc_n4.bin");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let layers: Vec<MhcLayerParams> = (0..12)
        .map(|layer_idx| {
            let mut res_logits = [0.0f32; 24];
            #[allow(clippy::needless_range_loop)] // Index needed for deterministic test data generation
            for i in 0..24 {
                res_logits[i] = ((layer_idx * 24 + i) as f32 * 0.7).sin() * 3.0;
            }
            MhcLayerParams::N4(MhcLiteN4::from_weights(
                res_logits, [0.0; 4], [0.5; 4], [0.0; 4], [0.5; 4],
            ))
        })
        .collect();

    save_mhc_file(&path, 4, &layers).unwrap();
    let (header, loaded) = load_mhc_file(&path).unwrap();

    assert_eq!(header.n_layers, 12);
    assert_eq!(header.n_streams, 4);
    assert_eq!(loaded.len(), 12);

    // Verify all loaded mHC layers produce DS matrices
    let mut matrices = Vec::new();
    for (i, layer) in loaded.iter().enumerate() {
        if let MhcLayerParams::N4(mhc) = layer {
            let h = mhc.h_res();
            verify_doubly_stochastic(&h, 1e-5)
                .unwrap_or_else(|e| panic!("loaded layer {} not DS: {}", i, e));
            matrices.push(h);
        } else {
            panic!("layer {} type mismatch", i);
        }
    }

    // Composite gain should be bounded
    let gain = mhc_lite::composite_amax_gain(&matrices);
    assert!(
        gain <= 1.0 + 1e-4,
        "loaded mHC composite gain {} exceeds bound",
        gain
    );

    std::fs::remove_file(&path).ok();
}

// ───────────────────── Pack → PlanarWeights Roundtrip ─────────────────────

#[test]
fn roundtrip_pack_planar_consistency() {
    let rows = 32;
    let cols = 256;
    let gs = 128;
    let w = gen_weights(rows, cols, 7);

    // Path A: float → PlanarWeights (packs internally)
    let pw_a = PlanarWeights::from_row_major(&w, rows, cols, gs);

    // Path B: float → PackedMatrix → PlanarWeights::from_packed
    let pm = pack_matrix(&w, rows, cols, gs);
    let pw_b = PlanarWeights::from_packed(&pm.packed, &pm.scales, rows, cols, gs);

    // Both should produce identical packed data
    assert_eq!(pw_a.data.len(), pw_b.data.len());
    for i in 0..pw_a.data.len() {
        assert_eq!(
            pw_a.data[i], pw_b.data[i],
            "packed data mismatch at byte {}",
            i
        );
    }

    // And identical GEMV results
    let x = gen_activations(cols);
    let mut y_a = vec![0.0f32; rows];
    let mut y_b = vec![0.0f32; rows];
    gemv_scalar_ref(&pw_a, &x, 1.0 / 127.0, &mut y_a);
    gemv_scalar_ref(&pw_b, &x, 1.0 / 127.0, &mut y_b);

    for r in 0..rows {
        assert!(
            (y_a[r] - y_b[r]).abs() < 1e-6,
            "row {}: path_a={}, path_b={}",
            r,
            y_a[r],
            y_b[r]
        );
    }
}
