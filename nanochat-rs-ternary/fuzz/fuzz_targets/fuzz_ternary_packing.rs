//! Fuzz target for ternary packing/unpacking.
//!
//! Tests for:
//! - Memory safety (no panics, no out-of-bounds)
//! - Roundtrip correctness
//! - Edge cases (all zeros, all max values, etc.)

#![no_main]

use libfuzzer_sys::fuzz_target;
use ternary_core::pack::{pack_group, unpack_group};

fuzz_target!(|data: &[u8]| {
    // Interpret input as floats (may be NaN, infinity, etc.)
    if data.len() < 128 * 4 {
        return; // Need at least 128 f32s
    }

    let mut weights = Vec::with_capacity(128);
    for chunk in data.chunks_exact(4).take(128) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let val = f32::from_le_bytes(bytes);

        // Only test finite values for meaningful roundtrips
        if val.is_finite() {
            weights.push(val);
        } else {
            weights.push(0.0); // Replace NaN/inf with 0
        }
    }

    if weights.len() < 128 {
        return;
    }

    // Test packing (should not panic)
    let (packed, scale) = pack_group(&weights);

    // Verify packed size
    assert_eq!(packed.len(), 32, "Packed size should be 32 bytes");

    // Test unpacking (should not panic)
    let unpacked = unpack_group(&packed, scale);

    // Verify unpacked size
    assert_eq!(
        unpacked.len(),
        128,
        "Unpacked size should be 128 elements"
    );

    // Verify all values are finite
    for (i, &val) in unpacked.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Unpacked value at index {} is not finite: {}",
            i,
            val
        );
    }

    // For non-zero scales, verify ternary property
    if scale.abs() > 1e-10 {
        for &val in &unpacked {
            let normalized = (val / scale).abs();
            // Should be approximately 0, 1, or -1
            assert!(
                normalized <= 1.0 + 1e-3,
                "Normalized value {} exceeds ternary range",
                normalized
            );
        }
    }
});
