//! Pack/unpack ternary weights with group quantization.
//!
//! Group size is typically 128 (BitNet b1.58 standard). For each group of
//! `group_size` float weights:
//!   1. Compute absmean scale = mean(|w|)
//!   2. Quantize each weight to {-1, 0, +1} using thresholds +/-0.5 of normalized value
//!   3. Pack 4 trits per byte (LSB-first)
//!
//! The packed representation stores `group_size / 4` bytes per group plus one f32 scale.

use crate::encode::{decode_trit, encode_trit, pack_4_trits, unpack_4_trits};

/// A fully packed matrix of ternary weights with per-group scales.
#[derive(Debug, Clone)]
pub struct PackedMatrix {
    /// Packed ternary bytes, row-major. Each row has `cols / 4` bytes.
    pub packed: Vec<u8>,
    /// Per-group scales, row-major. Each row has `cols / group_size` scales.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

impl PackedMatrix {
    /// Number of packed bytes per row.
    pub fn bytes_per_row(&self) -> usize {
        assert!(self.cols > 0, "cols must be > 0");
        self.cols / 4
    }

    /// Number of groups per row.
    pub fn groups_per_row(&self) -> usize {
        assert!(self.group_size > 0, "group_size must be > 0");
        self.cols / self.group_size
    }

    /// Get packed bytes for a specific row.
    pub fn row_bytes(&self, row: usize) -> &[u8] {
        assert!(
            row < self.rows,
            "row index {} out of bounds {}",
            row,
            self.rows
        );
        let bpr = self.bytes_per_row();
        let start = row
            .checked_mul(bpr)
            .expect("row_bytes start offset overflow");
        let end = start
            .checked_add(bpr)
            .expect("row_bytes end offset overflow");
        assert!(
            end <= self.packed.len(),
            "row_bytes range [{}..{}) out of packed bounds {}",
            start,
            end,
            self.packed.len()
        );
        &self.packed[start..end]
    }

    /// Get scales for a specific row.
    pub fn row_scales(&self, row: usize) -> &[f32] {
        assert!(
            row < self.rows,
            "row index {} out of bounds {}",
            row,
            self.rows
        );
        let gpr = self.groups_per_row();
        let start = row
            .checked_mul(gpr)
            .expect("row_scales start offset overflow");
        let end = start
            .checked_add(gpr)
            .expect("row_scales end offset overflow");
        assert!(
            end <= self.scales.len(),
            "row_scales range [{}..{}) out of scales bounds {}",
            start,
            end,
            self.scales.len()
        );
        &self.scales[start..end]
    }
}

/// Quantize a single float weight relative to its group's absmean scale.
///
/// `inv` is `1.0 / absmean` (or 0 if absmean is negligible).
/// Returns an i8 in {-1, 0, +1}.
#[inline]
pub fn quantize_one(w: f32, inv: f32) -> i8 {
    let s = w * inv;
    if s > 0.5 {
        1
    } else if s < -0.5 {
        -1
    } else {
        0
    }
}

/// Pack 4 quantized i8 trits into one byte.
#[inline]
pub fn pack_4(t0: i8, t1: i8, t2: i8, t3: i8) -> u8 {
    pack_4_trits(
        encode_trit(t0),
        encode_trit(t1),
        encode_trit(t2),
        encode_trit(t3),
    )
}

/// Unpack one byte into 4 decoded trit values.
#[inline]
pub fn unpack_4(byte: u8) -> (i8, i8, i8, i8) {
    let (t0, t1, t2, t3) = unpack_4_trits(byte);
    (
        decode_trit(t0),
        decode_trit(t1),
        decode_trit(t2),
        decode_trit(t3),
    )
}

/// Quantize and pack a group of `group_size` float weights.
///
/// Uses absmean quantization:
///   scale = mean(|w_i|)
///   quantized = sign(w_i / scale) rounded to {-1, 0, +1} with threshold 0.5
///
/// Returns (packed_bytes, scale).
/// `weights.len()` must be a multiple of 4 (and typically == `group_size`).
pub fn pack_group(weights: &[f32]) -> (Vec<u8>, f32) {
    assert!(
        weights.len().is_multiple_of(4),
        "pack_group: weights.len()={} must be multiple of 4",
        weights.len()
    );

    let gs = weights.len();

    // Compute absmean
    let asum: f32 = weights.iter().map(|w| w.abs()).sum();
    let amean = asum / gs as f32;
    let inv = if amean > 1e-10 { 1.0 / amean } else { 0.0 };

    // Quantize to trits and pack
    let n_bytes = gs / 4;
    let mut packed = Vec::with_capacity(n_bytes);

    for chunk in weights.chunks_exact(4) {
        let t0 = quantize_one(chunk[0], inv);
        let t1 = quantize_one(chunk[1], inv);
        let t2 = quantize_one(chunk[2], inv);
        let t3 = quantize_one(chunk[3], inv);
        packed.push(pack_4(t0, t1, t2, t3));
    }

    (packed, amean)
}

/// Unpack a group of packed bytes back to float values using the given scale.
///
/// Each byte produces 4 float values: `decode_trit(bits) * scale`.
pub fn unpack_group(packed: &[u8], scale: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(packed.len() * 4);
    for &byte in packed {
        let (t0, t1, t2, t3) = unpack_4(byte);
        result.push(t0 as f32 * scale);
        result.push(t1 as f32 * scale);
        result.push(t2 as f32 * scale);
        result.push(t3 as f32 * scale);
    }
    result
}

/// Pack a full row-major weight matrix into ternary packed format with per-group scales.
///
/// Requirements:
/// - `weights.len() == rows * cols`
/// - `cols % group_size == 0`
/// - `group_size % 4 == 0`
pub fn pack_matrix(weights: &[f32], rows: usize, cols: usize, group_size: usize) -> PackedMatrix {
    let rows_cols = rows
        .checked_mul(cols)
        .expect("pack_matrix: rows*cols overflow");
    assert_eq!(
        weights.len(),
        rows_cols,
        "pack_matrix: weights.len()={} != rows*cols={}",
        weights.len(),
        rows_cols
    );
    assert!(
        cols.is_multiple_of(group_size),
        "pack_matrix: cols={} not divisible by group_size={}",
        cols,
        group_size
    );
    assert!(
        group_size.is_multiple_of(4),
        "pack_matrix: group_size={} not divisible by 4",
        group_size
    );

    let bytes_per_row = cols / 4;
    let groups_per_row = cols / group_size;
    let bytes_per_group = group_size / 4;

    let packed_len = rows
        .checked_mul(bytes_per_row)
        .expect("pack_matrix: rows*bytes_per_row overflow");
    let scales_len = rows
        .checked_mul(groups_per_row)
        .expect("pack_matrix: rows*groups_per_row overflow");
    let mut packed = vec![0u8; packed_len];
    let mut scales = vec![0.0f32; scales_len];

    for r in 0..rows {
        let row_start = r * cols;
        for g in 0..groups_per_row {
            let g_start = row_start + g * group_size;
            let g_end = g_start + group_size;
            let group_weights = &weights[g_start..g_end];

            let (group_packed, scale) = pack_group(group_weights);

            // Copy packed bytes into the output
            let byte_offset = r * bytes_per_row + g * bytes_per_group;
            packed[byte_offset..byte_offset + bytes_per_group].copy_from_slice(&group_packed);

            scales[r * groups_per_row + g] = scale;
        }
    }

    PackedMatrix {
        packed,
        scales,
        rows,
        cols,
        group_size,
    }
}

/// Unpack a PackedMatrix back to row-major floats.
pub fn unpack_matrix(pm: &PackedMatrix) -> Vec<f32> {
    let out_len = pm
        .rows
        .checked_mul(pm.cols)
        .expect("unpack_matrix: rows*cols overflow");
    let mut result = vec![0.0f32; out_len];
    let bytes_per_group = pm.group_size / 4;
    let groups_per_row = pm.groups_per_row();
    let bytes_per_row = pm.bytes_per_row();

    for r in 0..pm.rows {
        for g in 0..groups_per_row {
            let scale = pm.scales[r * groups_per_row + g];
            let byte_offset = r * bytes_per_row + g * bytes_per_group;
            let group_packed = &pm.packed[byte_offset..byte_offset + bytes_per_group];
            let group_floats = unpack_group(group_packed, scale);

            let out_offset = r * pm.cols + g * pm.group_size;
            result[out_offset..out_offset + pm.group_size].copy_from_slice(&group_floats);
        }
    }

    result
}

/// Quantize a row-major f32 weight matrix to Q1_58 format (ternary + scales).
///
/// Q1_58 format: interleaved packed bytes and scales per row.
/// Row format: [packed_bytes (cols/4)] [scales_f32 (cols/group_size)]
///
/// This is the GGUF-compatible format for ternary tensors.
///
/// # Arguments
/// * `weights` - Row-major f32 weights [rows * cols]
/// * `rows` - Number of output rows
/// * `cols` - Number of input columns (must be divisible by group_size and 4)
/// * `group_size` - Quantization group size (typically 128)
///
/// # Returns
/// Packed bytes: rows * (cols/4 + cols/group_size * 4) bytes
pub fn quantize_row_q1_58(
    weights: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<Vec<u8>, String> {
    if !cols.is_multiple_of(group_size) {
        return Err(format!(
            "cols {} must be divisible by group_size {}",
            cols, group_size
        ));
    }
    if !cols.is_multiple_of(4) {
        return Err(format!("cols {} must be divisible by 4", cols));
    }
    let rows_cols = rows
        .checked_mul(cols)
        .ok_or_else(|| "rows*cols overflow in quantize_row_q1_58".to_string())?;
    if weights.len() != rows_cols {
        return Err(format!(
            "weights.len() {} != rows {} * cols {}",
            weights.len(),
            rows,
            cols
        ));
    }

    let bytes_per_row = cols / 4;
    let groups_per_row = cols / group_size;
    let scales_bytes_per_row = groups_per_row
        .checked_mul(4)
        .ok_or_else(|| "groups_per_row*4 overflow in quantize_row_q1_58".to_string())?;
    let total_bytes_per_row = bytes_per_row
        .checked_add(scales_bytes_per_row)
        .ok_or_else(|| "row byte size overflow in quantize_row_q1_58".to_string())?;
    let total_capacity = rows
        .checked_mul(total_bytes_per_row)
        .ok_or_else(|| "rows*total_bytes_per_row overflow in quantize_row_q1_58".to_string())?;

    let mut result = Vec::with_capacity(total_capacity);
    let group_bytes = group_size / 4;

    for r in 0..rows {
        let row_start = r * cols;
        let row_weights = &weights[row_start..row_start + cols];

        let packed_offset = result.len();
        result.resize(packed_offset + bytes_per_row, 0u8);
        let scales_offset = result.len();
        result.resize(scales_offset + scales_bytes_per_row, 0u8);

        for g in 0..groups_per_row {
            let group_start = g * group_size;
            let group_end = group_start + group_size;
            let group_weights = &row_weights[group_start..group_end];

            // Inline quantization to avoid per-group temporary allocations.
            let asum: f32 = group_weights.iter().map(|w| w.abs()).sum();
            let scale = asum / group_size as f32;
            let inv = if scale > 1e-10 { 1.0 / scale } else { 0.0 };

            let dst_start = packed_offset + g * group_bytes;
            for (j, chunk) in group_weights.chunks_exact(4).enumerate() {
                let t0 = quantize_one(chunk[0], inv);
                let t1 = quantize_one(chunk[1], inv);
                let t2 = quantize_one(chunk[2], inv);
                let t3 = quantize_one(chunk[3], inv);
                result[dst_start + j] = pack_4(t0, t1, t2, t3);
            }

            let scale_bytes = scale.to_le_bytes();
            let scale_start = scales_offset + g * 4;
            result[scale_start..scale_start + 4].copy_from_slice(&scale_bytes);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_one() {
        // val * inv > 0.5 => +1
        assert_eq!(quantize_one(1.0, 1.0), 1); // 1.0 > 0.5
                                               // val * inv < -0.5 => -1
        assert_eq!(quantize_one(-1.0, 1.0), -1); // -1.0 < -0.5
                                                 // val * inv in [-0.5, 0.5] => 0
        assert_eq!(quantize_one(0.0, 1.0), 0);
        assert_eq!(quantize_one(0.3, 1.0), 0); // 0.3 < 0.5
        assert_eq!(quantize_one(-0.3, 1.0), 0);
    }

    #[test]
    fn test_quantize_one_threshold() {
        // Exactly at threshold
        assert_eq!(quantize_one(0.5, 1.0), 0); // 0.5 is NOT > 0.5
        assert_eq!(quantize_one(-0.5, 1.0), 0); // -0.5 is NOT < -0.5
        assert_eq!(quantize_one(0.51, 1.0), 1);
        assert_eq!(quantize_one(-0.51, 1.0), -1);
    }

    #[test]
    fn test_pack_4_unpack_4() {
        for t0 in [-1i8, 0, 1] {
            for t1 in [-1i8, 0, 1] {
                for t2 in [-1i8, 0, 1] {
                    for t3 in [-1i8, 0, 1] {
                        let byte = pack_4(t0, t1, t2, t3);
                        let (u0, u1, u2, u3) = unpack_4(byte);
                        assert_eq!((u0, u1, u2, u3), (t0, t1, t2, t3));
                    }
                }
            }
        }
    }

    #[test]
    fn test_pack_group_simple() {
        // 4 weights: [1.0, -1.0, 0.0, 1.0]
        // absmean = (1+1+0+1)/4 = 0.75
        // inv = 1/0.75 = 1.333...
        // quantized: 1*1.333=1.333>0.5 -> +1, -1*1.333=-1.333<-0.5 -> -1,
        //            0*1.333=0 -> 0, 1*1.333=1.333>0.5 -> +1
        let weights = vec![1.0, -1.0, 0.0, 1.0];
        let (packed, scale) = pack_group(&weights);
        assert_eq!(packed.len(), 1);
        assert!((scale - 0.75).abs() < 1e-6); // absmean = (1+1+0+1)/4 = 0.75

        let (t0, t1, t2, t3) = unpack_4(packed[0]);
        assert_eq!(t0, 1);
        assert_eq!(t1, -1);
        assert_eq!(t2, 0);
        assert_eq!(t3, 1);
    }

    #[test]
    fn test_pack_group_all_zeros() {
        let weights = vec![0.0; 128];
        let (packed, scale) = pack_group(&weights);
        assert_eq!(packed.len(), 32);
        assert!(scale.abs() < 1e-10);
        // All bytes should be 0
        for &b in &packed {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn test_pack_unpack_group_roundtrip() {
        // Create weights that are exactly {-1, 0, +1} * scale
        let scale = 0.42;
        let trits = [1i8, -1, 0, 1, 0, 0, -1, -1, 1, 0, -1, 1];
        let weights: Vec<f32> = trits.iter().map(|&t| t as f32 * scale).collect();

        let (packed, recovered_scale) = pack_group(&weights);
        let unpacked = unpack_group(&packed, recovered_scale);

        // The absmean of the weights is scale * (count_nonzero / len)
        // With 8 nonzero out of 12: absmean = scale * 8/12 = scale * 2/3
        // Then inv = 1/(scale * 2/3) = 3/(2*scale)
        // Each ternary weight w * inv = t * scale * 3 / (2*scale) = t * 1.5
        // |1.5| > 0.5 so all nonzero survive, 0 * 1.5 = 0 stays 0. Good.

        for (i, (&orig_trit, &recovered)) in trits.iter().zip(unpacked.iter()).enumerate() {
            let expected = orig_trit as f32 * recovered_scale;
            assert!(
                (recovered - expected).abs() < 1e-6,
                "mismatch at index {}: expected {}, got {}",
                i,
                expected,
                recovered
            );
        }
    }

    #[test]
    fn test_pack_group_128_random() {
        // Create a group of 128 floats with known ternary values
        let scale = 1.5;
        let mut weights = vec![0.0f32; 128];
        for (i, weight) in weights.iter_mut().enumerate() {
            *weight = match i % 3 {
                0 => scale,
                1 => -scale,
                _ => 0.0,
            };
        }

        let (packed, _recovered_scale) = pack_group(&weights);
        assert_eq!(packed.len(), 32); // 128 / 4

        // Unpack and check signs match
        let unpacked = unpack_group(&packed, _recovered_scale);
        for (i, &val) in unpacked.iter().enumerate() {
            let expected_sign = match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
            let actual_sign = if val > 1e-10 {
                1
            } else if val < -1e-10 {
                -1
            } else {
                0
            };
            assert_eq!(actual_sign, expected_sign, "sign mismatch at index {}", i);
        }
    }

    #[test]
    fn test_pack_matrix_basic() {
        // 2 rows, 8 cols, group_size=4
        let weights = vec![
            1.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, // row 0
            0.0, 1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, // row 1
        ];

        let pm = pack_matrix(&weights, 2, 8, 4);
        assert_eq!(pm.rows, 2);
        assert_eq!(pm.cols, 8);
        assert_eq!(pm.group_size, 4);
        assert_eq!(pm.packed.len(), 2 * 2); // 2 rows * (8/4) bytes per row
        assert_eq!(pm.scales.len(), 2 * 2); // 2 rows * (8/4) groups per row
    }

    #[test]
    fn test_pack_unpack_matrix_roundtrip() {
        // Matrix with values that are exactly ternary * some scale
        let rows = 4;
        let cols = 128;
        let group_size = 128;
        let scale = 0.7;

        let mut weights = vec![0.0f32; rows * cols];
        for (i, weight) in weights.iter_mut().enumerate() {
            *weight = match i % 5 {
                0 => scale,
                1 => -scale,
                2 => 0.0,
                3 => scale,
                _ => -scale,
            };
        }

        let pm = pack_matrix(&weights, rows, cols, group_size);
        let unpacked = unpack_matrix(&pm);

        // Check that the sign pattern is preserved
        for i in 0..weights.len() {
            let orig_sign = if weights[i] > 1e-10 {
                1
            } else if weights[i] < -1e-10 {
                -1
            } else {
                0
            };
            let recov_sign = if unpacked[i] > 1e-10 {
                1
            } else if unpacked[i] < -1e-10 {
                -1
            } else {
                0
            };
            assert_eq!(
                orig_sign, recov_sign,
                "sign mismatch at index {}: orig={}, recov={}",
                i, weights[i], unpacked[i]
            );
        }
    }

    #[test]
    fn test_pack_matrix_multiple_groups() {
        // 2 rows, 256 cols, group_size=128 -> 2 groups per row
        let rows = 2;
        let cols = 256;
        let group_size = 128;

        let mut weights = vec![0.0f32; rows * cols];
        // First group of each row: positive
        // Second group: negative
        for r in 0..rows {
            for c in 0..cols {
                if c < 128 {
                    weights[r * cols + c] = 1.0;
                } else {
                    weights[r * cols + c] = -1.0;
                }
            }
        }

        let pm = pack_matrix(&weights, rows, cols, group_size);
        assert_eq!(pm.scales.len(), 2 * 2); // 2 rows * 2 groups

        let unpacked = unpack_matrix(&pm);
        for r in 0..rows {
            for c in 0..cols {
                let val = unpacked[r * cols + c];
                if c < 128 {
                    assert!(
                        val > 0.0,
                        "row {} col {} should be positive, got {}",
                        r,
                        c,
                        val
                    );
                } else {
                    assert!(
                        val < 0.0,
                        "row {} col {} should be negative, got {}",
                        r,
                        c,
                        val
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "must be multiple of 4")]
    fn test_pack_group_bad_size() {
        let weights = vec![1.0f32; 5]; // not multiple of 4
        pack_group(&weights);
    }

    #[test]
    #[should_panic(expected = "not divisible by group_size")]
    fn test_pack_matrix_bad_group_size() {
        let weights = vec![1.0f32; 2 * 128];
        pack_matrix(&weights, 2, 128, 3); // group_size=3 does not divide cols
    }

    #[test]
    #[should_panic(expected = "not divisible by 4")]
    fn test_pack_matrix_bad_group_size_mod4() {
        let weights = vec![1.0f32; 2 * 6];
        pack_matrix(&weights, 2, 6, 6); // group_size=6, 6%4!=0
    }

    #[test]
    fn test_packed_matrix_accessors() {
        let rows = 3;
        let cols = 128;
        let group_size = 128;
        let weights = vec![1.0f32; rows * cols];

        let pm = pack_matrix(&weights, rows, cols, group_size);

        assert_eq!(pm.bytes_per_row(), 32);
        assert_eq!(pm.groups_per_row(), 1);

        for r in 0..rows {
            assert_eq!(pm.row_bytes(r).len(), 32);
            assert_eq!(pm.row_scales(r).len(), 1);
        }
    }
}
