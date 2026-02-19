//! BitNet ternary encoding/decoding.
//!
//! Encoding table (BitNet standard, GGUF-compatible):
//!   -1 -> 0b11
//!    0 -> 0b00
//!   +1 -> 0b01
//! 0b10 -> invalid (decodes as 0)
//!
//! Each trit occupies 2 bits. The low bit indicates non-zero,
//! the high bit indicates the sign (1 = negative).

/// Encode a ternary value (-1, 0, or +1) to its 2-bit representation.
///
/// Panics (debug) if `val` is not in {-1, 0, +1}.
#[inline]
pub fn encode_trit(val: i8) -> u8 {
    debug_assert!(
        val == -1 || val == 0 || val == 1,
        "encode_trit: val must be -1, 0, or +1, got {}",
        val
    );
    match val {
        -1 => 0b11,
        0 => 0b00,
        1 => 0b01,
        _ => 0b00, // fallback for release mode
    }
}

/// Decode a 2-bit code back to a ternary value.
///
/// - 0b00 ->  0
/// - 0b01 -> +1
/// - 0b10 ->  0 (invalid codepoint, treated as zero)
/// - 0b11 -> -1
///
/// This matches the C reference:
/// ```c
/// int8_t nz = bits & 1;
/// int8_t sg = (bits >> 1) & 1;
/// return nz - 2 * (nz & sg);
/// ```
#[inline]
pub fn decode_trit(bits: u8) -> i8 {
    let nz = (bits & 1) as i8;
    let sg = ((bits >> 1) & 1) as i8;
    nz - 2 * (nz & sg)
}

/// Branchless encode of a ternary value to 2-bit code.
///
/// For val in {-1, 0, +1}:
///   bit0 (non-zero) = val != 0 = (val as u8) & 1 | ((val as u8) >> 7) & 1
///   bit1 (sign)     = val < 0
///   result = bit0 | (bit1 << 1)
///
/// This is suitable for SIMD preparation where branches are costly.
#[inline]
pub fn encode_trit_branchless(val: i8) -> u8 {
    // nz: 1 if val != 0.  For val in {-1,0,1}:
    //   val=-1: unsigned byte = 0xFF, bit0=1
    //   val=0:  unsigned byte = 0x00, bit0=0
    //   val=1:  unsigned byte = 0x01, bit0=1
    // nz = (val != 0) as u8  (but branchless)
    let u = val as u8; // wrapping cast: -1 -> 0xFF, 0 -> 0x00, 1 -> 0x01
    let nz = (u & 1) | ((u >> 7) & 1); // bit0 set for 0x01 and 0xFF
    let sg = (u >> 7) & 1; // sign bit: 1 for negative
    nz | (sg << 1)
}

/// Pack 4 trits (each 2 bits) into a single byte, LSB-first.
///
/// byte = trit0 | (trit1 << 2) | (trit2 << 4) | (trit3 << 6)
#[inline]
pub fn pack_4_trits(t0: u8, t1: u8, t2: u8, t3: u8) -> u8 {
    (t0 & 0x03) | ((t1 & 0x03) << 2) | ((t2 & 0x03) << 4) | ((t3 & 0x03) << 6)
}

/// Unpack a byte into 4 trits (each 2 bits), LSB-first.
///
/// Returns (trit0, trit1, trit2, trit3) as 2-bit codes.
#[inline]
pub fn unpack_4_trits(byte: u8) -> (u8, u8, u8, u8) {
    (
        byte & 0x03,
        (byte >> 2) & 0x03,
        (byte >> 4) & 0x03,
        (byte >> 6) & 0x03,
    )
}

/// Compile-time lookup table: maps a 2-bit code to its decoded i8 value.
///
/// Index 0b00 →  0, 0b01 → +1, 0b10 → 0 (invalid), 0b11 → -1
pub const DECODE_TRIT_LUT: [i8; 4] = [0, 1, 0, -1];

/// Compile-time lookup table: maps every possible byte (4 packed trits) to
/// four decoded i8 values. Eliminates per-trit branching in hot decode loops.
///
/// Usage: `let [t0, t1, t2, t3] = DECODE_BYTE_LUT[byte as usize];`
pub const DECODE_BYTE_LUT: [[i8; 4]; 256] = {
    let mut lut = [[0i8; 4]; 256];
    let mut b: usize = 0;
    while b < 256 {
        lut[b][0] = DECODE_TRIT_LUT[b & 0x03];
        lut[b][1] = DECODE_TRIT_LUT[(b >> 2) & 0x03];
        lut[b][2] = DECODE_TRIT_LUT[(b >> 4) & 0x03];
        lut[b][3] = DECODE_TRIT_LUT[(b >> 6) & 0x03];
        b += 1;
    }
    lut
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        for val in [-1i8, 0, 1] {
            let encoded = encode_trit(val);
            let decoded = decode_trit(encoded);
            assert_eq!(decoded, val, "roundtrip failed for val={}", val);
        }
    }

    #[test]
    fn test_encode_specific_values() {
        assert_eq!(encode_trit(-1), 0b11);
        assert_eq!(encode_trit(0), 0b00);
        assert_eq!(encode_trit(1), 0b01);
    }

    #[test]
    fn test_decode_specific_values() {
        assert_eq!(decode_trit(0b00), 0);
        assert_eq!(decode_trit(0b01), 1);
        assert_eq!(decode_trit(0b10), 0); // invalid -> 0
        assert_eq!(decode_trit(0b11), -1);
    }

    #[test]
    fn test_invalid_codepoint_decodes_to_zero() {
        assert_eq!(decode_trit(0b10), 0);
    }

    #[test]
    fn test_branchless_encode_matches_branching() {
        for val in [-1i8, 0, 1] {
            assert_eq!(
                encode_trit_branchless(val),
                encode_trit(val),
                "branchless mismatch for val={}",
                val
            );
        }
    }

    #[test]
    fn test_branchless_encode_specific() {
        assert_eq!(encode_trit_branchless(-1), 0b11);
        assert_eq!(encode_trit_branchless(0), 0b00);
        assert_eq!(encode_trit_branchless(1), 0b01);
    }

    #[test]
    fn test_pack_4_trits() {
        // Pack: -1, 0, +1, -1 => 0b11, 0b00, 0b01, 0b11
        let byte = pack_4_trits(0b11, 0b00, 0b01, 0b11);
        // byte = 0b11_01_00_11 = 0xD3
        assert_eq!(byte, 0b11_01_00_11);

        let (t0, t1, t2, t3) = unpack_4_trits(byte);
        assert_eq!(t0, 0b11);
        assert_eq!(t1, 0b00);
        assert_eq!(t2, 0b01);
        assert_eq!(t3, 0b11);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        for t0 in 0..4u8 {
            for t1 in 0..4u8 {
                for t2 in 0..4u8 {
                    for t3 in 0..4u8 {
                        let byte = pack_4_trits(t0, t1, t2, t3);
                        let (u0, u1, u2, u3) = unpack_4_trits(byte);
                        assert_eq!((u0, u1, u2, u3), (t0, t1, t2, t3));
                    }
                }
            }
        }
    }

    #[test]
    fn test_decode_all_possible_2bit_values() {
        // Exhaustive: all 4 possible 2-bit patterns
        let expected = [(0b00, 0i8), (0b01, 1), (0b10, 0), (0b11, -1)];
        for (bits, exp) in expected {
            assert_eq!(decode_trit(bits), exp, "decode_trit({:#04b}) failed", bits);
        }
    }

    #[test]
    fn test_pack_all_zeros() {
        let byte = pack_4_trits(0b00, 0b00, 0b00, 0b00);
        assert_eq!(byte, 0x00);
    }

    #[test]
    fn test_pack_all_negative_ones() {
        let byte = pack_4_trits(0b11, 0b11, 0b11, 0b11);
        assert_eq!(byte, 0xFF);
    }

    #[test]
    fn test_pack_all_positive_ones() {
        let byte = pack_4_trits(0b01, 0b01, 0b01, 0b01);
        assert_eq!(byte, 0b01_01_01_01);
        assert_eq!(byte, 0x55);
    }

    #[test]
    fn test_encode_trit_fallback_in_release() {
        // In release mode, invalid values hit the _ => 0b00 fallback
        // In debug mode, this would panic from debug_assert.
        // We test the valid cases thoroughly here.
        assert_eq!(encode_trit(-1), 0b11);
        assert_eq!(encode_trit(0), 0b00);
        assert_eq!(encode_trit(1), 0b01);
    }
}
