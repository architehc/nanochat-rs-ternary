// ternary_gemv_avx2.c — AVX2 PSHUFB-based ternary GEMV kernel
//
// Algorithm: Adapted from VPERMW approach, using 256-bit registers.
// - Row blocking: 16 rows per iteration (fits in YMM after widening to int16/int32)
// - LUT strategy: build_lut16() per column group, split lo/hi bytes for PSHUFB
// - LUT hoisting: Pre-build all gpp LUTs per group (fits L1)
// - Lookup: _mm256_shuffle_epi8 (PSHUFB) for byte-level LUT
// - Accumulation: int32 accumulators (safe for any group size)
// - Scaling: FMA with group-major scales
//
// Dispatch priority: below AVX-512 VPERMW/VPERMB, above LUT-Grouped.

#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

// Build a 16-entry LUT mapping 4-bit nibble → dot product contribution.
// nibble bits select which of the 4 activations to sum.
__attribute__((target("avx2,fma")))
static void avx2_build_lut16(int16_t a0, int16_t a1, int16_t a2, int16_t a3,
                              int16_t out[16]) {
    for (int m = 0; m < 16; m++) {
        int16_t s = 0;
        if (m & 1) s += a0;
        if (m & 2) s += a1;
        if (m & 4) s += a2;
        if (m & 8) s += a3;
        out[m] = s;
    }
}

// Scalar nibble extraction helpers (for tail handling)
static inline uint8_t avx2_pos_nibble(uint8_t p) {
    uint8_t nz = p & 0x55, sg = (p >> 1) & 0x55, pos = nz & ~sg;
    return (pos & 1) | ((pos >> 1) & 2) | ((pos >> 2) & 4) | ((pos >> 3) & 8);
}

static inline uint8_t avx2_neg_nibble(uint8_t p) {
    uint8_t nz = p & 0x55, sg = (p >> 1) & 0x55, neg = nz & sg;
    return (neg & 1) | ((neg >> 1) & 2) | ((neg >> 2) & 4) | ((neg >> 3) & 8);
}

// Extract nibble from packed byte: take bits at positions 0,2,4,6 and
// compact them into bits 0,1,2,3 of the result.
//
// Input byte: b7 b6 b5 b4 b3 b2 b1 b0
// We want:    0  0  0  0  b6 b4 b2 b0
#define AVX2_COMPACT(r, h01, h02, h04, h08) \
    _mm_or_si128( \
        _mm_or_si128( \
            _mm_and_si128(r, h01), \
            _mm_and_si128(_mm_srli_epi16(r, 1), h02)), \
        _mm_or_si128( \
            _mm_and_si128(_mm_srli_epi16(r, 2), h04), \
            _mm_and_si128(_mm_srli_epi16(r, 3), h08)))

__attribute__((target("avx2,fma")))
void gemv_avx2(
    const uint8_t * __restrict__ Wt,
    const float   * __restrict__ scales_gm,
    const int8_t  * __restrict__ x,
    float act_scale,
    float * __restrict__ y,
    int M, int K, int gs, int rows_padded
) {
    int gprow = K / gs;
    int gpp   = gs / 4;

    // Zero output
    for (int r = 0; r < M; r++) y[r] = 0;

    // 128-bit constants for COMPACT operations on 16-byte chunks
    const __m128i h55_128 = _mm_set1_epi8(0x55);
    const __m128i h01_128 = _mm_set1_epi8(0x01);
    const __m128i h02_128 = _mm_set1_epi8(0x02);
    const __m128i h04_128 = _mm_set1_epi8(0x04);
    const __m128i h08_128 = _mm_set1_epi8(0x08);

    // Pre-allocate LUT storage (max gpp=32 for gs=128)
    int16_t pd_all[32][16] __attribute__((aligned(32)));
    int16_t nd_all[32][16] __attribute__((aligned(32)));

    // Pre-built byte-level PSHUFB LUTs: lo and hi bytes, 16 entries each,
    // duplicated across both 128-bit lanes of YMM for PSHUFB compatibility.
    // But we only use 128-bit PSHUFB (_mm_shuffle_epi8), so 16 bytes each.
    uint8_t plo_all[32][16] __attribute__((aligned(16)));
    uint8_t phi_all[32][16] __attribute__((aligned(16)));
    uint8_t nlo_all[32][16] __attribute__((aligned(16)));
    uint8_t nhi_all[32][16] __attribute__((aligned(16)));

    for (int g = 0; g < gprow; g++) {
        const float *sc_base = &scales_gm[(size_t)g * rows_padded];

        // Build all LUTs for this group ONCE (hoisted out of row loop)
        for (int j = 0; j < gpp; j++) {
            int cg = g * gpp + j;
            int b = cg * 4;
            avx2_build_lut16(x[b], x[b + 1], x[b + 2], x[b + 3], pd_all[j]);
            for (int i = 0; i < 16; i++) nd_all[j][i] = -pd_all[j][i];

            // Build byte-level LUTs for PSHUFB
            for (int i = 0; i < 16; i++) {
                uint16_t pv = (uint16_t)pd_all[j][i];
                uint16_t nv = (uint16_t)nd_all[j][i];
                plo_all[j][i] = (uint8_t)(pv & 0xFF);
                phi_all[j][i] = (uint8_t)(pv >> 8);
                nlo_all[j][i] = (uint8_t)(nv & 0xFF);
                nhi_all[j][i] = (uint8_t)(nv >> 8);
            }
        }

        // Process 16 rows per block.
        // 16 packed bytes → 16 nibble indices → 16 PSHUFB lookups → 16 int16 results
        // → widen to 16 int32 in 2 YMM accumulators (8 int32 each).
        for (int rblk = 0; rblk < M; rblk += 16) {
            int rc = (rblk + 16 <= M) ? 16 : M - rblk;

            // 2 YMM accumulators for 16 int32 values (8 per YMM)
            __m256i a0 = _mm256_setzero_si256();
            __m256i a1 = _mm256_setzero_si256();

            for (int j = 0; j < gpp; j++) {
                int cg = g * gpp + j;
                const uint8_t *col = &Wt[(size_t)cg * rows_padded + rblk];

                if (rc == 16) {
                    // Load 16 bytes of column-major packed data
                    __m128i pk = _mm_loadu_si128((const __m128i *)col);

                    // Extract value and sign bits (BitNet: bit0=nz, bit1=sign)
                    __m128i val = _mm_and_si128(pk, h55_128);
                    __m128i sgn = _mm_and_si128(
                        _mm_srli_epi16(pk, 1), h55_128);
                    __m128i pr = _mm_andnot_si128(sgn, val);   // positive: nz & ~sign
                    __m128i nr = _mm_and_si128(val, sgn);      // negative: nz & sign

                    // COMPACT: extract bits at positions 0,2,4,6 → 4-bit nibbles
                    __m128i pn = AVX2_COMPACT(pr, h01_128, h02_128, h04_128, h08_128);
                    __m128i nn = AVX2_COMPACT(nr, h01_128, h02_128, h04_128, h08_128);

                    // Load byte-level PSHUFB LUTs (16 bytes each)
                    __m128i p_lut_lo = _mm_load_si128((const __m128i *)plo_all[j]);
                    __m128i p_lut_hi = _mm_load_si128((const __m128i *)phi_all[j]);
                    __m128i n_lut_lo = _mm_load_si128((const __m128i *)nlo_all[j]);
                    __m128i n_lut_hi = _mm_load_si128((const __m128i *)nhi_all[j]);

                    // PSHUFB lookups: each byte of pn/nn selects from the 16-byte LUT
                    // Result: 16 lo bytes and 16 hi bytes for positive contribution
                    __m128i pres_lo = _mm_shuffle_epi8(p_lut_lo, pn);
                    __m128i pres_hi = _mm_shuffle_epi8(p_lut_hi, pn);
                    __m128i nres_lo = _mm_shuffle_epi8(n_lut_lo, nn);
                    __m128i nres_hi = _mm_shuffle_epi8(n_lut_hi, nn);

                    // Combine lo+hi bytes → int16 using unpacklo/unpackhi
                    // unpacklo interleaves: lo[0],hi[0],lo[1],hi[1],...,lo[7],hi[7]
                    // This gives us proper little-endian int16 values!
                    __m128i pres16_lo = _mm_unpacklo_epi8(pres_lo, pres_hi);  // rows 0-7
                    __m128i pres16_hi = _mm_unpackhi_epi8(pres_lo, pres_hi);  // rows 8-15
                    __m128i nres16_lo = _mm_unpacklo_epi8(nres_lo, nres_hi);
                    __m128i nres16_hi = _mm_unpackhi_epi8(nres_lo, nres_hi);

                    // Sum positive and negative contributions (int16)
                    __m128i res16_lo = _mm_add_epi16(pres16_lo, nres16_lo);
                    __m128i res16_hi = _mm_add_epi16(pres16_hi, nres16_hi);

                    // Widen int16 → int32 and accumulate
                    // cvtepi16_epi32 takes low 4 int16 from 128-bit → 8 int32 in 256-bit
                    __m256i r32_0 = _mm256_cvtepi16_epi32(res16_lo);  // rows 0-7
                    __m256i r32_1 = _mm256_cvtepi16_epi32(res16_hi);  // rows 8-15

                    a0 = _mm256_add_epi32(a0, r32_0);
                    a1 = _mm256_add_epi32(a1, r32_1);
                } else {
                    // Tail: scalar fallback for < 16 rows
                    int32_t buf[16];
                    _mm256_storeu_si256((__m256i *)(buf),     a0);
                    _mm256_storeu_si256((__m256i *)(buf + 8), a1);

                    for (int r = 0; r < rc; r++) {
                        buf[r] += pd_all[j][avx2_pos_nibble(col[r])]
                                + nd_all[j][avx2_neg_nibble(col[r])];
                    }

                    a0 = _mm256_loadu_si256((const __m256i *)(buf));
                    a1 = _mm256_loadu_si256((const __m256i *)(buf + 8));
                }
            }

            // Scale and store: convert int32 → float, FMA with group-major scales
            if (rc == 16) {
                __m256 as_vec = _mm256_set1_ps(act_scale);

                __m256 sc0 = _mm256_mul_ps(
                    _mm256_loadu_ps(&sc_base[rblk]), as_vec);
                __m256 sc1 = _mm256_mul_ps(
                    _mm256_loadu_ps(&sc_base[rblk + 8]), as_vec);

                _mm256_storeu_ps(&y[rblk],
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(a0), sc0,
                                    _mm256_loadu_ps(&y[rblk])));
                _mm256_storeu_ps(&y[rblk + 8],
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(a1), sc1,
                                    _mm256_loadu_ps(&y[rblk + 8])));
            } else {
                // Tail: scalar store
                int32_t buf[16];
                _mm256_storeu_si256((__m256i *)(buf),     a0);
                _mm256_storeu_si256((__m256i *)(buf + 8), a1);
                for (int r = 0; r < rc; r++)
                    y[rblk + r] += (float)buf[r]
                        * sc_base[rblk + r] * act_scale;
            }
        }
    }
}

#endif // x86_64
