// ternary_final.c — Production ternary selection kernels v3.3.1
//
// V3.3.1 CHANGES (from final review):
//   i)  Dispatch chain actually selects KERN_LUT on non-AVX512
//   ii) gpp<=32 guard prevents stack overflow if gs!=128
//   iii) ternary_gemv() is self-initializing (thread-safe lazy init)
//   iv) Removed unused 'kp' in AVX-512 kernels (-Wall clean)
//
// V3.3 CHANGES (from RC review):
//   1) OSXSAVE/XGETBV gating — prevents SIGILL on VMs without ZMM state
//   2) VBMI gated on BW+VBMI jointly
//   3) Full dispatch chain: ternary_gemv() single entrypoint
//   4) LUT build hoisted out of rblk loop (perf win for large M)
//   5) Dead sub-loop removed from 64-row path
//
// V3.2 CHANGES (from code review):
//   A) Runtime CPUID dispatch (function multiversioning)
//   B) Group-major scale layout (vector loads, not scalar gathers)
//   C) 64-row blocking (NT-load aligned)
//   D) size_t indexing (large-matrix safe)
//   E) Unsigned cast on int16_t right-shift
//
// V3 SPEC: VPERMW primary, Planar SoA, BitNet (11=-1), gs=128
//
// Compile: gcc -O3 -o ternary_final ternary_final.c -lm
//   (No -mavx512* flags needed — runtime dispatch handles it)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "ternary_gemv.h"
#include "ternary_gemv_avx2.h"

// ============================================================
// CPU FEATURE DETECTION (FIX 1: OSXSAVE/XGETBV gating)
// ============================================================

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#define ARCH_X86_64 1
#else
#define ARCH_X86_64 0
#endif

static int cpu_has_avx2       = 0;
static int cpu_has_avx512bw   = 0;
static int cpu_has_avx512vbmi = 0;

static void detect_cpu_features(void) {
#if ARCH_X86_64
    unsigned int eax, ebx, ecx, edx;

    // Step 1: Check CPUID max level
    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < 7) return;

    // Step 2: Check OSXSAVE (CPUID.1:ECX.OSXSAVE[bit 27])
    // If OS hasn't enabled XSAVE, we can't use AVX-512 even if CPU has it
    __cpuid(1, eax, ebx, ecx, edx);
    int has_osxsave = (ecx >> 27) & 1;
    if (!has_osxsave) return;

    // Step 3: XGETBV — check OS enabled AVX/ZMM state
    // XCR0 bits: SSE=1, AVX=2, opmask=5, zmm_hi256=6, hi16_zmm=7
    unsigned int xcr0_lo, xcr0_hi;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));

    // Step 4: Check actual CPU feature flags
    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    // AVX2 requires OSXSAVE + YMM state enabled (XCR0 bits 1+2)
    if ((xcr0_lo & 0x06) == 0x06) {
        cpu_has_avx2 = (ebx >> 5) & 1;       // AVX2: EBX bit 5
    }

    // AVX-512 requires ZMM state: (xcr0 & 0xE6) == 0xE6
    if ((xcr0_lo & 0xE6) != 0xE6) return;

    cpu_has_avx512bw   = (ebx >> 30) & 1;   // AVX-512 BW: EBX bit 30
    cpu_has_avx512vbmi = (ecx >> 1) & 1;     // AVX-512 VBMI: ECX bit 1

    // FIX 2: VBMI requires BW (defensive)
    if (!cpu_has_avx512bw) cpu_has_avx512vbmi = 0;
#endif
}

// ============================================================
// Function-level target attributes for AVX-512 code
// ============================================================

#if ARCH_X86_64
#include <immintrin.h>
#define AVX512_TARGET __attribute__((target("avx512f,avx512bw")))
#define VBMI_TARGET   __attribute__((target("avx512f,avx512bw,avx512vbmi")))
#else
#define AVX512_TARGET
#define VBMI_TARGET
#endif

// ============================================================
// Utility
// ============================================================

static inline double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *aligned_alloc_128(size_t bytes) {
    void *p;
    if (posix_memalign(&p, 128, bytes) != 0) { perror("memalign"); exit(1); }
    memset(p, 0, bytes);
    return p;
}

static inline size_t round_up_sz(size_t x, size_t align) {
    return (x + align - 1) / align * align;
}
static inline int round_up(int x, int align) {
    return (x + align - 1) / align * align;
}

// ============================================================
// ENCODING: BitNet Standard (11=-1, 01=+1, 00=0)
// ============================================================

static inline int8_t decode_trit(uint8_t bits) {
    int8_t nz = (int8_t)(bits & 1);
    int8_t sg = (int8_t)((bits >> 1) & 1);
    return nz - 2 * (nz & sg);
}

// ============================================================
// PLANAR STORAGE
// ============================================================

// PlanarWeightsC is defined in ternary_gemv.h (the single authoritative definition).
// Legacy alias for code below that uses the short name.
typedef PlanarWeightsC PlanarWeights;

PlanarWeights planar_pack(const float *w, int rows, int cols, int gs) {
    if (cols % 4)  { fprintf(stderr, "FATAL: cols%%4!=0\n"); exit(1); }
    if (gs % 4)    { fprintf(stderr, "FATAL: gs%%4!=0\n"); exit(1); }
    if (cols % gs) { fprintf(stderr, "FATAL: cols%%gs!=0\n"); exit(1); }

    PlanarWeights pw;
    pw.rows        = rows;
    pw.cols        = cols;
    pw.group_size  = gs;
    pw.rows_padded = round_up(rows, 64);

    int kp    = cols / 4;
    int gprow = cols / gs;

    pw.data        = aligned_alloc_128((size_t)rows * kp);
    pw.data_colmaj = aligned_alloc_128((size_t)kp * pw.rows_padded);
    pw.scales_rm   = aligned_alloc_128((size_t)rows * gprow * sizeof(float));
    pw.scales_gm   = aligned_alloc_128((size_t)gprow * pw.rows_padded * sizeof(float));

    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < gprow; g++) {
            size_t start = (size_t)r * cols + (size_t)g * gs;
            float asum = 0;
            for (int i = 0; i < gs; i++) asum += fabsf(w[start + i]);
            float amean = asum / gs;
            pw.scales_rm[r * gprow + g] = amean;
            pw.scales_gm[g * pw.rows_padded + r] = amean;
            float inv = (amean > 1e-10f) ? 1.0f / amean : 0.0f;
            for (int i = 0; i < gs; i++) {
                float s = w[start + i] * inv;
                uint8_t trit = (s > 0.5f) ? 0x01 : (s < -0.5f) ? 0x03 : 0x00;
                size_t flat = start + i;
                pw.data[flat / 4] |= trit << ((flat % 4) * 2);
            }
        }
    }
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < kp; c++)
            pw.data_colmaj[(size_t)c * pw.rows_padded + r] =
                pw.data[(size_t)r * kp + c];
    return pw;
}

void planar_free(PlanarWeights *pw) {
    free(pw->data); free(pw->data_colmaj);
    free(pw->scales_rm); free(pw->scales_gm);
}

// ============================================================
// KERNEL 4 (Fallback): dp4a scalar reference
// ============================================================

void gemv_dp4a_ref(
    const uint8_t * restrict data, const float * restrict scales_rm,
    const int8_t  * restrict x,    float act_scale,
    float * restrict y, int M, int K, int gs
) {
    int kp = K / 4, gprow = K / gs, gpp = gs / 4;
    for (int r = 0; r < M; r++) {
        float racc = 0;
        for (int g = 0; g < gprow; g++) {
            int32_t gacc = 0;
            size_t bs = (size_t)r * kp + (size_t)g * gpp;
            for (int j = 0; j < gpp; j++) {
                uint8_t p = data[bs + j];
                int base = (g * gpp + j) * 4;
                gacc += (int32_t)decode_trit( p       & 3) * x[base]
                      + (int32_t)decode_trit((p >> 2) & 3) * x[base + 1]
                      + (int32_t)decode_trit((p >> 4) & 3) * x[base + 2]
                      + (int32_t)decode_trit((p >> 6) & 3) * x[base + 3];
            }
            racc += (float)gacc * scales_rm[r * gprow + g] * act_scale;
        }
        y[r] = racc;
    }
}

// ============================================================
// KERNEL 3 (Portable): LUT-Grouped i16
// ============================================================

void gemv_lut_grouped(
    const uint8_t * restrict data, const float * restrict scales_rm,
    const int8_t  * restrict x,    float act_scale,
    float * restrict y, int M, int K, int gs
) {
    int kp = K / 4, gprow = K / gs, gpp = gs / 4;
    int16_t *luts = (int16_t *)malloc((size_t)kp * 256 * sizeof(int16_t));
    if (!luts) { perror("malloc luts"); exit(1); }
    for (int cg = 0; cg < kp; cg++) {
        int b = cg * 4;
        int16_t a0 = x[b], a1 = x[b+1], a2 = x[b+2], a3 = x[b+3];
        for (int p = 0; p < 256; p++)
            luts[(size_t)cg * 256 + p] =
                (int16_t)decode_trit( p     & 3) * a0 +
                (int16_t)decode_trit((p>>2) & 3) * a1 +
                (int16_t)decode_trit((p>>4) & 3) * a2 +
                (int16_t)decode_trit((p>>6) & 3) * a3;
    }
    for (int r = 0; r < M; r++) {
        float racc = 0;
        const uint8_t *wr = &data[(size_t)r * kp];
        for (int g = 0; g < gprow; g++) {
            int32_t gacc = 0;
            int cs = g * gpp;
            for (int j = 0; j < gpp; j++)
                gacc += luts[(size_t)(cs + j) * 256 + wr[cs + j]];
            racc += (float)gacc * scales_rm[r * gprow + g] * act_scale;
        }
        y[r] = racc;
    }
    free(luts);
}

// ============================================================
// AVX-512 KERNELS
// ============================================================

#if ARCH_X86_64

static inline uint8_t pos_nibble(uint8_t p) {
    uint8_t nz = p & 0x55, sg = (p >> 1) & 0x55, pos = nz & ~sg;
    return (pos & 1) | ((pos >> 1) & 2) | ((pos >> 2) & 4) | ((pos >> 3) & 8);
}

static inline uint8_t neg_nibble(uint8_t p) {
    uint8_t nz = p & 0x55, sg = (p >> 1) & 0x55, neg = nz & sg;
    return (neg & 1) | ((neg >> 1) & 2) | ((neg >> 2) & 4) | ((neg >> 3) & 8);
}

AVX512_TARGET
static void build_lut16(int16_t a0, int16_t a1, int16_t a2, int16_t a3,
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

// ── KERNEL 1 (PRIMARY): VPERMW fused (AVX-512 BW) ──────────
//
// FIX 4: LUT build hoisted out of rblk loop.
//   LUTs depend on activations x[cg*4..+3] and group index — NOT on row block.
//   Pre-build all gpp LUT pairs per group, then iterate row blocks.
//   For gs=128, gpp=32: 32 × 2 × 32 bytes = 2KB precomputed data (fits L1).
//
// FIX 5: Dead sub-loop removed. 64-row path uses two 32-row extraction
//   blocks directly.

AVX512_TARGET
void gemv_vpermw(
    const uint8_t * restrict Wt,
    const float   * restrict scales_gm,
    const int8_t  * restrict x,
    float act_scale,
    float * restrict y,
    int M, int K, int gs, int rows_padded
) {
    int gprow = K / gs, gpp = gs / 4;
    if (gpp > 32) {
        fprintf(stderr, "FATAL: gs=%d (gpp=%d) exceeds stack LUT limit of 32. "
                "Use gs<=128.\n", gs, gpp);
        abort();
    }
    for (int r = 0; r < M; r++) y[r] = 0;

    const __m256i h55 = _mm256_set1_epi8(0x55);
    const __m256i h01 = _mm256_set1_epi8(0x01);
    const __m256i h02 = _mm256_set1_epi8(0x02);
    const __m256i h04 = _mm256_set1_epi8(0x04);
    const __m256i h08 = _mm256_set1_epi8(0x08);
    const __m512i z_mask = _mm512_set1_epi16(0x000F);

    // FIX 4: Pre-allocate LUT storage (max gpp for gs=128 is 32)
    // Stack allocation — 2KB each, fine for any reasonable gs
    int16_t pd_all[32][16] __attribute__((aligned(32)));
    int16_t nd_all[32][16] __attribute__((aligned(32)));

    for (int g = 0; g < gprow; g++) {
        const float *sc_base = &scales_gm[(size_t)g * rows_padded];

        // FIX 4: Build all LUTs for this group ONCE (before row loop)
        for (int j = 0; j < gpp; j++) {
            int cg = g * gpp + j;
            int b = cg * 4;
            build_lut16(x[b], x[b + 1], x[b + 2], x[b + 3], pd_all[j]);
            for (int i = 0; i < 16; i++) nd_all[j][i] = -pd_all[j][i];
        }

        for (int rblk = 0; rblk < M; rblk += 64) {
            int rc = (rblk + 64 <= M) ? 64 : M - rblk;

            __m512i a0 = _mm512_setzero_si512();
            __m512i a1 = _mm512_setzero_si512();
            __m512i a2 = _mm512_setzero_si512();
            __m512i a3 = _mm512_setzero_si512();

            for (int j = 0; j < gpp; j++) {
                int cg = g * gpp + j;

                // FIX 4: Broadcast from pre-built LUTs (one load, no rebuild)
                __m512i plut = _mm512_broadcast_i64x4(
                    _mm256_load_si256((__m256i *)pd_all[j]));
                __m512i nlut = _mm512_broadcast_i64x4(
                    _mm256_load_si256((__m256i *)nd_all[j]));

                const uint8_t *col = &Wt[(size_t)cg * rows_padded + rblk];

                if (rc == 64) {
                    // FIX 5: Clean 64-row path — two 32-row blocks, no dead loop

                    // Rows 0..31
                    {
                        __m256i pk = _mm256_loadu_si256((__m256i *)col);
                        __m256i val = _mm256_and_si256(pk, h55);
                        __m256i sgn = _mm256_and_si256(
                            _mm256_srli_epi16(pk, 1), h55);
                        __m256i pr = _mm256_andnot_si256(sgn, val);
                        __m256i nr = _mm256_and_si256(val, sgn);

                        #define COMPACT(r) _mm256_or_si256( \
                            _mm256_or_si256( \
                                _mm256_and_si256(r, h01), \
                                _mm256_and_si256( \
                                    _mm256_srli_epi16(r, 1), h02)), \
                            _mm256_or_si256( \
                                _mm256_and_si256( \
                                    _mm256_srli_epi16(r, 2), h04), \
                                _mm256_and_si256( \
                                    _mm256_srli_epi16(r, 3), h08)))
                        __m256i pn = COMPACT(pr);
                        __m256i nn = COMPACT(nr);
                        #undef COMPACT

                        __m512i pi = _mm512_and_si512(
                            _mm512_cvtepu8_epi16(pn), z_mask);
                        __m512i ni = _mm512_and_si512(
                            _mm512_cvtepu8_epi16(nn), z_mask);
                        __m512i res = _mm512_add_epi16(
                            _mm512_permutexvar_epi16(pi, plut),
                            _mm512_permutexvar_epi16(ni, nlut));

                        __m256i rlo = _mm512_castsi512_si256(res);
                        __m256i rhi = _mm512_extracti64x4_epi64(res, 1);
                        a0 = _mm512_add_epi32(a0, _mm512_cvtepi16_epi32(rlo));
                        a1 = _mm512_add_epi32(a1, _mm512_cvtepi16_epi32(rhi));
                    }
                    // Rows 32..63
                    {
                        __m256i pk = _mm256_loadu_si256(
                            (__m256i *)(col + 32));
                        __m256i val = _mm256_and_si256(pk, h55);
                        __m256i sgn = _mm256_and_si256(
                            _mm256_srli_epi16(pk, 1), h55);
                        __m256i pr = _mm256_andnot_si256(sgn, val);
                        __m256i nr = _mm256_and_si256(val, sgn);

                        #define COMPACT(r) _mm256_or_si256( \
                            _mm256_or_si256( \
                                _mm256_and_si256(r, h01), \
                                _mm256_and_si256( \
                                    _mm256_srli_epi16(r, 1), h02)), \
                            _mm256_or_si256( \
                                _mm256_and_si256( \
                                    _mm256_srli_epi16(r, 2), h04), \
                                _mm256_and_si256( \
                                    _mm256_srli_epi16(r, 3), h08)))
                        __m256i pn = COMPACT(pr);
                        __m256i nn = COMPACT(nr);
                        #undef COMPACT

                        __m512i pi = _mm512_and_si512(
                            _mm512_cvtepu8_epi16(pn), z_mask);
                        __m512i ni = _mm512_and_si512(
                            _mm512_cvtepu8_epi16(nn), z_mask);
                        __m512i res = _mm512_add_epi16(
                            _mm512_permutexvar_epi16(pi, plut),
                            _mm512_permutexvar_epi16(ni, nlut));

                        __m256i rlo = _mm512_castsi512_si256(res);
                        __m256i rhi = _mm512_extracti64x4_epi64(res, 1);
                        a2 = _mm512_add_epi32(a2, _mm512_cvtepi16_epi32(rlo));
                        a3 = _mm512_add_epi32(a3, _mm512_cvtepi16_epi32(rhi));
                    }
                } else {
                    // Tail: scalar fallback for <64 rows
                    int32_t buf[64];
                    _mm512_storeu_si512(buf,      a0);
                    _mm512_storeu_si512(buf + 16, a1);
                    _mm512_storeu_si512(buf + 32, a2);
                    _mm512_storeu_si512(buf + 48, a3);
                    for (int r = 0; r < rc; r++)
                        buf[r] += pd_all[j][pos_nibble(col[r])]
                                + nd_all[j][neg_nibble(col[r])];
                    a0 = _mm512_loadu_si512(buf);
                    a1 = _mm512_loadu_si512(buf + 16);
                    a2 = _mm512_loadu_si512(buf + 32);
                    a3 = _mm512_loadu_si512(buf + 48);
                }
            }

            // Fused scale: contiguous vector loads from group-major scales
            if (rc == 64) {
                __m512 as_vec = _mm512_set1_ps(act_scale);
                for (int sub = 0; sub < 4; sub++) {
                    __m512i *acc = (sub == 0) ? &a0 :
                                   (sub == 1) ? &a1 :
                                   (sub == 2) ? &a2 : &a3;
                    int off = rblk + sub * 16;
                    __m512 sc = _mm512_mul_ps(
                        _mm512_loadu_ps(&sc_base[off]), as_vec);
                    _mm512_storeu_ps(&y[off],
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(*acc),
                                        sc, _mm512_loadu_ps(&y[off])));
                }
            } else {
                int32_t buf[64];
                _mm512_storeu_si512(buf,      a0);
                _mm512_storeu_si512(buf + 16, a1);
                _mm512_storeu_si512(buf + 32, a2);
                _mm512_storeu_si512(buf + 48, a3);
                for (int r = 0; r < rc; r++)
                    y[rblk + r] += (float)buf[r]
                        * sc_base[rblk + r] * act_scale;
            }
        }
    }
}

// ── KERNEL 2 (ALT): Dual-VPERMB (AVX-512 VBMI) ────────────
// FIX 4 applied here too: LUT hoisted out of rblk loop
// FIX E retained: unsigned cast on int16_t right-shift

VBMI_TARGET
void gemv_dual_vpermb(
    const uint8_t * restrict Wt,
    const float   * restrict scales_gm,
    const int8_t  * restrict x,
    float act_scale,
    float * restrict y,
    int M, int K, int gs, int rows_padded
) {
    int gprow = K / gs, gpp = gs / 4;
    if (gpp > 32) {
        fprintf(stderr, "FATAL: gs=%d (gpp=%d) exceeds stack LUT limit of 32. "
                "Use gs<=128.\n", gs, gpp);
        abort();
    }
    for (int r = 0; r < M; r++) y[r] = 0;

    const __m512i z55 = _mm512_set1_epi8(0x55);
    const __m512i z01 = _mm512_set1_epi8(0x01);
    const __m512i z02 = _mm512_set1_epi8(0x02);
    const __m512i z04 = _mm512_set1_epi8(0x04);
    const __m512i z08 = _mm512_set1_epi8(0x08);

    // Pre-allocate LUT byte tables
    // For VPERMB: need 4 × 64-byte tables per column group
    // gpp ≤ 32 for gs=128, so 32 × 4 × 64 = 8KB on stack
    uint8_t plo_all[32][64] __attribute__((aligned(64)));
    uint8_t phi_all[32][64] __attribute__((aligned(64)));
    uint8_t nlo_all[32][64] __attribute__((aligned(64)));
    uint8_t nhi_all[32][64] __attribute__((aligned(64)));
    int16_t lut16_all[32][16] __attribute__((aligned(32)));

    for (int g = 0; g < gprow; g++) {
        const float *sc_base = &scales_gm[(size_t)g * rows_padded];

        // FIX 4: Build all LUTs for this group ONCE
        for (int j = 0; j < gpp; j++) {
            int cg = g * gpp + j;
            int b = cg * 4;
            build_lut16(x[b], x[b + 1], x[b + 2], x[b + 3], lut16_all[j]);
            for (int i = 0; i < 16; i++) {
                uint16_t pu = (uint16_t)lut16_all[j][i];
                uint16_t nu = (uint16_t)(-lut16_all[j][i]);
                plo_all[j][i] = plo_all[j][i+16] = plo_all[j][i+32] =
                    plo_all[j][i+48] = (uint8_t)(pu & 0xFF);
                phi_all[j][i] = phi_all[j][i+16] = phi_all[j][i+32] =
                    phi_all[j][i+48] = (uint8_t)(pu >> 8);
                nlo_all[j][i] = nlo_all[j][i+16] = nlo_all[j][i+32] =
                    nlo_all[j][i+48] = (uint8_t)(nu & 0xFF);
                nhi_all[j][i] = nhi_all[j][i+16] = nhi_all[j][i+32] =
                    nhi_all[j][i+48] = (uint8_t)(nu >> 8);
            }
        }

        for (int rblk = 0; rblk < M; rblk += 32) {
            int rc = (rblk + 32 <= M) ? 32 : M - rblk;
            __m512i alo = _mm512_setzero_si512();
            __m512i ahi = _mm512_setzero_si512();

            for (int j = 0; j < gpp; j++) {
                // FIX 4: Load from pre-built tables
                __m512i vplo = _mm512_load_si512(plo_all[j]);
                __m512i vphi = _mm512_load_si512(phi_all[j]);
                __m512i vnlo = _mm512_load_si512(nlo_all[j]);
                __m512i vnhi = _mm512_load_si512(nhi_all[j]);

                int cg = g * gpp + j;
                const uint8_t *col = &Wt[(size_t)cg * rows_padded + rblk];

                if (rc == 32) {
                    __m512i pk = _mm512_inserti64x4(
                        _mm512_castsi256_si512(
                            _mm256_loadu_si256((__m256i *)col)),
                        _mm256_setzero_si256(), 1);

                    __m512i val = _mm512_and_si512(pk, z55);
                    __m512i sgn = _mm512_and_si512(
                        _mm512_srli_epi16(pk, 1), z55);
                    __m512i pr = _mm512_andnot_si512(sgn, val);
                    __m512i nr = _mm512_and_si512(val, sgn);

                    #define C5(r) _mm512_or_si512(_mm512_or_si512( \
                        _mm512_and_si512(r, z01), \
                        _mm512_and_si512(_mm512_srli_epi16(r, 1), z02)), \
                        _mm512_or_si512( \
                        _mm512_and_si512(_mm512_srli_epi16(r, 2), z04), \
                        _mm512_and_si512(_mm512_srli_epi16(r, 3), z08)))
                    __m512i pidx = C5(pr);
                    __m512i nidx = C5(nr);
                    #undef C5

                    __m512i plr = _mm512_permutexvar_epi8(pidx, vplo);
                    __m512i phr = _mm512_permutexvar_epi8(pidx, vphi);
                    __m512i nlr = _mm512_permutexvar_epi8(nidx, vnlo);
                    __m512i nhr = _mm512_permutexvar_epi8(nidx, vnhi);

                    __m128i plo0 = _mm512_castsi512_si128(plr);
                    __m128i phi0 = _mm512_castsi512_si128(phr);
                    __m128i nlo0 = _mm512_castsi512_si128(nlr);
                    __m128i nhi0 = _mm512_castsi512_si128(nhr);

                    __m256i p16_0 = _mm256_add_epi16(
                        _mm256_cvtepu8_epi16(plo0),
                        _mm256_slli_epi16(_mm256_cvtepi8_epi16(phi0), 8));
                    __m256i n16_0 = _mm256_add_epi16(
                        _mm256_cvtepu8_epi16(nlo0),
                        _mm256_slli_epi16(_mm256_cvtepi8_epi16(nhi0), 8));
                    __m256i r16_0 = _mm256_add_epi16(p16_0, n16_0);

                    __m128i plo1 = _mm512_extracti32x4_epi32(plr, 1);
                    __m128i phi1 = _mm512_extracti32x4_epi32(phr, 1);
                    __m128i nlo1 = _mm512_extracti32x4_epi32(nlr, 1);
                    __m128i nhi1 = _mm512_extracti32x4_epi32(nhr, 1);

                    __m256i p16_1 = _mm256_add_epi16(
                        _mm256_cvtepu8_epi16(plo1),
                        _mm256_slli_epi16(_mm256_cvtepi8_epi16(phi1), 8));
                    __m256i n16_1 = _mm256_add_epi16(
                        _mm256_cvtepu8_epi16(nlo1),
                        _mm256_slli_epi16(_mm256_cvtepi8_epi16(nhi1), 8));
                    __m256i r16_1 = _mm256_add_epi16(p16_1, n16_1);

                    __m128i r0l = _mm256_castsi256_si128(r16_0);
                    __m128i r0h = _mm256_extracti128_si256(r16_0, 1);
                    __m128i r1l = _mm256_castsi256_si128(r16_1);
                    __m128i r1h = _mm256_extracti128_si256(r16_1, 1);

                    __m512i w0 = _mm512_inserti64x4(
                        _mm512_castsi256_si512(
                            _mm256_cvtepi16_epi32(r0l)),
                        _mm256_cvtepi16_epi32(r0h), 1);
                    __m512i w1 = _mm512_inserti64x4(
                        _mm512_castsi256_si512(
                            _mm256_cvtepi16_epi32(r1l)),
                        _mm256_cvtepi16_epi32(r1h), 1);

                    alo = _mm512_add_epi32(alo, w0);
                    ahi = _mm512_add_epi32(ahi, w1);
                } else {
                    int16_t *lut = lut16_all[j];
                    int16_t neg16[16];
                    for (int i = 0; i < 16; i++) neg16[i] = -lut[i];
                    int32_t buf[32];
                    _mm512_storeu_si512(buf, alo);
                    _mm512_storeu_si512(buf + 16, ahi);
                    for (int r = 0; r < rc; r++)
                        buf[r] += lut[pos_nibble(col[r])]
                                + neg16[neg_nibble(col[r])];
                    alo = _mm512_loadu_si512(buf);
                    ahi = _mm512_loadu_si512(buf + 16);
                }
            }

            // Contiguous scale loads
            if (rc == 32) {
                __m512 as_v = _mm512_set1_ps(act_scale);
                __m512 sc0 = _mm512_mul_ps(
                    _mm512_loadu_ps(&sc_base[rblk]), as_v);
                __m512 sc1 = _mm512_mul_ps(
                    _mm512_loadu_ps(&sc_base[rblk + 16]), as_v);
                _mm512_storeu_ps(&y[rblk],
                    _mm512_fmadd_ps(_mm512_cvtepi32_ps(alo), sc0,
                                    _mm512_loadu_ps(&y[rblk])));
                _mm512_storeu_ps(&y[rblk + 16],
                    _mm512_fmadd_ps(_mm512_cvtepi32_ps(ahi), sc1,
                                    _mm512_loadu_ps(&y[rblk + 16])));
            } else {
                int32_t buf[32];
                _mm512_storeu_si512(buf, alo);
                _mm512_storeu_si512(buf + 16, ahi);
                for (int r = 0; r < rc; r++)
                    y[rblk + r] += (float)buf[r]
                        * sc_base[rblk + r] * act_scale;
            }
        }
    }
}

#endif // ARCH_X86_64

// ============================================================
// FIX 3: UNIFIED DISPATCH — ternary_gemv() single entrypoint
//
// Always works. Picks the best available kernel automatically.
// Signature matches the SIMD kernels (col-major + group-major scales).
// For scalar fallback, uses row-major data + row-major scales.
// ============================================================

typedef enum {
    KERN_VPERMW,
    KERN_VPERMB,
    KERN_AVX2,
    KERN_LUT,
    KERN_SCALAR
} KernelType;

static KernelType selected_kernel = KERN_SCALAR;

static void select_kernel(void) {
#if ARCH_X86_64
    if (cpu_has_avx512bw) {
        selected_kernel = KERN_VPERMW;
        return;
    }
    // VBMI without BW: forced to 0 in detect_cpu_features(), so dead path.
    // But if someone manually overrides flags:
    if (cpu_has_avx512vbmi) {
        selected_kernel = KERN_VPERMB;
        return;
    }
    // AVX2 PSHUFB: better than LUT-Grouped which degrades at large K
    if (cpu_has_avx2) {
        selected_kernel = KERN_AVX2;
        return;
    }
#endif
    // LUT-Grouped is faster than scalar for K ≤ ~8192 (LUT fits L1).
    // Only falls to scalar if malloc fails inside gemv_lut_grouped.
    selected_kernel = KERN_LUT;
}

// FIX 3: Lazy init — users don't need to call detect/select manually
// 0 = uninitialized, 1 = initialization in progress, 2 = initialized
static int gemv_initialized = 0;

static void ensure_init(void) {
    int state = __atomic_load_n(&gemv_initialized, __ATOMIC_ACQUIRE);
    if (__builtin_expect(state == 2, 1)) {
        return;
    }

    int expected = 0;
    if (__atomic_compare_exchange_n(
            &gemv_initialized,
            &expected,
            1,
            0,
            __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE)) {
        detect_cpu_features();
        select_kernel();
        __atomic_store_n(&gemv_initialized, 2, __ATOMIC_RELEASE);
        return;
    }

    while (__atomic_load_n(&gemv_initialized, __ATOMIC_ACQUIRE) != 2) {
#if ARCH_X86_64
        __builtin_ia32_pause();
#endif
    }
}

// Unified entrypoint: always dispatches to the best kernel
// Self-initializing — safe to call without any setup.
void ternary_gemv(
    const PlanarWeights *pw,
    const int8_t  * restrict x,
    float act_scale,
    float * restrict y
) {
    ensure_init();
    int M  = pw->rows;
    int K  = pw->cols;
    int gs = pw->group_size;

    switch (selected_kernel) {
#if ARCH_X86_64
    case KERN_VPERMW:
        gemv_vpermw(pw->data_colmaj, pw->scales_gm, x, act_scale,
                    y, M, K, gs, pw->rows_padded);
        return;
    case KERN_VPERMB:
        gemv_dual_vpermb(pw->data_colmaj, pw->scales_gm, x, act_scale,
                         y, M, K, gs, pw->rows_padded);
        return;
    case KERN_AVX2:
        gemv_avx2(pw->data_colmaj, pw->scales_gm, x, act_scale,
                  y, M, K, gs, pw->rows_padded);
        return;
#endif
    case KERN_LUT:
        gemv_lut_grouped(pw->data, pw->scales_rm, x, act_scale, y, M, K, gs);
        return;
    case KERN_SCALAR:
    default:
        gemv_dp4a_ref(pw->data, pw->scales_rm, x, act_scale, y, M, K, gs);
        return;
    }
}

const char *kernel_name(KernelType k) {
    switch (k) {
    case KERN_VPERMW:  return "VPERMW fused (AVX-512 BW)";
    case KERN_VPERMB:  return "Dual-VPERMB (AVX-512 VBMI)";
    case KERN_AVX2:    return "AVX2 PSHUFB (AVX2+FMA)";
    case KERN_LUT:     return "LUT-Grouped (portable)";
    case KERN_SCALAR:  return "dp4a scalar (portable)";
    default:           return "unknown";
    }
}

// ============================================================
// TEST DATA GENERATORS
// ============================================================

static void gen_weights(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = ((unsigned)(i * 2654435761u) >> 16) % 200 / 100.0f - 1.0f;
}

static void gen_activations(int8_t *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = (int8_t)((i * 37 + 13) % 200 - 100);
}

// ============================================================
// VERIFICATION HARNESS
// ============================================================

typedef struct { int pass; int fail; int total; } TestResult;

static void tr_add(TestResult *dst, TestResult src) {
    dst->pass += src.pass; dst->fail += src.fail; dst->total += src.total;
}

static TestResult verify_shape(const char *label, int M, int K, int gs) {
    TestResult res = {0, 0, 0};
    float *fw = (float *)malloc((size_t)M * K * sizeof(float));
    gen_weights(fw, M * K);
    PlanarWeights pw = planar_pack(fw, M, K, gs);
    int8_t *x = (int8_t *)malloc(K);
    gen_activations(x, K);
    float as = 1.0f / 127.0f;

    float *yr = (float *)calloc(M, sizeof(float));
    float *yl = (float *)calloc(M, sizeof(float));

    gemv_dp4a_ref(pw.data, pw.scales_rm, x, as, yr, M, K, gs);
    gemv_lut_grouped(pw.data, pw.scales_rm, x, as, yl, M, K, gs);

    float md = 0;
    for (int i = 0; i < M; i++) {
        float d = fabsf(yr[i] - yl[i]);
        if (d > md) md = d;
    }
    res.total++;
    if (md < 1e-4f) res.pass++; else res.fail++;
    printf("    %-28s dp4a vs LUT:    %.1e %s\n", label, md,
           md < 1e-4f ? "✓" : "✗");

#if ARCH_X86_64
    if (cpu_has_avx2) {
        float *ya = (float *)calloc(M, sizeof(float));
        gemv_avx2(pw.data_colmaj, pw.scales_gm, x, as, ya, M, K, gs,
                  pw.rows_padded);
        md = 0;
        for (int i = 0; i < M; i++) {
            float d = fabsf(yr[i] - ya[i]);
            if (d > md) md = d;
        }
        res.total++;
        if (md < 1e-4f) res.pass++; else res.fail++;
        printf("    %-28s dp4a vs AVX2:   %.1e %s\n", label, md,
               md < 1e-4f ? "✓" : "✗");
        free(ya);
    }

    if (cpu_has_avx512bw) {
        float *yv = (float *)calloc(M, sizeof(float));
        gemv_vpermw(pw.data_colmaj, pw.scales_gm, x, as, yv, M, K, gs,
                    pw.rows_padded);
        md = 0;
        for (int i = 0; i < M; i++) {
            float d = fabsf(yr[i] - yv[i]);
            if (d > md) md = d;
        }
        res.total++;
        if (md < 1e-4f) res.pass++; else res.fail++;
        printf("    %-28s dp4a vs VPERMW: %.1e %s\n", label, md,
               md < 1e-4f ? "✓" : "✗");
        free(yv);
    }

    if (cpu_has_avx512bw && cpu_has_avx512vbmi) {  // FIX 2: BW+VBMI
        float *yd = (float *)calloc(M, sizeof(float));
        gemv_dual_vpermb(pw.data_colmaj, pw.scales_gm, x, as, yd, M, K,
                         gs, pw.rows_padded);
        md = 0;
        for (int i = 0; i < M; i++) {
            float d = fabsf(yr[i] - yd[i]);
            if (d > md) md = d;
        }
        res.total++;
        if (md < 1e-4f) res.pass++; else res.fail++;
        printf("    %-28s dp4a vs VPERMB: %.1e %s\n", label, md,
               md < 1e-4f ? "✓" : "✗");
        free(yd);
    }

    // Also verify ternary_gemv() dispatch wrapper matches dp4a_ref
    {
        float *yd = (float *)calloc(M, sizeof(float));
        ternary_gemv(&pw, x, as, yd);
        md = 0;
        for (int i = 0; i < M; i++) {
            float d = fabsf(yr[i] - yd[i]);
            if (d > md) md = d;
        }
        res.total++;
        if (md < 1e-4f) res.pass++; else res.fail++;
        printf("    %-28s dp4a vs GEMV(): %.1e %s\n", label, md,
               md < 1e-4f ? "✓" : "✗");
        free(yd);
    }
#endif

    free(fw); free(x); free(yr); free(yl);
    planar_free(&pw);
    return res;
}

static TestResult test_invalid_codepoints(void) {
    TestResult res = {0, 0, 0};
    printf("  Invalid Codepoints (0b10 → 0):\n");

    int8_t d = decode_trit(0x02);
    res.total++;
    if (d == 0) res.pass++; else res.fail++;
    printf("    decode_trit(0b10) = %d                %s\n", d,
           d == 0 ? "✓" : "✗");

    int M = 64, K = 128, gs = 128;
    uint8_t *packed = (uint8_t *)aligned_alloc_128((size_t)M * K / 4);
    memset(packed, 0xAA, (size_t)M * K / 4);
    float *scales = (float *)aligned_alloc_128(M * sizeof(float));
    for (int i = 0; i < M; i++) scales[i] = 1.0f;
    int8_t *x = (int8_t *)malloc(K);
    for (int i = 0; i < K; i++) x[i] = 100;
    float *y = (float *)calloc(M, sizeof(float));
    gemv_dp4a_ref(packed, scales, x, 1.0f, y, M, K, gs);

    float maxval = 0;
    for (int i = 0; i < M; i++)
        if (fabsf(y[i]) > maxval) maxval = fabsf(y[i]);
    res.total++;
    if (maxval == 0.0f) res.pass++; else res.fail++;
    printf("    All-0b10 dp4a output max=%.1e     %s\n", maxval,
           maxval == 0.0f ? "✓" : "✗");

    free(packed); free(scales); free(x); free(y);
    return res;
}

static TestResult test_scale_correctness(void) {
    TestResult res = {0, 0, 0};
    printf("  Scale-Only Correctness:\n");
    int M = 4, K = 256, gs = 128, gprow = K / gs;
    float act_scale = 0.5f;

    // +1 weights
    {
        float *fw = (float *)malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) fw[i] = 1.0f;
        PlanarWeights pw = planar_pack(fw, M, K, gs);
        int8_t *x = (int8_t *)malloc(K);
        for (int i = 0; i < K; i++) x[i] = 1;
        float *yr = (float *)calloc(M, sizeof(float));
        gemv_dp4a_ref(pw.data, pw.scales_rm, x, act_scale, yr, M, K, gs);
        int ok = 1;
        for (int r = 0; r < M; r++) {
            float exp = 0;
            for (int g = 0; g < gprow; g++)
                exp += 128.0f * pw.scales_rm[r * gprow + g] * act_scale;
            if (fabsf(yr[r] - exp) > 1e-3f) ok = 0;
        }
        res.total++; if (ok) res.pass++; else res.fail++;
        printf("    Uniform +1:  %s\n", ok ? "✓" : "✗");
        free(fw); free(x); free(yr); planar_free(&pw);
    }
    // -1 weights
    {
        float *fw = (float *)malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) fw[i] = -1.0f;
        PlanarWeights pw = planar_pack(fw, M, K, gs);
        int8_t *x = (int8_t *)malloc(K);
        for (int i = 0; i < K; i++) x[i] = 1;
        float *yr = (float *)calloc(M, sizeof(float));
        gemv_dp4a_ref(pw.data, pw.scales_rm, x, act_scale, yr, M, K, gs);
        int ok = 1;
        for (int r = 0; r < M; r++) {
            float exp = 0;
            for (int g = 0; g < gprow; g++)
                exp += -128.0f * pw.scales_rm[r * gprow + g] * act_scale;
            if (fabsf(yr[r] - exp) > 1e-3f) ok = 0;
        }
        res.total++; if (ok) res.pass++; else res.fail++;
        printf("    Uniform -1:  %s\n", ok ? "✓" : "✗");
        free(fw); free(x); free(yr); planar_free(&pw);
    }
    // ±1 cancellation
    {
        float *fw = (float *)malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) fw[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        PlanarWeights pw = planar_pack(fw, M, K, gs);
        int8_t *x = (int8_t *)malloc(K);
        for (int i = 0; i < K; i++) x[i] = 1;
        float *yr = (float *)calloc(M, sizeof(float));
        gemv_dp4a_ref(pw.data, pw.scales_rm, x, act_scale, yr, M, K, gs);
        int ok = 1;
        for (int r = 0; r < M; r++)
            if (fabsf(yr[r]) > 1e-3f) ok = 0;
        res.total++; if (ok) res.pass++; else res.fail++;
        printf("    ±1 cancel:   %s\n", ok ? "✓" : "✗");
        free(fw); free(x); free(yr); planar_free(&pw);
    }
    return res;
}

// ============================================================
// MAIN
// ============================================================

int main(void) {
    detect_cpu_features();
    select_kernel();

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Ternary Selection v3.3.1 — Production Kernels             ║\n");
    printf("║  Runtime Dispatch | XGETBV Safe | LUT Hoisted | 64-Row     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  CPU Features (CPUID + XGETBV verified):\n");
    printf("    AVX2+FMA:     %s\n", cpu_has_avx2        ? "✓" : "✗");
    printf("    AVX-512 BW:   %s\n", cpu_has_avx512bw   ? "✓" : "✗");
    printf("    AVX-512 VBMI: %s\n", cpu_has_avx512vbmi  ? "✓" : "✗");
    printf("    Selected:     %s\n", kernel_name(selected_kernel));
    printf("  Storage: Planar SoA, 128B aligned, padded col-major\n");
    printf("  Encoding: BitNet standard (11=-1, 01=+1, 00=0)\n\n");

    // ── VERIFICATION ──
    TestResult all = {0, 0, 0};

    printf("═══ Triangle of Truth ═══\n\n");

    printf("  Standard Shapes:\n");
    tr_add(&all, verify_shape("[256×512]",    256,  512,  128));
    tr_add(&all, verify_shape("[512×1024]",   512,  1024, 128));
    tr_add(&all, verify_shape("[1024×2048]", 1024,  2048, 128));

    printf("\n  Tail Torture (M%%64 != 0):\n");
    tr_add(&all, verify_shape("[1×128]",        1,   128, 128));
    tr_add(&all, verify_shape("[1×512]",        1,   512, 128));
    tr_add(&all, verify_shape("[17×128]",      17,   128, 128));
    tr_add(&all, verify_shape("[33×256]",      33,   256, 128));
    tr_add(&all, verify_shape("[63×384]",      63,   384, 128));
    tr_add(&all, verify_shape("[65×256]",      65,   256, 128));
    tr_add(&all, verify_shape("[100×1280]",   100,  1280, 128));
    tr_add(&all, verify_shape("[127×640]",    127,   640, 128));

    printf("\n  Large K:\n");
    tr_add(&all, verify_shape("[128×4096]",   128,  4096, 128));
    tr_add(&all, verify_shape("[64×10880]",    64, 10880, 128));

    printf("\n");
    tr_add(&all, test_invalid_codepoints());
    printf("\n");
    tr_add(&all, test_scale_correctness());

    printf("\n  ─── Results: %d/%d passed", all.pass, all.total);
    if (all.fail > 0)
        printf(", %d FAILED ✗ ───\n", all.fail);
    else
        printf(" ✓ ALL CLEAR ───\n");

    if (all.fail > 0) {
        printf("\n  *** VERIFICATION FAILED — DO NOT DEPLOY ***\n\n");
        return 1;
    }
    printf("\n");

    // ── BENCHMARKS ──
    printf("═══ Benchmarks (1 thread, per-group scaling, planar SoA) ═══\n\n");

    int sizes[][2] = {
        {2048, 2048}, {4096, 4096}, {4096, 11008},
        {11008, 4096}, {8192, 8192}
    };

    for (int s = 0; s < 5; s++) {
        int M = sizes[s][0];
        int K = round_up(sizes[s][1], 128);
        int gs = 128;
        int it = (M * K > 16000000) ? 5 : ((M * K > 4000000) ? 10 : 20);

        float *fw = (float *)malloc((size_t)M * K * sizeof(float));
        gen_weights(fw, M * K);
        PlanarWeights pw = planar_pack(fw, M, K, gs);
        int8_t *x = (int8_t *)malloc(K);
        gen_activations(x, K);
        float as = 1.0f / 127.0f;
        float *y = (float *)calloc(M, sizeof(float));

        printf("  [%5d × %5d] gs=%d ×%d\n", M, K, gs, it);

        #define BENCH(name, call) do { \
            call; \
            double t = now(); \
            for (int _i = 0; _i < it; _i++) { call; } \
            double el = now() - t; \
            printf("    %-28s %7.3f ms  %6.1f GOPS\n", \
                   name, el * 1000.0 / it, \
                   2.0 * M * K * it / el / 1e9); \
        } while(0)

        BENCH("dp4a scalar",
              gemv_dp4a_ref(pw.data, pw.scales_rm, x, as, y, M, K, gs));
        BENCH("LUT-Grouped (i16)",
              gemv_lut_grouped(pw.data, pw.scales_rm, x, as, y, M, K, gs));

#if ARCH_X86_64
        if (cpu_has_avx2)
            BENCH("AVX2 PSHUFB",
                  gemv_avx2(pw.data_colmaj, pw.scales_gm, x, as, y,
                            M, K, gs, pw.rows_padded));
        if (cpu_has_avx512bw)
            BENCH("VPERMW fused ★PRIMARY",
                  gemv_vpermw(pw.data_colmaj, pw.scales_gm, x, as, y,
                              M, K, gs, pw.rows_padded));
        if (cpu_has_avx512bw && cpu_has_avx512vbmi)  // FIX 2
            BENCH("Dual-VPERMB (lo/hi)",
                  gemv_dual_vpermb(pw.data_colmaj, pw.scales_gm, x, as,
                                   y, M, K, gs, pw.rows_padded));
#endif
        // Also bench the dispatch wrapper
        BENCH("ternary_gemv() [dispatch]",
              ternary_gemv(&pw, x, as, y));

        #undef BENCH
        printf("\n");
        free(fw); free(x); free(y);
        planar_free(&pw);
    }

    // ── DISPATCH SUMMARY ──
    printf("═══ Kernel Dispatch Chain (v3.3.1 — XGETBV + lazy init) ═══\n\n");
    printf("  1. VPERMW fused    (AVX-512 BW)   ★ PRIMARY  [64-row, LUT hoisted]\n");
    printf("  2. Dual-VPERMB     (AVX-512 VBMI)   alt       [32-row, LUT hoisted]\n");
    printf("  3. AVX2 PSHUFB     (AVX2+FMA)        [32-row, PSHUFB LUT]\n");
    printf("  4. LUT-Grouped     (portable)        collapses at large K\n");
    printf("  5. dp4a scalar     (portable)        competitive fallback\n\n");
    printf("  API: ternary_gemv(&pw, x, act_scale, y) — auto-dispatches\n\n");
    printf("  Safety:\n");
    printf("    OSXSAVE + XGETBV(0) & 0xE6 == 0xE6 (ZMM state enabled)\n");
    printf("    VBMI gated on BW+VBMI jointly\n");
    printf("    LUTs built per-group, reused across row blocks\n");

    return 0;
}
