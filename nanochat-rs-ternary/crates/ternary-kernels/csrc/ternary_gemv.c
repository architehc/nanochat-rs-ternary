// ternary_final.c — Production ternary selection kernels v3.4.0
//
// V3.4.0 CHANGES (The AVX2 Rebirth):
//   1) Restored highly-optimized AVX2 Kernel ("Nibble-Split" LUT).
//   2) 16-bit vertical accumulation eliminates YMM register spilling & FMA in hot loop.
//   3) Fixed XGETBV trap that locked out AVX2 if OS lacked AVX512 state.
//   4) Unconditionalized inner loops (zero-padded buffers safely allow tail vectorization).
//
// Compile: gcc -O3 -o ternary_final ternary_final.c -lm -mavx2 -mfma
//   (No -mavx512* flags needed — runtime dispatch handles it natively)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// ============================================================
// CPU FEATURE DETECTION (Safe & Segmented)
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

    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < 7) return;

    __cpuid(1, eax, ebx, ecx, edx);
    int has_osxsave = (ecx >> 27) & 1;
    int has_fma     = (ecx >> 12) & 1; // Require FMA for fused scaling
    if (!has_osxsave) return;

    unsigned int xcr0_lo, xcr0_hi;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));

    // Separate checking for OS support of YMM (bit 1,2) vs ZMM (bit 5,6,7)
    int os_avx_support    = ((xcr0_lo & 6) == 6);
    int os_avx512_support = ((xcr0_lo & 0xE6) == 0xE6);

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    
    // Fix: AVX2 activates correctly even if AVX512 isn't supported by Hypervisor/CPU
    if (os_avx_support) {
        cpu_has_avx2 = ((ebx >> 5) & 1) && has_fma;
    }
    
    if (os_avx512_support) {
        cpu_has_avx512bw   = (ebx >> 30) & 1;
        cpu_has_avx512vbmi = (ecx >> 1) & 1;
        if (!cpu_has_avx512bw) cpu_has_avx512vbmi = 0;
    }
#endif
}

// ============================================================
// Target Attributes
// ============================================================

#if ARCH_X86_64
#include <immintrin.h>
#define AVX2_TARGET   __attribute__((target("avx2,fma")))
#define AVX512_TARGET __attribute__((target("avx512f,avx512bw")))
#define VBMI_TARGET   __attribute__((target("avx512f,avx512bw,avx512vbmi")))
#else
#define AVX2_TARGET
#define AVX512_TARGET
#define VBMI_TARGET
#endif

// ============================================================
// Utility & Memory
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

static inline int round_up(int x, int align) {
    return (x + align - 1) / align * align;
}

static inline int8_t decode_trit(uint8_t bits) {
    int8_t nz = (int8_t)(bits & 1);
    int8_t sg = (int8_t)((bits >> 1) & 1);
    return nz - 2 * (nz & sg);
}

// ============================================================
// PLANAR STORAGE
// ============================================================

typedef struct {
    uint8_t *data;          
    uint8_t *data_colmaj;   
    float   *scales_rm;     
    float   *scales_gm;     
    int      rows, cols, group_size;
    int      rows_padded;
} PlanarWeights;

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
            pw.data_colmaj[(size_t)c * pw.rows_padded + r] = pw.data[(size_t)r * kp + c];
    return pw;
}

void planar_free(PlanarWeights *pw) {
    free(pw->data); free(pw->data_colmaj);
    free(pw->scales_rm); free(pw->scales_gm);
}

// ============================================================
// PORTABLE KERNELS 
// ============================================================

void gemv_dp4a_ref(const uint8_t * restrict data, const float * restrict scales_rm,
                   const int8_t  * restrict x,    float act_scale,
                   float * restrict y, int M, int K, int gs) {
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

void gemv_lut_grouped(const uint8_t * restrict data, const float * restrict scales_rm,
                      const int8_t  * restrict x,    float act_scale,
                      float * restrict y, int M, int K, int gs) {
    int kp = K / 4, gprow = K / gs, gpp = gs / 4;
    int16_t *luts = (int16_t *)malloc((size_t)kp * 256 * sizeof(int16_t));
    for (int cg = 0; cg < kp; cg++) {
        int b = cg * 4;
        int16_t a0 = x[b], a1 = x[b+1], a2 = x[b+2], a3 = x[b+3];
        for (int p = 0; p < 256; p++)
            luts[(size_t)cg * 256 + p] = 
                (int16_t)decode_trit( p     & 3) * a0 + (int16_t)decode_trit((p>>2) & 3) * a1 +
                (int16_t)decode_trit((p>>4) & 3) * a2 + (int16_t)decode_trit((p>>6) & 3) * a3;
    }
    for (int r = 0; r < M; r++) {
        float racc = 0;
        const uint8_t *wr = &data[(size_t)r * kp];
        for (int g = 0; g < gprow; g++) {
            int32_t gacc = 0;
            int cs = g * gpp;
            for (int j = 0; j < gpp; j++) gacc += luts[(size_t)(cs + j) * 256 + wr[cs + j]];
            racc += (float)gacc * scales_rm[r * gprow + g] * act_scale;
        }
        y[r] = racc;
    }
    free(luts);
}

// ============================================================
// SIMD HELPERS
// ============================================================

static void build_lut16(int16_t a0, int16_t a1, int16_t a2, int16_t a3, int16_t out[16]) {
    for (int m = 0; m < 16; m++) {
        int16_t s = 0;
        if (m & 1) s += a0; if (m & 2) s += a1;
        if (m & 4) s += a2; if (m & 8) s += a3;
        out[m] = s;
    }
}

// ============================================================
// NEW AVX2 KERNEL ("Nibble-Split" MatMul-Free)
// ============================================================

#if ARCH_X86_64

AVX2_TARGET
void gemv_avx2_lut(const uint8_t * restrict Wt, const float * restrict scales_gm,
                   const int8_t  * restrict x,  float act_scale,
                   float * restrict y, int M, int K, int gs, int rows_padded) {
    int gprow = K / gs, gpp = gs / 4;
    if (gpp > 64) { fprintf(stderr, "FATAL: gs=%d exceeds AVX2 16-bit acc limits.\n", gs); abort(); }
    for (int r = 0; r < M; r++) y[r] = 0;

    const __m256i m0F = _mm256_set1_epi8(0x0F);

    __m256i v_llo[64] __attribute__((aligned(32))); __m256i v_lhi[64] __attribute__((aligned(32)));
    __m256i v_hlo[64] __attribute__((aligned(32))); __m256i v_hhi[64] __attribute__((aligned(32)));

    for (int g = 0; g < gprow; g++) {
        const float *sc_base = &scales_gm[(size_t)g * rows_padded];

        // Hoisted Nibble-Split LUT Generation 
        for (int j = 0; j < gpp; j++) {
            int b = (g * gpp + j) * 4;
            int16_t a0 = x[b], a1 = x[b+1], a2 = x[b+2], a3 = x[b+3];
            uint8_t llo[32], lhi[32], hlo[32], hhi[32];
            
            for (int m = 0; m < 16; m++) {
                int16_t s_lo = decode_trit(m & 3) * a0 + decode_trit((m >> 2) & 3) * a1;
                int16_t s_hi = decode_trit(m & 3) * a2 + decode_trit((m >> 2) & 3) * a3;
                
                // Mirroring to both 128-bit lanes (0-15 and 16-31) bypasses AVX2 lane-crossing penalties
                llo[m] = llo[m+16] = (uint8_t)(s_lo & 0xFF);
                lhi[m] = lhi[m+16] = (uint8_t)((s_lo >> 8) & 0xFF);
                hlo[m] = hlo[m+16] = (uint8_t)(s_hi & 0xFF);
                hhi[m] = hhi[m+16] = (uint8_t)((s_hi >> 8) & 0xFF);
            }
            v_llo[j] = _mm256_loadu_si256((__m256i*)llo); v_lhi[j] = _mm256_loadu_si256((__m256i*)lhi);
            v_hlo[j] = _mm256_loadu_si256((__m256i*)hlo); v_hhi[j] = _mm256_loadu_si256((__m256i*)hhi);
        }

        for (int rblk = 0; rblk < M; rblk += 32) {
            int rc = (rblk + 32 <= M) ? 32 : M - rblk;
            
            // Magic Trick: Accumulators natively target int16_t, preventing all register spilling
            __m256i acc_lo = _mm256_setzero_si256();
            __m256i acc_hi = _mm256_setzero_si256();

            // Unconditional vector loop safely sweeps padded zero values without branches
            for (int j = 0; j < gpp; j++) {
                __m256i pk = _mm256_loadu_si256((__m256i*)&Wt[(size_t)(g * gpp + j) * rows_padded + rblk]);

                __m256i idx_lo = _mm256_and_si256(pk, m0F);
                __m256i idx_hi = _mm256_and_si256(_mm256_srli_epi16(pk, 4), m0F);

                __m256i slo_lo = _mm256_shuffle_epi8(v_llo[j], idx_lo);
                __m256i slo_hi = _mm256_shuffle_epi8(v_lhi[j], idx_lo);
                __m256i shi_lo = _mm256_shuffle_epi8(v_hlo[j], idx_hi);
                __m256i shi_hi = _mm256_shuffle_epi8(v_hhi[j], idx_hi);

                // Natively reconstruct 16-bit integers with unpacking. Zero FP operations inside the loop! 
                __m256i sum_16_lo = _mm256_add_epi16(_mm256_unpacklo_epi8(slo_lo, slo_hi), _mm256_unpacklo_epi8(shi_lo, shi_hi));
                __m256i sum_16_hi = _mm256_add_epi16(_mm256_unpackhi_epi8(slo_lo, slo_hi), _mm256_unpackhi_epi8(shi_lo, shi_hi));

                acc_lo = _mm256_add_epi16(acc_lo, sum_16_lo);
                acc_hi = _mm256_add_epi16(acc_hi, sum_16_hi);
            }

            // Expanding 16-bit accumulations cleanly out to 32-bit Float arrays
            __m256 f0_7   = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_lo)));
            __m256 f16_23 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_lo, 1)));
            __m256 f8_15  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_hi)));
            __m256 f24_31 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_hi, 1)));

            __m256 vas = _mm256_set1_ps(act_scale);

            if (rc == 32) {
                _mm256_storeu_ps(&y[rblk + 0],  _mm256_fmadd_ps(f0_7,   _mm256_mul_ps(_mm256_loadu_ps(&sc_base[rblk + 0]), vas),  _mm256_loadu_ps(&y[rblk + 0])));
                _mm256_storeu_ps(&y[rblk + 8],  _mm256_fmadd_ps(f8_15,  _mm256_mul_ps(_mm256_loadu_ps(&sc_base[rblk + 8]), vas),  _mm256_loadu_ps(&y[rblk + 8])));
                _mm256_storeu_ps(&y[rblk + 16], _mm256_fmadd_ps(f16_23, _mm256_mul_ps(_mm256_loadu_ps(&sc_base[rblk + 16]), vas), _mm256_loadu_ps(&y[rblk + 16])));
                _mm256_storeu_ps(&y[rblk + 24], _mm256_fmadd_ps(f24_31, _mm256_mul_ps(_mm256_loadu_ps(&sc_base[rblk + 24]), vas), _mm256_loadu_ps(&y[rblk + 24])));
            } else {
                float buf[32];
                _mm256_storeu_ps(buf + 0, f0_7); _mm256_storeu_ps(buf + 8, f8_15);
                _mm256_storeu_ps(buf + 16, f16_23); _mm256_storeu_ps(buf + 24, f24_31);
                for (int r = 0; r < rc; r++) y[rblk + r] += buf[r] * sc_base[rblk + r] * act_scale;
            }
        }
    }
}

// ── AVX-512 VPERMW (Unconditionalized) ──────────
AVX512_TARGET
void gemv_vpermw(const uint8_t * restrict Wt, const float * restrict scales_gm, const int8_t * restrict x, float act_scale, float * restrict y, int M, int K, int gs, int rows_padded) {
    int gprow = K / gs, gpp = gs / 4;
    if (gpp > 32) abort();
    for (int r = 0; r < M; r++) y[r] = 0;

    const __m256i h55 = _mm256_set1_epi8(0x55), h01 = _mm256_set1_epi8(0x01), h02 = _mm256_set1_epi8(0x02), h04 = _mm256_set1_epi8(0x04), h08 = _mm256_set1_epi8(0x08);
    const __m512i z_mask = _mm512_set1_epi16(0x000F);
    int16_t pd_all[32][16] __attribute__((aligned(32))); int16_t nd_all[32][16] __attribute__((aligned(32)));

    for (int g = 0; g < gprow; g++) {
        const float *sc_base = &scales_gm[(size_t)g * rows_padded];
        for (int j = 0; j < gpp; j++) {
            build_lut16(x[(g * gpp + j) * 4], x[(g * gpp + j) * 4 + 1], x[(g * gpp + j) * 4 + 2], x[(g * gpp + j) * 4 + 3], pd_all[j]);
            for (int i = 0; i < 16; i++) nd_all[j][i] = -pd_all[j][i];
        }
        for (int rblk = 0; rblk < M; rblk += 64) {
            int rc = (rblk + 64 <= M) ? 64 : M - rblk;
            __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512(), a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();

            for (int j = 0; j < gpp; j++) {
                __m512i plut = _mm512_broadcast_i64x4(_mm256_load_si256((__m256i *)pd_all[j])), nlut = _mm512_broadcast_i64x4(_mm256_load_si256((__m256i *)nd_all[j]));
                const uint8_t *col = &Wt[(size_t)(g * gpp + j) * rows_padded + rblk];

                for (int sub = 0; sub < 2; sub++) {
                    __m256i pk = _mm256_loadu_si256((__m256i *)(col + sub * 32));
                    __m256i val = _mm256_and_si256(pk, h55), sgn = _mm256_and_si256(_mm256_srli_epi16(pk, 1), h55);
                    __m256i pr = _mm256_andnot_si256(sgn, val), nr = _mm256_and_si256(val, sgn);
                    
                    #define COMPACT(r) _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(r, h01), _mm256_and_si256(_mm256_srli_epi16(r, 1), h02)), _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(r, 2), h04), _mm256_and_si256(_mm256_srli_epi16(r, 3), h08)))
                    __m512i res = _mm512_add_epi16(_mm512_permutexvar_epi16(_mm512_and_si512(_mm512_cvtepu8_epi16(COMPACT(pr)), z_mask), plut), _mm512_permutexvar_epi16(_mm512_and_si512(_mm512_cvtepu8_epi16(COMPACT(nr)), z_mask), nlut));
                    #undef COMPACT

                    if (sub == 0) { a0 = _mm512_add_epi32(a0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(res))); a1 = _mm512_add_epi32(a1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(res, 1))); } 
                    else { a2 = _mm512_add_epi32(a2, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(res))); a3 = _mm512_add_epi32(a3, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(res, 1))); }
                }
            }

            if (rc == 64) {
                __m512 as_vec = _mm512_set1_ps(act_scale);
                for (int sub = 0; sub < 4; sub++) {
                    __m512i *acc = (sub == 0) ? &a0 : (sub == 1) ? &a1 : (sub == 2) ? &a2 : &a3;
                    __m512 sc = _mm512_mul_ps(_mm512_loadu_ps(&sc_base[rblk + sub * 16]), as_vec);
                    _mm512_storeu_ps(&y[rblk + sub * 16], _mm512_fmadd_ps(_mm512_cvtepi32_ps(*acc), sc, _mm512_loadu_ps(&y[rblk + sub * 16])));
                }
            } else {
                float buf[64];
                _mm512_storeu_ps(buf,      _mm512_cvtepi32_ps(a0)); _mm512_storeu_ps(buf + 16, _mm512_cvtepi32_ps(a1));
                _mm512_storeu_ps(buf + 32, _mm512_cvtepi32_ps(a2)); _mm512_storeu_ps(buf + 48, _mm512_cvtepi32_ps(a3));
                for (int r = 0; r < rc; r++) y[rblk + r] += buf[r] * sc_base[rblk + r] * act_scale;
            }
        }
    }
}

// ── KERNEL 2: Dual-VPERMB (AVX-512 VBMI) ────────────
VBMI_TARGET
void gemv_dual_vpermb(const uint8_t * restrict Wt, const float * restrict scales_gm, const int8_t * restrict x, float act_scale, float * restrict y, int M, int K, int gs, int rows_padded) {
    int gprow = K / gs, gpp = gs / 4;
    if (gpp > 32) abort();
    for (int r = 0; r < M; r++) y[r] = 0;

    const __m512i z55 = _mm512_set1_epi8(0x55), z01 = _mm512_set1_epi8(0x01), z02 = _mm512_set1_epi8(0x02), z04 = _mm512_set1_epi8(0x04), z08 = _mm512_set1_epi8(0x08);
    uint8_t plo[32][64] __attribute__((aligned(64))), phi[32][64] __attribute__((aligned(64))), nlo[32][64] __attribute__((aligned(64))), nhi[32][64] __attribute__((aligned(64)));
    int16_t lut16[32][16];

    for (int g = 0; g < gprow; g++) {
        const float *sc_base = &scales_gm[(size_t)g * rows_padded];
        for (int j = 0; j < gpp; j++) {
            build_lut16(x[(g * gpp + j) * 4], x[(g * gpp + j) * 4 + 1], x[(g * gpp + j) * 4 + 2], x[(g * gpp + j) * 4 + 3], lut16[j]);
            for (int i = 0; i < 16; i++) {
                uint16_t pu = (uint16_t)lut16[j][i], nu = (uint16_t)(-lut16[j][i]);
                plo[j][i] = plo[j][i+16] = plo[j][i+32] = plo[j][i+48] = (uint8_t)(pu & 0xFF);
                phi[j][i] = phi[j][i+16] = phi[j][i+32] = phi[j][i+48] = (uint8_t)(pu >> 8);
                nlo[j][i] = nlo[j][i+16] = nlo[j][i+32] = nlo[j][i+48] = (uint8_t)(nu & 0xFF);
                nhi[j][i] = nhi[j][i+16] = nhi[j][i+32] = nhi[j][i+48] = (uint8_t)(nu >> 8);
            }
        }
        for (int rblk = 0; rblk < M; rblk += 32) {
            int rc = (rblk + 32 <= M) ? 32 : M - rblk;
            __m512i alo = _mm512_setzero_si512(), ahi = _mm512_setzero_si512();

            for (int j = 0; j < gpp; j++) {
                __m512i vplo = _mm512_load_si512(plo[j]), vphi = _mm512_load_si512(phi[j]), vnlo = _mm512_load_si512(nlo[j]), vnhi = _mm512_load_si512(nhi[j]);
                __m512i pk = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((__m256i *)&Wt[(size_t)(g * gpp + j) * rows_padded + rblk])), _mm256_setzero_si256(), 1);
                
                __m512i val = _mm512_and_si512(pk, z55), sgn = _mm512_and_si512(_mm512_srli_epi16(pk, 1), z55);
                __m512i pr = _mm512_andnot_si512(sgn, val), nr = _mm512_and_si512(val, sgn);

                #define C5(r) _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(r, z01), _mm512_and_si512(_mm512_srli_epi16(r, 1), z02)), _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(r, 2), z04), _mm512_and_si512(_mm512_srli_epi16(r, 3), z08)))
                __m512i plr = _mm512_permutexvar_epi8(C5(pr), vplo), phr = _mm512_permutexvar_epi8(C5(pr), vphi);
                __m512i nlr = _mm512_permutexvar_epi8(C5(nr), vnlo), nhr = _mm512_permutexvar_epi8(C5(nr), vnhi);
                #undef C5

                __m256i r16_0 = _mm256_add_epi16(_mm256_add_epi16(_mm256_cvtepu8_epi16(_mm512_castsi512_si128(plr)), _mm256_slli_epi16(_mm256_cvtepi8_epi16(_mm512_castsi512_si128(phr)), 8)), _mm256_add_epi16(_mm256_cvtepu8_epi16(_mm512_castsi512_si128(nlr)), _mm256_slli_epi16(_mm256_cvtepi8_epi16(_mm512_castsi512_si128(nhr)), 8)));
                __m256i r16_1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_cvtepu8_epi16(_mm512_extracti32x4_epi32(plr, 1)), _mm256_slli_epi16(_mm256_cvtepi8_epi16(_mm512_extracti32x4_epi32(phr, 1)), 8)), _mm256_add_epi16(_mm256_cvtepu8_epi16(_mm512_extracti32x4_epi32(nlr, 1)), _mm256_slli_epi16(_mm256_cvtepi8_epi16(_mm512_extracti32x4_epi32(nhr, 1)), 8)));

                alo = _mm512_add_epi32(alo, _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(r16_0))), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r16_0, 1)), 1));
                ahi = _mm512_add_epi32(ahi, _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(r16_1))), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r16_1, 1)), 1));
            }

            if (rc == 32) {
                __m512 as_v = _mm512_set1_ps(act_scale);
                _mm512_storeu_ps(&y[rblk],      _mm512_fmadd_ps(_mm512_cvtepi32_ps(alo), _mm512_mul_ps(_mm512_loadu_ps(&sc_base[rblk]), as_v),      _mm512_loadu_ps(&y[rblk])));
                _mm512_storeu_ps(&y[rblk + 16], _mm512_fmadd_ps(_mm512_cvtepi32_ps(ahi), _mm512_mul_ps(_mm512_loadu_ps(&sc_base[rblk + 16]), as_v), _mm512_loadu_ps(&y[rblk + 16])));
            } else {
                float buf[32];
                _mm512_storeu_ps(buf, _mm512_cvtepi32_ps(alo)); _mm512_storeu_ps(buf + 16, _mm512_cvtepi32_ps(ahi));
                for (int r = 0; r < rc; r++) y[rblk + r] += buf[r] * sc_base[rblk + r] * act_scale;
            }
        }
    }
}
#endif // ARCH_X86_64

// ============================================================
// DISPATCH CHAIN
// ============================================================

typedef enum { KERN_VPERMW, KERN_VPERMB, KERN_AVX2, KERN_LUT, KERN_SCALAR } KernelType;

static KernelType selected_kernel = KERN_SCALAR;
static volatile int gemv_initialized = 0;

static void select_kernel(void) {
#if ARCH_X86_64
    if (cpu_has_avx512bw)   { selected_kernel = KERN_VPERMW; return; }
    if (cpu_has_avx512vbmi) { selected_kernel = KERN_VPERMB; return; }
    if (cpu_has_avx2)       { selected_kernel = KERN_AVX2; return; }
#endif
    selected_kernel = KERN_LUT;
}

static void ensure_init(void) {
    if (__builtin_expect(!gemv_initialized, 0)) {
        detect_cpu_features(); select_kernel();
        __atomic_store_n(&gemv_initialized, 1, __ATOMIC_RELEASE);
    }
}

void ternary_gemv(const PlanarWeights *pw, const int8_t * restrict x, float act_scale, float * restrict y) {
    ensure_init();
    switch (selected_kernel) {
#if ARCH_X86_64
    case KERN_VPERMW: gemv_vpermw(pw->data_colmaj, pw->scales_gm, x, act_scale, y, pw->rows, pw->cols, pw->group_size, pw->rows_padded); return;
    case KERN_VPERMB: gemv_dual_vpermb(pw->data_colmaj, pw->scales_gm, x, act_scale, y, pw->rows, pw->cols, pw->group_size, pw->rows_padded); return;
    case KERN_AVX2:   gemv_avx2_lut(pw->data_colmaj, pw->scales_gm, x, act_scale, y, pw->rows, pw->cols, pw->group_size, pw->rows_padded); return;
#endif
    case KERN_LUT:    gemv_lut_grouped(pw->data, pw->scales_rm, x, act_scale, y, pw->rows, pw->cols, pw->group_size); return;
    default:          gemv_dp4a_ref(pw->data, pw->scales_rm, x, act_scale, y, pw->rows, pw->cols, pw->group_size); return;
    }
}

const char *kernel_name(KernelType k) {
    switch (k) {
    case KERN_VPERMW: return "VPERMW fused (AVX-512 BW)";
    case KERN_VPERMB: return "Dual-VPERMB (AVX-512 VBMI)";
    case KERN_AVX2:   return "Nibble-Split LUT (AVX2)";
    case KERN_LUT:    return "LUT-Grouped (portable)";
    default:          return "dp4a scalar (portable)";
    }
}

// ============================================================
// VERIFICATION HARNESS & BENCHMARK
// ============================================================

static void gen_weights(float *w, int n) {
    for (int i = 0; i < n; i++) w[i] = ((unsigned)(i * 2654435761u) >> 16) % 200 / 100.0f - 1.0f;
}

static void gen_activations(int8_t *x, int n) {
    for (int i = 0; i < n; i++) x[i] = (int8_t)((i * 37 + 13) % 200 - 100);
}

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
    for (int i = 0; i < M; i++) { float d = fabsf(yr[i] - yl[i]); if (d > md) md = d; }
    res.total++; if (md < 1e-4f) res.pass++; else res.fail++;
    printf("    %-28s dp4a vs LUT:    %.1e %s\n", label, md, md < 1e-4f ? "✓" : "✗");

#if ARCH_X86_64
    if (cpu_has_avx2) {
        float *ya = (float *)calloc(M, sizeof(float));
        gemv_avx2_lut(pw.data_colmaj, pw.scales_gm, x, as, ya, M, K, gs, pw.rows_padded);
        md = 0;
        for (int i = 0; i < M; i++) { float d = fabsf(yr[i] - ya[i]); if (d > md) md = d; }
        res.total++; if (md < 1e-4f) res.pass++; else res.fail++;
        printf("    %-28s dp4a vs AVX2:   %.1e %s\n", label, md, md < 1e-4f ? "✓" : "✗");
        free(ya);
    }
    if (cpu_has_avx512bw) {
        float *yv = (float *)calloc(M, sizeof(float));
        gemv_vpermw(pw.data_colmaj, pw.scales_gm, x, as, yv, M, K, gs, pw.rows_padded);
        md = 0;
        for (int i = 0; i < M; i++) { float d = fabsf(yr[i] - yv[i]); if (d > md) md = d; }
        res.total++; if (md < 1e-4f) res.pass++; else res.fail++;
        printf("    %-28s dp4a vs VPERMW: %.1e %s\n", label, md, md < 1e-4f ? "✓" : "✗");
        free(yv);
    }
#endif
    {
        float *yd = (float *)calloc(M, sizeof(float));
        ternary_gemv(&pw, x, as, yd);
        md = 0;
        for (int i = 0; i < M; i++) { float d = fabsf(yr[i] - yd[i]); if (d > md) md = d; }
        res.total++; if (md < 1e-4f) res.pass++; else res.fail++;
        printf("    %-28s dp4a vs GEMV(): %.1e %s\n", label, md, md < 1e-4f ? "✓" : "✗");
        free(yd);
    }

    free(fw); free(x); free(yr); free(yl); planar_free(&pw);
    return res;
}

int main(void) {
    detect_cpu_features();
    select_kernel();

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Ternary Selection v3.4.0 — Production Kernels               ║\n");
    printf("║  Runtime Dispatch | AVX2 Restored | 16-bit Fast Vertical Acc ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  CPU Features Detected:\n");
    printf("    AVX2 (+FMA):  %s\n", cpu_has_avx2       ? "✓" : "✗");
    printf("    AVX-512 BW:   %s\n", cpu_has_avx512bw   ? "✓" : "✗");
    printf("    AVX-512 VBMI: %s\n", cpu_has_avx512vbmi ? "✓" : "✗");
    printf("    Selected:     %s\n", kernel_name(selected_kernel));
    printf("  Storage: Planar SoA, 128B aligned, padded col-major\n\n");

    TestResult all = {0, 0, 0};

    printf("═══ Triangle of Truth ═══\n\n");
    printf("  Standard Shapes:\n");
    tr_add(&all, verify_shape("[256×512]",    256,  512,  128));
    tr_add(&all, verify_shape("[512×1024]",   512,  1024, 128));

    printf("\n  Tail Torture (M%%64 != 0):\n");
    tr_add(&all, verify_shape("[65×256]",     65,   256, 128));

    printf("\n  ─── Results: %d/%d passed", all.pass, all.total);
    if (all.fail > 0) {
        printf(", %d FAILED ✗ ───\n\n", all.fail);
        return 1;
    } else {
        printf(" ✓ ALL CLEAR ───\n\n");
    }

    printf("═══ Benchmarks (1 thread, per-group scaling, planar SoA) ═══\n\n");
    int sizes[][2] = { {2048, 2048}, {4096, 4096}, {8192, 8192} };

    for (int s = 0; s < 3; s++) {
        int M = sizes[s][0], K = round_up(sizes[s][1], 128), gs = 128;
        int it = (M * K > 16000000) ? 5 : ((M * K > 4000000) ? 10 : 20);

        float *fw = (float *)malloc((size_t)M * K * sizeof(float));
        gen_weights(fw, M * K);
        PlanarWeights pw = planar_pack(fw, M, K, gs);
        int8_t *x = (int8_t *)malloc(K);
        gen_activations(x, K);
        float as = 1.0f / 127.0f, *y = (float *)calloc(M, sizeof(float));

        printf("  [%5d × %5d] gs=%d ×%d\n", M, K, gs, it);

        #define BENCH(name, call) do { \
            call; double t = now(); \
            for (int _i = 0; _i < it; _i++) { call; } \
            double el = now() - t; \
            printf("    %-28s %7.3f ms  %6.1f GOPS\n", name, el * 1000.0 / it, 2.0 * M * K * it / el / 1e9); \
        } while(0)

        BENCH("dp4a scalar", gemv_dp4a_ref(pw.data, pw.scales_rm, x, as, y, M, K, gs));
        BENCH("LUT-Grouped (i16)", gemv_lut_grouped(pw.data, pw.scales_rm, x, as, y, M, K, gs));

#if ARCH_X86_64
        if (cpu_has_avx2)     BENCH("AVX2 LUT (Nibble-Split) ★", gemv_avx2_lut(pw.data_colmaj, pw.scales_gm, x, as, y, M, K, gs, pw.rows_padded));
        if (cpu_has_avx512bw) BENCH("VPERMW fused (AVX-512)", gemv_vpermw(pw.data_colmaj, pw.scales_gm, x, as, y, M, K, gs, pw.rows_padded));
#endif
        BENCH("ternary_gemv() [dispatch]", ternary_gemv(&pw, x, as, y));

        #undef BENCH
        printf("\n");
        free(fw); free(x); free(y); planar_free(&pw);
    }

    printf("═══ Kernel Dispatch Chain (v3.4.0) ═══\n\n");
    printf("  1. VPERMW fused    (AVX-512 BW)   [64-row, unconditional SIMD]\n");
    printf("  2. Dual-VPERMB     (AVX-512 VBMI) [32-row, unconditional SIMD]\n");
    printf("  3. Nibble-Split    (AVX2)         ★ [32-row, 16-bit Vertical Acc]\n");
    printf("  4. LUT-Grouped     (portable)     [L1 cached fallback]\n");

    return 0;
}