/*
 * FWHT (Fast Walsh-Hadamard Transform) and Haar DWT kernels.
 *
 * Both transforms use only additions and subtractions — no multiplications
 * in the butterfly stages. This aligns with the ternary architecture's
 * integer-only compute thesis.
 *
 * FWHT: Global token mixing via Walsh-Hadamard basis (+1/-1 coefficients).
 *   Butterfly: a' = a + b, b' = a - b. Self-inverse (up to 1/N scaling).
 *
 * Haar DWT: Multi-scale localized transform.
 *   Low = a + b, High = a - b. Multi-level decomposition.
 *
 * Both are O(N log N).
 *
 * SIMD acceleration: AVX2 for x86_64, NEON for AArch64.
 */

#include "fwht_haar.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================
 *  FWHT — Fast Walsh-Hadamard Transform
 * ============================================================ */

/* Scalar in-place FWHT for int32. */
static void fwht_i32_scalar(int32_t *data, int len) {
    for (int half = 1; half < len; half <<= 1) {
        for (int i = 0; i < len; i += 2 * half) {
            for (int j = 0; j < half; j++) {
                int32_t a = data[i + j];
                int32_t b = data[i + j + half];
                data[i + j]        = a + b;
                data[i + j + half] = a - b;
            }
        }
    }
}

/* Scalar in-place FWHT for float. */
static void fwht_f32_scalar(float *data, int len) {
    for (int half = 1; half < len; half <<= 1) {
        for (int i = 0; i < len; i += 2 * half) {
            for (int j = 0; j < half; j++) {
                float a = data[i + j];
                float b = data[i + j + half];
                data[i + j]        = a + b;
                data[i + j + half] = a - b;
            }
        }
    }
}

/* --- AVX2 FWHT --- */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

__attribute__((target("avx2")))
static void fwht_i32_avx2(int32_t *data, int len) {
    for (int half = 1; half < len; half <<= 1) {
        if (half >= 8) {
            /* Process 8 butterflies in parallel */
            for (int i = 0; i < len; i += 2 * half) {
                for (int j = 0; j < half; j += 8) {
                    __m256i a = _mm256_loadu_si256((__m256i *)(data + i + j));
                    __m256i b = _mm256_loadu_si256((__m256i *)(data + i + j + half));
                    __m256i sum  = _mm256_add_epi32(a, b);
                    __m256i diff = _mm256_sub_epi32(a, b);
                    _mm256_storeu_si256((__m256i *)(data + i + j), sum);
                    _mm256_storeu_si256((__m256i *)(data + i + j + half), diff);
                }
            }
        } else {
            /* Scalar fallback for small half */
            for (int i = 0; i < len; i += 2 * half) {
                for (int j = 0; j < half; j++) {
                    int32_t a = data[i + j];
                    int32_t b = data[i + j + half];
                    data[i + j]        = a + b;
                    data[i + j + half] = a - b;
                }
            }
        }
    }
}

__attribute__((target("avx2")))
static void fwht_f32_avx2(float *data, int len) {
    for (int half = 1; half < len; half <<= 1) {
        if (half >= 8) {
            for (int i = 0; i < len; i += 2 * half) {
                for (int j = 0; j < half; j += 8) {
                    __m256 a = _mm256_loadu_ps(data + i + j);
                    __m256 b = _mm256_loadu_ps(data + i + j + half);
                    __m256 sum  = _mm256_add_ps(a, b);
                    __m256 diff = _mm256_sub_ps(a, b);
                    _mm256_storeu_ps(data + i + j, sum);
                    _mm256_storeu_ps(data + i + j + half, diff);
                }
            }
        } else {
            for (int i = 0; i < len; i += 2 * half) {
                for (int j = 0; j < half; j++) {
                    float a = data[i + j];
                    float b = data[i + j + half];
                    data[i + j]        = a + b;
                    data[i + j + half] = a - b;
                }
            }
        }
    }
}

static int has_avx2_cached = -1;

static int detect_avx2(void) {
    if (has_avx2_cached >= 0) return has_avx2_cached;
    __builtin_cpu_init();
    has_avx2_cached = __builtin_cpu_supports("avx2") ? 1 : 0;
    return has_avx2_cached;
}

#endif /* x86_64 */

/* --- NEON FWHT --- */
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

static void fwht_i32_neon(int32_t *data, int len) {
    for (int half = 1; half < len; half <<= 1) {
        if (half >= 4) {
            for (int i = 0; i < len; i += 2 * half) {
                for (int j = 0; j < half; j += 4) {
                    int32x4_t a = vld1q_s32(data + i + j);
                    int32x4_t b = vld1q_s32(data + i + j + half);
                    int32x4_t sum  = vaddq_s32(a, b);
                    int32x4_t diff = vsubq_s32(a, b);
                    vst1q_s32(data + i + j, sum);
                    vst1q_s32(data + i + j + half, diff);
                }
            }
        } else {
            for (int i = 0; i < len; i += 2 * half) {
                for (int j = 0; j < half; j++) {
                    int32_t a = data[i + j];
                    int32_t b = data[i + j + half];
                    data[i + j]        = a + b;
                    data[i + j + half] = a - b;
                }
            }
        }
    }
}

static void fwht_f32_neon(float *data, int len) {
    for (int half = 1; half < len; half <<= 1) {
        if (half >= 4) {
            for (int i = 0; i < len; i += 2 * half) {
                for (int j = 0; j < half; j += 4) {
                    float32x4_t a = vld1q_f32(data + i + j);
                    float32x4_t b = vld1q_f32(data + i + j + half);
                    float32x4_t sum  = vaddq_f32(a, b);
                    float32x4_t diff = vsubq_f32(a, b);
                    vst1q_f32(data + i + j, sum);
                    vst1q_f32(data + i + j + half, diff);
                }
            }
        } else {
            for (int i = 0; i < len; i += 2 * half) {
                for (int j = 0; j < half; j++) {
                    float a = data[i + j];
                    float b = data[i + j + half];
                    data[i + j]        = a + b;
                    data[i + j + half] = a - b;
                }
            }
        }
    }
}
#endif /* aarch64 */

/* --- Public dispatch functions --- */

void fwht_i32(int32_t *data, int len) {
    if (len <= 1) return;
#if defined(__x86_64__) || defined(_M_X64)
    if (detect_avx2() && len >= 16)
        fwht_i32_avx2(data, len);
    else
        fwht_i32_scalar(data, len);
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (len >= 8)
        fwht_i32_neon(data, len);
    else
        fwht_i32_scalar(data, len);
#else
    fwht_i32_scalar(data, len);
#endif
}

void fwht_f32(float *data, int len) {
    if (len <= 1) return;
#if defined(__x86_64__) || defined(_M_X64)
    if (detect_avx2() && len >= 16)
        fwht_f32_avx2(data, len);
    else
        fwht_f32_scalar(data, len);
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (len >= 8)
        fwht_f32_neon(data, len);
    else
        fwht_f32_scalar(data, len);
#else
    fwht_f32_scalar(data, len);
#endif
}

void fwht_convolve_i32(const int32_t *signal, const int32_t *kernel,
                       int32_t *output, int len) {
    /* Allocate temp buffers for transformed signal and kernel */
    int32_t *sig_t = (int32_t *)malloc(len * sizeof(int32_t));
    int32_t *ker_t = (int32_t *)malloc(len * sizeof(int32_t));
    memcpy(sig_t, signal, len * sizeof(int32_t));
    memcpy(ker_t, kernel, len * sizeof(int32_t));

    /* Forward FWHT both */
    fwht_i32(sig_t, len);
    fwht_i32(ker_t, len);

    /* Pointwise multiply */
    for (int i = 0; i < len; i++) {
        output[i] = sig_t[i] * ker_t[i];
    }

    /* Inverse FWHT (FWHT is self-inverse up to 1/N scaling) */
    fwht_i32(output, len);

    /* Scale by 1/N (integer division) */
    for (int i = 0; i < len; i++) {
        output[i] /= len;
    }

    free(sig_t);
    free(ker_t);
}

void fwht_convolve_f32(const float *signal, const float *kernel,
                       float *output, int len) {
    float *sig_t = (float *)malloc(len * sizeof(float));
    float *ker_t = (float *)malloc(len * sizeof(float));
    memcpy(sig_t, signal, len * sizeof(float));
    memcpy(ker_t, kernel, len * sizeof(float));

    fwht_f32(sig_t, len);
    fwht_f32(ker_t, len);

    for (int i = 0; i < len; i++) {
        output[i] = sig_t[i] * ker_t[i];
    }

    fwht_f32(output, len);

    float inv_n = 1.0f / (float)len;
    for (int i = 0; i < len; i++) {
        output[i] *= inv_n;
    }

    free(sig_t);
    free(ker_t);
}

/* ============================================================
 *  Haar DWT — Discrete Haar Wavelet Transform
 * ============================================================ */

/* Scalar forward Haar DWT (one level): pairs [a, b] -> [a+b, a-b] */
static void haar_forward_one_level_i32(int32_t *data, int n) {
    int32_t *tmp = (int32_t *)malloc(n * sizeof(int32_t));
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        tmp[i]        = data[2 * i] + data[2 * i + 1]; /* low (approximation) */
        tmp[half + i] = data[2 * i] - data[2 * i + 1]; /* high (detail) */
    }
    memcpy(data, tmp, n * sizeof(int32_t));
    free(tmp);
}

static void haar_forward_one_level_f32(float *data, int n) {
    float *tmp = (float *)malloc(n * sizeof(float));
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        tmp[i]        = data[2 * i] + data[2 * i + 1];
        tmp[half + i] = data[2 * i] - data[2 * i + 1];
    }
    memcpy(data, tmp, n * sizeof(float));
    free(tmp);
}

/* Scalar inverse Haar DWT (one level): [low, high] -> interleaved pairs */
static void haar_inverse_one_level_i32(int32_t *data, int n) {
    int32_t *tmp = (int32_t *)malloc(n * sizeof(int32_t));
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        /* low = data[i], high = data[half + i]
         * Reconstruct: a = (low + high) / 2, b = (low - high) / 2
         * But we didn't normalize forward, so:
         * a = (low + high) / 2, b = (low - high) / 2 */
        tmp[2 * i]     = (data[i] + data[half + i]) / 2;
        tmp[2 * i + 1] = (data[i] - data[half + i]) / 2;
    }
    memcpy(data, tmp, n * sizeof(int32_t));
    free(tmp);
}

static void haar_inverse_one_level_f32(float *data, int n) {
    float *tmp = (float *)malloc(n * sizeof(float));
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        tmp[2 * i]     = (data[i] + data[half + i]) * 0.5f;
        tmp[2 * i + 1] = (data[i] - data[half + i]) * 0.5f;
    }
    memcpy(data, tmp, n * sizeof(float));
    free(tmp);
}

/* --- AVX2 Haar --- */
#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("avx2")))
static void haar_forward_one_level_f32_avx2(float *data, int n) {
    float *tmp = (float *)malloc(n * sizeof(float));
    int half = n / 2;
    int i;
    /* Process 8 pairs (16 elements) at a time */
    for (i = 0; i + 7 < half; i += 8) {
        /* Load 16 elements: [a0,b0,a1,b1,...,a7,b7] */
        __m256 lo = _mm256_loadu_ps(data + 2 * i);      /* a0,b0,a1,b1,a2,b2,a3,b3 */
        __m256 hi = _mm256_loadu_ps(data + 2 * i + 8);  /* a4,b4,a5,b5,a6,b6,a7,b7 */

        /* Deinterleave: separate even and odd elements */
        __m256 even_lo = _mm256_shuffle_ps(lo, lo, 0x88); /* a0,a1,a2,a3,... (not quite) */
        __m256 odd_lo  = _mm256_shuffle_ps(lo, lo, 0xDD);
        __m256 even_hi = _mm256_shuffle_ps(hi, hi, 0x88);
        __m256 odd_hi  = _mm256_shuffle_ps(hi, hi, 0xDD);

        /* Pack evens and odds into contiguous vectors */
        __m256 evens = _mm256_permute2f128_ps(
            _mm256_unpacklo_ps(even_lo, even_hi),
            _mm256_unpackhi_ps(even_lo, even_hi), 0x20);
        __m256 odds = _mm256_permute2f128_ps(
            _mm256_unpacklo_ps(odd_lo, odd_hi),
            _mm256_unpackhi_ps(odd_lo, odd_hi), 0x20);

        /* This shuffle approach doesn't deinterleave perfectly.
         * Use a simpler gather approach instead. */
        (void)evens; (void)odds;

        /* Simple scalar-in-SIMD approach: just process pairs manually.
         * The overhead is in the memory access pattern, not the add/sub. */
        for (int j = 0; j < 8; j++) {
            tmp[i + j]        = data[2 * (i + j)] + data[2 * (i + j) + 1];
            tmp[half + i + j] = data[2 * (i + j)] - data[2 * (i + j) + 1];
        }
    }
    /* Scalar remainder */
    for (; i < half; i++) {
        tmp[i]        = data[2 * i] + data[2 * i + 1];
        tmp[half + i] = data[2 * i] - data[2 * i + 1];
    }
    memcpy(data, tmp, n * sizeof(float));
    free(tmp);
}

__attribute__((target("avx2")))
static void haar_inverse_one_level_f32_avx2(float *data, int n) {
    float *tmp = (float *)malloc(n * sizeof(float));
    int half = n / 2;
    int i;
    __m256 half_vec = _mm256_set1_ps(0.5f);
    for (i = 0; i + 7 < half; i += 8) {
        __m256 lo = _mm256_loadu_ps(data + i);
        __m256 hi = _mm256_loadu_ps(data + half + i);
        __m256 sum  = _mm256_mul_ps(_mm256_add_ps(lo, hi), half_vec);
        __m256 diff = _mm256_mul_ps(_mm256_sub_ps(lo, hi), half_vec);
        /* Interleave sum and diff back into pairs */
        for (int j = 0; j < 8; j++) {
            tmp[2 * (i + j)]     = ((float *)&sum)[j];
            tmp[2 * (i + j) + 1] = ((float *)&diff)[j];
        }
    }
    for (; i < half; i++) {
        tmp[2 * i]     = (data[i] + data[half + i]) * 0.5f;
        tmp[2 * i + 1] = (data[i] - data[half + i]) * 0.5f;
    }
    memcpy(data, tmp, n * sizeof(float));
    free(tmp);
}
#endif /* x86_64 Haar AVX2 */

/* --- NEON Haar --- */
#if defined(__aarch64__) || defined(_M_ARM64)

static void haar_forward_one_level_f32_neon(float *data, int n) {
    float *tmp = (float *)malloc(n * sizeof(float));
    int half = n / 2;
    int i;
    for (i = 0; i + 3 < half; i += 4) {
        /* Load 8 elements, deinterleave */
        float32x4x2_t pairs = vld2q_f32(data + 2 * i);
        float32x4_t sum  = vaddq_f32(pairs.val[0], pairs.val[1]);
        float32x4_t diff = vsubq_f32(pairs.val[0], pairs.val[1]);
        vst1q_f32(tmp + i, sum);
        vst1q_f32(tmp + half + i, diff);
    }
    for (; i < half; i++) {
        tmp[i]        = data[2 * i] + data[2 * i + 1];
        tmp[half + i] = data[2 * i] - data[2 * i + 1];
    }
    memcpy(data, tmp, n * sizeof(float));
    free(tmp);
}

static void haar_inverse_one_level_f32_neon(float *data, int n) {
    float *tmp = (float *)malloc(n * sizeof(float));
    int half = n / 2;
    float32x4_t half_vec = vdupq_n_f32(0.5f);
    int i;
    for (i = 0; i + 3 < half; i += 4) {
        float32x4_t lo = vld1q_f32(data + i);
        float32x4_t hi = vld1q_f32(data + half + i);
        float32x4_t sum  = vmulq_f32(vaddq_f32(lo, hi), half_vec);
        float32x4_t diff = vmulq_f32(vsubq_f32(lo, hi), half_vec);
        /* Interleave */
        float32x4x2_t pairs;
        pairs.val[0] = sum;
        pairs.val[1] = diff;
        vst2q_f32(tmp + 2 * i, pairs);
    }
    for (; i < half; i++) {
        tmp[2 * i]     = (data[i] + data[half + i]) * 0.5f;
        tmp[2 * i + 1] = (data[i] - data[half + i]) * 0.5f;
    }
    memcpy(data, tmp, n * sizeof(float));
    free(tmp);
}
#endif /* aarch64 Haar NEON */

/* --- Public Haar functions --- */

void haar_forward_i32(int32_t *data, int len, int levels) {
    int n = len;
    for (int l = 0; l < levels && n >= 2; l++) {
        haar_forward_one_level_i32(data, n);
        n /= 2;
    }
}

void haar_forward_f32(float *data, int len, int levels) {
    int n = len;
    for (int l = 0; l < levels && n >= 2; l++) {
#if defined(__x86_64__) || defined(_M_X64)
        if (detect_avx2() && n >= 16)
            haar_forward_one_level_f32_avx2(data, n);
        else
            haar_forward_one_level_f32(data, n);
#elif defined(__aarch64__) || defined(_M_ARM64)
        if (n >= 8)
            haar_forward_one_level_f32_neon(data, n);
        else
            haar_forward_one_level_f32(data, n);
#else
        haar_forward_one_level_f32(data, n);
#endif
        n /= 2;
    }
}

void haar_inverse_i32(int32_t *data, int len, int levels) {
    /* Compute the size at deepest level */
    int n = len;
    for (int l = 0; l < levels - 1 && n >= 2; l++) {
        n /= 2;
    }
    /* Inverse from deepest level back up */
    for (int l = 0; l < levels && n <= len; l++) {
        haar_inverse_one_level_i32(data, n);
        n *= 2;
    }
}

void haar_inverse_f32(float *data, int len, int levels) {
    int n = len;
    for (int l = 0; l < levels - 1 && n >= 2; l++) {
        n /= 2;
    }
    for (int l = 0; l < levels && n <= len; l++) {
#if defined(__x86_64__) || defined(_M_X64)
        if (detect_avx2() && n >= 16)
            haar_inverse_one_level_f32_avx2(data, n);
        else
            haar_inverse_one_level_f32(data, n);
#elif defined(__aarch64__) || defined(_M_ARM64)
        if (n >= 8)
            haar_inverse_one_level_f32_neon(data, n);
        else
            haar_inverse_one_level_f32(data, n);
#else
        haar_inverse_one_level_f32(data, n);
#endif
        n *= 2;
    }
}

void haar_convolve_i32(const int32_t *signal, const int32_t *kernel,
                       int32_t *output, int len, int levels) {
    int32_t *sig_t = (int32_t *)malloc(len * sizeof(int32_t));
    int32_t *ker_t = (int32_t *)malloc(len * sizeof(int32_t));
    memcpy(sig_t, signal, len * sizeof(int32_t));
    memcpy(ker_t, kernel, len * sizeof(int32_t));

    haar_forward_i32(sig_t, len, levels);
    haar_forward_i32(ker_t, len, levels);

    for (int i = 0; i < len; i++) {
        output[i] = sig_t[i] * ker_t[i];
    }

    haar_inverse_i32(output, len, levels);

    free(sig_t);
    free(ker_t);
}

void haar_convolve_f32(const float *signal, const float *kernel,
                       float *output, int len, int levels) {
    float *sig_t = (float *)malloc(len * sizeof(float));
    float *ker_t = (float *)malloc(len * sizeof(float));
    memcpy(sig_t, signal, len * sizeof(float));
    memcpy(ker_t, kernel, len * sizeof(float));

    haar_forward_f32(sig_t, len, levels);
    haar_forward_f32(ker_t, len, levels);

    for (int i = 0; i < len; i++) {
        output[i] = sig_t[i] * ker_t[i];
    }

    haar_inverse_f32(output, len, levels);

    free(sig_t);
    free(ker_t);
}
