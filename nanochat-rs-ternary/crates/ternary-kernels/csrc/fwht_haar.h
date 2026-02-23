#pragma once
#include <stdint.h>

/*
 * FWHT (Fast Walsh-Hadamard Transform) and Haar DWT kernels.
 *
 * Both transforms use only additions and subtractions (integer-only compatible).
 * All lengths must be powers of 2.
 */

/* === FWHT: Fast Walsh-Hadamard Transform === */

/* In-place FWHT on int32 array. len must be power of 2. */
void fwht_i32(int32_t *data, int len);

/* In-place FWHT on float array. len must be power of 2. */
void fwht_f32(float *data, int len);

/* FWHT convolution: FWHT(signal), FWHT(kernel), pointwise mul, inverse FWHT.
 * signal, kernel, output are all [len]. output must NOT alias signal or kernel. */
void fwht_convolve_i32(const int32_t *restrict signal, const int32_t *restrict kernel,
                       int32_t *restrict output, int len);

/* FWHT convolution on floats. output must NOT alias signal or kernel. */
void fwht_convolve_f32(const float *restrict signal, const float *restrict kernel,
                       float *restrict output, int len);

/* === Haar DWT: Discrete Haar Wavelet Transform === */

/* Forward Haar DWT in-place. levels = number of decomposition levels (max log2(len)). */
void haar_forward_i32(int32_t *data, int len, int levels);
void haar_forward_f32(float *data, int len, int levels);

/* Inverse Haar DWT in-place. */
void haar_inverse_i32(int32_t *data, int len, int levels);
void haar_inverse_f32(float *data, int len, int levels);

/* Haar convolution: forward DWT, pointwise mul, inverse DWT.
 * output must NOT alias signal or kernel. */
void haar_convolve_i32(const int32_t *restrict signal, const int32_t *restrict kernel,
                       int32_t *restrict output, int len, int levels);
void haar_convolve_f32(const float *restrict signal, const float *restrict kernel,
                       float *restrict output, int len, int levels);

/* === Buffer-reuse variants (no internal malloc) === */

/* FWHT convolution with caller-provided scratch buffers.
 * scratch1 and scratch2 must each be at least len elements.
 * output must NOT alias signal, kernel, scratch1, or scratch2. */
void fwht_convolve_f32_buf(const float *restrict signal, const float *restrict kernel,
                           float *restrict output, int len,
                           float *restrict scratch1, float *restrict scratch2);

/* Haar convolution with caller-provided scratch buffer.
 * scratch must be at least len elements (used by forward/inverse levels).
 * output must NOT alias signal, kernel, or scratch. */
void haar_convolve_f32_buf(const float *restrict signal, const float *restrict kernel,
                           float *restrict output, int len, int levels,
                           float *restrict scratch);
