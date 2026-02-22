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
 * signal, kernel, output are all [len]. output may alias neither input. */
void fwht_convolve_i32(const int32_t *signal, const int32_t *kernel,
                       int32_t *output, int len);

/* FWHT convolution on floats. */
void fwht_convolve_f32(const float *signal, const float *kernel,
                       float *output, int len);

/* === Haar DWT: Discrete Haar Wavelet Transform === */

/* Forward Haar DWT in-place. levels = number of decomposition levels (max log2(len)). */
void haar_forward_i32(int32_t *data, int len, int levels);
void haar_forward_f32(float *data, int len, int levels);

/* Inverse Haar DWT in-place. */
void haar_inverse_i32(int32_t *data, int len, int levels);
void haar_inverse_f32(float *data, int len, int levels);

/* Haar convolution: forward DWT, pointwise mul, inverse DWT. */
void haar_convolve_i32(const int32_t *signal, const int32_t *kernel,
                       int32_t *output, int len, int levels);
void haar_convolve_f32(const float *signal, const float *kernel,
                       float *output, int len, int levels);
