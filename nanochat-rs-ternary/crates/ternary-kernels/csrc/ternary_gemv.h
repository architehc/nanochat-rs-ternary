#pragma once
#include <stdint.h>

typedef struct {
    uint8_t *data;          /* [rows * cols/4] row-major, 128B aligned */
    uint8_t *data_colmaj;   /* [cols/4 * rows_padded] col-major, 128B aligned */
    float   *scales_rm;     /* [rows * gprow] row-major (scalar kernels) */
    float   *scales_gm;     /* [gprow * rows_padded] group-major (SIMD) */
    int      rows, cols, group_size;
    int      rows_padded;
} PlanarWeightsC;

/* Unified entrypoint — self-initializing, selects best kernel at first call */
void ternary_gemv(
    const PlanarWeightsC *pw,
    const int8_t  *x,        /* quantized activations [K] */
    float act_scale,          /* activation scale factor */
    float *y                  /* output [M], caller-allocated */
);

/* Expose for testing */
void gemv_dp4a_ref(
    const uint8_t *data, const float *scales_rm,
    const int8_t  *x,    float act_scale,
    float *y, int M, int K, int gs
);
