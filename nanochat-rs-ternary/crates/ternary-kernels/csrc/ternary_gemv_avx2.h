#pragma once
#include <stdint.h>

// AVX2 PSHUFB-based ternary GEMV kernel.
// Uses column-major packed data + group-major scales (same as VPERMW path).
void gemv_avx2(
    const uint8_t *Wt,          // column-major packed ternary data, 128B aligned
    const float   *scales_gm,   // group-major scales, 128B aligned
    const int8_t  *x,           // quantized activations [K]
    float act_scale,            // activation scale factor
    float *y,                   // output [M], caller-allocated
    int M, int K, int gs, int rows_padded
);
