/**
 * ternary_dp4a.cu — GPU ternary GEMV decode kernel using dp4a.
 *
 * Decodes packed 2-bit ternary weights on the fly using a constant-memory
 * 256-entry LUT (byte → 4 signed int8 trits), then uses dp4a for efficient
 * INT8 dot products with per-group scale accumulation.
 *
 * One warp per output row, K-reduction across 32 lanes, warp-shuffle reduction.
 * Optimized for autoregressive decode (N=1).
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// 256-entry decode LUT: packed byte -> 4 signed int8 trits
// Encoding: 00=0, 01=+1, 11=-1, 10=invalid(0)
// Each entry produces 4 trits packed as an int32 (4 x i8)
__constant__ int32_t DECODE_LUT[256];

// Host-side LUT for initialization
static int32_t h_decode_lut[256];
static int lut_initialized = 0;

static void init_lut_host(void) {
    if (lut_initialized) return;
    for (int b = 0; b < 256; b++) {
        int8_t trits[4];
        for (int j = 0; j < 4; j++) {
            int bits = (b >> (j * 2)) & 3;
            if (bits == 0x01) trits[j] = 1;
            else if (bits == 0x03) trits[j] = -1;
            else trits[j] = 0; // 00 or 10(invalid)
        }
        int32_t packed;
        memcpy(&packed, trits, 4);
        h_decode_lut[b] = packed;
    }
    lut_initialized = 1;
}

/**
 * Ternary GEMV kernel: y = W * x * act_scale
 *
 * Each warp handles one output row.
 * data: [M, K/4] packed bytes (row-major)
 * scales: [M, n_groups] where n_groups = K / group_size (row-major)
 * x: [K] int8 activations
 * y: [M] float output
 */
__global__ void ternary_gemv_dp4a_kernel(
    const uint8_t* __restrict__ data,
    const float*   __restrict__ scales,
    const int8_t*  __restrict__ x,
    float          act_scale,
    float*         __restrict__ y,
    int M, int K, int group_size)
{
    int row = blockIdx.x;
    if (row >= M) return;

    int lane = threadIdx.x; // 0..31 within warp
    int kp = K / 4; // packed bytes per row
    int n_groups = K / group_size;
    int trits_per_group = group_size / 4; // packed bytes per group

    float row_sum = 0.0f;

    // Each lane processes every 32nd packed byte
    const uint8_t* row_data = data + row * kp;
    const float* row_scales = scales + row * n_groups;

    for (int g = 0; g < n_groups; g++) {
        float scale = row_scales[g];
        int group_start = g * trits_per_group;
        int group_end = group_start + trits_per_group;

        int acc = 0;
        for (int p = group_start + lane; p < group_end; p += 32) {
            uint8_t packed = row_data[p];
            int32_t w4 = DECODE_LUT[packed];

            // Load 4 consecutive int8 activations
            int32_t x4;
            memcpy(&x4, &x[p * 4], 4);

            // dp4a: dot product of 4 int8 pairs, accumulated into int32
            acc = __dp4a(w4, x4, acc);
        }

        // Warp-shuffle reduction for this group
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }

        if (lane == 0) {
            row_sum += (float)acc * scale;
        }
    }

    if (lane == 0) {
        y[row] = row_sum * act_scale;
    }
}

// ─── Host C wrappers (extern "C" for Rust FFI) ────────────────────

extern "C" {

int cuda_ternary_gemv_init(void) {
    init_lut_host();
    cudaError_t err = cudaMemcpyToSymbol(DECODE_LUT, h_decode_lut, sizeof(h_decode_lut));
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_ternary_gemv_init: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

void cuda_ternary_gemv(
    const uint8_t* d_data,
    const float*   d_scales,
    const int8_t*  d_x,
    float          act_scale,
    float*         d_y,
    int M, int K, int group_size)
{
    // One warp (32 threads) per output row
    dim3 grid(M);
    dim3 block(32);
    ternary_gemv_dp4a_kernel<<<grid, block>>>(
        d_data, d_scales, d_x, act_scale, d_y, M, K, group_size);
}

void* cuda_alloc(size_t bytes) {
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_alloc(%zu): %s\n", bytes, cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

int cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_synchronize(void) {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

} // extern "C"
