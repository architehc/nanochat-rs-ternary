// Custom CUDA sigmoid kernel for Candle integration
// Provides sigmoid operation when Candle's built-in CUDA sigmoid is unavailable

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return err; \
        } \
    } while(0)

// Sigmoid kernel for f32
__global__ void sigmoid_kernel_f32(const float* input, float* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-val));
    }
}

// Sigmoid kernel for f16
__global__ void sigmoid_kernel_f16(const __half* input, __half* output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = __half2float(input[idx]);
        float result = 1.0f / (1.0f + expf(-val));
        output[idx] = __float2half(result);
    }
}

// Sigmoid backward kernel for f32
__global__ void sigmoid_backward_kernel_f32(
    const float* grad_output,
    const float* sigmoid_output,
    float* grad_input,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float sig = sigmoid_output[idx];
        grad_input[idx] = grad_output[idx] * sig * (1.0f - sig);
    }
}

// Sigmoid backward kernel for f16
__global__ void sigmoid_backward_kernel_f16(
    const __half* grad_output,
    const __half* sigmoid_output,
    __half* grad_input,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float sig = __half2float(sigmoid_output[idx]);
        float grad = __half2float(grad_output[idx]);
        float result = grad * sig * (1.0f - sig);
        grad_input[idx] = __float2half(result);
    }
}

extern "C" {

// Forward pass: f32
cudaError_t sigmoid_f32(
    const float* input,
    float* output,
    size_t numel,
    cudaStream_t cuda_stream
) {
    if (numel == 0) return cudaSuccess;

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    sigmoid_kernel_f32<<<blocks, threads, 0, cuda_stream>>>(input, output, numel);

    return cudaGetLastError();
}

// Forward pass: f16
cudaError_t sigmoid_f16(
    const __half* input,
    __half* output,
    size_t numel,
    cudaStream_t cuda_stream
) {
    if (numel == 0) return cudaSuccess;

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    sigmoid_kernel_f16<<<blocks, threads, 0, cuda_stream>>>(input, output, numel);

    return cudaGetLastError();
}

// Backward pass: f32
cudaError_t sigmoid_backward_f32(
    const float* grad_output,
    const float* sigmoid_output,
    float* grad_input,
    size_t numel,
    cudaStream_t cuda_stream
) {
    if (numel == 0) return cudaSuccess;

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    sigmoid_backward_kernel_f32<<<blocks, threads, 0, cuda_stream>>>(
        grad_output, sigmoid_output, grad_input, numel
    );

    return cudaGetLastError();
}

// Backward pass: f16
cudaError_t sigmoid_backward_f16(
    const __half* grad_output,
    const __half* sigmoid_output,
    __half* grad_input,
    size_t numel,
    cudaStream_t cuda_stream
) {
    if (numel == 0) return cudaSuccess;

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    sigmoid_backward_kernel_f16<<<blocks, threads, 0, cuda_stream>>>(
        grad_output, sigmoid_output, grad_input, numel
    );

    return cudaGetLastError();
}

} // extern "C"
