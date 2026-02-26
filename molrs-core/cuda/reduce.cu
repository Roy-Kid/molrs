#include "common.cuh"
#include "include/molrs_kernels.h"

// Kernel: reduce an array of doubles to a single sum
__global__ void kernel_reduce_sum_d(const double* input, double* output, int n) {
    double val = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        val += input[i];
    }
    val = block_reduce_sum_d(val);
    if (threadIdx.x == 0) {
        atomicAdd_d(output, val);
    }
}

// Kernel: compute max absolute value of a float array
__global__ void kernel_absmax(const float* input, float* output, int n) {
    float val = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        val = fmaxf(val, fabsf(input[i]));
    }
    val = block_reduce_absmax(val);
    if (threadIdx.x == 0) {
        // atomicMax for float via int reinterpretation (works for non-negative)
        int* addr = (int*)output;
        int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                            __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }
}

extern "C" double molrs_energy_reduce(const double* partial, int n, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(double), s));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 256) blocks = 256;

    kernel_reduce_sum_d<<<blocks, threads, 0, s>>>(partial, d_result, n);

    double result;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_result, sizeof(double),
                                cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}

extern "C" float molrs_grad_max_norm(const float* grad, int n, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(float), s));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 256) blocks = 256;

    kernel_absmax<<<blocks, threads, 0, s>>>(grad, d_result, n);

    float result;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_result, sizeof(float),
                                cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}
