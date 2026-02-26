#pragma once

#include <cstdio>
#include <cuda_runtime.h>

// Check CUDA errors
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
        }                                                                        \
    } while (0)

// Warp-level reduction (sum)
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__ double warp_reduce_sum_d(double val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp-level reduction (max of absolute values)
__inline__ __device__ float warp_reduce_absmax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(fabsf(val), fabsf(other));
    }
    return val;
}

// Block-level reduction (sum) using shared memory
__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // one per warp
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces across warps
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

__inline__ __device__ double block_reduce_sum_d(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum_d(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
    if (wid == 0) val = warp_reduce_sum_d(val);

    return val;
}

// Block-level reduction (absmax)
__inline__ __device__ float block_reduce_absmax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_absmax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_absmax(val);

    return val;
}

// Atomic add for double (needed on older architectures)
#if __CUDA_ARCH__ < 600
__inline__ __device__ double atomicAdd_d(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__inline__ __device__ double atomicAdd_d(double* address, double val) {
    return atomicAdd(address, val);
}
#endif
