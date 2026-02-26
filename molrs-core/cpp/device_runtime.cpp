#include "include/molrs_runtime.h"

#include <cuda_runtime.h>
#include <cstdio>

extern "C" int molrs_device_init(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" int molrs_device_get_name(int device_id, char* buf, int len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return -1;
    std::snprintf(buf, len, "%s", prop.name);
    return 0;
}

extern "C" void* molrs_malloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "molrs_malloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

extern "C" void molrs_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

extern "C" void molrs_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

extern "C" void molrs_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

extern "C" void molrs_memset_zero(void* ptr, size_t size) {
    cudaMemset(ptr, 0, size);
}
