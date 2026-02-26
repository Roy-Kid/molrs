#include "common.cuh"
#include "include/molrs_kernels.h"

// ---------------------------------------------------------------------------
// GPUSimBox — GPU-resident simulation box
//
// Holds device memory for box matrix H, its inverse, origin, and PBC flags.
// GPU kernels (e.g. barostat) can write directly to the device pointers.
// The neighbor list reads from these buffers via constant memory broadcast.
// ---------------------------------------------------------------------------

struct GPUSimBox {
    float* h;         // device: [9] row-major H (columns = lattice vectors)
    float* inv_h;     // device: [9] row-major H^{-1}
    float* origin;    // device: [3] cell origin in Cartesian
    int*   periodic;  // device: [3] PBC flags (0 or 1)
};

// ---------------------------------------------------------------------------
// Create / Destroy
// ---------------------------------------------------------------------------

extern "C" GPUSimBox* molrs_simbox_create(void)
{
    GPUSimBox* sb = new GPUSimBox;
    CUDA_CHECK(cudaMalloc(&sb->h,        9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sb->inv_h,    9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sb->origin,   3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sb->periodic, 3 * sizeof(int)));
    // Zero-initialize
    CUDA_CHECK(cudaMemset(sb->h,        0, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(sb->inv_h,    0, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(sb->origin,   0, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(sb->periodic, 0, 3 * sizeof(int)));
    return sb;
}

extern "C" void molrs_simbox_destroy(GPUSimBox* sb)
{
    if (!sb) return;
    CUDA_CHECK(cudaFree(sb->h));
    CUDA_CHECK(cudaFree(sb->inv_h));
    CUDA_CHECK(cudaFree(sb->origin));
    CUDA_CHECK(cudaFree(sb->periodic));
    delete sb;
}

// ---------------------------------------------------------------------------
// Upload from host (for initialization / CPU-side box updates)
// Uses synchronous memcpy — safe for stack-allocated host arrays.
// ---------------------------------------------------------------------------

extern "C" void molrs_simbox_upload(
    GPUSimBox* sb,
    const float* h_host,
    const float* inv_h_host,
    const float* origin_host,
    const int*   periodic_host,
    void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    CUDA_CHECK(cudaMemcpyAsync(sb->h,        h_host,        9 * sizeof(float), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(sb->inv_h,    inv_h_host,    9 * sizeof(float), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(sb->origin,   origin_host,   3 * sizeof(float), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(sb->periodic, periodic_host, 3 * sizeof(int),   cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ---------------------------------------------------------------------------
// Device pointer accessors — for GPU kernels to read/write directly
// ---------------------------------------------------------------------------

extern "C" float* molrs_simbox_h_ptr(GPUSimBox* sb)        { return sb->h; }
extern "C" float* molrs_simbox_inv_ptr(GPUSimBox* sb)       { return sb->inv_h; }
extern "C" float* molrs_simbox_origin_ptr(GPUSimBox* sb)    { return sb->origin; }
extern "C" int*   molrs_simbox_periodic_ptr(GPUSimBox* sb)  { return sb->periodic; }

extern "C" const float* molrs_simbox_h_ptr_const(const GPUSimBox* sb)        { return sb->h; }
extern "C" const float* molrs_simbox_inv_ptr_const(const GPUSimBox* sb)       { return sb->inv_h; }
extern "C" const float* molrs_simbox_origin_ptr_const(const GPUSimBox* sb)    { return sb->origin; }
extern "C" const int*   molrs_simbox_periodic_ptr_const(const GPUSimBox* sb)  { return sb->periodic; }

// ---------------------------------------------------------------------------
// GPU kernel: Compute H^{-1} from H on device
// (Call after barostat updates H on GPU)
// Single-thread kernel — the 3x3 inversion is trivial.
// ---------------------------------------------------------------------------

__global__ void kernel_invert_3x3(
    const float* __restrict__ m,
    float*       __restrict__ inv)
{
    // Cofactor matrix (transposed for inverse)
    float m00 = m[0], m01 = m[1], m02 = m[2];
    float m10 = m[3], m11 = m[4], m12 = m[5];
    float m20 = m[6], m21 = m[7], m22 = m[8];

    float c00 =  (m11 * m22 - m12 * m21);
    float c01 = -(m10 * m22 - m12 * m20);
    float c02 =  (m10 * m21 - m11 * m20);

    float c10 = -(m01 * m22 - m02 * m21);
    float c11 =  (m00 * m22 - m02 * m20);
    float c12 = -(m00 * m21 - m01 * m20);

    float c20 =  (m01 * m12 - m02 * m11);
    float c21 = -(m00 * m12 - m02 * m10);
    float c22 =  (m00 * m11 - m01 * m10);

    float det = m00 * c00 + m01 * c01 + m02 * c02;
    float inv_det = 1.0f / det;

    // Transpose of cofactor matrix / det
    inv[0] = c00 * inv_det;  inv[1] = c10 * inv_det;  inv[2] = c20 * inv_det;
    inv[3] = c01 * inv_det;  inv[4] = c11 * inv_det;  inv[5] = c21 * inv_det;
    inv[6] = c02 * inv_det;  inv[7] = c12 * inv_det;  inv[8] = c22 * inv_det;
}

extern "C" void molrs_simbox_update_inverse(GPUSimBox* sb, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    kernel_invert_3x3<<<1, 1, 0, s>>>(sb->h, sb->inv_h);
}

// ---------------------------------------------------------------------------
// GPU kernel: Compute cell grid dimensions from H matrix
// Writes [nx, ny, nz] to output. Single-thread kernel.
//
// nearest_plane_distance[d] = volume / |cross(a_{d+1}, a_{d+2})|
// cell_n[d] = max(1, floor(npd[d] / cutoff))
//
// Mirrors SimBox::nearest_plane_distance_impl (simbox.rs)
// ---------------------------------------------------------------------------

__global__ void kernel_compute_cell_dims(
    const float* __restrict__ h,
    float cutoff,
    int* __restrict__ cell_dims)
{
    // Lattice vectors = columns of H (row-major storage)
    float a1x = h[0], a1y = h[3], a1z = h[6];  // column 0
    float a2x = h[1], a2y = h[4], a2z = h[7];  // column 1
    float a3x = h[2], a3y = h[5], a3z = h[8];  // column 2

    // Volume = |det(H)|
    float vol = fabsf(
        h[0] * (h[4] * h[8] - h[5] * h[7]) -
        h[1] * (h[3] * h[8] - h[5] * h[6]) +
        h[2] * (h[3] * h[7] - h[4] * h[6]));

    // a2 × a3
    float c0x = a2y * a3z - a2z * a3y;
    float c0y = a2z * a3x - a2x * a3z;
    float c0z = a2x * a3y - a2y * a3x;
    float npd0 = vol / sqrtf(c0x * c0x + c0y * c0y + c0z * c0z);

    // a3 × a1
    float c1x = a3y * a1z - a3z * a1y;
    float c1y = a3z * a1x - a3x * a1z;
    float c1z = a3x * a1y - a3y * a1x;
    float npd1 = vol / sqrtf(c1x * c1x + c1y * c1y + c1z * c1z);

    // a1 × a2
    float c2x = a1y * a2z - a1z * a2y;
    float c2y = a1z * a2x - a1x * a2z;
    float c2z = a1x * a2y - a1y * a2x;
    float npd2 = vol / sqrtf(c2x * c2x + c2y * c2y + c2z * c2z);

    cell_dims[0] = max(1, (int)floorf(npd0 / cutoff));
    cell_dims[1] = max(1, (int)floorf(npd1 / cutoff));
    cell_dims[2] = max(1, (int)floorf(npd2 / cutoff));
}

extern "C" void molrs_simbox_compute_cell_dims(
    const GPUSimBox* sb,
    float cutoff,
    int* cell_dims_host,
    void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int* d_cell_dims;
    CUDA_CHECK(cudaMalloc(&d_cell_dims, 3 * sizeof(int)));

    kernel_compute_cell_dims<<<1, 1, 0, s>>>(sb->h, cutoff, d_cell_dims);

    CUDA_CHECK(cudaMemcpyAsync(cell_dims_host, d_cell_dims, 3 * sizeof(int),
                                cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d_cell_dims));
}
