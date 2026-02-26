#include "common.cuh"
#include "include/molrs_kernels.h"

// One thread per bond: compute gradient contribution
__global__ void kernel_bond_harmonic_gradient(
    const float* __restrict__ positions,
    float* __restrict__ gradients,
    const int* __restrict__ atom_i,
    const int* __restrict__ atom_j,
    const float* __restrict__ k,
    const float* __restrict__ r0,
    int n_bonds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bonds) return;

    int i = atom_i[idx];
    int j = atom_j[idx];
    float ki = k[idx];
    float r0i = r0[idx];

    float dx = positions[j * 3]     - positions[i * 3];
    float dy = positions[j * 3 + 1] - positions[i * 3 + 1];
    float dz = positions[j * 3 + 2] - positions[i * 3 + 2];

    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    if (r < 1e-12f) return;

    // E = 0.5 * k * (r - r0)^2
    // dE/dr = k * (r - r0)
    // dE/dx_j = k * (r - r0) * dx / r  (gradient, not force)
    float factor = ki * (r - r0i) / r;
    float gx = factor * dx;
    float gy = factor * dy;
    float gz = factor * dz;

    atomicAdd(&gradients[j * 3],     gx);
    atomicAdd(&gradients[j * 3 + 1], gy);
    atomicAdd(&gradients[j * 3 + 2], gz);
    atomicAdd(&gradients[i * 3],     -gx);
    atomicAdd(&gradients[i * 3 + 1], -gy);
    atomicAdd(&gradients[i * 3 + 2], -gz);
}

// One thread per bond: compute energy contribution
__global__ void kernel_bond_harmonic_energy(
    const float* __restrict__ positions,
    double* __restrict__ energy_out,
    const int* __restrict__ atom_i,
    const int* __restrict__ atom_j,
    const float* __restrict__ k,
    const float* __restrict__ r0,
    int n_bonds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double e = 0.0;
    if (idx < n_bonds) {
        int i = atom_i[idx];
        int j = atom_j[idx];
        float ki = k[idx];
        float r0i = r0[idx];

        float dx = positions[j * 3]     - positions[i * 3];
        float dy = positions[j * 3 + 1] - positions[i * 3 + 1];
        float dz = positions[j * 3 + 2] - positions[i * 3 + 2];

        float r = sqrtf(dx * dx + dy * dy + dz * dz);
        double dr = (double)r - (double)r0i;
        e = 0.5 * (double)ki * dr * dr;
    }

    // Block reduction
    e = block_reduce_sum_d(e);
    if (threadIdx.x == 0) {
        atomicAdd_d(energy_out, e);
    }
}

extern "C" void molrs_bond_harmonic_gradient(
    const float* positions,
    float* gradients,
    const int* atom_i, const int* atom_j,
    const float* k, const float* r0,
    int n_bonds, int n_atoms,
    void* stream)
{
    if (n_bonds == 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (n_bonds + threads - 1) / threads;
    kernel_bond_harmonic_gradient<<<blocks, threads, 0, s>>>(
        positions, gradients, atom_i, atom_j, k, r0, n_bonds);
}

extern "C" double molrs_bond_harmonic_energy(
    const float* positions,
    const int* atom_i, const int* atom_j,
    const float* k, const float* r0,
    int n_bonds, int n_atoms,
    void* stream)
{
    if (n_bonds == 0) return 0.0;
    cudaStream_t s = (cudaStream_t)stream;

    double* d_energy;
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_energy, 0, sizeof(double), s));

    int threads = 256;
    int blocks = (n_bonds + threads - 1) / threads;
    kernel_bond_harmonic_energy<<<blocks, threads, 0, s>>>(
        positions, d_energy, atom_i, atom_j, k, r0, n_bonds);

    double result;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_energy, sizeof(double),
                                cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d_energy));
    return result;
}
