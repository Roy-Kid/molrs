#include "common.cuh"
#include "include/molrs_kernels.h"

// One thread per pair: compute gradient (dE/dx convention)
__global__ void kernel_pair_lj126_gradient(
    const float* __restrict__ positions,
    float* __restrict__ gradients,
    const int* __restrict__ atom_i,
    const int* __restrict__ atom_j,
    const float* __restrict__ epsilon,
    const float* __restrict__ sigma,
    int n_pairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;

    int i = atom_i[idx];
    int j = atom_j[idx];
    float eps = epsilon[idx];
    float sig = sigma[idx];

    float dx = positions[j * 3]     - positions[i * 3];
    float dy = positions[j * 3 + 1] - positions[i * 3 + 1];
    float dz = positions[j * 3 + 2] - positions[i * 3 + 2];

    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < 1e-24f) return;

    float sr2 = sig * sig / r2;
    float sr6 = sr2 * sr2 * sr2;
    float sr12 = sr6 * sr6;

    // gradient factor: 4*eps*(-12*sr12 + 6*sr6) / r2
    float factor = 4.0f * eps * (-12.0f * sr12 + 6.0f * sr6) / r2;
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

// One thread per pair: compute energy
__global__ void kernel_pair_lj126_energy(
    const float* __restrict__ positions,
    double* __restrict__ energy_out,
    const int* __restrict__ atom_i,
    const int* __restrict__ atom_j,
    const float* __restrict__ epsilon,
    const float* __restrict__ sigma,
    int n_pairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double e = 0.0;
    if (idx < n_pairs) {
        int i = atom_i[idx];
        int j = atom_j[idx];
        double eps = (double)epsilon[idx];
        double sig = (double)sigma[idx];

        float dx = positions[j * 3]     - positions[i * 3];
        float dy = positions[j * 3 + 1] - positions[i * 3 + 1];
        float dz = positions[j * 3 + 2] - positions[i * 3 + 2];

        double r2 = (double)dx * dx + (double)dy * dy + (double)dz * dz;
        if (r2 > 1e-24) {
            double sr2 = sig * sig / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            e = 4.0 * eps * (sr12 - sr6);
        }
    }

    e = block_reduce_sum_d(e);
    if (threadIdx.x == 0) {
        atomicAdd_d(energy_out, e);
    }
}

extern "C" void molrs_pair_lj126_gradient(
    const float* positions,
    float* gradients,
    const int* atom_i, const int* atom_j,
    const float* epsilon, const float* sigma,
    int n_pairs, int n_atoms,
    void* stream)
{
    if (n_pairs == 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (n_pairs + threads - 1) / threads;
    kernel_pair_lj126_gradient<<<blocks, threads, 0, s>>>(
        positions, gradients, atom_i, atom_j, epsilon, sigma, n_pairs);
}

extern "C" double molrs_pair_lj126_energy(
    const float* positions,
    const int* atom_i, const int* atom_j,
    const float* epsilon, const float* sigma,
    int n_pairs, int n_atoms,
    void* stream)
{
    if (n_pairs == 0) return 0.0;
    cudaStream_t s = (cudaStream_t)stream;

    double* d_energy;
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_energy, 0, sizeof(double), s));

    int threads = 256;
    int blocks = (n_pairs + threads - 1) / threads;
    kernel_pair_lj126_energy<<<blocks, threads, 0, s>>>(
        positions, d_energy, atom_i, atom_j, epsilon, sigma, n_pairs);

    double result;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_energy, sizeof(double),
                                cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d_energy));
    return result;
}
