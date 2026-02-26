#include "common.cuh"
#include "include/molrs_kernels.h"

// One thread per angle: compute gradient
__global__ void kernel_angle_harmonic_gradient(
    const float* __restrict__ positions,
    float* __restrict__ gradients,
    const int* __restrict__ atom_i,
    const int* __restrict__ atom_j,
    const int* __restrict__ atom_k,
    const float* __restrict__ k_spring,
    const float* __restrict__ theta0,
    int n_angles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_angles) return;

    int ai = atom_i[idx];
    int aj = atom_j[idx]; // central atom
    int ak = atom_k[idx];
    float kk = k_spring[idx];
    float t0 = theta0[idx];

    // Vectors from central atom j to i and k
    float rji_x = positions[ai * 3]     - positions[aj * 3];
    float rji_y = positions[ai * 3 + 1] - positions[aj * 3 + 1];
    float rji_z = positions[ai * 3 + 2] - positions[aj * 3 + 2];

    float rjk_x = positions[ak * 3]     - positions[aj * 3];
    float rjk_y = positions[ak * 3 + 1] - positions[aj * 3 + 1];
    float rjk_z = positions[ak * 3 + 2] - positions[aj * 3 + 2];

    float rji2 = rji_x * rji_x + rji_y * rji_y + rji_z * rji_z;
    float rjk2 = rjk_x * rjk_x + rjk_y * rjk_y + rjk_z * rjk_z;

    if (rji2 < 1e-24f || rjk2 < 1e-24f) return;

    float rji_inv = rsqrtf(rji2);
    float rjk_inv = rsqrtf(rjk2);

    float cos_theta = (rji_x * rjk_x + rji_y * rjk_y + rji_z * rjk_z)
                      * rji_inv * rjk_inv;
    // Clamp to avoid NaN in acos
    cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
    float theta = acosf(cos_theta);

    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    if (sin_theta < 1e-6f) sin_theta = 1e-6f;

    // E = 0.5 * k * (theta - theta0)^2
    // dE/dtheta = k * (theta - theta0)
    float dEdtheta = kk * (theta - t0);

    // d(theta)/d(cos_theta) = -1/sin_theta
    // dE/d(cos_theta) = -dEdtheta / sin_theta
    float prefactor = -dEdtheta / sin_theta;

    // Gradient w.r.t. atom i:
    // d(cos_theta)/d(r_i) = (rjk/(|rji||rjk|) - cos_theta * rji/|rji|^2)
    float gi_x = prefactor * (rjk_x * rji_inv * rjk_inv - cos_theta * rji_x * rji_inv * rji_inv);
    float gi_y = prefactor * (rjk_y * rji_inv * rjk_inv - cos_theta * rji_y * rji_inv * rji_inv);
    float gi_z = prefactor * (rjk_z * rji_inv * rjk_inv - cos_theta * rji_z * rji_inv * rji_inv);

    // Gradient w.r.t. atom k:
    float gk_x = prefactor * (rji_x * rji_inv * rjk_inv - cos_theta * rjk_x * rjk_inv * rjk_inv);
    float gk_y = prefactor * (rji_y * rji_inv * rjk_inv - cos_theta * rjk_y * rjk_inv * rjk_inv);
    float gk_z = prefactor * (rji_z * rji_inv * rjk_inv - cos_theta * rjk_z * rjk_inv * rjk_inv);

    // Gradient w.r.t. atom j = -(gi + gk)
    float gj_x = -(gi_x + gk_x);
    float gj_y = -(gi_y + gk_y);
    float gj_z = -(gi_z + gk_z);

    atomicAdd(&gradients[ai * 3],     gi_x);
    atomicAdd(&gradients[ai * 3 + 1], gi_y);
    atomicAdd(&gradients[ai * 3 + 2], gi_z);
    atomicAdd(&gradients[aj * 3],     gj_x);
    atomicAdd(&gradients[aj * 3 + 1], gj_y);
    atomicAdd(&gradients[aj * 3 + 2], gj_z);
    atomicAdd(&gradients[ak * 3],     gk_x);
    atomicAdd(&gradients[ak * 3 + 1], gk_y);
    atomicAdd(&gradients[ak * 3 + 2], gk_z);
}

// One thread per angle: compute energy
__global__ void kernel_angle_harmonic_energy(
    const float* __restrict__ positions,
    double* __restrict__ energy_out,
    const int* __restrict__ atom_i,
    const int* __restrict__ atom_j,
    const int* __restrict__ atom_k,
    const float* __restrict__ k_spring,
    const float* __restrict__ theta0,
    int n_angles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double e = 0.0;
    if (idx < n_angles) {
        int ai = atom_i[idx];
        int aj = atom_j[idx];
        int ak = atom_k[idx];

        float rji_x = positions[ai * 3]     - positions[aj * 3];
        float rji_y = positions[ai * 3 + 1] - positions[aj * 3 + 1];
        float rji_z = positions[ai * 3 + 2] - positions[aj * 3 + 2];

        float rjk_x = positions[ak * 3]     - positions[aj * 3];
        float rjk_y = positions[ak * 3 + 1] - positions[aj * 3 + 1];
        float rjk_z = positions[ak * 3 + 2] - positions[aj * 3 + 2];

        float rji2 = rji_x * rji_x + rji_y * rji_y + rji_z * rji_z;
        float rjk2 = rjk_x * rjk_x + rjk_y * rjk_y + rjk_z * rjk_z;

        if (rji2 > 1e-24f && rjk2 > 1e-24f) {
            float rji_inv = rsqrtf(rji2);
            float rjk_inv = rsqrtf(rjk2);
            float cos_theta = (rji_x * rjk_x + rji_y * rjk_y + rji_z * rjk_z)
                              * rji_inv * rjk_inv;
            cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
            double theta = acos((double)cos_theta);
            double dt = theta - (double)theta0[idx];
            e = 0.5 * (double)k_spring[idx] * dt * dt;
        }
    }

    e = block_reduce_sum_d(e);
    if (threadIdx.x == 0) {
        atomicAdd_d(energy_out, e);
    }
}

extern "C" void molrs_angle_harmonic_gradient(
    const float* positions,
    float* gradients,
    const int* atom_i, const int* atom_j, const int* atom_k,
    const float* k_spring, const float* theta0,
    int n_angles, int n_atoms,
    void* stream)
{
    if (n_angles == 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (n_angles + threads - 1) / threads;
    kernel_angle_harmonic_gradient<<<blocks, threads, 0, s>>>(
        positions, gradients, atom_i, atom_j, atom_k,
        k_spring, theta0, n_angles);
}

extern "C" double molrs_angle_harmonic_energy(
    const float* positions,
    const int* atom_i, const int* atom_j, const int* atom_k,
    const float* k_spring, const float* theta0,
    int n_angles, int n_atoms,
    void* stream)
{
    if (n_angles == 0) return 0.0;
    cudaStream_t s = (cudaStream_t)stream;

    double* d_energy;
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_energy, 0, sizeof(double), s));

    int threads = 256;
    int blocks = (n_angles + threads - 1) / threads;
    kernel_angle_harmonic_energy<<<blocks, threads, 0, s>>>(
        positions, d_energy, atom_i, atom_j, atom_k,
        k_spring, theta0, n_angles);

    double result;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_energy, sizeof(double),
                                cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d_energy));
    return result;
}
