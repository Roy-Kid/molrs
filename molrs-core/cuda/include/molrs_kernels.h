#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Bond harmonic ---- */
void molrs_bond_harmonic_gradient(
    const float* positions,
    float* gradients,
    const int* atom_i, const int* atom_j,
    const float* k, const float* r0,
    int n_bonds, int n_atoms,
    void* stream);

double molrs_bond_harmonic_energy(
    const float* positions,
    const int* atom_i, const int* atom_j,
    const float* k, const float* r0,
    int n_bonds, int n_atoms,
    void* stream);

/* ---- Angle harmonic ---- */
void molrs_angle_harmonic_gradient(
    const float* positions,
    float* gradients,
    const int* atom_i, const int* atom_j, const int* atom_k,
    const float* k_spring, const float* theta0,
    int n_angles, int n_atoms,
    void* stream);

double molrs_angle_harmonic_energy(
    const float* positions,
    const int* atom_i, const int* atom_j, const int* atom_k,
    const float* k_spring, const float* theta0,
    int n_angles, int n_atoms,
    void* stream);

/* ---- Pair LJ 12-6 ---- */
void molrs_pair_lj126_gradient(
    const float* positions,
    float* gradients,
    const int* atom_i, const int* atom_j,
    const float* epsilon, const float* sigma,
    int n_pairs, int n_atoms,
    void* stream);

double molrs_pair_lj126_energy(
    const float* positions,
    const int* atom_i, const int* atom_j,
    const float* epsilon, const float* sigma,
    int n_pairs, int n_atoms,
    void* stream);

/* ---- Energy reduction ---- */
double molrs_energy_reduce(const double* partial, int n, void* stream);

/* ---- Gradient max norm ---- */
float molrs_grad_max_norm(const float* grad, int n, void* stream);

/* ---- GPU-resident SimBox ---- */
typedef struct GPUSimBox GPUSimBox;

GPUSimBox* molrs_simbox_create(void);
void       molrs_simbox_destroy(GPUSimBox* sb);

/* Upload box parameters from host arrays (synchronous) */
void molrs_simbox_upload(
    GPUSimBox* sb,
    const float* h_host,          /* host: [9] row-major H */
    const float* inv_h_host,      /* host: [9] row-major H^{-1} */
    const float* origin_host,     /* host: [3] */
    const int*   periodic_host,   /* host: [3] (0 or 1) */
    void* stream);

/* Device pointer accessors — GPU kernels read/write directly */
float* molrs_simbox_h_ptr(GPUSimBox* sb);
float* molrs_simbox_inv_ptr(GPUSimBox* sb);
float* molrs_simbox_origin_ptr(GPUSimBox* sb);
int*   molrs_simbox_periodic_ptr(GPUSimBox* sb);

/* Const accessors for read-only access */
const float* molrs_simbox_h_ptr_const(const GPUSimBox* sb);
const float* molrs_simbox_inv_ptr_const(const GPUSimBox* sb);
const float* molrs_simbox_origin_ptr_const(const GPUSimBox* sb);
const int*   molrs_simbox_periodic_ptr_const(const GPUSimBox* sb);

/* Recompute H^{-1} from H on GPU (call after barostat updates H) */
void molrs_simbox_update_inverse(GPUSimBox* sb, void* stream);

/* Compute cell grid dimensions from H matrix on GPU */
void molrs_simbox_compute_cell_dims(
    const GPUSimBox* sb,
    float cutoff,
    int* cell_dims_host,          /* host: [3] output (nx, ny, nz) */
    void* stream);

/* ---- GPU-resident neighbor list (Eastman & Pande 2010) ---- */
void molrs_neighborlist_build(
    const float* positions,        /* device: n_atoms * 3 */
    int n_atoms,
    const GPUSimBox* simbox,       /* GPU-resident box handle */
    float cutoff,
    int** out_pair_i, int** out_pair_j,
    int*  out_n_pairs,
    void* stream);

#ifdef __cplusplus
}
#endif
