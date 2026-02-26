#include "common.cuh"
#include "include/molrs_kernels.h"

#include <cub/cub.cuh>

// ---------------------------------------------------------------------------
// Constant memory for box parameters (broadcast reads across all threads)
// ---------------------------------------------------------------------------
__constant__ float c_box_matrix[9];      // H (row-major 3x3)
__constant__ float c_inv_box_matrix[9];  // H^{-1} (row-major 3x3)
__constant__ float c_origin[3];
__constant__ int   c_periodic[3];        // 0 or 1

// ---------------------------------------------------------------------------
// Device helpers — mirror SimBox methods exactly
// ---------------------------------------------------------------------------

/// mat3 * vec3 (row-major)
__device__ __forceinline__ void mat3_mul_vec(
    const float* __restrict__ m,
    float vx, float vy, float vz,
    float& ox, float& oy, float& oz)
{
    ox = m[0] * vx + m[1] * vy + m[2] * vz;
    oy = m[3] * vx + m[4] * vy + m[5] * vz;
    oz = m[6] * vx + m[7] * vy + m[8] * vz;
}

/// Cartesian -> fractional, wrapped to [0,1)
/// Mirrors SimBox::make_fractional_impl (simbox.rs)
__device__ __forceinline__ void cart_to_frac(
    float px, float py, float pz,
    float& fx, float& fy, float& fz)
{
    float dx = px - c_origin[0];
    float dy = py - c_origin[1];
    float dz = pz - c_origin[2];
    mat3_mul_vec(c_inv_box_matrix, dx, dy, dz, fx, fy, fz);
    fx -= floorf(fx);
    fy -= floorf(fy);
    fz -= floorf(fz);
}

/// Minimum-image displacement vector (b - a)
/// Mirrors SimBox::shortest_vector_impl (simbox.rs)
__device__ __forceinline__ void min_image_dr(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float& drx, float& dry, float& drz)
{
    float dx = bx - ax;
    float dy = by - ay;
    float dz = bz - az;

    float sx, sy, sz;
    mat3_mul_vec(c_inv_box_matrix, dx, dy, dz, sx, sy, sz);

    if (c_periodic[0]) sx -= rintf(sx);
    if (c_periodic[1]) sy -= rintf(sy);
    if (c_periodic[2]) sz -= rintf(sz);

    mat3_mul_vec(c_box_matrix, sx, sy, sz, drx, dry, drz);
}

// ---------------------------------------------------------------------------
// Stage 1: Assign atoms to cells via fractional coordinates
// ---------------------------------------------------------------------------
__global__ void kernel_assign_cells(
    const float* __restrict__ positions,
    int* __restrict__ cell_indices,
    int* __restrict__ atom_indices,
    int nx, int ny, int nz,
    int n_atoms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_atoms) return;

    float px = positions[idx * 3];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];

    float fx, fy, fz;
    cart_to_frac(px, py, pz, fx, fy, fz);

    int cx = min((int)(fx * nx), nx - 1);
    int cy = min((int)(fy * ny), ny - 1);
    int cz = min((int)(fz * nz), nz - 1);
    cx = max(0, cx);
    cy = max(0, cy);
    cz = max(0, cz);

    // Cell index: z-major ordering (matches CPU linkcell.rs coord_to_index)
    cell_indices[idx] = cz * ny * nx + cy * nx + cx;
    atom_indices[idx] = idx;
}

// ---------------------------------------------------------------------------
// Stage 3: Reorder positions into sorted order for coalesced access
// ---------------------------------------------------------------------------
__global__ void kernel_reorder_positions(
    const float* __restrict__ positions,
    const int*   __restrict__ sorted_atom_indices,
    float*       __restrict__ sorted_positions,
    int n_atoms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_atoms) return;

    int orig = sorted_atom_indices[idx];
    sorted_positions[idx * 3]     = positions[orig * 3];
    sorted_positions[idx * 3 + 1] = positions[orig * 3 + 1];
    sorted_positions[idx * 3 + 2] = positions[orig * 3 + 2];
}

// ---------------------------------------------------------------------------
// Stage 4: Find cell boundaries in sorted array
// ---------------------------------------------------------------------------
__global__ void kernel_find_cell_bounds(
    const int* __restrict__ sorted_cell_indices,
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    int n_atoms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_atoms) return;

    int cell = sorted_cell_indices[idx];

    if (idx == 0 || cell != sorted_cell_indices[idx - 1]) {
        cell_start[cell] = idx;
    }
    if (idx == n_atoms - 1 || cell != sorted_cell_indices[idx + 1]) {
        cell_end[cell] = idx + 1;
    }
}

// ---------------------------------------------------------------------------
// Cell coordinate helpers with PBC support
// ---------------------------------------------------------------------------

__device__ __forceinline__ int wrap_cell(int c, int dim, int periodic_flag)
{
    if (!periodic_flag) return c;
    if (c < 0) return c + dim;
    if (c >= dim) return c - dim;
    return c;
}

__device__ __forceinline__ bool cell_in_range(int c, int dim, int periodic_flag)
{
    if (periodic_flag) return true;
    return c >= 0 && c < dim;
}

// ---------------------------------------------------------------------------
// Stage 5: Count pairs per sorted atom (half-shell: sorted_j > sorted_i)
// ---------------------------------------------------------------------------
__global__ void kernel_count_pairs(
    const float* __restrict__ sorted_positions,
    const int*   __restrict__ sorted_cell_indices,
    const int*   __restrict__ cell_start,
    const int*   __restrict__ cell_end,
    int*         __restrict__ pair_counts,
    float cutoff_sq,
    int nx, int ny, int nz,
    int n_atoms)
{
    int sorted_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_i >= n_atoms) return;

    float ax = sorted_positions[sorted_i * 3];
    float ay = sorted_positions[sorted_i * 3 + 1];
    float az = sorted_positions[sorted_i * 3 + 2];

    int cell_i = sorted_cell_indices[sorted_i];
    int cz_i = cell_i / (ny * nx);
    int rem   = cell_i % (ny * nx);
    int cy_i = rem / nx;
    int cx_i = rem % nx;

    int count = 0;

    for (int dz = -1; dz <= 1; dz++) {
        int ncz = cz_i + dz;
        if (!cell_in_range(ncz, nz, c_periodic[2])) continue;
        ncz = wrap_cell(ncz, nz, c_periodic[2]);

        for (int dy = -1; dy <= 1; dy++) {
            int ncy = cy_i + dy;
            if (!cell_in_range(ncy, ny, c_periodic[1])) continue;
            ncy = wrap_cell(ncy, ny, c_periodic[1]);

            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx_i + dx;
                if (!cell_in_range(ncx, nx, c_periodic[0])) continue;
                ncx = wrap_cell(ncx, nx, c_periodic[0]);

                int neighbor_cell = ncz * ny * nx + ncy * nx + ncx;
                int start = cell_start[neighbor_cell];
                int end   = cell_end[neighbor_cell];

                for (int sorted_j = start; sorted_j < end; sorted_j++) {
                    if (sorted_j <= sorted_i) continue;  // half-shell

                    float bx = sorted_positions[sorted_j * 3];
                    float by = sorted_positions[sorted_j * 3 + 1];
                    float bz = sorted_positions[sorted_j * 3 + 2];

                    float drx, dry, drz;
                    min_image_dr(ax, ay, az, bx, by, bz, drx, dry, drz);
                    float r2 = drx * drx + dry * dry + drz * drz;

                    if (r2 < cutoff_sq) {
                        count++;
                    }
                }
            }
        }
    }

    pair_counts[sorted_i] = count;
}

// ---------------------------------------------------------------------------
// Stage 7: Collect pairs (same iteration as count, writes output)
// ---------------------------------------------------------------------------
__global__ void kernel_collect_pairs(
    const float* __restrict__ sorted_positions,
    const int*   __restrict__ sorted_cell_indices,
    const int*   __restrict__ sorted_atom_indices,
    const int*   __restrict__ cell_start,
    const int*   __restrict__ cell_end,
    const int*   __restrict__ pair_offsets,
    int*         __restrict__ out_pair_i,
    int*         __restrict__ out_pair_j,
    float cutoff_sq,
    int nx, int ny, int nz,
    int n_atoms)
{
    int sorted_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_i >= n_atoms) return;

    float ax = sorted_positions[sorted_i * 3];
    float ay = sorted_positions[sorted_i * 3 + 1];
    float az = sorted_positions[sorted_i * 3 + 2];

    int cell_i = sorted_cell_indices[sorted_i];
    int cz_i = cell_i / (ny * nx);
    int rem   = cell_i % (ny * nx);
    int cy_i = rem / nx;
    int cx_i = rem % nx;

    int orig_i = sorted_atom_indices[sorted_i];
    int write_pos = pair_offsets[sorted_i];

    for (int dz = -1; dz <= 1; dz++) {
        int ncz = cz_i + dz;
        if (!cell_in_range(ncz, nz, c_periodic[2])) continue;
        ncz = wrap_cell(ncz, nz, c_periodic[2]);

        for (int dy = -1; dy <= 1; dy++) {
            int ncy = cy_i + dy;
            if (!cell_in_range(ncy, ny, c_periodic[1])) continue;
            ncy = wrap_cell(ncy, ny, c_periodic[1]);

            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx_i + dx;
                if (!cell_in_range(ncx, nx, c_periodic[0])) continue;
                ncx = wrap_cell(ncx, nx, c_periodic[0]);

                int neighbor_cell = ncz * ny * nx + ncy * nx + ncx;
                int start = cell_start[neighbor_cell];
                int end   = cell_end[neighbor_cell];

                for (int sorted_j = start; sorted_j < end; sorted_j++) {
                    if (sorted_j <= sorted_i) continue;  // half-shell

                    float bx_p = sorted_positions[sorted_j * 3];
                    float by_p = sorted_positions[sorted_j * 3 + 1];
                    float bz_p = sorted_positions[sorted_j * 3 + 2];

                    float drx, dry, drz;
                    min_image_dr(ax, ay, az, bx_p, by_p, bz_p, drx, dry, drz);
                    float r2 = drx * drx + dry * dry + drz * drz;

                    if (r2 < cutoff_sq) {
                        int orig_j = sorted_atom_indices[sorted_j];
                        // Canonical order: smaller original index first
                        if (orig_i < orig_j) {
                            out_pair_i[write_pos] = orig_i;
                            out_pair_j[write_pos] = orig_j;
                        } else {
                            out_pair_i[write_pos] = orig_j;
                            out_pair_j[write_pos] = orig_i;
                        }
                        write_pos++;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Orchestrator: GPU-resident neighbor list pipeline
//
// Algorithm (Eastman & Pande, J. Comput. Chem. 2010):
// 1. kernel_assign_cells    - fractional coords -> cell index
// 2. CUB RadixSort          - sort (cell_index, atom_index) pairs
// 3. kernel_reorder_positions - gather positions into sorted order
// 4. kernel_find_cell_bounds - detect cell_start/cell_end boundaries
// 5. kernel_count_pairs     - iterate 27 neighbor cells, count pairs
// 6. CUB ExclusiveSum       - prefix sum -> write offsets
// 7. kernel_collect_pairs   - same iteration, write (orig_i, orig_j)
//
// Sync points:
// - Reading cell_dims (3 ints) after compute_cell_dims kernel
// - Reading total_pairs (2 ints) after prefix sum
// ---------------------------------------------------------------------------
extern "C" void molrs_neighborlist_build(
    const float* positions,
    int n_atoms,
    const GPUSimBox* simbox,
    float cutoff,
    int** out_pair_i, int** out_pair_j,
    int*  out_n_pairs,
    void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;

    if (n_atoms == 0) {
        *out_pair_i = nullptr;
        *out_pair_j = nullptr;
        *out_n_pairs = 0;
        return;
    }

    // --- Compute cell grid dimensions on GPU ---
    int h_cell_dims[3];
    molrs_simbox_compute_cell_dims(simbox, cutoff, h_cell_dims, stream);
    int cell_nx = h_cell_dims[0];
    int cell_ny = h_cell_dims[1];
    int cell_nz = h_cell_dims[2];
    int n_cells = cell_nx * cell_ny * cell_nz;

    // --- Copy box parameters from GPUSimBox to constant memory ---
    const float* sb_h   = molrs_simbox_h_ptr_const(simbox);
    const float* sb_inv = molrs_simbox_inv_ptr_const(simbox);
    const float* sb_org = molrs_simbox_origin_ptr_const(simbox);
    const int*   sb_pbc = molrs_simbox_periodic_ptr_const(simbox);

    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_box_matrix, sb_h, 9 * sizeof(float),
                                        0, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_inv_box_matrix, sb_inv, 9 * sizeof(float),
                                        0, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_origin, sb_org, 3 * sizeof(float),
                                        0, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_periodic, sb_pbc, 3 * sizeof(int),
                                        0, cudaMemcpyDeviceToDevice, s));

    float cutoff_sq = cutoff * cutoff;
    const int BLOCK = 256;
    int grid_atoms = (n_atoms + BLOCK - 1) / BLOCK;

    // --- Allocate intermediates ---
    int* d_cell_indices;
    int* d_atom_indices;
    int* d_sorted_cell;
    int* d_sorted_atom;
    float* d_sorted_pos;
    int* d_cell_start;
    int* d_cell_end;
    int* d_pair_counts;
    int* d_pair_offsets;

    CUDA_CHECK(cudaMalloc(&d_cell_indices, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_atom_indices, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_cell,  n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_atom,  n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_pos,   n_atoms * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_start,   n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_end,     n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_counts,  n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_offsets, (n_atoms + 1) * sizeof(int)));

    CUDA_CHECK(cudaMemsetAsync(d_cell_start, 0, n_cells * sizeof(int), s));
    CUDA_CHECK(cudaMemsetAsync(d_cell_end,   0, n_cells * sizeof(int), s));

    // --- Stage 1: Assign cells ---
    kernel_assign_cells<<<grid_atoms, BLOCK, 0, s>>>(
        positions, d_cell_indices, d_atom_indices,
        cell_nx, cell_ny, cell_nz, n_atoms);

    // --- Stage 2: CUB Radix Sort ---
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_bytes,
        d_cell_indices, d_sorted_cell,
        d_atom_indices, d_sorted_atom,
        n_atoms, 0, sizeof(int) * 8, s);

    void* d_sort_temp;
    CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));

    cub::DeviceRadixSort::SortPairs(
        d_sort_temp, sort_temp_bytes,
        d_cell_indices, d_sorted_cell,
        d_atom_indices, d_sorted_atom,
        n_atoms, 0, sizeof(int) * 8, s);

    CUDA_CHECK(cudaFree(d_sort_temp));

    // --- Stage 3: Reorder positions ---
    kernel_reorder_positions<<<grid_atoms, BLOCK, 0, s>>>(
        positions, d_sorted_atom, d_sorted_pos, n_atoms);

    // --- Stage 4: Find cell boundaries ---
    kernel_find_cell_bounds<<<grid_atoms, BLOCK, 0, s>>>(
        d_sorted_cell, d_cell_start, d_cell_end, n_atoms);

    // --- Stage 5: Count pairs ---
    kernel_count_pairs<<<grid_atoms, BLOCK, 0, s>>>(
        d_sorted_pos, d_sorted_cell,
        d_cell_start, d_cell_end,
        d_pair_counts, cutoff_sq,
        cell_nx, cell_ny, cell_nz, n_atoms);

    // --- Stage 6: CUB Exclusive Sum ---
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, scan_temp_bytes,
        d_pair_counts, d_pair_offsets,
        n_atoms, s);

    void* d_scan_temp;
    CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));

    cub::DeviceScan::ExclusiveSum(
        d_scan_temp, scan_temp_bytes,
        d_pair_counts, d_pair_offsets,
        n_atoms, s);

    CUDA_CHECK(cudaFree(d_scan_temp));

    // --- Read total pair count (sync point: 2 ints D2H) ---
    int h_last_offset = 0;
    int h_last_count  = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_last_offset, d_pair_offsets + (n_atoms - 1),
                                sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(&h_last_count,  d_pair_counts  + (n_atoms - 1),
                                sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    int total_pairs = h_last_offset + h_last_count;

    // --- Allocate output + Stage 7: Collect pairs ---
    int* d_out_pair_i = nullptr;
    int* d_out_pair_j = nullptr;
    if (total_pairs > 0) {
        CUDA_CHECK(cudaMalloc(&d_out_pair_i, total_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_pair_j, total_pairs * sizeof(int)));

        kernel_collect_pairs<<<grid_atoms, BLOCK, 0, s>>>(
            d_sorted_pos, d_sorted_cell, d_sorted_atom,
            d_cell_start, d_cell_end,
            d_pair_offsets,
            d_out_pair_i, d_out_pair_j,
            cutoff_sq,
            cell_nx, cell_ny, cell_nz, n_atoms);
    }

    *out_pair_i  = d_out_pair_i;
    *out_pair_j  = d_out_pair_j;
    *out_n_pairs = total_pairs;

    // --- Free intermediates ---
    CUDA_CHECK(cudaFree(d_cell_indices));
    CUDA_CHECK(cudaFree(d_atom_indices));
    CUDA_CHECK(cudaFree(d_sorted_cell));
    CUDA_CHECK(cudaFree(d_sorted_atom));
    CUDA_CHECK(cudaFree(d_sorted_pos));
    CUDA_CHECK(cudaFree(d_cell_start));
    CUDA_CHECK(cudaFree(d_cell_end));
    CUDA_CHECK(cudaFree(d_pair_counts));
    CUDA_CHECK(cudaFree(d_pair_offsets));
}
