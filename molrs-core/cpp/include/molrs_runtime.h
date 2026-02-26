#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Device management ---- */
int molrs_device_init(int device_id);
int molrs_device_get_name(int device_id, char* buf, int len);
void* molrs_malloc(size_t size);
void molrs_free(void* ptr);
void molrs_memcpy_h2d(void* dst, const void* src, size_t size);
void molrs_memcpy_d2h(void* dst, const void* src, size_t size);
void molrs_memset_zero(void* ptr, size_t size);

/* ---- Generic potential runtime ---- */
#define MOLRS_POTENTIAL_TYPE_PAIR_LJ126     1u
#define MOLRS_POTENTIAL_TYPE_BOND_HARMONIC  2u
#define MOLRS_POTENTIAL_TYPE_ANGLE_HARMONIC 3u

void* molrs_potential_create(
    uint32_t type,
    const float* param0,
    const float* param1,
    int n_items,
    uint32_t flags);

void molrs_potential_destroy(void* handle);

double molrs_potential_energy(
    void* handle,
    const float* positions,
    const int* atom_i, const int* atom_j,
    int n_items, int n_atoms,
    void* stream);

void molrs_potential_gradient(
    void* handle,
    const float* positions,
    float* gradients,
    const int* atom_i, const int* atom_j,
    int n_items, int n_atoms,
    void* stream);

#ifdef __cplusplus
}
#endif
