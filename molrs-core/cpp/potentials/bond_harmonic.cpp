#include "bond_harmonic.hpp"

#include <limits>

#include "../../cuda/include/molrs_kernels.h"

namespace molrs::core::potentials {

BondHarmonicPotential::BondHarmonicPotential(
    const float* k,
    const float* r0,
    int n_bonds,
    std::uint32_t flags)
    : PotentialBase(flags),
      k_(k),
      r0_(r0),
      n_bonds_(n_bonds) {}

bool BondHarmonicPotential::validate_runtime(const RuntimeView& runtime) const {
    if (!has_flag(PotentialFlag::ValidateInput)) {
        return true;
    }
    if (runtime.positions == nullptr || runtime.atom_i == nullptr || runtime.atom_j == nullptr) {
        return false;
    }
    if (runtime.n_items < 0 || runtime.n_atoms < 0) {
        return false;
    }
    if (k_ == nullptr || r0_ == nullptr) {
        return false;
    }
    if (n_bonds_ >= 0 && runtime.n_items != n_bonds_) {
        return false;
    }
    return true;
}

double BondHarmonicPotential::energy(const RuntimeView& runtime) const {
    if (!validate_runtime(runtime)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (runtime.n_items == 0) {
        return 0.0;
    }
    return molrs_bond_harmonic_energy(
        runtime.positions,
        runtime.atom_i,
        runtime.atom_j,
        k_,
        r0_,
        runtime.n_items,
        runtime.n_atoms,
        runtime.stream);
}

void BondHarmonicPotential::gradient(const RuntimeView& runtime) const {
    if (!validate_runtime(runtime) || runtime.n_items == 0 || runtime.gradients == nullptr) {
        return;
    }
    molrs_bond_harmonic_gradient(
        runtime.positions,
        runtime.gradients,
        runtime.atom_i,
        runtime.atom_j,
        k_,
        r0_,
        runtime.n_items,
        runtime.n_atoms,
        runtime.stream);
}

} // namespace molrs::core::potentials
