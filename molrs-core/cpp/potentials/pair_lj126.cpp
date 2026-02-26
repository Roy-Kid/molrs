#include "pair_lj126.hpp"

#include <limits>

#include "../../cuda/include/molrs_kernels.h"

namespace molrs::core::potentials {

PairLJ126Potential::PairLJ126Potential(
    const float* epsilon,
    const float* sigma,
    int n_pairs,
    std::uint32_t flags)
    : PotentialBase(flags),
      epsilon_(epsilon),
      sigma_(sigma),
      n_pairs_(n_pairs) {}

bool PairLJ126Potential::validate_runtime(const RuntimeView& runtime) const {
    if (!has_flag(PotentialFlag::ValidateInput)) {
        return true;
    }
    if (runtime.positions == nullptr || runtime.atom_i == nullptr || runtime.atom_j == nullptr) {
        return false;
    }
    if (runtime.n_items < 0 || runtime.n_atoms < 0) {
        return false;
    }
    if (epsilon_ == nullptr || sigma_ == nullptr) {
        return false;
    }
    if (n_pairs_ >= 0 && runtime.n_items != n_pairs_) {
        return false;
    }
    return true;
}

double PairLJ126Potential::energy(const RuntimeView& runtime) const {
    if (!validate_runtime(runtime)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (runtime.n_items == 0) {
        return 0.0;
    }
    return molrs_pair_lj126_energy(
        runtime.positions,
        runtime.atom_i,
        runtime.atom_j,
        epsilon_,
        sigma_,
        runtime.n_items,
        runtime.n_atoms,
        runtime.stream);
}

void PairLJ126Potential::gradient(const RuntimeView& runtime) const {
    if (!validate_runtime(runtime) || runtime.n_items == 0 || runtime.gradients == nullptr) {
        return;
    }
    molrs_pair_lj126_gradient(
        runtime.positions,
        runtime.gradients,
        runtime.atom_i,
        runtime.atom_j,
        epsilon_,
        sigma_,
        runtime.n_items,
        runtime.n_atoms,
        runtime.stream);
}

} // namespace molrs::core::potentials
