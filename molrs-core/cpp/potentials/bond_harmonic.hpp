#pragma once

#include <cstdint>

#include "potential_base.hpp"

namespace molrs::core::potentials {

class BondHarmonicPotential final : public PotentialBase {
public:
    BondHarmonicPotential(
        const float* k,
        const float* r0,
        int n_bonds,
        std::uint32_t flags);

    double energy(const RuntimeView& runtime) const override;
    void gradient(const RuntimeView& runtime) const override;

private:
    bool validate_runtime(const RuntimeView& runtime) const;

    const float* k_;
    const float* r0_;
    int n_bonds_;
};

} // namespace molrs::core::potentials
