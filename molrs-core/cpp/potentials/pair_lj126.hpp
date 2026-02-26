#pragma once

#include <cstdint>

#include "potential_base.hpp"

namespace molrs::core::potentials {

class PairLJ126Potential final : public PotentialBase {
public:
    PairLJ126Potential(
        const float* epsilon,
        const float* sigma,
        int n_pairs,
        std::uint32_t flags);

    double energy(const RuntimeView& runtime) const override;
    void gradient(const RuntimeView& runtime) const override;

private:
    bool validate_runtime(const RuntimeView& runtime) const;

    const float* epsilon_;
    const float* sigma_;
    int n_pairs_;
};

} // namespace molrs::core::potentials
