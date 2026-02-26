#pragma once

#include <cstdint>
#include <memory>

#include "potential_base.hpp"

namespace molrs::core::potentials {

enum class PotentialType : std::uint32_t {
    PairLJ126 = 1u,
    BondHarmonic = 2u,
    AngleHarmonic = 3u,
};

struct PotentialSpec {
    PotentialType type;
    const float* param0;
    const float* param1;
    int n_items;
    std::uint32_t flags;
};

class Potential final {
public:
    static std::unique_ptr<Potential> create(const PotentialSpec& spec);

    explicit Potential(std::unique_ptr<PotentialBase> impl);
    ~Potential();

    double energy(const PotentialBase::RuntimeView& runtime) const;
    void gradient(const PotentialBase::RuntimeView& runtime) const;

private:
    std::unique_ptr<PotentialBase> impl_;
};

} // namespace molrs::core::potentials
