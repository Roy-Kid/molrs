#include "potential.hpp"

#include <limits>

#include "pair_lj126.hpp"
#include "bond_harmonic.hpp"

namespace molrs::core::potentials {

std::unique_ptr<Potential> Potential::create(const PotentialSpec& spec) {
    switch (spec.type) {
        case PotentialType::PairLJ126:
            return std::make_unique<Potential>(std::make_unique<PairLJ126Potential>(
                spec.param0, spec.param1, spec.n_items, spec.flags));
        case PotentialType::BondHarmonic:
            return std::make_unique<Potential>(std::make_unique<BondHarmonicPotential>(
                spec.param0, spec.param1, spec.n_items, spec.flags));
        default:
            return nullptr;
    }
}

Potential::Potential(std::unique_ptr<PotentialBase> impl) : impl_(std::move(impl)) {}
Potential::~Potential() = default;

double Potential::energy(const PotentialBase::RuntimeView& runtime) const {
    if (!impl_) return std::numeric_limits<double>::quiet_NaN();
    return impl_->energy(runtime);
}

void Potential::gradient(const PotentialBase::RuntimeView& runtime) const {
    if (!impl_) return;
    impl_->gradient(runtime);
}

} // namespace molrs::core::potentials

extern "C" void* molrs_potential_create(
    std::uint32_t type,
    const float* param0,
    const float* param1,
    int n_items,
    std::uint32_t flags)
{
    using namespace molrs::core::potentials;
    const PotentialSpec spec{
        static_cast<PotentialType>(type),
        param0,
        param1,
        n_items,
        flags,
    };
    auto potential = Potential::create(spec);
    return potential.release();
}

extern "C" void molrs_potential_destroy(void* handle) {
    using namespace molrs::core::potentials;
    auto* potential = static_cast<Potential*>(handle);
    delete potential;
}

extern "C" double molrs_potential_energy(
    void* handle,
    const float* positions,
    const int* atom_i,
    const int* atom_j,
    int n_items,
    int n_atoms,
    void* stream)
{
    using namespace molrs::core::potentials;
    auto* potential = static_cast<Potential*>(handle);
    if (!potential) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const PotentialBase::RuntimeView runtime{
        positions,
        nullptr,
        atom_i,
        atom_j,
        n_items,
        n_atoms,
        stream,
    };
    return potential->energy(runtime);
}

extern "C" void molrs_potential_gradient(
    void* handle,
    const float* positions,
    float* gradients,
    const int* atom_i,
    const int* atom_j,
    int n_items,
    int n_atoms,
    void* stream)
{
    using namespace molrs::core::potentials;
    auto* potential = static_cast<Potential*>(handle);
    if (!potential) return;
    const PotentialBase::RuntimeView runtime{
        positions,
        gradients,
        atom_i,
        atom_j,
        n_items,
        n_atoms,
        stream,
    };
    potential->gradient(runtime);
}
