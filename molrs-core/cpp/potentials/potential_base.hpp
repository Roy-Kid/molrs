#pragma once

#include <cstdint>

namespace molrs::core::potentials {

enum class PotentialFlag : std::uint32_t {
    None = 0u,
    ValidateInput = 1u << 0,
};

class PotentialBase {
public:
    explicit PotentialBase(std::uint32_t flags) : flags_(flags) {}
    virtual ~PotentialBase() = default;

    struct RuntimeView {
        const float* positions;
        float* gradients;
        const int* atom_i;
        const int* atom_j;
        int n_items;
        int n_atoms;
        void* stream;
    };

    virtual double energy(const RuntimeView& runtime) const = 0;
    virtual void gradient(const RuntimeView& runtime) const = 0;

protected:
    bool has_flag(PotentialFlag flag) const {
        return (flags_ & static_cast<std::uint32_t>(flag)) != 0u;
    }

private:
    std::uint32_t flags_;
};

} // namespace molrs::core::potentials
