"""Dielectric susceptibility computation (dipole moment, EH/GK spectra, decomposition).

All computation is in Rust; these are thin Python re-exports.
"""

from .molrs import (
    dielectric_compute_current_density as compute_current_density,
    dielectric_compute_dipole_moment as compute_dipole_moment,
    dielectric_decompose_current as decompose_current,
    dielectric_einstein_helfand_spectrum as einstein_helfand_spectrum,
    dielectric_green_kubo_spectrum as green_kubo_spectrum,
    dielectric_static_dielectric_constant as static_dielectric_constant,
)
