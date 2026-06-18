"""Dielectric & ionic-conductivity kernels.

All computation is in Rust. The kernels are grouped under a single
:class:`Dielectric` namespace (a thin set of static methods over the compiled
``molrs.molrs`` PyO3 functions) so callers reach them as
``molrs.dielectric.Dielectric.static_dielectric_constant(...)`` rather than
through a flat list of free functions.

The bundled ``einstein_helfand_conductivity`` was removed in
compute-fit-03-cleanup: compose :class:`molrs.EinsteinConductivity` (raw
collective-dipole MSD) with :class:`molrs.LinearFit` and a
``slope/(6·V·k_B·T)`` MD→SI prefactor instead. The frequency-dependent ε(ω)
spectrum routes (``einstein_helfand_spectrum`` / ``green_kubo_spectrum``)
remain.
"""

from .molrs import (
    dielectric_compute_current_density,
    dielectric_compute_dipole_moment,
    dielectric_decompose_current,
    dielectric_einstein_helfand_spectrum,
    dielectric_green_kubo_spectrum,
    dielectric_static_dielectric_constant,
)


class Dielectric:
    """Namespace of dielectric / ionic-conductivity kernels (all static).

    Grouping the kernels under one class keeps the public ``molrs.dielectric``
    surface to a single name instead of a flat list of free functions. Every
    method is a ``staticmethod`` forwarding straight to the Rust implementation.
    """

    compute_dipole_moment = staticmethod(dielectric_compute_dipole_moment)
    compute_current_density = staticmethod(dielectric_compute_current_density)
    decompose_current = staticmethod(dielectric_decompose_current)
    static_dielectric_constant = staticmethod(dielectric_static_dielectric_constant)
    einstein_helfand_spectrum = staticmethod(dielectric_einstein_helfand_spectrum)
    green_kubo_spectrum = staticmethod(dielectric_green_kubo_spectrum)


__all__ = ["Dielectric"]
