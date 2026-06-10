//! Physical constants (CODATA 2018 / SI-2019 exact where applicable).

use crate::types::F;

/// Avogadro constant `N_A` (exact, SI-2019), in mol⁻¹.
pub const AVOGADRO: F = 6.022_140_76e23;

/// Boltzmann constant `k_B` (exact, SI-2019), in J/K.
pub const BOLTZMANN: F = 1.380_649e-23;

/// Molar gas constant `R = N_A * k_B`, in J/(mol·K).
pub const GAS_CONSTANT: F = AVOGADRO * BOLTZMANN;
