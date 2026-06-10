//! Physical constants (CODATA 2018 / SI-2019 exact where applicable).
//!
//! Reference: SI Brochure, 9th edition (2019) for the exact defining
//! constants; CODATA 2018 recommended values,
//! <https://physics.nist.gov/cuu/Constants/>.
//!
//! # Examples
//!
//! ```
//! use molrs_core::units::constants::GAS_CONSTANT;
//!
//! // R = N_A · k_B = 8.314 462 618... J/(mol·K), exact under SI-2019.
//! assert!((GAS_CONSTANT - 8.314_462_618_153_24).abs() < 1e-12);
//! ```

use crate::types::F;

/// Avogadro constant `N_A` (exact, SI-2019), in mol⁻¹.
pub const AVOGADRO: F = 6.022_140_76e23;

/// Boltzmann constant `k_B` (exact, SI-2019), in J/K.
pub const BOLTZMANN: F = 1.380_649e-23;

/// Molar gas constant `R = N_A · k_B` (exact, SI-2019), in J/(mol·K).
pub const GAS_CONSTANT: F = AVOGADRO * BOLTZMANN;
