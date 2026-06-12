//! Numeric constants shared across the MMFF energy kernels (`mmff::energy`) and
//! the pair / bonded potential adapters (`potential`). Values match RDKit's
//! MMFF94 implementation exactly so the two evaluation paths stay numerically
//! identical — keeping them in one place prevents the rounding drift that
//! creeps in when the same literal is re-typed per module.

/// mdyne·Å → kcal/mol (RDKit `MDYNE_A_TO_KCAL_MOL`, `Params.h`).
pub(crate) const MDYNE_A_TO_KCAL: f64 = 143.9325;

/// Coulomb constant `e²/(4π·ε₀)` in kcal·Å·mol⁻¹·e⁻² (RDKit `Nonbonded.cpp`).
/// Distinct from the CODATA-derived [`molrs::units::constants::COULOMB_REAL`]
/// used by the generic Coulomb pair potential — RDKit rounds it differently and
/// the MMFF parity tests pin this exact value.
pub(crate) const COULOMB_MMFF: f64 = 332.0716;

/// Electrostatic buffering distance δ in Å (RDKit `calcEleEnergy`).
pub(crate) const ELE_BUFFER: f64 = 0.05;

/// degrees → radians (RDKit `DEG2RAD`).
pub(crate) const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// radians → degrees (RDKit `RAD2DEG`).
pub(crate) const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;
