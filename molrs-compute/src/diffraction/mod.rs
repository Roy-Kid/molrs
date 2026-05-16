//! Diffraction / structure-factor analyzers ported from
//! `freud.diffraction`.
//!
//! Currently implemented:
//! - [`StaticStructureFactorDebye`](debye::StaticStructureFactorDebye) — the
//!   closed-form Debye structure factor `S(k) = N⁻¹ Σ_{i,j} sin(k r_ij) /
//!   (k r_ij)` evaluated on a user-supplied k grid.
//!
//! Phase 9 will follow with `StaticStructureFactorDirect` and
//! `DiffractionPattern` (both FFT-based).

pub mod debye;

pub use debye::{StaticStructureFactorDebye, StaticStructureFactorDebyeResult};
