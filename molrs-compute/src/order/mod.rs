//! Bond-orientational order parameters ported from `freud.order`.
//!
//! Currently implemented:
//! - [`Steinhardt`](steinhardt::Steinhardt) — per-particle `q_ℓ`, `w_ℓ`,
//!   averaged variants, and the underlying `q_ℓm` array. Drives the
//!   downstream `SolidLiquid` and `ContinuousCoordination` analyzers.
//!
//! Future phases will add `Nematic`, `Hexatic`, `Cubatic`, `SolidLiquid`,
//! `ContinuousCoordination`, `RotationalAutocorrelation`.

pub mod hexatic;
pub mod nematic;
pub mod solid_liquid;
pub mod steinhardt;

pub use hexatic::{Hexatic, HexaticResult};
pub use nematic::{Nematic, NematicResult};
pub use solid_liquid::{SolidLiquid, SolidLiquidResult};
pub use steinhardt::{Steinhardt, SteinhardtResult, compute_qlm};
