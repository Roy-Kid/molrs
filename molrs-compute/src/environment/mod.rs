//! Local-environment analyzers ported from `freud.environment`.
//!
//! Currently implemented:
//! - [`AngularSeparation`](angular_separation::AngularSeparationGlobal) —
//!   pairwise angular separation between two sets of unit quaternions
//!   (rotational orientations). Both the global variant (all-vs-all
//!   reference) and a per-neighbor variant are exposed.
//! - [`BondOrder`](bond_order::BondOrder) — 2-D `(θ, φ)` histogram of
//!   neighbor bond vectors in the local frame.
//!
//! Future phases will add `LocalBondProjection`, `LocalDescriptors`, and
//! `MatchEnv`.

pub mod angular_separation;
pub mod bond_order;

pub use angular_separation::{
    AngularSeparationGlobal, AngularSeparationGlobalResult, AngularSeparationNeighbor,
    AngularSeparationNeighborResult,
};
pub use bond_order::{BondOrder, BondOrderResult};
