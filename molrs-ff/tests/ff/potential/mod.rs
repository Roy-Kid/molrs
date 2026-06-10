//! Integration tests for the `potential` module tree.
//!
//! Mirrors `src/potential/`. Each leaf builds inputs in code, evaluates the
//! public `Potential` API, and checks energy / forces against analytical
//! values plus finite-difference gradients and Newton's third law.

#[path = "geometry.rs"]
mod geometry;

#[path = "angle.rs"]
mod angle;
#[path = "bond.rs"]
mod bond;
#[path = "mmff.rs"]
mod mmff;
#[path = "opls.rs"]
mod opls;
#[path = "pair.rs"]
mod pair;
#[path = "pme.rs"]
mod pme;
