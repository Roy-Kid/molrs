//! Density-related analyzers ported from `freud.density`.
//!
//! Currently implemented:
//! - [`LocalDensity`](local_density::LocalDensity) — per-particle number
//!   density inside a sphere of radius `r_max`.
//! - [`GaussianDensity`](gaussian_density::GaussianDensity) — 3-D / 2-D grid
//!   smearing of point particles with a Gaussian kernel.
//! - [`CorrelationFunction`](correlation_function::CorrelationFunction) —
//!   generic distance-binned correlation `⟨A·B*⟩(r)` for arbitrary scalar
//!   (real or complex) per-particle fields.
//!
//! [`crate::RDF`] (the radial distribution function) lives in `crate::rdf`
//! and is the prototypical member of this family; the others reuse the same
//! histogram and SimBox conventions.

pub mod correlation_function;
pub mod gaussian_density;
pub mod local_density;

pub use correlation_function::{CorrelationFunction, CorrelationFunctionResult};
pub use gaussian_density::{GaussianDensity, GaussianDensityResult};
pub use local_density::{LocalDensity, LocalDensityResult};
