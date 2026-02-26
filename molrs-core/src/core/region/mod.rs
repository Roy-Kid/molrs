//! Region module exports
//!
//! This module collects geometric region data types and traits.

// Expose the ndarray-based Box merged into simbox.rs
pub mod simbox;
pub use simbox::SimBox;
// Generic region traits and shapes
#[allow(clippy::module_inception)]
pub mod region;
pub use crate::core::types::{Bounds3f, PointsNx3f};
pub use region::{AndRegion, HollowSphere, NotRegion, OrRegion, Region, Sphere};
