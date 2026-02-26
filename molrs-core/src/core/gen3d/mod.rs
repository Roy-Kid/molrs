//! 3D coordinate generation pipeline for molecular graphs.
//!
//! This module provides a practical `generate_3d` API using a staged workflow:
//! initial coordinate build -> coarse minimization -> rotor sampling ->
//! final minimization -> stereo sanity checks.

mod builder;
mod distance_geometry;
mod fragment_data;
mod geom;
mod optimizer;
mod options;
mod pipeline;
mod report;
mod rotor_search;
mod stereo_guard;

pub use options::{EmbedAlgorithm, ForceFieldKind, Gen3DOptions, Gen3DSpeed};
pub use report::{Gen3DReport, StageKind, StageReport};

use super::molgraph::MolGraph;
use crate::error::MolRsError;

/// Generate 3D coordinates for a molecular graph.
///
/// Returns the generated molecule and a stage-by-stage report.
pub fn generate_3d(
    mol: &MolGraph,
    opts: &Gen3DOptions,
) -> Result<(MolGraph, Gen3DReport), MolRsError> {
    pipeline::generate_3d_impl(mol, opts)
}
