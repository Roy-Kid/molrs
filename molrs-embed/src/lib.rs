//! 3D coordinate generation pipeline for molecular graphs.
//!
//! This module provides a practical `generate_3d` API using a staged workflow:
//! initial coordinate build -> coarse minimization -> rotor sampling ->
//! final minimization -> stereo sanity checks.

// Retired FragmentRules pipeline modules. `generate_3d` no longer routes
// through these (it uses the ETKDG pipeline in `etkdg`), but they are kept on
// disk because a concurrent work-stream has uncommitted changes here. They are
// `#[allow(dead_code)]` at the module level so clippy stays quiet without
// removing any code. See spec mmff94-etkdg-04-embed for the migration plan.
#[allow(dead_code)]
mod builder;
#[allow(dead_code)]
mod distance_geometry;
pub mod distgeom;
#[allow(dead_code)]
mod fragment_data;
#[allow(dead_code)]
mod geom;
#[allow(dead_code)]
mod optimizer;
mod options;
#[allow(dead_code)]
mod pipeline;
mod report;
#[allow(dead_code)]
mod rotor_search;
#[allow(dead_code)]
mod stereo_guard;

/// ETKDGv3 conformer-embedding pipeline (the active `generate_3d` backend).
pub mod etkdg;

pub use options::{EmbedAlgorithm, EmbedOptions, EmbedSpeed, ForceFieldKind};
pub use report::{EmbedReport, StageKind, StageReport};

use molrs::atomistic::Atomistic;
use molrs::error::MolRsError;

/// Generate 3D coordinates for an all-atom molecular graph.
///
/// Returns the generated molecule (with 3D coordinates) and a stage-by-stage
/// report. The input molecule is not modified.
///
/// Requires [`Atomistic`] (not raw `MolGraph`) because embed depends on
/// element symbols for bond-length estimation, ring geometry, and force-field
/// selection.
pub fn generate_3d(
    mol: &Atomistic,
    opts: &EmbedOptions,
) -> Result<(Atomistic, EmbedReport), MolRsError> {
    // Pipeline works on &MolGraph internally (via Deref).
    let (mol_out, report) = etkdg::generate_3d_impl(mol, opts)?;
    // embed only adds atoms via add_hydrogens (which sets "element") and never
    // removes "element" from existing atoms, so the invariant holds.
    let atomistic = Atomistic::try_from_molgraph(mol_out)
        .expect("embed pipeline preserves Atomistic invariant");
    Ok((atomistic, report))
}
