use crate::Frame;

use super::error::ComputeError;

/// Unified analysis trait for all compute operations.
///
/// `Args` is a GAT that specifies what additional data the compute needs
/// beyond the `Frame`:
///
/// - `()` for frame-only analyses (e.g. MSD)
/// - `&'a NeighborList` for pair-based analyses (e.g. RDF, Cluster)
/// - `&'a ClusterResult` for cluster property analyses
///
/// `&self` is an immutable parameter container (bins, cutoffs, masses, etc.).
/// Returns an owned result struct — no hidden mutable state.
pub trait Compute {
    /// Additional arguments beyond the Frame (e.g. `&NeighborList`, `&ClusterResult`).
    type Args<'a>;
    /// The per-frame result type.
    type Output;

    /// Run the analysis on a single frame with the given arguments.
    fn compute(&self, frame: &Frame, args: Self::Args<'_>) -> Result<Self::Output, ComputeError>;
}
