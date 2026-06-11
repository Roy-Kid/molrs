//! Molecular typifiers.
//!
//! Bridges [`MolGraph`](crate::molgraph::MolGraph) to typed
//! [`Frame`](crate::frame::Frame) representations by assigning
//! integer type IDs to atoms, bonds, angles, dihedrals, and impropers.

use molrs::Atomistic;
use molrs::store::frame::Frame;

pub mod mmff;

/// A typifier assigns force-field type IDs to a molecular graph and produces
/// a fully typed [`Frame`].
pub trait Typifier {
    /// Typify an all-atom molecule, returning a [`Frame`] with topology blocks
    /// and type labels ready for force-field compilation.
    fn typify(&self, mol: &Atomistic) -> Result<Frame, String>;
}
