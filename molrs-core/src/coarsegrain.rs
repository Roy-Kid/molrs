//! Coarse-grained molecular graph.
//!
//! [`CoarseGrain`] wraps the domain-agnostic [`MolGraph`] where every node is a
//! bead (a group of atoms). It registers its own `bonds` kind and exposes the
//! bead / CG-bond vocabulary; `MolGraph` itself stays chemistry-agnostic.
//!
//! Generic graph methods (`nodes`, `neighbors`, …) remain available via
//! `Deref`/`DerefMut`.
//!
//! # Examples
//!
//! ```
//! use molrs_core::coarsegrain::CoarseGrain;
//!
//! let mut cg = CoarseGrain::new();
//! let b1 = cg.add_bead("W", 0.0, 0.0, 0.0);
//! let b2 = cg.add_bead("W", 3.0, 0.0, 0.0);
//! cg.add_bond(b1, b2).unwrap();
//!
//! assert_eq!(cg.n_beads(), 2);
//! assert_eq!(cg.n_bonds(), 1);
//! ```

use std::ops::{Deref, DerefMut};

use crate::atomistic::{Bond, BondId};
use crate::error::MolRsError;
use crate::molgraph::{Atom, KindId, MolGraph, NodeId};

/// Handle to a bead (a graph node).
pub type BeadId = NodeId;

/// Coarse-grained molecular graph.
///
/// Invariant: every node has a `"bead_type"` property.
#[derive(Debug, Clone)]
pub struct CoarseGrain {
    graph: MolGraph,
    bond: KindId,
}

impl Deref for CoarseGrain {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.graph
    }
}

impl DerefMut for CoarseGrain {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }
}

impl Default for CoarseGrain {
    fn default() -> Self {
        Self::new()
    }
}

impl CoarseGrain {
    /// Create an empty coarse-grained molecular graph with the CG `bonds` kind
    /// registered.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        let bond = graph.register_kind("bonds", 2);
        Self { graph, bond }
    }

    /// Add a bead with type name and 3D coordinates.
    pub fn add_bead(&mut self, bead_type: &str, x: f64, y: f64, z: f64) -> BeadId {
        let mut a = Atom::new();
        a.set("bead_type", bead_type);
        a.set("x", x);
        a.set("y", y);
        a.set("z", z);
        self.graph.add_node_with(a)
    }

    /// Add a bead with type name only (no coordinates).
    pub fn add_bead_bare(&mut self, bead_type: &str) -> BeadId {
        let mut a = Atom::new();
        a.set("bead_type", bead_type);
        self.graph.add_node_with(a)
    }

    /// Remove a bead and all incident CG bonds.
    pub fn remove_bead(&mut self, id: BeadId) -> Result<Atom, MolRsError> {
        self.graph.remove_node(id)
    }

    /// Get a reference to a bead.
    pub fn get_bead(&self, id: BeadId) -> Result<&Atom, MolRsError> {
        self.graph.get_node(id)
    }

    /// Iterate over all `(BeadId, &Atom)` pairs.
    pub fn beads(&self) -> impl Iterator<Item = (BeadId, &Atom)> {
        self.graph.nodes()
    }

    /// Number of beads.
    pub fn n_beads(&self) -> usize {
        self.graph.n_nodes()
    }

    /// Add a CG bond between two existing beads.
    pub fn add_bond(&mut self, a: BeadId, b: BeadId) -> Result<BondId, MolRsError> {
        self.graph.add_relation(self.bond, &[a, b])
    }

    /// Get a reference to a CG bond.
    pub fn get_bond(&self, id: BondId) -> Result<&Bond, MolRsError> {
        self.graph.get_relation(self.bond, id)
    }

    /// Iterate over all `(BondId, &Bond)` pairs.
    pub fn bonds(&self) -> impl Iterator<Item = (BondId, &Bond)> {
        self.graph.relations(self.bond)
    }

    /// Number of CG bonds.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    /// Promote from a [`MolGraph`], validating all nodes have `"bead_type"`.
    pub fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.nodes() {
            if atom.get_str("bead_type").is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing 'bead_type' property",
                    id
                )));
            }
        }
        let bond = mol.kind_id("bonds").unwrap_or(KindId(0));
        Ok(Self { graph: mol, bond })
    }

    /// Unwrap to the inner [`MolGraph`] (zero cost).
    pub fn into_inner(self) -> MolGraph {
        self.graph
    }

    /// Borrow the inner [`MolGraph`].
    pub fn as_molgraph(&self) -> &MolGraph {
        &self.graph
    }

    /// Mutably borrow the inner [`MolGraph`].
    pub fn as_molgraph_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_add_bead() {
        let mut cg = CoarseGrain::new();
        let b1 = cg.add_bead("W", 0.0, 0.0, 0.0);
        let b2 = cg.add_bead("P1", 3.0, 0.0, 0.0);
        cg.add_bond(b1, b2).unwrap();
        assert_eq!(cg.n_beads(), 2);
        assert_eq!(cg.n_bonds(), 1);
    }

    #[test]
    fn test_bead_has_type() {
        let mut cg = CoarseGrain::new();
        let b = cg.add_bead("W", 1.0, 2.0, 3.0);
        let bead = cg.get_bead(b).unwrap();
        assert_eq!(bead.get_str("bead_type"), Some("W"));
        assert_eq!(bead.get_f64("x"), Some(1.0));
    }

    #[test]
    fn test_try_from_molgraph_missing_bead_type() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node_with(Atom::new());
        assert!(CoarseGrain::try_from_molgraph(g).is_err());
    }

    #[test]
    fn test_into_inner() {
        let mut cg = CoarseGrain::new();
        cg.add_bead("W", 0.0, 0.0, 0.0);
        let g: MolGraph = cg.into_inner();
        assert_eq!(g.n_nodes(), 1);
    }
}
