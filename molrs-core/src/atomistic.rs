//! All-atom molecular graph with element-level chemistry semantics.
//!
//! [`Atomistic`] wraps the domain-agnostic [`MolGraph`] and owns **all** atom /
//! bond / angle / dihedral / improper vocabulary: it registers those relation
//! kinds at construction (caching their [`KindId`]s) and exposes the typed
//! convenience API (`add_bond`, `bonds`, `get_bond`, …). `MolGraph` itself knows
//! nothing of bonds — the chemistry lives here.
//!
//! Generic graph methods (`nodes`, `neighbors`, `translate`, `add_relation`, …)
//! remain available via `Deref`/`DerefMut`.
//!
//! # Examples
//!
//! ```
//! use molrs_core::atomistic::Atomistic;
//!
//! let mut mol = Atomistic::new();
//! let c = mol.add_atom_bare("C");
//! let h = mol.add_atom_bare("H");
//! mol.add_bond(c, h).unwrap();
//!
//! assert_eq!(mol.n_atoms(), 2);
//! assert_eq!(mol.n_bonds(), 1);
//! ```

use std::ops::{Deref, DerefMut};

use crate::error::MolRsError;
use crate::frame::Frame;
use crate::molgraph::{Atom, KindId, MolGraph, NodeId, PropValue, Relation, RelationId};

/// Handle to an atom (a graph node).
pub type AtomId = NodeId;
/// Handle to a bond (a relation). Distinct-key semantics for `HashSet<BondId>`.
pub type BondId = RelationId;
/// Handle to an angle (a relation).
pub type AngleId = RelationId;
/// Handle to a dihedral (a relation).
pub type DihedralId = RelationId;
/// Handle to an improper (a relation).
pub type ImproperId = RelationId;

/// A bond — a 2-ary relation.
pub type Bond = Relation;
/// An angle (i-j-k) — a 3-ary relation.
pub type Angle = Relation;
/// A dihedral (i-j-k-l) — a 4-ary relation.
pub type Dihedral = Relation;
/// An improper (i-j-k-l) — a 4-ary relation, distinct from a dihedral by kind.
pub type Improper = Relation;

/// All-atom molecular graph.
///
/// Invariant: every atom carries an `"element"` property (element symbol).
#[derive(Debug, Clone)]
pub struct Atomistic {
    graph: MolGraph,
    bond: KindId,
    angle: KindId,
    dihedral: KindId,
    improper: KindId,
}

impl Deref for Atomistic {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.graph
    }
}

impl DerefMut for Atomistic {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }
}

impl Default for Atomistic {
    fn default() -> Self {
        Self::new()
    }
}

impl Atomistic {
    /// Create an empty all-atom molecular graph with the bond / angle /
    /// dihedral / improper kinds registered.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        let bond = graph.register_kind("bonds", 2);
        let angle = graph.register_kind("angles", 3);
        let dihedral = graph.register_kind("dihedrals", 4);
        let improper = graph.register_kind("impropers", 4);
        Self {
            graph,
            bond,
            angle,
            dihedral,
            improper,
        }
    }

    // ---- atoms (nodes) ----

    /// Add an atom carrying a property bag.
    pub fn add_atom(&mut self, atom: Atom) -> AtomId {
        self.graph.add_node_with(atom)
    }

    /// Add an atom with element symbol and 3D coordinates.
    pub fn add_atom_xyz(&mut self, symbol: &str, x: f64, y: f64, z: f64) -> AtomId {
        self.graph.add_node_with(Atom::xyz(symbol, x, y, z))
    }

    /// Add an atom with element symbol only (no coordinates).
    pub fn add_atom_bare(&mut self, symbol: &str) -> AtomId {
        let mut a = Atom::new();
        a.set("element", symbol);
        self.graph.add_node_with(a)
    }

    /// Remove an atom and all incident bonds / angles / dihedrals / impropers.
    pub fn remove_atom(&mut self, id: AtomId) -> Result<Atom, MolRsError> {
        self.graph.remove_node(id)
    }

    /// Materialize an atom's property bag (owned copy of its set components).
    pub fn get_atom(&self, id: AtomId) -> Result<Atom, MolRsError> {
        self.graph.get_node(id)
    }

    /// Set a single component on an atom.
    pub fn set_atom(
        &mut self,
        id: AtomId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_node(id, key, val)
    }

    /// Clear a single component on an atom (no-op if absent).
    pub fn clear_atom(&mut self, id: AtomId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_node(id, key)
    }

    /// Iterate over all `(AtomId, Atom)` pairs (each property bag materialized).
    pub fn atoms(&self) -> impl Iterator<Item = (AtomId, Atom)> + '_ {
        self.graph.nodes()
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.graph.n_nodes()
    }

    // ---- bonds ----

    /// Add a bond between two existing atoms (default order 1.0).
    pub fn add_bond(&mut self, a: AtomId, b: AtomId) -> Result<BondId, MolRsError> {
        let bid = self.graph.add_relation(self.bond, &[a, b])?;
        self.graph
            .get_relation_mut(self.bond, bid)?
            .props
            .insert("order".to_owned(), PropValue::F64(1.0));
        Ok(bid)
    }

    /// Remove a bond.
    pub fn remove_bond(&mut self, id: BondId) -> Result<Bond, MolRsError> {
        self.graph.remove_relation(self.bond, id)
    }

    /// Get a reference to a bond.
    pub fn get_bond(&self, id: BondId) -> Result<&Bond, MolRsError> {
        self.graph.get_relation(self.bond, id)
    }

    /// Get a mutable reference to a bond.
    pub fn get_bond_mut(&mut self, id: BondId) -> Result<&mut Bond, MolRsError> {
        self.graph.get_relation_mut(self.bond, id)
    }

    /// Iterate over all `(BondId, &Bond)` pairs.
    pub fn bonds(&self) -> impl Iterator<Item = (BondId, &Bond)> {
        self.graph.relations(self.bond)
    }

    /// Number of bonds.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    /// Iterate over `(neighbor_id, bond_order)` for a given atom (order read
    /// from the `"order"` prop, default 1.0).
    pub fn neighbor_bonds(&self, id: AtomId) -> impl Iterator<Item = (AtomId, f64)> + '_ {
        self.graph
            .neighbor_relations(id)
            .filter(move |(kind, _, _)| *kind == self.bond)
            .map(move |(kind, rid, other)| {
                let order = self
                    .graph
                    .get_relation(kind, rid)
                    .ok()
                    .and_then(|r| match r.props.get("order") {
                        Some(PropValue::F64(v)) => Some(*v),
                        _ => None,
                    })
                    .unwrap_or(1.0);
                (other, order)
            })
    }

    // ---- angles ----

    /// Add an angle (i-j-k, j central).
    pub fn add_angle(&mut self, i: AtomId, j: AtomId, k: AtomId) -> Result<AngleId, MolRsError> {
        self.graph.add_relation(self.angle, &[i, j, k])
    }

    /// Remove an angle.
    pub fn remove_angle(&mut self, id: AngleId) -> Result<Angle, MolRsError> {
        self.graph.remove_relation(self.angle, id)
    }

    /// Get a reference to an angle.
    pub fn get_angle(&self, id: AngleId) -> Result<&Angle, MolRsError> {
        self.graph.get_relation(self.angle, id)
    }

    /// Get a mutable reference to an angle.
    pub fn get_angle_mut(&mut self, id: AngleId) -> Result<&mut Angle, MolRsError> {
        self.graph.get_relation_mut(self.angle, id)
    }

    /// Iterate over all `(AngleId, &Angle)` pairs.
    pub fn angles(&self) -> impl Iterator<Item = (AngleId, &Angle)> {
        self.graph.relations(self.angle)
    }

    /// Number of angles.
    pub fn n_angles(&self) -> usize {
        self.graph.n_relations(self.angle)
    }

    // ---- dihedrals ----

    /// Add a dihedral (i-j-k-l).
    pub fn add_dihedral(
        &mut self,
        i: AtomId,
        j: AtomId,
        k: AtomId,
        l: AtomId,
    ) -> Result<DihedralId, MolRsError> {
        self.graph.add_relation(self.dihedral, &[i, j, k, l])
    }

    /// Remove a dihedral.
    pub fn remove_dihedral(&mut self, id: DihedralId) -> Result<Dihedral, MolRsError> {
        self.graph.remove_relation(self.dihedral, id)
    }

    /// Get a reference to a dihedral.
    pub fn get_dihedral(&self, id: DihedralId) -> Result<&Dihedral, MolRsError> {
        self.graph.get_relation(self.dihedral, id)
    }

    /// Get a mutable reference to a dihedral.
    pub fn get_dihedral_mut(&mut self, id: DihedralId) -> Result<&mut Dihedral, MolRsError> {
        self.graph.get_relation_mut(self.dihedral, id)
    }

    /// Iterate over all `(DihedralId, &Dihedral)` pairs.
    pub fn dihedrals(&self) -> impl Iterator<Item = (DihedralId, &Dihedral)> {
        self.graph.relations(self.dihedral)
    }

    /// Number of dihedrals.
    pub fn n_dihedrals(&self) -> usize {
        self.graph.n_relations(self.dihedral)
    }

    // ---- impropers ----

    /// Add an improper dihedral (i-j-k-l; i conventionally central).
    pub fn add_improper(
        &mut self,
        i: AtomId,
        j: AtomId,
        k: AtomId,
        l: AtomId,
    ) -> Result<ImproperId, MolRsError> {
        self.graph.add_relation(self.improper, &[i, j, k, l])
    }

    /// Remove an improper.
    pub fn remove_improper(&mut self, id: ImproperId) -> Result<Improper, MolRsError> {
        self.graph.remove_relation(self.improper, id)
    }

    /// Get a reference to an improper.
    pub fn get_improper(&self, id: ImproperId) -> Result<&Improper, MolRsError> {
        self.graph.get_relation(self.improper, id)
    }

    /// Get a mutable reference to an improper.
    pub fn get_improper_mut(&mut self, id: ImproperId) -> Result<&mut Improper, MolRsError> {
        self.graph.get_relation_mut(self.improper, id)
    }

    /// Iterate over all `(ImproperId, &Improper)` pairs.
    pub fn impropers(&self) -> impl Iterator<Item = (ImproperId, &Improper)> {
        self.graph.relations(self.improper)
    }

    /// Number of impropers.
    pub fn n_impropers(&self) -> usize {
        self.graph.n_relations(self.improper)
    }

    // ---- kind handles (for callers that go through the generic API) ----

    /// The bond relation kind.
    pub fn bond_kind(&self) -> KindId {
        self.bond
    }

    // ---- frame conversion ----

    /// Export to a tabular [`Frame`] (atoms / bonds / angles / dihedrals /
    /// impropers blocks).
    pub fn to_frame(&self) -> Frame {
        self.graph.to_frame()
    }

    /// Build from a [`Frame`], registering the standard kinds first so bond /
    /// angle / dihedral / improper blocks are read back.
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let mut mol = Self::new();
        mol.graph.read_frame(frame)?;
        Ok(mol)
    }

    // ---- conversions ----

    /// Promote from a [`MolGraph`], validating all atoms have `"element"`. The
    /// graph's relation kinds are re-registered to the standard set.
    pub fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.nodes() {
            if atom.get_str("element").is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing 'element' property",
                    id
                )));
            }
        }
        let bond = mol.kind_id("bonds").unwrap_or(KindId(0));
        let angle = mol.kind_id("angles").unwrap_or(KindId(1));
        let dihedral = mol.kind_id("dihedrals").unwrap_or(KindId(2));
        let improper = mol.kind_id("impropers").unwrap_or(KindId(3));
        Ok(Self {
            graph: mol,
            bond,
            angle,
            dihedral,
            improper,
        })
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

    /// Perceive aromaticity (RDKit default `AROMATICITY_RDKIT` model) and
    /// annotate the graph in place. Returns the number of atoms flagged
    /// aromatic. See [`crate::aromaticity::perceive_aromaticity`].
    pub fn perceive_aromaticity(&mut self) -> usize {
        crate::aromaticity::perceive_aromaticity(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molgraph::Atom;

    #[test]
    fn test_new_and_add() {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        let h = mol.add_atom_bare("H");
        mol.add_bond(c, h).unwrap();
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
        assert_eq!(mol.get_atom(c).unwrap().get_str("element"), Some("C"));
    }

    #[test]
    fn test_add_atom_xyz() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let atom = mol.get_atom(o).unwrap();
        assert_eq!(atom.get_str("element"), Some("O"));
        assert_eq!(atom.get_f64("x"), Some(0.0));
    }

    #[test]
    fn test_full_topology_and_cascade() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_bare("C");
        let b = mol.add_atom_bare("C");
        let c = mol.add_atom_bare("C");
        let d = mol.add_atom_bare("C");
        mol.add_bond(a, b).unwrap();
        mol.add_bond(a, c).unwrap();
        mol.add_bond(a, d).unwrap();
        mol.add_angle(b, a, c).unwrap();
        mol.add_dihedral(b, a, c, d).unwrap();
        mol.add_improper(b, a, c, d).unwrap();
        assert_eq!(mol.n_bonds(), 3);
        assert_eq!(mol.n_angles(), 1);
        assert_eq!(mol.n_dihedrals(), 1);
        assert_eq!(mol.n_impropers(), 1);
        // typed BondId usable as a HashSet key
        let ids: std::collections::HashSet<BondId> = mol.bonds().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 3);
        // cascade on central atom
        mol.remove_atom(a).unwrap();
        assert_eq!(mol.n_bonds(), 0);
        assert_eq!(mol.n_angles(), 0);
        assert_eq!(mol.n_dihedrals(), 0);
        assert_eq!(mol.n_impropers(), 0);
    }

    #[test]
    fn test_neighbors_and_neighbor_bonds() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_bare("O");
        let h1 = mol.add_atom_bare("H");
        let h2 = mol.add_atom_bare("H");
        mol.add_bond(o, h1).unwrap();
        mol.add_bond(o, h2).unwrap();
        assert_eq!(mol.neighbors(o).count(), 2);
        let nb: Vec<(AtomId, f64)> = mol.neighbor_bonds(o).collect();
        assert_eq!(nb.len(), 2);
        assert!(nb.iter().all(|(_, order)| (*order - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_frame_roundtrip() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let b = mol.add_atom_xyz("C", 1.0, 0.0, 0.0);
        let c = mol.add_atom_xyz("C", 0.0, 1.0, 0.0);
        let d = mol.add_atom_xyz("C", 0.0, 0.0, 1.0);
        mol.add_bond(a, b).unwrap();
        mol.add_angle(a, b, c).unwrap();
        mol.add_improper(a, b, c, d).unwrap();
        let frame = mol.to_frame();
        let mol2 = Atomistic::from_frame(&frame).unwrap();
        assert_eq!(mol2.n_atoms(), 4);
        assert_eq!(mol2.n_bonds(), 1);
        assert_eq!(mol2.n_angles(), 1);
        assert_eq!(mol2.n_impropers(), 1);
    }

    #[test]
    fn test_try_from_molgraph_missing_element() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node_with(Atom::new()); // no element
        assert!(Atomistic::try_from_molgraph(g).is_err());
    }

    #[test]
    fn test_deref_generic_methods() {
        let mut mol = Atomistic::new();
        let c1 = mol.add_atom_bare("C");
        let c2 = mol.add_atom_bare("C");
        mol.add_bond(c1, c2).unwrap();
        // generic MolGraph methods via Deref
        assert_eq!(mol.n_nodes(), 2);
        assert_eq!(mol.neighbors(c1).count(), 1);
    }
}
