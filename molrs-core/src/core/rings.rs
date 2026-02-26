//! Ring detection for molecular graphs.
//!
//! Computes the **Smallest Set of Smallest Rings (SSSR)** — equivalently the
//! minimum cycle basis — via [`igraph_minimum_cycle_basis`].
//!
//! # Algorithm
//! 1. Assign consecutive integer IDs to atoms and bonds (matching igraph's
//!    0-based vertex/edge numbering).
//! 2. Build an undirected `igraph::Graph` from those edges.
//! 3. Call `graph.minimum_cycle_basis(use_cycle_order=true)` → `Vec<Vec<i64>>`
//!    of edge-ID lists, one per cycle, in traversal order.
//! 4. Convert each edge-ID list to an ordered `Vec<AtomId>` ring and build
//!    reverse-lookup maps for atoms and bonds.
//!
//! # Complexity
//! O(V + E) graph build + O(R · L) index build (R = rings, L = avg ring size).

use std::collections::HashMap;

use igraph::Graph;

use super::molgraph::{AtomId, BondId, MolGraph};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// All ring information for a [`MolGraph`], produced by [`find_rings`].
#[derive(Debug, Clone)]
pub struct RingInfo {
    /// Each ring is an ordered list of `AtomId`s forming a closed path.
    rings: Vec<Vec<AtomId>>,
    /// atom → indices of rings that contain it.
    atom_rings: HashMap<AtomId, Vec<usize>>,
    /// bond → indices of rings that contain it.
    bond_rings: HashMap<BondId, Vec<usize>>,
}

impl RingInfo {
    fn empty() -> Self {
        Self {
            rings: Vec::new(),
            atom_rings: HashMap::new(),
            bond_rings: HashMap::new(),
        }
    }

    /// Whether the atom belongs to any ring.
    pub fn is_atom_in_ring(&self, id: AtomId) -> bool {
        self.atom_rings.get(&id).is_some_and(|v| !v.is_empty())
    }

    /// Whether the bond belongs to any ring.
    pub fn is_bond_in_ring(&self, id: BondId) -> bool {
        self.bond_rings.get(&id).is_some_and(|v| !v.is_empty())
    }

    /// Number of rings containing this atom.
    pub fn num_atom_rings(&self, id: AtomId) -> usize {
        self.atom_rings.get(&id).map_or(0, Vec::len)
    }

    /// Number of rings containing this bond.
    pub fn num_bond_rings(&self, id: BondId) -> usize {
        self.bond_rings.get(&id).map_or(0, Vec::len)
    }

    /// Size (atom count) of every ring, in ascending order.
    pub fn ring_sizes(&self) -> Vec<usize> {
        self.rings.iter().map(Vec::len).collect()
    }

    /// All rings of exactly `n` atoms.
    pub fn rings_of_size(&self, n: usize) -> Vec<&Vec<AtomId>> {
        self.rings.iter().filter(|r| r.len() == n).collect()
    }

    /// Size of the smallest ring containing `id`, if any.
    pub fn smallest_ring_containing_atom(&self, id: AtomId) -> Option<usize> {
        self.atom_rings
            .get(&id)?
            .iter()
            .map(|&ri| self.rings[ri].len())
            .min()
    }

    /// Total number of rings detected.
    pub fn num_rings(&self) -> usize {
        self.rings.len()
    }

    /// All rings as slices of `AtomId`.
    pub fn rings(&self) -> &[Vec<AtomId>] {
        &self.rings
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compute the ring information (SSSR / minimum cycle basis) for `mol`.
pub fn find_rings(mol: &MolGraph) -> RingInfo {
    if mol.n_atoms() == 0 {
        return RingInfo::empty();
    }

    // ---- 1. Build stable atom/bond orderings --------------------------------
    // igraph numbers vertices 0..n-1 and edges 0..m-1 in insertion order,
    // so we keep parallel Vecs to map between igraph indices and slotmap IDs.
    let atom_vec: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    let atom_to_idx: HashMap<AtomId, i64> = atom_vec
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as i64))
        .collect();

    let bond_vec: Vec<BondId> = mol.bonds().map(|(id, _)| id).collect();

    // ---- 2. Build igraph::Graph ---------------------------------------------
    // Edge i in igraph corresponds to bond_vec[i] in MolGraph.
    let ig_edges: Vec<(i64, i64)> = bond_vec
        .iter()
        .map(|&bid| {
            let b = mol.bond(bid).expect("bond must exist");
            (atom_to_idx[&b.atoms[0]], atom_to_idx[&b.atoms[1]])
        })
        .collect();

    let graph = match Graph::from_edges(&ig_edges, atom_vec.len() as i64, false) {
        Ok(g) => g,
        Err(_) => return RingInfo::empty(),
    };

    // ---- 3. Minimum cycle basis (edge-ID lists in traversal order) ----------
    let cycle_edge_lists = match graph.minimum_cycle_basis(true) {
        Ok(v) => v,
        Err(_) => return RingInfo::empty(),
    };

    // ---- 4. Convert edge-ID cycles → AtomId rings --------------------------
    // Fast (AtomId, AtomId) → BondId lookup table.
    let mut bond_map: HashMap<(AtomId, AtomId), BondId> = HashMap::new();
    for &bid in &bond_vec {
        let b = mol.bond(bid).expect("bond must exist");
        let [a, bb] = b.atoms;
        bond_map.insert((a, bb), bid);
        bond_map.insert((bb, a), bid);
    }

    let mut raw_rings: Vec<Vec<AtomId>> = cycle_edge_lists
        .iter()
        .filter_map(|edge_ids| {
            let vids = edges_to_vertex_sequence(edge_ids, &graph)?;
            if vids.len() < 3 {
                return None;
            }
            Some(vids.iter().map(|&v| atom_vec[v as usize]).collect())
        })
        .collect();

    // Sort smallest first (SSSR convention).
    raw_rings.sort_by_key(Vec::len);

    // ---- 5. Build reverse-lookup maps --------------------------------------
    let mut atom_rings: HashMap<AtomId, Vec<usize>> = HashMap::new();
    let mut bond_rings: HashMap<BondId, Vec<usize>> = HashMap::new();

    for (ri, ring) in raw_rings.iter().enumerate() {
        let n = ring.len();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            atom_rings.entry(a).or_default().push(ri);
            if let Some(&bid) = bond_map.get(&(a, b)) {
                bond_rings.entry(bid).or_default().push(ri);
            }
        }
    }

    RingInfo {
        rings: raw_rings,
        atom_rings,
        bond_rings,
    }
}

// ---------------------------------------------------------------------------
// Helper: reconstruct vertex sequence from ordered edge IDs
// ---------------------------------------------------------------------------

/// Convert an ordered list of edge IDs (as returned by
/// `minimum_cycle_basis(use_cycle_order=true)`) into an ordered vertex
/// sequence.
///
/// igraph guarantees that consecutive edges in the list share exactly one
/// vertex.  We determine the start vertex by finding which endpoint of
/// `edge[0]` is NOT shared with `edge[1]`, then walk forward.
fn edges_to_vertex_sequence(edge_ids: &[i64], graph: &Graph) -> Option<Vec<i64>> {
    if edge_ids.len() < 2 {
        return None;
    }

    let (a0, b0) = graph.edge(edge_ids[0]).ok()?;
    let (a1, b1) = graph.edge(edge_ids[1]).ok()?;

    // Start at the endpoint of edge[0] that is NOT shared with edge[1].
    let start = if a1 == b0 || b1 == b0 { a0 } else { b0 };

    let mut vertices = Vec::with_capacity(edge_ids.len());
    let mut prev = start;
    for &eid in edge_ids {
        let (u, v) = graph.edge(eid).ok()?;
        vertices.push(prev);
        prev = if u == prev { v } else { u };
    }
    // `prev` is `start` again (closed cycle) — not pushed.

    Some(vertices)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::molgraph::{Atom, MolGraph};

    fn cycle(n: usize) -> MolGraph {
        let mut g = MolGraph::new();
        let ids: Vec<AtomId> = (0..n).map(|_| g.add_atom(Atom::new())).collect();
        for i in 0..n {
            g.add_bond(ids[i], ids[(i + 1) % n]);
        }
        g
    }

    #[test]
    fn test_single_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        assert_eq!(ri.num_rings(), 1);
        assert_eq!(ri.ring_sizes(), vec![6]);
    }

    #[test]
    fn test_linear_no_rings() {
        let mut g = MolGraph::new();
        let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(Atom::new())).collect();
        for i in 0..5 {
            g.add_bond(ids[i], ids[i + 1]);
        }
        assert_eq!(find_rings(&g).num_rings(), 0);
    }

    #[test]
    fn test_all_atoms_in_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        for (id, _) in g.atoms() {
            assert!(ri.is_atom_in_ring(id));
        }
    }

    #[test]
    fn test_all_bonds_in_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        for (bid, _) in g.bonds() {
            assert!(ri.is_bond_in_ring(bid));
        }
    }

    #[test]
    fn test_empty_mol() {
        assert_eq!(find_rings(&MolGraph::new()).num_rings(), 0);
    }
}
