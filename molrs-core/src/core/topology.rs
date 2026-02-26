//! Graph-based molecular topology using igraph.
//!
//! Provides graph-based representation of molecular connectivity with
//! automated detection of angles, dihedrals, and impropers via VF2
//! subisomorphism matching.

use igraph::Graph;

/// Graph-based molecular topology.
///
/// Wraps an undirected `igraph::Graph` where vertices are atoms and edges
/// are bonds. Angles, dihedrals, and impropers are detected automatically
/// from bond connectivity using VF2 subisomorphism.
pub struct Topology {
    graph: Graph,
}

impl Topology {
    /// Create an empty topology with no atoms or bonds.
    pub fn new() -> Self {
        Self {
            graph: Graph::empty(0, false).expect("failed to create empty graph"),
        }
    }

    /// Create a topology with `n` atoms and no bonds.
    pub fn with_atoms(n: usize) -> Self {
        Self {
            graph: Graph::empty(n as i64, false).expect("failed to create graph"),
        }
    }

    /// Create a topology from edge pairs.
    pub fn from_edges(n_atoms: usize, edges: &[[usize; 2]]) -> Self {
        let pairs: Vec<(i64, i64)> = edges.iter().map(|e| (e[0] as i64, e[1] as i64)).collect();
        Self {
            graph: Graph::from_edges(&pairs, n_atoms as i64, false)
                .expect("failed to create graph from edges"),
        }
    }

    // -----------------------------------------------------------------------
    // Count accessors
    // -----------------------------------------------------------------------

    /// Number of atoms (vertices).
    pub fn n_atoms(&self) -> usize {
        self.graph.vcount() as usize
    }

    /// Number of bonds (edges).
    pub fn n_bonds(&self) -> usize {
        self.graph.ecount() as usize
    }

    /// Number of unique angles (i-j-k triplets).
    pub fn n_angles(&self) -> usize {
        let pattern = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
        let count = self.graph.count_subisomorphisms_vf2(&pattern).unwrap();
        (count / 2) as usize
    }

    /// Number of unique proper dihedrals (i-j-k-l quartets).
    pub fn n_dihedrals(&self) -> usize {
        let pattern = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
        let count = self.graph.count_subisomorphisms_vf2(&pattern).unwrap();
        (count / 2) as usize
    }

    // -----------------------------------------------------------------------
    // List accessors
    // -----------------------------------------------------------------------

    /// All atom indices.
    pub fn atoms(&self) -> Vec<usize> {
        (0..self.n_atoms()).collect()
    }

    /// All bond pairs as `[i, j]`.
    pub fn bonds(&self) -> Vec<[usize; 2]> {
        self.graph
            .get_edgelist()
            .unwrap()
            .into_iter()
            .map(|(a, b)| [a as usize, b as usize])
            .collect()
    }

    /// All unique angle triplets `[i, j, k]`, deduplicated.
    pub fn angles(&self) -> Vec<[usize; 3]> {
        let pattern = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
        let matches = self.graph.get_subisomorphisms_vf2(&pattern).unwrap();
        matches
            .into_iter()
            .filter(|m| m[0] < m[2])
            .map(|m| [m[0] as usize, m[1] as usize, m[2] as usize])
            .collect()
    }

    /// All unique proper dihedral quartets `[i, j, k, l]`, deduplicated.
    pub fn dihedrals(&self) -> Vec<[usize; 4]> {
        let pattern = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], 4, false).unwrap();
        let matches = self.graph.get_subisomorphisms_vf2(&pattern).unwrap();
        matches
            .into_iter()
            .filter(|m| m[1] < m[2])
            .map(|m| [m[0] as usize, m[1] as usize, m[2] as usize, m[3] as usize])
            .collect()
    }

    /// All unique improper dihedral quartets `[center, i, j, k]`, deduplicated.
    pub fn impropers(&self) -> Vec<[usize; 4]> {
        let pattern = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], 4, false).unwrap();
        let matches = self.graph.get_subisomorphisms_vf2(&pattern).unwrap();
        if matches.is_empty() {
            return Vec::new();
        }
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for m in &matches {
            let mut tail = [m[1], m[2], m[3]];
            tail.sort();
            let key = (m[0], tail[0], tail[1], tail[2]);
            if seen.insert(key) {
                result.push([m[0] as usize, m[1] as usize, m[2] as usize, m[3] as usize]);
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Atom operations
    // -----------------------------------------------------------------------

    /// Add a single atom.
    pub fn add_atom(&mut self) {
        self.graph.add_vertices(1).expect("failed to add vertex");
    }

    /// Add `n` atoms.
    pub fn add_atoms(&mut self, n: usize) {
        self.graph
            .add_vertices(n as i64)
            .expect("failed to add vertices");
    }

    /// Delete an atom by index.
    pub fn delete_atom(&mut self, idx: usize) {
        self.graph
            .delete_vertices(&[idx as i64])
            .expect("failed to delete vertex");
    }

    // -----------------------------------------------------------------------
    // Bond operations
    // -----------------------------------------------------------------------

    /// Add a bond between atoms `i` and `j` if not already connected.
    pub fn add_bond(&mut self, i: usize, j: usize) {
        if !self.graph.are_adjacent(i as i64, j as i64).unwrap_or(false) {
            self.graph
                .add_edges(&[(i as i64, j as i64)])
                .expect("failed to add edge");
        }
    }

    /// Add multiple bonds from pairs. Skips duplicates.
    pub fn add_bonds(&mut self, pairs: &[[usize; 2]]) {
        for pair in pairs {
            self.add_bond(pair[0], pair[1]);
        }
    }

    /// Delete a bond by edge index.
    pub fn delete_bond(&mut self, idx: usize) {
        self.graph
            .delete_edges(&[idx as i64])
            .expect("failed to delete edge");
    }

    // -----------------------------------------------------------------------
    // Angle helpers
    // -----------------------------------------------------------------------

    /// Add an angle by ensuring bonds i-j and j-k exist.
    pub fn add_angle(&mut self, i: usize, j: usize, k: usize) {
        self.add_bond(i, j);
        self.add_bond(j, k);
    }

    /// Add multiple angles from triplets, ensuring all required bonds exist.
    pub fn add_angles(&mut self, triplets: &[[usize; 3]]) {
        for t in triplets {
            self.add_angle(t[0], t[1], t[2]);
        }
    }
}

impl Default for Topology {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_topology() {
        let topo = Topology::new();
        assert_eq!(topo.n_atoms(), 0);
        assert_eq!(topo.n_bonds(), 0);
    }

    #[test]
    fn test_with_atoms() {
        let topo = Topology::with_atoms(5);
        assert_eq!(topo.n_atoms(), 5);
        assert_eq!(topo.n_bonds(), 0);
    }

    #[test]
    fn test_add_atom() {
        let mut topo = Topology::new();
        topo.add_atom();
        assert_eq!(topo.n_atoms(), 1);
        topo.add_atoms(3);
        assert_eq!(topo.n_atoms(), 4);
    }

    #[test]
    fn test_delete_atom() {
        let mut topo = Topology::with_atoms(3);
        topo.delete_atom(1);
        assert_eq!(topo.n_atoms(), 2);
    }

    #[test]
    fn test_add_bond() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bond(0, 1);
        assert_eq!(topo.n_bonds(), 1);

        // Adding same bond again should not create duplicate
        topo.add_bond(0, 1);
        assert_eq!(topo.n_bonds(), 1);
    }

    #[test]
    fn test_add_bonds() {
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_bonds(), 3);
    }

    #[test]
    fn test_delete_bond() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bond(0, 1);
        topo.add_bond(1, 2);
        assert_eq!(topo.n_bonds(), 2);
        topo.delete_bond(0);
        assert_eq!(topo.n_bonds(), 1);
    }

    #[test]
    fn test_bonds_list() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bond(0, 1);
        topo.add_bond(1, 2);
        let bonds = topo.bonds();
        assert_eq!(bonds.len(), 2);
    }

    #[test]
    fn test_angles_3atom_chain() {
        // 0 - 1 - 2
        let mut topo = Topology::with_atoms(3);
        topo.add_bond(0, 1);
        topo.add_bond(1, 2);
        assert_eq!(topo.n_angles(), 1);
        let angles = topo.angles();
        assert_eq!(angles.len(), 1);
        assert_eq!(angles[0], [0, 1, 2]);
    }

    #[test]
    fn test_dihedrals_4atom_chain() {
        // 0 - 1 - 2 - 3
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_dihedrals(), 1);
        let dihedrals = topo.dihedrals();
        assert_eq!(dihedrals.len(), 1);
        assert_eq!(dihedrals[0], [0, 1, 2, 3]);
    }

    #[test]
    fn test_impropers_star() {
        // Central atom 0 bonded to 1, 2, 3
        let mut topo = Topology::with_atoms(4);
        topo.add_bond(0, 1);
        topo.add_bond(0, 2);
        topo.add_bond(0, 3);
        let impropers = topo.impropers();
        // One unique improper with center 0
        assert_eq!(impropers.len(), 1);
        assert_eq!(impropers[0][0], 0);
    }

    #[test]
    fn test_add_angle_creates_bonds() {
        let mut topo = Topology::with_atoms(3);
        topo.add_angle(0, 1, 2);
        assert_eq!(topo.n_bonds(), 2);
        assert_eq!(topo.n_angles(), 1);
    }

    #[test]
    fn test_methane_ch4() {
        let mut topo = Topology::with_atoms(5);
        topo.add_bond(0, 1);
        topo.add_bond(0, 2);
        topo.add_bond(0, 3);
        topo.add_bond(0, 4);

        assert_eq!(topo.n_atoms(), 5);
        assert_eq!(topo.n_bonds(), 4);
        assert_eq!(topo.n_angles(), 6);
        assert_eq!(topo.n_dihedrals(), 0);
        assert_eq!(topo.impropers().len(), 4);
    }

    #[test]
    fn test_ethane_c2h6() {
        let mut topo = Topology::with_atoms(8);
        topo.add_bond(0, 1);
        topo.add_bond(0, 2);
        topo.add_bond(0, 3);
        topo.add_bond(0, 4);
        topo.add_bond(1, 5);
        topo.add_bond(1, 6);
        topo.add_bond(1, 7);

        assert_eq!(topo.n_atoms(), 8);
        assert_eq!(topo.n_bonds(), 7);
        assert_eq!(topo.n_angles(), 12);
        assert_eq!(topo.n_dihedrals(), 9);
    }

    #[test]
    fn test_from_edges() {
        let topo = Topology::from_edges(4, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_atoms(), 4);
        assert_eq!(topo.n_bonds(), 3);
        assert_eq!(topo.n_angles(), 2);
        assert_eq!(topo.n_dihedrals(), 1);
    }
}
