//! Graph-based molecular topology.
//!
//! Provides graph-based representation of molecular connectivity with
//! automated detection of angles, dihedrals, and impropers via neighbor
//! traversal.

use std::collections::{HashMap, VecDeque};

/// Graph-based molecular topology.
///
/// Holds a native adjacency snapshot where vertices are contiguous atom
/// indices `0..n` and edges are bonds. Angles, dihedrals, and impropers are
/// detected automatically from bond connectivity using neighbor traversal.
pub struct Topology {
    /// Node count.
    n: usize,
    /// `adj[node]` = neighbor node indices, in insertion order.
    adj: Vec<Vec<usize>>,
    /// Edges in insertion order, `[i, j]` as added.
    edges: Vec<[usize; 2]>,
}

impl Topology {
    /// Create an empty topology with no atoms or bonds.
    pub fn new() -> Self {
        Self {
            n: 0,
            adj: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Create a topology with `n` atoms and no bonds.
    pub fn with_atoms(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
            edges: Vec::new(),
        }
    }

    /// Create a topology from edge pairs.
    pub fn from_edges(n_atoms: usize, edges: &[[usize; 2]]) -> Self {
        let mut topo = Self::with_atoms(n_atoms);
        // `from_edges` does not deduplicate (matching the prior graph backend),
        // so push every edge as-is.
        for e in edges {
            topo.edges.push([e[0], e[1]]);
            topo.adj[e[0]].push(e[1]);
            topo.adj[e[1]].push(e[0]);
        }
        topo
    }

    // -----------------------------------------------------------------------
    // Count accessors
    // -----------------------------------------------------------------------

    /// Number of atoms (vertices).
    pub fn n_atoms(&self) -> usize {
        self.n
    }

    /// Number of bonds (edges).
    pub fn n_bonds(&self) -> usize {
        self.edges.len()
    }

    /// Number of unique angles (i-j-k triplets).
    pub fn n_angles(&self) -> usize {
        self.angles().len()
    }

    /// Number of unique proper dihedrals (i-j-k-l quartets).
    pub fn n_dihedrals(&self) -> usize {
        self.dihedrals().len()
    }

    // -----------------------------------------------------------------------
    // List accessors
    // -----------------------------------------------------------------------

    /// All atom indices.
    pub fn atoms(&self) -> Vec<usize> {
        (0..self.n).collect()
    }

    /// All bond pairs as `[i, j]`.
    pub fn bonds(&self) -> Vec<[usize; 2]> {
        self.edges.clone()
    }

    /// All unique angle triplets `[i, j, k]`, deduplicated (i < k).
    pub fn angles(&self) -> Vec<[usize; 3]> {
        let mut result = Vec::new();
        for j in 0..self.n {
            let neighbors = &self.adj[j];
            for a in 0..neighbors.len() {
                for b in (a + 1)..neighbors.len() {
                    let i = neighbors[a];
                    let k = neighbors[b];
                    if i < k {
                        result.push([i, j, k]);
                    } else {
                        result.push([k, j, i]);
                    }
                }
            }
        }
        result
    }

    /// All unique proper dihedral quartets `[i, j, k, l]`, deduplicated (j < k).
    pub fn dihedrals(&self) -> Vec<[usize; 4]> {
        let mut result = Vec::new();
        for edge in &self.edges {
            let (a, b) = (edge[0], edge[1]);
            // Canonical ordering: j < k
            let (j, k) = if a < b { (a, b) } else { (b, a) };

            let j_neighbors: Vec<usize> = self.adj[j].iter().copied().filter(|&n| n != k).collect();
            let k_neighbors: Vec<usize> = self.adj[k].iter().copied().filter(|&n| n != j).collect();

            for &i in &j_neighbors {
                for &l in &k_neighbors {
                    if i != l {
                        result.push([i, j, k, l]);
                    }
                }
            }
        }
        result
    }

    /// All unique improper dihedral quartets `[center, i, j, k]`, deduplicated.
    ///
    /// For each atom with degree >= 3, iterate all sorted 3-combinations
    /// of its neighbors.
    pub fn impropers(&self) -> Vec<[usize; 4]> {
        let mut result = Vec::new();
        for center in 0..self.n {
            let mut neighbors: Vec<usize> = self.adj[center].clone();
            if neighbors.len() < 3 {
                continue;
            }
            neighbors.sort_unstable();
            let n = neighbors.len();
            for a in 0..n {
                for b in (a + 1)..n {
                    for c in (b + 1)..n {
                        result.push([center, neighbors[a], neighbors[b], neighbors[c]]);
                    }
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Query accessors
    // -----------------------------------------------------------------------

    /// Neighbor atom indices of atom `idx`.
    pub fn neighbors(&self, idx: usize) -> Vec<usize> {
        self.adj[idx].clone()
    }

    /// Degree (number of bonds) of atom `idx`.
    pub fn degree(&self, idx: usize) -> usize {
        self.adj[idx].len()
    }

    /// Whether atoms `i` and `j` are directly bonded.
    pub fn are_bonded(&self, i: usize, j: usize) -> bool {
        self.adj[i].contains(&j)
    }

    // -----------------------------------------------------------------------
    // Connected components (cluster by bond topology)
    // -----------------------------------------------------------------------

    /// Per-atom connected component labels.
    ///
    /// Returns a `Vec<i64>` of length `n_atoms`, where each element is the
    /// component ID (0-based, contiguous). Isolated atoms each form their own
    /// component.
    pub fn connected_components(&self) -> Vec<i64> {
        let n = self.n;
        let mut labels = vec![-1i64; n];
        let mut label = 0i64;

        for start in 0..n {
            if labels[start] >= 0 {
                continue;
            }
            let mut queue = VecDeque::new();
            queue.push_back(start);
            labels[start] = label;

            while let Some(current) = queue.pop_front() {
                for &ni in &self.adj[current] {
                    if labels[ni] < 0 {
                        labels[ni] = label;
                        queue.push_back(ni);
                    }
                }
            }
            label += 1;
        }
        labels
    }

    /// Single-source shortest-path distances (BFS over the unweighted bond
    /// graph).
    ///
    /// Returns a `Vec<i64>` of length `n_atoms`: the hop count from `source` to
    /// each atom, or `-1` for atoms unreachable from `source` (a different
    /// connected component). `source` itself has distance 0. An out-of-range
    /// `source` yields an all-`-1` vector.
    pub fn distances(&self, source: usize) -> Vec<i64> {
        let n = self.n;
        let mut dist = vec![-1i64; n];
        if source >= n {
            return dist;
        }
        dist[source] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(current) = queue.pop_front() {
            let d = dist[current];
            for &ni in &self.adj[current] {
                if dist[ni] < 0 {
                    dist[ni] = d + 1;
                    queue.push_back(ni);
                }
            }
        }
        dist
    }

    /// Number of connected components.
    pub fn n_components(&self) -> usize {
        self.connected_components()
            .iter()
            .max()
            .map_or(0, |&m| (m + 1) as usize)
    }

    // -----------------------------------------------------------------------
    // Ring detection (SSSR)
    // -----------------------------------------------------------------------

    /// Compute the Smallest Set of Smallest Rings (SSSR).
    ///
    /// Returns a [`TopologyRingInfo`] containing all detected rings and
    /// lookup tables for per-atom and per-bond ring membership.
    pub fn find_rings(&self) -> TopologyRingInfo {
        let n_nodes = self.n;
        let n_edges = self.edges.len();
        if n_nodes == 0 || n_edges == 0 {
            return TopologyRingInfo::empty();
        }

        // Expected cycle basis size = E - V + C
        let n_comp = self.n_components();
        let cycle_rank = n_edges as isize - n_nodes as isize + n_comp as isize;
        if cycle_rank <= 0 {
            return TopologyRingInfo::empty();
        }
        let cycle_rank = cycle_rank as usize;

        // Generate candidate cycles (Horton-style)
        let mut candidates: Vec<Vec<usize>> = Vec::new();
        for edge in &self.edges {
            let (u, v) = (edge[0], edge[1]);
            let skip = if u < v { (u, v) } else { (v, u) };
            if let Some(path) = self.bfs_skip_edge(u, v, skip) {
                candidates.push(path);
            }
        }
        candidates.sort_by_key(|c| c.len());

        // Edge lookup for bit vector construction
        let mut edge_lookup: HashMap<(usize, usize), usize> = HashMap::new();
        for (ei, edge) in self.edges.iter().enumerate() {
            let (a, b) = (edge[0], edge[1]);
            let key = if a < b { (a, b) } else { (b, a) };
            edge_lookup.insert(key, ei);
        }

        // Select linearly independent cycles via GF(2) Gaussian elimination
        let words = n_edges.div_ceil(64);
        let mut basis: Vec<Vec<u64>> = Vec::new();
        let mut selected: Vec<Vec<usize>> = Vec::new();

        for cycle in &candidates {
            if selected.len() >= cycle_rank {
                break;
            }
            let mut bitvec = vec![0u64; words];
            let n = cycle.len();
            for i in 0..n {
                let a = cycle[i];
                let b = cycle[(i + 1) % n];
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&ei) = edge_lookup.get(&key) {
                    bitvec[ei / 64] |= 1u64 << (ei % 64);
                }
            }
            if gf2_independent(&mut basis, bitvec, words) {
                selected.push(cycle.clone());
            }
        }

        selected.sort_by_key(Vec::len);

        // Build reverse-lookup maps
        let mut atom_rings: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut bond_rings: HashMap<usize, Vec<usize>> = HashMap::new();

        for (ri, ring) in selected.iter().enumerate() {
            let n = ring.len();
            for i in 0..n {
                atom_rings.entry(ring[i]).or_default().push(ri);
                let a = ring[i];
                let b = ring[(i + 1) % n];
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&ei) = edge_lookup.get(&key) {
                    bond_rings.entry(ei).or_default().push(ri);
                }
            }
        }

        TopologyRingInfo {
            rings: selected,
            atom_rings,
            bond_rings,
        }
    }

    /// BFS from `start` to `goal`, skipping one specific edge identified by its
    /// endpoint pair `skip = (min, max)`.
    fn bfs_skip_edge(&self, start: usize, goal: usize, skip: (usize, usize)) -> Option<Vec<usize>> {
        let mut visited = vec![false; self.n];
        let mut parent: Vec<i64> = vec![-1; self.n];
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            if current == goal {
                let mut path = vec![goal];
                let mut node = goal;
                while node != start {
                    node = parent[node] as usize;
                    path.push(node);
                }
                path.reverse();
                return Some(path);
            }
            for &neighbor in &self.adj[current] {
                let key = if current < neighbor {
                    (current, neighbor)
                } else {
                    (neighbor, current)
                };
                if key == skip {
                    continue;
                }
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = current as i64;
                    queue.push_back(neighbor);
                }
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Atom operations
    // -----------------------------------------------------------------------

    /// Add a single atom.
    pub fn add_atom(&mut self) {
        self.adj.push(Vec::new());
        self.n += 1;
    }

    /// Add `n` atoms.
    pub fn add_atoms(&mut self, n: usize) {
        for _ in 0..n {
            self.add_atom();
        }
    }

    /// Delete an atom by index.
    ///
    /// Note: uses swap-remove semantics — the last node is moved into the
    /// removed node's slot, so indices of other nodes may change.
    pub fn delete_atom(&mut self, idx: usize) {
        if idx >= self.n {
            return;
        }
        let last = self.n - 1;

        // Remove edges incident to `idx` and drop `idx` from every adj list.
        self.edges.retain(|e| e[0] != idx && e[1] != idx);
        for list in &mut self.adj {
            list.retain(|&x| x != idx);
        }

        // Relabel the last node into `idx` (swap-remove semantics).
        if idx != last {
            for e in &mut self.edges {
                if e[0] == last {
                    e[0] = idx;
                }
                if e[1] == last {
                    e[1] = idx;
                }
            }
            for list in &mut self.adj {
                for x in list.iter_mut() {
                    if *x == last {
                        *x = idx;
                    }
                }
            }
            self.adj.swap(idx, last);
        }

        self.adj.pop();
        self.n -= 1;
    }

    // -----------------------------------------------------------------------
    // Bond operations
    // -----------------------------------------------------------------------

    /// Add a bond between atoms `i` and `j` if not already connected.
    pub fn add_bond(&mut self, i: usize, j: usize) {
        if !self.are_bonded(i, j) {
            self.edges.push([i, j]);
            self.adj[i].push(j);
            self.adj[j].push(i);
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
        let [a, b] = self.edges[idx];
        // Remove one instance of the bond from each endpoint's adjacency list.
        if let Some(pos) = self.adj[a].iter().position(|&x| x == b) {
            self.adj[a].remove(pos);
        }
        if let Some(pos) = self.adj[b].iter().position(|&x| x == a) {
            self.adj[b].remove(pos);
        }
        // Swap-remove the edge slot (matching the prior graph backend).
        self.edges.swap_remove(idx);
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
// TopologyRingInfo
// ---------------------------------------------------------------------------

/// Ring information from SSSR detection on a [`Topology`] graph.
///
/// Rings are stored as ordered lists of atom indices forming closed paths.
#[derive(Debug, Clone)]
pub struct TopologyRingInfo {
    rings: Vec<Vec<usize>>,
    atom_rings: HashMap<usize, Vec<usize>>,
    bond_rings: HashMap<usize, Vec<usize>>,
}

impl TopologyRingInfo {
    fn empty() -> Self {
        Self {
            rings: Vec::new(),
            atom_rings: HashMap::new(),
            bond_rings: HashMap::new(),
        }
    }

    /// Whether the atom belongs to any ring.
    pub fn is_atom_in_ring(&self, idx: usize) -> bool {
        self.atom_rings.get(&idx).is_some_and(|v| !v.is_empty())
    }

    /// Number of rings containing this atom.
    pub fn num_atom_rings(&self, idx: usize) -> usize {
        self.atom_rings.get(&idx).map_or(0, Vec::len)
    }

    /// Whether the bond belongs to any ring.
    pub fn is_bond_in_ring(&self, idx: usize) -> bool {
        self.bond_rings.get(&idx).is_some_and(|v| !v.is_empty())
    }

    /// Number of rings containing this bond.
    pub fn num_bond_rings(&self, idx: usize) -> usize {
        self.bond_rings.get(&idx).map_or(0, Vec::len)
    }

    /// Size (atom count) of every ring, sorted ascending.
    pub fn ring_sizes(&self) -> Vec<usize> {
        self.rings.iter().map(Vec::len).collect()
    }

    /// All rings of exactly `n` atoms.
    pub fn rings_of_size(&self, n: usize) -> Vec<&Vec<usize>> {
        self.rings.iter().filter(|r| r.len() == n).collect()
    }

    /// Total number of rings detected.
    pub fn num_rings(&self) -> usize {
        self.rings.len()
    }

    /// All rings as slices of atom indices.
    pub fn rings(&self) -> &[Vec<usize>] {
        &self.rings
    }

    /// Per-atom boolean mask: true if the atom is in any ring.
    pub fn atom_ring_mask(&self, n_atoms: usize) -> Vec<bool> {
        let mut mask = vec![false; n_atoms];
        for &idx in self.atom_rings.keys() {
            if idx < n_atoms {
                mask[idx] = true;
            }
        }
        mask
    }

    /// Per-bond boolean mask: true if the bond is in any ring.
    pub fn bond_ring_mask(&self, n_bonds: usize) -> Vec<bool> {
        let mut mask = vec![false; n_bonds];
        for &idx in self.bond_rings.keys() {
            if idx < n_bonds {
                mask[idx] = true;
            }
        }
        mask
    }
}

// ---------------------------------------------------------------------------
// GF(2) linear independence helper
// ---------------------------------------------------------------------------

/// Try to add `vec` to `basis` over GF(2). Returns true if independent.
fn gf2_independent(basis: &mut Vec<Vec<u64>>, mut vec: Vec<u64>, words: usize) -> bool {
    for basis_vec in basis.iter() {
        if let Some(lead) = leading_bit(basis_vec, words)
            && (vec[lead / 64] >> (lead % 64)) & 1 == 1
        {
            for w in 0..words {
                vec[w] ^= basis_vec[w];
            }
        }
    }
    let is_nonzero = vec.iter().any(|&w| w != 0);
    if is_nonzero {
        basis.push(vec);
    }
    is_nonzero
}

fn leading_bit(vec: &[u64], words: usize) -> Option<usize> {
    for w in (0..words).rev() {
        if vec[w] != 0 {
            return Some(w * 64 + (63 - vec[w].leading_zeros() as usize));
        }
    }
    None
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
    fn test_distances_path_and_disconnected() {
        // Path 0-1-2-3 plus an isolated atom 4.
        let topo = Topology::from_edges(5, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.distances(0), vec![0, 1, 2, 3, -1]);
        assert_eq!(topo.distances(3), vec![3, 2, 1, 0, -1]);
        // Isolated atom: only itself reachable.
        assert_eq!(topo.distances(4), vec![-1, -1, -1, -1, 0]);
        // Out-of-range source -> all unreachable.
        assert_eq!(topo.distances(9), vec![-1; 5]);
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

    #[test]
    fn test_neighbors() {
        let topo = Topology::from_edges(4, &[[0, 1], [1, 2], [2, 3]]);
        let mut n = topo.neighbors(1);
        n.sort();
        assert_eq!(n, vec![0, 2]);
    }

    #[test]
    fn test_degree() {
        let topo = Topology::from_edges(4, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.degree(0), 1);
        assert_eq!(topo.degree(1), 2);
    }

    #[test]
    fn test_are_bonded() {
        let topo = Topology::from_edges(3, &[[0, 1], [1, 2]]);
        assert!(topo.are_bonded(0, 1));
        assert!(!topo.are_bonded(0, 2));
    }

    #[test]
    fn test_connected_components_single() {
        let topo = Topology::from_edges(3, &[[0, 1], [1, 2]]);
        let cc = topo.connected_components();
        assert_eq!(cc.len(), 3);
        assert_eq!(cc[0], cc[1]);
        assert_eq!(cc[1], cc[2]);
        assert_eq!(topo.n_components(), 1);
    }

    #[test]
    fn test_connected_components_two() {
        let topo = Topology::from_edges(4, &[[0, 1], [2, 3]]);
        let cc = topo.connected_components();
        assert_eq!(cc[0], cc[1]);
        assert_eq!(cc[2], cc[3]);
        assert_ne!(cc[0], cc[2]);
        assert_eq!(topo.n_components(), 2);
    }

    #[test]
    fn test_connected_components_isolated() {
        let topo = Topology::with_atoms(3); // no bonds
        assert_eq!(topo.n_components(), 3);
        let cc = topo.connected_components();
        assert_ne!(cc[0], cc[1]);
        assert_ne!(cc[1], cc[2]);
    }

    #[test]
    fn test_find_rings_linear() {
        let topo = Topology::from_edges(4, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.find_rings().num_rings(), 0);
    }

    #[test]
    fn test_find_rings_single_6ring() {
        // Hexagon: 0-1-2-3-4-5-0
        let topo = Topology::from_edges(6, &[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]);
        let ri = topo.find_rings();
        assert_eq!(ri.num_rings(), 1);
        assert_eq!(ri.ring_sizes(), vec![6]);
        for i in 0..6 {
            assert!(ri.is_atom_in_ring(i));
        }
        for i in 0..topo.n_bonds() {
            assert!(ri.is_bond_in_ring(i));
            assert_eq!(ri.num_bond_rings(i), 1);
        }
        assert_eq!(
            ri.bond_ring_mask(topo.n_bonds()),
            vec![true; topo.n_bonds()]
        );
    }

    #[test]
    fn test_find_rings_naphthalene() {
        // Two fused 6-membered rings
        let topo = Topology::from_edges(
            10,
            &[
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 0], // ring A
                [2, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 3], // ring B
            ],
        );
        let ri = topo.find_rings();
        assert_eq!(ri.num_rings(), 2);
        let mut sizes = ri.ring_sizes();
        sizes.sort();
        assert_eq!(sizes, vec![6, 6]);
        assert!(ri.bond_ring_mask(topo.n_bonds()).into_iter().all(|x| x));
        assert_eq!(ri.num_bond_rings(2), 2);
    }

    #[test]
    fn test_find_rings_empty() {
        let topo = Topology::new();
        assert_eq!(topo.find_rings().num_rings(), 0);
    }

    // -----------------------------------------------------------------------
    // Edge-case parity tests (native adjacency rewrite)
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph_enumerations() {
        // Empty graph: no angles/dihedrals/impropers, no panic.
        let topo = Topology::new();
        assert!(topo.angles().is_empty());
        assert!(topo.dihedrals().is_empty());
        assert!(topo.impropers().is_empty());
        assert_eq!(topo.n_components(), 0);
        assert_eq!(topo.find_rings().num_rings(), 0);
        // with_atoms(0) is equivalent to new() for enumeration.
        let topo0 = Topology::with_atoms(0);
        assert!(topo0.angles().is_empty());
        assert!(topo0.dihedrals().is_empty());
        assert!(topo0.impropers().is_empty());
    }

    #[test]
    fn test_single_edge_graph() {
        // A single bond 0-1: one angle path is impossible (degree 1 each), no
        // dihedral, no improper; two connected atoms.
        let topo = Topology::from_edges(2, &[[0, 1]]);
        assert_eq!(topo.n_atoms(), 2);
        assert_eq!(topo.n_bonds(), 1);
        assert!(topo.angles().is_empty());
        assert!(topo.dihedrals().is_empty());
        assert!(topo.impropers().is_empty());
        assert_eq!(topo.n_components(), 1);
        assert_eq!(topo.distances(0), vec![0, 1]);
        assert_eq!(topo.distances(1), vec![1, 0]);
        assert_eq!(topo.find_rings().num_rings(), 0);
    }

    #[test]
    fn test_disconnected_distances_multiple_sources() {
        // Two components: {0-1-2} and {3-4}.
        let topo = Topology::from_edges(5, &[[0, 1], [1, 2], [3, 4]]);
        assert_eq!(topo.n_components(), 2);
        // Source in first component never reaches second.
        assert_eq!(topo.distances(0), vec![0, 1, 2, -1, -1]);
        assert_eq!(topo.distances(2), vec![2, 1, 0, -1, -1]);
        // Source in second component never reaches first.
        assert_eq!(topo.distances(3), vec![-1, -1, -1, 0, 1]);
        assert_eq!(topo.distances(4), vec![-1, -1, -1, 1, 0]);
    }

    #[test]
    fn test_delete_atom_swap_remove_relabels() {
        // 0-1-2-3 chain; deleting atom 1 swaps node 3 into slot 1.
        let mut topo = Topology::from_edges(4, &[[0, 1], [1, 2], [2, 3]]);
        topo.delete_atom(1);
        assert_eq!(topo.n_atoms(), 3);
        // Edge [0,1] and [1,2] are removed; edge [2,3] survives but relabeled
        // (old node 3 -> slot 1). So node 2 and new node 1 stay bonded.
        assert_eq!(topo.n_bonds(), 1);
        assert!(topo.are_bonded(2, 1));
    }

    #[test]
    fn test_delete_bond_swap_remove() {
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_bonds(), 3);
        topo.delete_bond(0); // remove edge [0,1]; [2,3] swaps into slot 0
        assert_eq!(topo.n_bonds(), 2);
        assert!(!topo.are_bonded(0, 1));
        assert!(topo.are_bonded(1, 2));
        assert!(topo.are_bonded(2, 3));
    }
}
