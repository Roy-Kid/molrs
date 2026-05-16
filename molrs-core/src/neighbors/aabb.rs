// Tight 3-coord AABB loops read naturally with index-based access.
#![allow(clippy::needless_range_loop)]

//! Axis-Aligned Bounding-Box tree neighbor search.
//!
//! Mirrors `freud.locality.AABBQuery`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/locality/AABBQuery.cc)).
//!
//! Builds a balanced binary BVH over input points: each leaf wraps a
//! single point, each internal node owns the union AABB of its subtree.
//! A radial query descends the tree, pruning subtrees whose closest box
//! point is farther than the cutoff. Average query cost is
//! `O(log N + n_hits)`.
//!
//! # PBC handling
//!
//! When the [`SimBox`] declares any periodic axis, we expand the point
//! set with [`periodic_buffer`](super::periodic_buffer::periodic_buffer)
//! using the cutoff as the per-axis buffer distance, build the tree over
//! the extended set, then deduplicate (i, j_original) pairs to keep the
//! shortest minimum-image displacement. This matches `LinkCell`'s output
//! semantics on cross-image pairs.
//!
//! # Trade-offs vs `LinkCell`
//!
//! For uniform-density systems with bond-length cutoffs LinkCell is
//! faster (single hash lookup per cell visit). AABBQuery wins for
//! non-uniform systems and for ball queries where the cutoff varies per
//! query point — both are the cases freud's `AABBQuery` was designed
//! for.

use std::collections::HashMap;

use crate::neighbors::periodic_buffer::periodic_buffer;
use crate::neighbors::{NbListAlgo, NeighborList, QueryMode};
use crate::region::simbox::SimBox;
use crate::types::{F, FNx3, FNx3View};

#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: [F; 3],
    max: [F; 3],
}

impl Aabb {
    fn point(p: [F; 3]) -> Self {
        Self { min: p, max: p }
    }

    fn union(a: Aabb, b: Aabb) -> Self {
        Self {
            min: [
                a.min[0].min(b.min[0]),
                a.min[1].min(b.min[1]),
                a.min[2].min(b.min[2]),
            ],
            max: [
                a.max[0].max(b.max[0]),
                a.max[1].max(b.max[1]),
                a.max[2].max(b.max[2]),
            ],
        }
    }

    /// Squared distance from `p` to the closest point on the AABB
    /// (0 if `p` is inside).
    #[inline]
    fn dist_sq_to(&self, p: [F; 3]) -> F {
        let mut s: F = 0.0;
        for d in 0..3 {
            let v = if p[d] < self.min[d] {
                self.min[d] - p[d]
            } else if p[d] > self.max[d] {
                p[d] - self.max[d]
            } else {
                0.0
            };
            s += v * v;
        }
        s
    }
}

#[derive(Debug, Clone)]
enum Node {
    /// Leaf: holds the index into the point array.
    Leaf { point: u32, aabb: Aabb },
    /// Internal: indices into the `nodes` arena.
    Inner { left: u32, right: u32, aabb: Aabb },
}

impl Node {
    fn aabb(&self) -> Aabb {
        match *self {
            Node::Leaf { aabb, .. } => aabb,
            Node::Inner { aabb, .. } => aabb,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AabbTree {
    nodes: Vec<Node>,
    root: u32,
}

impl AabbTree {
    fn build(points: FNx3View<'_>) -> Self {
        let n = points.nrows();
        if n == 0 {
            return Self::default();
        }
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let mut tree = AabbTree {
            nodes: Vec::with_capacity(2 * n),
            root: 0,
        };
        tree.root = tree.build_recursive(&mut indices, points);
        tree
    }

    fn build_recursive(&mut self, idx: &mut [u32], points: FNx3View<'_>) -> u32 {
        if idx.len() == 1 {
            let p = [
                points[[idx[0] as usize, 0]],
                points[[idx[0] as usize, 1]],
                points[[idx[0] as usize, 2]],
            ];
            self.nodes.push(Node::Leaf {
                point: idx[0],
                aabb: Aabb::point(p),
            });
            return (self.nodes.len() - 1) as u32;
        }
        // Compute parent AABB and longest axis.
        let mut amin = [F::INFINITY; 3];
        let mut amax = [F::NEG_INFINITY; 3];
        for &i in idx.iter() {
            for d in 0..3 {
                let v = points[[i as usize, d]];
                if v < amin[d] {
                    amin[d] = v;
                }
                if v > amax[d] {
                    amax[d] = v;
                }
            }
        }
        let (mut ax, mut ext) = (0usize, amax[0] - amin[0]);
        for d in 1..3 {
            let e = amax[d] - amin[d];
            if e > ext {
                ax = d;
                ext = e;
            }
        }
        // Partition `idx` around the median along `ax`.
        idx.sort_unstable_by(|a, b| {
            points[[*a as usize, ax]]
                .partial_cmp(&points[[*b as usize, ax]])
                .unwrap()
        });
        let mid = idx.len() / 2;
        let (left_idx, right_idx) = idx.split_at_mut(mid);
        let left = self.build_recursive(left_idx, points);
        let right = self.build_recursive(right_idx, points);
        let aabb = Aabb::union(
            self.nodes[left as usize].aabb(),
            self.nodes[right as usize].aabb(),
        );
        self.nodes.push(Node::Inner { left, right, aabb });
        (self.nodes.len() - 1) as u32
    }

    /// Visit every leaf whose point is within `radius` of `p`.
    fn query(&self, p: [F; 3], radius_sq: F, mut hit: impl FnMut(u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(idx) = stack.pop() {
            match self.nodes[idx as usize] {
                Node::Leaf { point, aabb } => {
                    if aabb.dist_sq_to(p) <= radius_sq {
                        hit(point);
                    }
                }
                Node::Inner { left, right, aabb } => {
                    if aabb.dist_sq_to(p) <= radius_sq {
                        stack.push(left);
                        stack.push(right);
                    }
                }
            }
        }
    }
}

/// AABB-tree neighbor query.
#[derive(Debug, Clone)]
pub struct AabbQuery {
    cutoff: F,
    bx: Option<SimBox>,
    result: NeighborList,
    /// The actual points the tree was built on (may include ghosts).
    tree_points: FNx3,
    /// Map from `tree_points` row index back to the original point index.
    tree_to_orig: Vec<u32>,
    tree: AabbTree,
    /// Stored original positions (for `visit_pairs` on demand).
    stored_pos: FNx3,
}

impl AabbQuery {
    pub fn new(cutoff: F) -> Self {
        Self {
            cutoff,
            bx: None,
            result: NeighborList::empty(),
            tree_points: FNx3::zeros((0, 3)),
            tree_to_orig: Vec::new(),
            tree: AabbTree::default(),
            stored_pos: FNx3::zeros((0, 3)),
        }
    }

    pub fn cutoff(&self) -> F {
        self.cutoff
    }

    fn build_extended_set(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        let pbc = bx.pbc();
        let any_pbc = pbc.iter().any(|&p| p);
        if !any_pbc {
            self.tree_points = points.to_owned();
            self.tree_to_orig = (0..points.nrows() as u32).collect();
            return;
        }
        let buf = [self.cutoff; 3];
        let pb = periodic_buffer(points, bx, buf);
        self.tree_points = pb.positions;
        self.tree_to_orig = pb.indices;
    }

    fn compute_pairs(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.result.clear();
        self.build_extended_set(points, bx);
        self.tree = AabbTree::build(self.tree_points.view());

        let n = points.nrows();
        let cutoff_sq = self.cutoff * self.cutoff;

        // For self-query MIC dedup: for each (i, j_orig), keep shortest
        // displacement.
        let mut best: HashMap<(u32, u32), (F, [F; 3])> = HashMap::new();

        for i in 0..n {
            let pi = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            let tree_to_orig = &self.tree_to_orig;
            let tp = &self.tree_points;
            let mut hits: Vec<u32> = Vec::new();
            self.tree.query(pi, cutoff_sq, |idx| hits.push(idx));
            for tree_idx in hits {
                let j_orig = tree_to_orig[tree_idx as usize];
                if j_orig as usize == i {
                    // Skip the self-image (i, i)
                    if tp[[tree_idx as usize, 0]] == pi[0]
                        && tp[[tree_idx as usize, 1]] == pi[1]
                        && tp[[tree_idx as usize, 2]] == pi[2]
                    {
                        continue;
                    }
                }
                // i < j_orig for self-query half-shell.
                let i_u = i as u32;
                if i_u >= j_orig {
                    continue;
                }
                let dx = tp[[tree_idx as usize, 0]] - pi[0];
                let dy = tp[[tree_idx as usize, 1]] - pi[1];
                let dz = tp[[tree_idx as usize, 2]] - pi[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                let key = (i_u, j_orig);
                match best.get(&key) {
                    Some(&(prev_d2, _)) if prev_d2 <= d2 => {}
                    _ => {
                        best.insert(key, (d2, [dx, dy, dz]));
                    }
                }
            }
        }

        // Sort emitted pairs lexicographically for deterministic output.
        let mut keys: Vec<(u32, u32)> = best.keys().copied().collect();
        keys.sort_unstable();
        for (i, j) in keys {
            let (d2, dr) = best[&(i, j)];
            self.result.push(i, j, d2, dr);
        }
        self.result.mode = QueryMode::SelfQuery;
        self.result.num_points = n;
        self.result.num_query_points = n;
        self.bx = Some(bx.clone());
        self.stored_pos = points.to_owned();
    }
}

impl NbListAlgo for AabbQuery {
    fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.compute_pairs(points, bx);
    }

    fn update(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.build(points, bx);
    }

    fn query(&self) -> &NeighborList {
        &self.result
    }

    fn box_ref(&self) -> &SimBox {
        self.bx.as_ref().expect("box_ref called before build")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbors::{BruteForce, NbListAlgo};
    use ndarray::array;

    fn cube_bx(l: F, pbc: [bool; 3]) -> SimBox {
        SimBox::cube(l, array![0.0_f64, 0.0, 0.0], pbc).unwrap()
    }

    #[test]
    fn empty_input_is_empty_output() {
        let pts: FNx3 = ndarray::Array2::zeros((0, 3));
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        assert_eq!(aabb.query().n_pairs(), 0);
    }

    #[test]
    fn matches_brute_force_no_pbc() {
        let pts = array![
            [1.0_f64, 1.0, 1.0],
            [1.4, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [5.0, 5.0, 5.0],
            [5.4, 5.0, 5.0],
        ];
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let mut bf = BruteForce::new(1.0);
        bf.build(pts.view(), &bx);
        let mut a_pairs: Vec<(u32, u32)> = (0..aabb.query().n_pairs())
            .map(|k| {
                (
                    aabb.query().query_point_indices()[k],
                    aabb.query().point_indices()[k],
                )
            })
            .collect();
        let mut b_pairs: Vec<(u32, u32)> = (0..bf.query().n_pairs())
            .map(|k| {
                (
                    bf.query().query_point_indices()[k],
                    bf.query().point_indices()[k],
                )
            })
            .collect();
        a_pairs.sort_unstable();
        b_pairs.sort_unstable();
        assert_eq!(a_pairs, b_pairs);
    }

    #[test]
    fn matches_brute_force_with_pbc() {
        // Two atoms straddling the periodic boundary.
        let pts = array![[0.5_f64, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let bx = cube_bx(10.0, [true, true, true]);
        let mut aabb = AabbQuery::new(2.0);
        aabb.build(pts.view(), &bx);
        let mut bf = BruteForce::new(2.0);
        bf.build(pts.view(), &bx);
        assert_eq!(aabb.query().n_pairs(), bf.query().n_pairs());
        assert_eq!(aabb.query().n_pairs(), 1);
        // PBC distance is 1.0, not 9.0.
        let d2_a = aabb.query().dist_sq()[0];
        let d2_b = bf.query().dist_sq()[0];
        assert!((d2_a - 1.0).abs() < 1e-12);
        assert!((d2_a - d2_b).abs() < 1e-12);
    }

    #[test]
    fn larger_random_system_matches_brute_force() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(7);
        let n = 200;
        let mut pts = FNx3::zeros((n, 3));
        for i in 0..n {
            pts[[i, 0]] = rng.random::<F>() * 10.0;
            pts[[i, 1]] = rng.random::<F>() * 10.0;
            pts[[i, 2]] = rng.random::<F>() * 10.0;
        }
        let bx = cube_bx(10.0, [true, true, true]);
        let mut aabb = AabbQuery::new(1.5);
        aabb.build(pts.view(), &bx);
        let mut bf = BruteForce::new(1.5);
        bf.build(pts.view(), &bx);
        // Pair counts must agree.
        assert_eq!(aabb.query().n_pairs(), bf.query().n_pairs());
        // Sets of (i, j) must agree.
        let mut a: Vec<(u32, u32)> = (0..aabb.query().n_pairs())
            .map(|k| {
                (
                    aabb.query().query_point_indices()[k],
                    aabb.query().point_indices()[k],
                )
            })
            .collect();
        let mut b: Vec<(u32, u32)> = (0..bf.query().n_pairs())
            .map(|k| {
                (
                    bf.query().query_point_indices()[k],
                    bf.query().point_indices()[k],
                )
            })
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn dist_sq_to_aabb() {
        let a = Aabb {
            min: [0.0_f64, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        // Inside → 0.
        assert!(a.dist_sq_to([0.5, 0.5, 0.5]).abs() < 1e-12);
        // Distance from (2, 0.5, 0.5) → (2 - 1)² = 1.
        assert!((a.dist_sq_to([2.0, 0.5, 0.5]) - 1.0).abs() < 1e-12);
        // Diagonal from (2, 2, 2) → 3 · 1².
        assert!((a.dist_sq_to([2.0, 2.0, 2.0]) - 3.0).abs() < 1e-12);
    }
}
