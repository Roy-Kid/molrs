// Union-Find traversal and per-bucket O(b²) compare loops read more
// clearly with explicit indexing than iterator combinators.
#![allow(clippy::needless_range_loop, clippy::if_same_then_else)]

//! Environment matching by sorted-bond-magnitude fingerprint.
//!
//! Mirrors `freud.environment.EnvironmentCluster` / `MatchEnv`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/MatchEnv.cc))
//! in its **no-rotation** mode: two particles' environments are
//! considered identical when their sorted bond-magnitude vectors agree
//! pair-wise within a tolerance. Particles are then clustered into
//! environment classes by union-find on the equivalence relation.
//!
//! This is the simplest useful form of freud's
//! `EnvironmentCluster` and matches its `registration=False`,
//! `global=False` defaults. A full rotation-invariant Kabsch / Hungarian
//! variant lives in the upstream `Registration.h`; porting that requires
//! a 3-D SVD that molrs doesn't yet expose, so it is a follow-up.
//!
//! # Algorithm
//!
//! 1. For each particle `i`, collect the lengths of all bonds `(i, j)`
//!    that appear in the supplied [`NeighborList`].
//! 2. Sort that vector — the **fingerprint** of particle `i`.
//! 3. Two particles match if their fingerprints are the same length and
//!    every pair of sorted entries differs by `≤ rmsd_threshold`.
//! 4. Cluster by union-find: each particle starts as its own component;
//!    every matched pair merges their components.
//!
//! Output: `cluster_idx[i]` is the environment class label for particle
//! `i`, and `n_clusters` counts the distinct classes.

use std::collections::HashMap;

use molrs::frame_access::FrameAccess;
use molrs::neighbors::NeighborList;
use molrs::types::F;

use crate::error::ComputeError;
use crate::result::ComputeResult;
use crate::traits::Compute;
use crate::util::get_positions_ref;

/// Per-frame environment-matching result.
#[derive(Debug, Clone, Default)]
pub struct MatchEnvResult {
    /// Particle → environment-class label (`0..n_clusters`).
    pub cluster_idx: Vec<u32>,
    /// Number of distinct environment classes.
    pub n_clusters: usize,
    /// Per-particle sorted-bond fingerprint (the raw feature).
    pub fingerprints: Vec<Vec<F>>,
}

impl ComputeResult for MatchEnvResult {}

/// `MatchEnv` analyzer (sorted-bond-magnitude mode, no rotation).
#[derive(Debug, Clone, Copy)]
pub struct MatchEnv {
    rmsd_threshold: F,
}

impl MatchEnv {
    pub fn new(rmsd_threshold: F) -> Result<Self, ComputeError> {
        if rmsd_threshold.is_nan() || rmsd_threshold < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "MatchEnv::rmsd_threshold",
                value: rmsd_threshold.to_string(),
            });
        }
        Ok(Self { rmsd_threshold })
    }

    pub fn rmsd_threshold(&self) -> F {
        self.rmsd_threshold
    }
}

/// Sorted-bond-magnitude fingerprint for one particle.
fn fingerprint(particle: usize, nlist: &NeighborList) -> Vec<F> {
    let mut bonds: Vec<F> = Vec::new();
    let i_idx = nlist.query_point_indices();
    let j_idx = nlist.point_indices();
    let dist_sq = nlist.dist_sq();
    let symmetric = matches!(nlist.mode(), molrs::neighbors::QueryMode::SelfQuery);
    for k in 0..nlist.n_pairs() {
        if i_idx[k] as usize == particle {
            bonds.push(dist_sq[k].sqrt());
        } else if symmetric && j_idx[k] as usize == particle {
            bonds.push(dist_sq[k].sqrt());
        }
    }
    bonds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bonds
}

/// Per-coordinate max difference between two sorted fingerprints, treating
/// mismatched length as `+∞` (no match possible).
fn rmsd(a: &[F], b: &[F]) -> F {
    if a.len() != b.len() {
        return F::INFINITY;
    }
    if a.is_empty() {
        return 0.0;
    }
    let mut s: F = 0.0;
    for k in 0..a.len() {
        let d = a[k] - b[k];
        s += d * d;
    }
    (s / a.len() as F).sqrt()
}

/// Single-pass disjoint-set union for environment clustering.
struct Dsu {
    parent: Vec<u32>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let next = self.parent[x as usize];
            self.parent[x as usize] = self.parent[next as usize];
            x = next;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
}

impl MatchEnv {
    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &NeighborList,
    ) -> Result<MatchEnvResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();

        let fingerprints: Vec<Vec<F>> = (0..n).map(|i| fingerprint(i, nlist)).collect();

        // Group particles by fingerprint length first — only same-length
        // pairs can match.
        let mut by_len: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, fp) in fingerprints.iter().enumerate() {
            by_len.entry(fp.len()).or_default().push(i);
        }

        let mut dsu = Dsu::new(n);
        for bucket in by_len.values() {
            // Pair-wise compare within each length bucket (O(b²) per bucket;
            // good enough for typical neighborhood sizes ≤ ~12 across
            // thousands of particles).
            for ai in 0..bucket.len() {
                let a = bucket[ai];
                for bi in (ai + 1)..bucket.len() {
                    let b = bucket[bi];
                    if rmsd(&fingerprints[a], &fingerprints[b]) <= self.rmsd_threshold {
                        dsu.union(a as u32, b as u32);
                    }
                }
            }
        }

        // Compact root labels into 0..n_clusters.
        let mut label_for: HashMap<u32, u32> = HashMap::new();
        let mut cluster_idx = vec![0_u32; n];
        for i in 0..n {
            let r = dsu.find(i as u32);
            let next = label_for.len() as u32;
            let lbl = *label_for.entry(r).or_insert(next);
            cluster_idx[i] = lbl;
        }
        let n_clusters = label_for.len();

        Ok(MatchEnvResult {
            cluster_idx,
            n_clusters,
            fingerprints,
        })
    }
}

impl Compute for MatchEnv {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<MatchEnvResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a Vec<NeighborList>,
    ) -> Result<Vec<MatchEnvResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if frames.len() != nlists.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: nlists.len(),
                what: "neighbor-list count",
            });
        }
        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::block::Block;
    use molrs::neighbors::{LinkCell, NbListAlgo};
    use molrs::region::simbox::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
        let xp = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(<F as molrs::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let yp = frame
            .get("atoms")
            .unwrap()
            .get("y")
            .and_then(<F as molrs::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let zp = frame
            .get("atoms")
            .unwrap()
            .get("z")
            .and_then(<F as molrs::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let n = xp.len();
        let mut pos = ndarray::Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = xp[i];
            pos[[i, 1]] = yp[i];
            pos[[i, 2]] = zp[i];
        }
        let simbox = frame.simbox.as_ref().unwrap();
        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build(pos.view(), simbox);
        lc.query().clone()
    }

    /// Two identical octahedra at different centres should be classed
    /// together; their satellite atoms (each with a single bond) form a
    /// second class.
    fn paired_octahedra() -> Frame {
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (10.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]); // centre (6 bonds of length 1)
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ] {
                p.push([cx + dx, cy + dy, cz + dz]); // satellite (1 bond)
            }
        }
        frame_with(&p, 20.0)
    }

    #[test]
    fn two_octahedra_centres_share_a_class() {
        let frame = paired_octahedra();
        let nl = build_nlist(&frame, 1.2);
        let r = &MatchEnv::new(1e-9)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        // Centres are particles 0 and 7. Their fingerprints are both
        // [1,1,1,1,1,1] → same class.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[7]);
        // Each satellite has fingerprint [1] → another (single) class
        // shared by all 12 satellites.
        for k in (1..=6).chain(8..=13) {
            assert_eq!(r.cluster_idx[k], r.cluster_idx[1]);
        }
        // Centres vs satellites must differ (different fingerprint length).
        assert_ne!(r.cluster_idx[0], r.cluster_idx[1]);
        assert_eq!(r.n_clusters, 2);
    }

    #[test]
    fn distinct_fingerprints_get_distinct_classes() {
        // Three particles forming a chain: 0 has 1 neighbour, 1 has 2, 2 has 1.
        let frame = frame_with(
            &[[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            10.0,
        );
        let nl = build_nlist(&frame, 1.5);
        let r = &MatchEnv::new(1e-9)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        // 0 and 2 each have one bond of length 1 → same class.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[2]);
        // 1 has two bonds → different class.
        assert_ne!(r.cluster_idx[1], r.cluster_idx[0]);
        assert_eq!(r.n_clusters, 2);
    }

    #[test]
    fn threshold_merges_near_matches() {
        // Two centres with slightly perturbed bond lengths: 1.00 vs 1.01.
        // Threshold 0.05 merges them; threshold 0.005 keeps them apart.
        let mut p: Vec<[F; 3]> = Vec::new();
        for (cx, off) in &[(5.0_f64, 1.00_f64), (10.0_f64, 1.01_f64)] {
            p.push([*cx, 5.0, 5.0]);
            for &(dx, dy, dz) in &[
                (*off, 0.0, 0.0),
                (-*off, 0.0, 0.0),
                (0.0, *off, 0.0),
                (0.0, -*off, 0.0),
            ] {
                p.push([*cx + dx, 5.0 + dy, 5.0 + dz]);
            }
        }
        let frame = frame_with(&p, 20.0);
        let nl = build_nlist(&frame, 1.5);
        let merged = &MatchEnv::new(0.05)
            .unwrap()
            .compute(&[&frame], &vec![nl.clone()])
            .unwrap()[0];
        assert_eq!(merged.cluster_idx[0], merged.cluster_idx[5]);

        let split = &MatchEnv::new(0.001)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        assert_ne!(split.cluster_idx[0], split.cluster_idx[5]);
    }

    #[test]
    fn invalid_threshold_errors() {
        assert!(MatchEnv::new(-1.0).is_err());
    }

    #[test]
    fn empty_frames_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = MatchEnv::new(0.1)
            .unwrap()
            .compute(&frames, &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
