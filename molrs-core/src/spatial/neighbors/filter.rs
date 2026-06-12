//! Post-processing filters that prune a [`NeighborList`] using
//! geometric criteria.
//!
//! Two filters are exposed, mirroring `freud.locality`:
//!
//! - [`filter_sann`]: van Meel et al.'s Solid-Angle Nearest-Neighbour
//!   construction
//!   ([source](https://github.com/glotzerlab/freud/blob/main/freud/locality/FilterSANN.cc)).
//!   For each query point we keep the smallest set of nearest neighbors
//!   whose sphere-cap solid angles sum to `4π`.
//! - [`filter_rad`]: Higham–Henchman Relative-Angular-Distance filter
//!   ([source](https://github.com/glotzerlab/freud/blob/main/freud/locality/FilterRAD.cc)).
//!   For each query point we walk neighbors in distance order and drop a
//!   neighbor `j` if there exists an already-accepted closer neighbor `k`
//!   such that the bond `r_ij` lies "behind" `r_ik` (i.e. the angle
//!   `∠(r̂_ik, r̂_ij)` is < some acceptance threshold). The default
//!   threshold is `arccos(½) = π/3 = 60°`, removing redundant second-
//!   shell artifacts that share a half-space with a closer first-shell
//!   bond.
//!
//! Both filters preserve the query-point ordering of pairs but may
//! shrink the list. The output [`NeighborList`] inherits the `mode`,
//! `num_points`, and `num_query_points` of the input.

use std::collections::HashMap;

use crate::spatial::neighbors::NeighborList;
use crate::types::F;

/// SANN filter: keep the smallest set of nearest neighbors whose solid
/// angles sum to `4π`.
///
/// Algorithm (van Meel et al., *J. Chem. Phys.* **136**, 234107 (2012)):
/// for each query point `i`, sort neighbors by distance `r_ij`, then find
/// the smallest integer `m ≥ 3` such that
///
/// ```text
///   Σ_{j=1..m} r_ij ≥ R_m · m
///   R_m = max(r_ij for j ≤ m)
/// ```
///
/// (equivalently `R_m = r_im` once sorted). The first `m` neighbors are
/// kept. If no such `m` exists (typically because the query point has
/// fewer than 3 neighbors), the entire neighbor set for that query point
/// is kept.
pub fn filter_sann(nlist: &NeighborList) -> NeighborList {
    // Group pair indices by query point.
    let mut by_query: HashMap<u32, Vec<usize>> = HashMap::new();
    for k in 0..nlist.n_pairs() {
        by_query
            .entry(nlist.query_point_indices()[k])
            .or_default()
            .push(k);
    }

    let mut out =
        NeighborList::with_mode(nlist.mode(), nlist.num_points(), nlist.num_query_points());
    let dist_sq = nlist.dist_sq();
    let vectors = nlist.vectors();
    let j_idx = nlist.point_indices();

    // For deterministic output, walk query indices in ascending order.
    let mut qs: Vec<u32> = by_query.keys().copied().collect();
    qs.sort_unstable();
    for q in qs {
        let mut pair_ks = by_query.remove(&q).unwrap();
        pair_ks.sort_unstable_by(|&a, &b| dist_sq[a].partial_cmp(&dist_sq[b]).unwrap());
        let m = sann_cutoff(&pair_ks, dist_sq);
        for &k in &pair_ks[..m] {
            out.push(
                q,
                j_idx[k],
                dist_sq[k],
                [vectors[[k, 0]], vectors[[k, 1]], vectors[[k, 2]]],
            );
        }
    }
    out
}

/// Find the smallest `m ≥ 3` satisfying the SANN inequality. If no `m`
/// exists, returns the full length so the caller keeps everything.
fn sann_cutoff(pair_ks_sorted: &[usize], dist_sq: &[F]) -> usize {
    let n = pair_ks_sorted.len();
    if n < 3 {
        return n;
    }
    let mut sum_r: F = 0.0;
    for k in 0..n {
        sum_r += dist_sq[pair_ks_sorted[k]].sqrt();
        if k + 1 >= 3 {
            let r_m = dist_sq[pair_ks_sorted[k]].sqrt();
            // sum_r ≥ r_m · (k+1)  →  cap = k+1
            if sum_r >= r_m * (k + 1) as F {
                return k + 1;
            }
        }
    }
    n
}

/// RAD filter: drop a neighbor `j` whenever there is a strictly closer
/// accepted neighbor `k` such that the bond `r_ij` lies inside the cone
/// of half-angle `acceptance` around `r_ik`.
///
/// `acceptance` is in **radians**. The default (`std::f64::consts::FRAC_PI_3`,
/// `60°`) reproduces the Higham–Henchman tuned criterion.
pub fn filter_rad(nlist: &NeighborList, acceptance: F) -> NeighborList {
    let cos_thresh = acceptance.cos();
    let mut by_query: HashMap<u32, Vec<usize>> = HashMap::new();
    for k in 0..nlist.n_pairs() {
        by_query
            .entry(nlist.query_point_indices()[k])
            .or_default()
            .push(k);
    }

    let mut out =
        NeighborList::with_mode(nlist.mode(), nlist.num_points(), nlist.num_query_points());
    let dist_sq = nlist.dist_sq();
    let vectors = nlist.vectors();
    let j_idx = nlist.point_indices();

    let mut qs: Vec<u32> = by_query.keys().copied().collect();
    qs.sort_unstable();
    for q in qs {
        let mut pair_ks = by_query.remove(&q).unwrap();
        pair_ks.sort_unstable_by(|&a, &b| dist_sq[a].partial_cmp(&dist_sq[b]).unwrap());

        // Walk neighbors in increasing distance and accept those that are
        // not "shadowed" by an already-accepted closer one.
        let mut accepted: Vec<[F; 3]> = Vec::new();
        for &k in &pair_ks {
            let r = dist_sq[k].sqrt();
            if r == 0.0 {
                continue;
            }
            let nx = vectors[[k, 0]] / r;
            let ny = vectors[[k, 1]] / r;
            let nz = vectors[[k, 2]] / r;
            let mut shadowed = false;
            for a in &accepted {
                let dot = a[0] * nx + a[1] * ny + a[2] * nz;
                if dot > cos_thresh {
                    shadowed = true;
                    break;
                }
            }
            if !shadowed {
                accepted.push([nx, ny, nz]);
                out.push(
                    q,
                    j_idx[k],
                    dist_sq[k],
                    [vectors[[k, 0]], vectors[[k, 1]], vectors[[k, 2]]],
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::neighbors::QueryMode;

    fn make_nlist(pairs: &[(u32, u32, F, [F; 3])], n_points: usize) -> NeighborList {
        let mut nl = NeighborList::with_mode(QueryMode::CrossQuery, n_points, n_points);
        for &(i, j, d2, dr) in pairs {
            nl.push(i, j, d2, dr);
        }
        nl
    }

    // ---- SANN -------------------------------------------------------------

    #[test]
    fn sann_keeps_minimum_three_when_present() {
        // 4 neighbors at distances 1, 1, 1, 2. The SANN sum:
        //   k=0: r=1, sum=1, 1 ≥ 1·1 = 1 → m = 1, but require m ≥ 3
        //   k=1: r=1, sum=2, 2 ≥ 1·2 → m = 2, still need ≥ 3
        //   k=2: r=1, sum=3, 3 ≥ 1·3 → m = 3 (first ≥ 3 success)
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
            (0, 3, 1.0, [0.0, 0.0, 1.0]),
            (0, 4, 4.0, [2.0, 0.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 5);
        let f = filter_sann(&nl);
        assert_eq!(f.n_pairs(), 3);
    }

    #[test]
    fn sann_keeps_all_when_below_three() {
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 3);
        let f = filter_sann(&nl);
        assert_eq!(f.n_pairs(), 2);
    }

    #[test]
    fn sann_preserves_query_split() {
        // Two query points, each with 4 NN: the filter independently caps
        // each at 3.
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
            (0, 3, 1.0, [0.0, 0.0, 1.0]),
            (0, 4, 4.0, [2.0, 0.0, 0.0]),
            (1, 0, 1.0, [-1.0, 0.0, 0.0]),
            (1, 5, 1.0, [0.0, 1.0, 0.0]),
            (1, 6, 1.0, [0.0, 0.0, 1.0]),
            (1, 7, 4.0, [2.0, 0.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 8);
        let f = filter_sann(&nl);
        let q0_count = (0..f.n_pairs())
            .filter(|&k| f.query_point_indices()[k] == 0)
            .count();
        let q1_count = (0..f.n_pairs())
            .filter(|&k| f.query_point_indices()[k] == 1)
            .count();
        assert_eq!(q0_count, 3);
        assert_eq!(q1_count, 3);
    }

    // ---- RAD --------------------------------------------------------------

    #[test]
    fn rad_keeps_orthogonal_neighbors() {
        // Three neighbors along +x, +y, +z (pairwise angles 90°, so cosines
        // are 0 < cos(60°) = 0.5). All three should be retained.
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
            (0, 3, 1.0, [0.0, 0.0, 1.0]),
        ];
        let nl = make_nlist(&pairs, 4);
        let f = filter_rad(&nl, std::f64::consts::FRAC_PI_3);
        assert_eq!(f.n_pairs(), 3);
    }

    #[test]
    fn rad_drops_a_second_neighbor_in_the_same_cone() {
        // Two neighbors along +x; the second is at twice the distance and
        // lies inside the 60° cone of the first → dropped.
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 4.0, [2.0_f64, 0.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 3);
        let f = filter_rad(&nl, std::f64::consts::FRAC_PI_3);
        assert_eq!(f.n_pairs(), 1);
        assert_eq!(f.point_indices()[0], 1);
    }

    #[test]
    fn rad_threshold_change_alters_acceptance() {
        // Two neighbors 30° apart along +x and a tilted +x direction.
        // With acceptance=60°, the second falls inside the cone (cos 30° ≈
        // 0.866 > cos 60° = 0.5) → dropped. With acceptance=10° (cos ≈
        // 0.985), it survives.
        let c = (30.0_f64).to_radians().cos();
        let s = (30.0_f64).to_radians().sin();
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0_f64, [c, s, 0.0_f64]),
        ];
        let nl = make_nlist(&pairs, 3);
        let f60 = filter_rad(&nl, std::f64::consts::FRAC_PI_3);
        let f10 = filter_rad(&nl, 10.0_f64.to_radians());
        assert_eq!(f60.n_pairs(), 1);
        assert_eq!(f10.n_pairs(), 2);
    }

    #[test]
    fn empty_input_returns_empty() {
        let nl = NeighborList::with_mode(QueryMode::SelfQuery, 0, 0);
        let f = filter_sann(&nl);
        assert_eq!(f.n_pairs(), 0);
        let g = filter_rad(&nl, std::f64::consts::FRAC_PI_3);
        assert_eq!(g.n_pairs(), 0);
    }
}
