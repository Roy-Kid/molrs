//! O(N²) brute-force neighbor search — reference implementation.
//!
//! Checks every pair `(i, j)` with `i < j`. Useful for correctness testing
//! against the cell-list algorithm and for very small systems where the
//! overhead of cell construction is not worthwhile.

use crate::core::region::simbox::SimBox;
use crate::core::types::{F, FNx3, FNx3View};
use crate::neighbors::{NbListAlgo, NeighborResult, PairVisitor};

/// Brute-force O(N²) neighbor search.
///
/// Iterates over all unique pairs and keeps those within the cutoff.
/// Results are cached after [`build`](NbListAlgo::build) /
/// [`update`](NbListAlgo::update) so that [`query`](NbListAlgo::query)
/// returns a cheap reference.
#[derive(Debug, Clone)]
pub struct BruteForce {
    /// Interaction cutoff distance.
    pub cutoff: F,
    /// Simulation box from the last build/update.
    bx: Option<SimBox>,
    /// Cached pair results.
    result: NeighborResult,
    /// Stored positions for visit_pairs (set by update_index).
    stored_pos: FNx3,
}

impl BruteForce {
    /// Create a new `BruteForce` with the given cutoff distance.
    pub fn new(cutoff: F) -> Self {
        Self {
            cutoff,
            bx: None,
            result: NeighborResult::empty(),
            stored_pos: FNx3::zeros((0, 3)),
        }
    }

    /// Scan all pairs and store those within cutoff.
    fn compute(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        let n = points.nrows();
        let cutoff2 = self.cutoff * self.cutoff;

        self.result.clear();

        for i in 0..n {
            let pi = points.row(i);
            for j in (i + 1)..n {
                let pj = points.row(j);
                let dr = bx.shortest_vector_fast(pi, pj);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= cutoff2 {
                    self.result
                        .push(i as u32, j as u32, d2, [dr[0], dr[1], dr[2]]);
                }
            }
        }

        self.bx = Some(bx.clone());
    }
}

impl NbListAlgo for BruteForce {
    fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.compute(points, bx);
    }

    fn update(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.build(points, bx);
    }

    fn query(&self) -> &NeighborResult {
        &self.result
    }

    fn box_ref(&self) -> &SimBox {
        self.bx.as_ref().expect("box_ref called before build")
    }

    /// Store positions and box without computing pairs.
    fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.update_index(points, bx);
    }

    /// Store positions and box without computing pairs.
    fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.stored_pos = points.to_owned();
        self.bx = Some(bx.clone());
    }

    /// On-demand O(N²) pair traversal using stored positions.
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let bx = match &self.bx {
            Some(b) => b,
            None => return,
        };
        let n = self.stored_pos.nrows();
        if n == 0 {
            // Fall back to pre-stored result if no stored_pos
            if self.result.n_pairs() > 0 {
                let diff = self.result.diff();
                for k in 0..self.result.n_pairs() {
                    visitor.visit_pair(
                        self.result.idx_i()[k],
                        self.result.idx_j()[k],
                        self.result.dist_sq()[k],
                        [diff[[k, 0]], diff[[k, 1]], diff[[k, 2]]],
                    );
                }
            }
            return;
        }

        let cutoff2 = self.cutoff * self.cutoff;
        for i in 0..n {
            let pi = self.stored_pos.row(i);
            for j in (i + 1)..n {
                let pj = self.stored_pos.row(j);
                let dr = bx.shortest_vector_fast(pi, pj);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= cutoff2 {
                    visitor.visit_pair(i as u32, j as u32, d2, [dr[0], dr[1], dr[2]]);
                }
            }
        }
    }
}
