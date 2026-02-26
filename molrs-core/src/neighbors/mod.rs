//! Neighbor-list algorithms for pairwise distance queries.
//!
//! This module provides a trait-based framework for neighbor search. Two
//! algorithms are available:
//!
//! - [`LinkCell`] — O(N) cell-list algorithm, parallelized with rayon when the
//!   `rayon` feature is enabled (default). Recommended for production use.
//! - [`BruteForce`] — O(N²) all-pairs reference implementation, useful for
//!   correctness testing and small systems.
//!
//! Both implement [`NbListAlgo`] and can be wrapped in [`NeighborList`] for a
//! uniform API.
//!
//! # Example
//!
//! ```ignore
//! let mut nl = NeighborList(LinkCell::new().cutoff(3.0));
//! nl.build(points.view(), &simbox);
//! let result = nl.query();
//! ```

use crate::core::region::simbox::SimBox;
use crate::core::types::{F, FNx3View};
use ndarray::ArrayView2;

pub mod bruteforce;
mod linkcell;

pub use bruteforce::BruteForce;
pub use linkcell::LinkCell;

// ---------------------------------------------------------------------------
// PairVisitor -- zero-allocation callback for on-demand pair traversal
// ---------------------------------------------------------------------------

/// Callback for on-demand pair traversal (zero-allocation alternative to
/// [`NeighborResult`]).
///
/// Implement this trait to process pairs without storing them. Used by
/// [`NbListAlgo::visit_pairs`].
pub trait PairVisitor {
    /// Called for each pair `(i, j)` within the cutoff.
    ///
    /// * `i`, `j` — original particle indices
    /// * `dist_sq` — squared distance between particles
    /// * `diff` — displacement vector `r_j - r_i` (minimum-image)
    fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]);
}

/// Blanket impl: any `FnMut(u32, u32, F, [F; 3])` is a `PairVisitor`.
impl<T: FnMut(u32, u32, F, [F; 3])> PairVisitor for T {
    #[inline(always)]
    fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]) {
        self(i, j, dist_sq, diff)
    }
}

// ---------------------------------------------------------------------------
// NbListAlgo trait
// ---------------------------------------------------------------------------

/// Trait implemented by neighbor-list algorithms.
///
/// Each algorithm maintains internal state and caches pair results after
/// [`build`](NbListAlgo::build) or [`update`](NbListAlgo::update), so that
/// [`query`](NbListAlgo::query) returns a cheap reference.
pub trait NbListAlgo {
    /// Build the neighbor list from scratch.
    ///
    /// # Panics
    /// Panics if the cutoff is not set or `points` has wrong shape.
    fn build(&mut self, points: FNx3View<'_>, bx: &SimBox);

    /// Rebuild the neighbor list with new positions.
    ///
    /// Implementations may reuse internal buffers for efficiency.
    fn update(&mut self, points: FNx3View<'_>, bx: &SimBox);

    /// Return a reference to the cached pair results.
    fn query(&self) -> &NeighborResult;

    /// Return a reference to the simulation box used in the last build/update.
    fn box_ref(&self) -> &SimBox;

    /// Build the spatial index only — NO pair pre-computation.
    ///
    /// After calling this, [`visit_pairs`](NbListAlgo::visit_pairs) can
    /// traverse pairs on-demand without allocating a [`NeighborResult`].
    /// The default falls back to a full [`build`](NbListAlgo::build).
    fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.build(points, bx);
    }

    /// Rebuild the spatial index only — NO pair pre-computation.
    ///
    /// Equivalent to [`update`](NbListAlgo::update) but skips pair enumeration.
    /// The default falls back to a full [`update`](NbListAlgo::update).
    fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.update(points, bx);
    }

    /// Traverse pairs on-demand, calling the visitor for each pair within
    /// the cutoff. Zero allocation — no [`NeighborResult`] is built.
    ///
    /// The default iterates the pre-stored pairs from [`query`](NbListAlgo::query).
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let result = self.query();
        let diff = result.diff();
        for k in 0..result.n_pairs() {
            visitor.visit_pair(
                result.idx_i()[k],
                result.idx_j()[k],
                result.dist_sq()[k],
                [diff[[k, 0]], diff[[k, 1]], diff[[k, 2]]],
            );
        }
    }
}

/// Algorithm-generic neighbor list wrapper.
///
/// Thin facade that delegates every call to the underlying [`NbListAlgo`]
/// implementation. Construct via `NeighborList(algo)`.
#[derive(Debug, Clone)]
pub struct NeighborList<A: NbListAlgo>(pub A);

impl<A: NbListAlgo> NeighborList<A> {
    /// Build the neighbor list from scratch. See [`NbListAlgo::build`].
    pub fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.build(points, bx)
    }

    /// Rebuild with new positions. See [`NbListAlgo::update`].
    pub fn update(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.update(points, bx)
    }

    /// Return cached pair results. See [`NbListAlgo::query`].
    pub fn query(&self) -> &NeighborResult {
        self.0.query()
    }

    /// Return the simulation box. See [`NbListAlgo::box_ref`].
    pub fn box_ref(&self) -> &SimBox {
        self.0.box_ref()
    }

    /// Build spatial index only (no pair pre-computation). See [`NbListAlgo::build_index`].
    pub fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.build_index(points, bx)
    }

    /// Rebuild spatial index only. See [`NbListAlgo::update_index`].
    pub fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.update_index(points, bx)
    }

    /// Traverse pairs on-demand. See [`NbListAlgo::visit_pairs`].
    pub fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        self.0.visit_pairs(visitor)
    }
}

/// Cached neighbor query results.
///
/// Stores all pairs `(i, j)` with `i < j` whose squared distance is within the
/// cutoff, together with the squared distance and the displacement vector
/// `r_j - r_i` (minimum-image).
///
/// Fields are `pub(crate)` for internal mutation by algorithm implementations;
/// consumers access data through the public accessor methods.
#[derive(Debug, Clone)]
pub struct NeighborResult {
    /// First particle index per pair.
    pub(crate) idx_i: Vec<u32>,
    /// Second particle index per pair (always > idx_i for same-cell pairs).
    pub(crate) idx_j: Vec<u32>,
    /// Squared distances, one per pair.
    pub(crate) dist_sq: Vec<F>,
    /// Flat displacement vectors `[dx0,dy0,dz0, dx1,dy1,dz1, ...]`, stride-3.
    pub(crate) diff_flat: Vec<F>,
}

impl NeighborResult {
    /// Create an empty result with no pairs.
    pub(crate) fn empty() -> Self {
        Self {
            idx_i: Vec::new(),
            idx_j: Vec::new(),
            dist_sq: Vec::new(),
            diff_flat: Vec::new(),
        }
    }

    /// Clear all buffers for reuse.
    pub(crate) fn clear(&mut self) {
        self.idx_i.clear();
        self.idx_j.clear();
        self.dist_sq.clear();
        self.diff_flat.clear();
    }

    /// Push a neighbor pair into the result.
    #[inline(always)]
    pub(crate) fn push(&mut self, i: u32, j: u32, d2: F, dr: [F; 3]) {
        self.idx_i.push(i);
        self.idx_j.push(j);
        self.dist_sq.push(d2);
        self.diff_flat.extend(dr);
    }

    /// Number of neighbor pairs.
    #[inline]
    pub fn n_pairs(&self) -> usize {
        self.idx_i.len()
    }

    /// First particle indices, one per pair.
    #[inline]
    pub fn idx_i(&self) -> &[u32] {
        &self.idx_i
    }

    /// Second particle indices, one per pair.
    #[inline]
    pub fn idx_j(&self) -> &[u32] {
        &self.idx_j
    }

    /// Squared distances, one per pair.
    #[inline]
    pub fn dist_sq(&self) -> &[F] {
        &self.dist_sq
    }

    /// Displacement vectors as an N×3 array view.
    #[inline]
    pub fn diff(&self) -> FNx3View<'_> {
        ArrayView2::from_shape((self.n_pairs(), 3), &self.diff_flat)
            .expect("diff_flat shape mismatch")
    }
}
