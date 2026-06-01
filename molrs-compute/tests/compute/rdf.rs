//! End-to-end RDF integration tests against known analytical results.
//!
//! Inputs are simple cubic lattices built in code. A simple-cubic lattice with
//! spacing `a` has sharp coordination shells: 6 neighbours at `a`, 12 at
//! `a·√2`, 8 at `a·√3`, … g(r) must therefore be a comb of peaks separated by
//! empty bins — a far stronger statement than the ideal-gas plateau covered by
//! the inline unit tests.

use molrs::Frame;
use molrs::block::Block;
use molrs::neighbors::{LinkCell, NbListAlgo, NeighborList};
use molrs::region::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array1, Array2, array};

use molrs_compute::rdf::RDF;
use molrs_compute::traits::Compute;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Simple-cubic lattice of `reps^3` points with spacing `a`, in a periodic box
/// of edge `reps * a` (so the lattice tiles the box seamlessly).
fn simple_cubic(reps: usize, a: F) -> Frame {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    for i in 0..reps {
        for j in 0..reps {
            for k in 0..reps {
                xs.push(i as F * a);
                ys.push(j as F * a);
                zs.push(k as F * a);
            }
        }
    }
    let mut block = Block::new();
    block.insert("x", Array1::from_vec(xs).into_dyn()).unwrap();
    block.insert("y", Array1::from_vec(ys).into_dyn()).unwrap();
    block.insert("z", Array1::from_vec(zs).into_dyn()).unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", block);
    let box_len = reps as F * a;
    frame.simbox =
        Some(SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());
    frame
}

fn positions(frame: &Frame) -> Array2<F> {
    let atoms = frame.get("atoms").unwrap();
    let xs = atoms.get_float("x").unwrap();
    let ys = atoms.get_float("y").unwrap();
    let zs = atoms.get_float("z").unwrap();
    let n = xs.len();
    let mut pos = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        pos[[i, 0]] = xs[[i]];
        pos[[i, 1]] = ys[[i]];
        pos[[i, 2]] = zs[[i]];
    }
    pos
}

fn build_nlist(frame: &Frame, r_max: F) -> NeighborList {
    let pos = positions(frame);
    let simbox = frame.simbox.as_ref().unwrap();
    let mut lc = LinkCell::new().cutoff(r_max);
    lc.build(pos.view(), simbox);
    lc.query().clone()
}

/// Largest g(r) over all bins whose center lies within `tol` of `r`. Robust to
/// a lattice distance landing exactly on a bin edge (counts split into the
/// adjacent bin).
fn peak_near(result: &molrs_compute::rdf::RDFResult, r: F, tol: F) -> F {
    result
        .bin_centers
        .iter()
        .zip(result.rdf.iter())
        .filter(|(c, _)| (**c - r).abs() <= tol)
        .map(|(_, g)| *g)
        .fold(0.0, F::max)
}

/// Total raw pair count over all bins whose center lies within `tol` of `r`.
fn counts_near(result: &molrs_compute::rdf::RDFResult, r: F, tol: F) -> F {
    result
        .bin_centers
        .iter()
        .zip(result.n_r.iter())
        .filter(|(c, _)| (**c - r).abs() <= tol)
        .map(|(_, n)| *n)
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn simple_cubic_rdf_peaks_at_lattice_shells() {
    let a: F = 2.0;
    let reps = 6; // 216 atoms, box edge 12 Å.
    let frame = simple_cubic(reps, a);

    // Cover the first two shells: a (=2) and a√2 (≈2.83). Stop before a√3.
    let r_max: F = 3.0;
    let n_bins = 60; // bin width 0.05 Å — narrow enough to resolve shells.
    let nlist = build_nlist(&frame, r_max);

    let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();
    let result = rdf.compute(&[&frame], &vec![nlist]).unwrap();

    // First shell at r = a (6 neighbours), second at a√2 (12 neighbours).
    // Use a one-bin-wide tolerance so a distance landing on a bin edge is
    // still captured.
    let tol = result.bin_edges[1] - result.bin_edges[0];
    let peak1 = peak_near(&result, a, tol);
    let peak2 = peak_near(&result, a * 2.0_f64.sqrt(), tol);

    // Peaks are strongly > 1 (a delta-like crystal lattice, not a gas).
    assert!(peak1 > 5.0, "first shell g(r≈{a:.2}) peak = {peak1:.3}");
    assert!(
        peak2 > 5.0,
        "second shell g(r≈{:.2}) peak = {peak2:.3}",
        a * 2.0_f64.sqrt()
    );

    // The gap between the two shells (r ≈ 2.4 Å) must be empty for a perfect
    // lattice — no pair distance lands there.
    let gap_counts = counts_near(&result, 2.4, tol);
    assert!(
        gap_counts.abs() < 1e-9,
        "gap region r≈2.4 should hold no pairs, got n_r sum = {gap_counts}"
    );
}

#[test]
fn simple_cubic_first_shell_coordination_is_six() {
    // The integral of g(r)·ρ·4πr² dr over the first shell must equal the
    // coordination number. For a simple-cubic lattice that's exactly 6.
    // We check it directly from raw counts: 2·(pairs in first shell) / N = z.
    let a: F = 2.0;
    let reps = 6;
    let n_atoms = reps * reps * reps;
    let frame = simple_cubic(reps, a);

    // Window the histogram tightly around the first shell only.
    let r_max: F = a * 1.2; // below a√2 ≈ 2.83, so only the a-shell contributes.
    let n_bins = 24;
    let nlist = build_nlist(&frame, r_max);

    let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();
    let result = rdf.compute(&[&frame], &vec![nlist]).unwrap();

    // Self-query neighbor lists count each pair once; coordination = 2·pairs/N.
    let total_pairs: F = result.n_r.sum();
    let coordination = 2.0 * total_pairs / n_atoms as F;
    assert!(
        (coordination - 6.0).abs() < 1e-6,
        "simple-cubic first-shell coordination = {coordination}, expected 6"
    );
}

#[test]
fn empty_frames_is_error() {
    let rdf = RDF::new(10, 4.0, 0.0).unwrap();
    let frames: Vec<&Frame> = Vec::new();
    let nlists: Vec<NeighborList> = Vec::new();
    let err = rdf.compute(&frames, &nlists).unwrap_err();
    assert!(matches!(
        err,
        molrs_compute::error::ComputeError::EmptyInput
    ));
}
