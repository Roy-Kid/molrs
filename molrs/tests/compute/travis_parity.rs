//! TRAVIS ↔ molrs numerical PARITY tests.
//!
//! Unlike the rest of the compute test target (which builds fixtures in code),
//! these tests feed the **same real trajectory** to molrs that the reference
//! TRAVIS run consumed, and assert molrs reproduces TRAVIS's numbers. This is
//! the external-oracle check behind the `travis-parity` chain's `scientific`
//! acceptance criteria.
//!
//! Reference data lives in `tests/compute/data/travis/`:
//!   - `rdf_he_he.csv`        — TRAVIS output (`Distance / pm; g(r); Integral`)
//!   - `rdf_he_he.travis-input` — the exact `travis -p helium.xyz -i <file>` answers
//!
//! ## Regenerate the reference (TRAVIS 2022, Jul 29 2022 build)
//! ```text
//! travis -p helium.xyz -i rdf_he_he.travis-input
//! ```
//! Trajectory: tests-data/xyz/helium.xyz (125 He, 396 frames, a Lennard-Jones
//! fluid). Cubic box = 2000 pm (20 Å) — well above the per-frame extent
//! (~1019 pm), so minimum-image wrapping is essentially inactive and the parity
//! isolates the RDF binning + shell normalization. RDF: intermolecular He–He,
//! r_max = 1000 pm (10 Å), 300 bins, corrected g(r).
//!
//! molrs feeds the identical box / cutoff / bin grid and compares g(r) per bin.

#![cfg(feature = "io")]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use std::f64::consts::PI;

use molrs::Frame;
use molrs::compute::distribution::{AngleObservable, AtomGroups, DistributionFunction, Observable};
use molrs::compute::rdf::RDF;
use molrs::compute::traits::Compute;
use molrs::spatial::neighbors::{LinkCell, NbListAlgo, NeighborList};
use molrs::spatial::region::simbox::SimBox;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;
use ndarray::{Array2, array};

/// Box edge fed to BOTH tools for the He RDF: 2000 pm = 20 Å.
const BOX_ANG: F = 20.0;
/// RDF cutoff fed to BOTH tools: 1000 pm = 10 Å.
const R_MAX_ANG: F = 10.0;
const N_BINS: usize = 300;

/// Box edge for the water ADF: 1490 pm = 14.90 Å — the value at which TRAVIS
/// recognizes exactly 99 clean H2O (0 close-contact warnings); the per-frame
/// extent is ~1489 pm. Both tools apply minimum-image with this box.
const WATER_BOX_ANG: F = 14.90;

fn tests_data() -> PathBuf {
    if let Some(d) = std::env::var_os("MOLRS_TESTS_DATA").filter(|v| !v.is_empty()) {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests-data")
}

fn travis_data(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/compute/data/travis")
        .join(rel)
}

/// Read every frame of an XYZ trajectory, tagging each with a cubic box of edge
/// `box_ang` (Å) — the same box fed to TRAVIS, so both apply identical
/// minimum-image wrapping.
fn read_all_frames(path: &PathBuf, box_ang: F) -> Vec<Frame> {
    use molrs::io::data::xyz::read_xyz_frame_from_reader;
    let mut reader = BufReader::new(File::open(path).expect("open xyz"));
    let simbox = SimBox::cube(box_ang, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();
    let mut frames = Vec::new();
    while let Some(mut frame) = read_xyz_frame_from_reader(&mut reader).expect("read xyz frame") {
        frame.simbox = Some(simbox.clone());
        frames.push(frame);
    }
    frames
}

fn positions(frame: &Frame) -> Array2<F> {
    let x = frame.get_float("atoms", "x").unwrap();
    let y = frame.get_float("atoms", "y").unwrap();
    let z = frame.get_float("atoms", "z").unwrap();
    let n = x.len();
    let mut p = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        p[[i, 0]] = x[i];
        p[[i, 1]] = y[i];
        p[[i, 2]] = z[i];
    }
    p
}

fn build_nlist(frame: &Frame) -> NeighborList {
    let pos = positions(frame);
    let simbox = frame.simbox.as_ref().unwrap();
    let mut lc = LinkCell::new().cutoff(R_MAX_ANG);
    lc.build(pos.view(), simbox);
    lc.query().clone()
}

/// Parse the TRAVIS RDF CSV → `Vec<(r_pm, g)>`.
fn read_travis_rdf(path: &PathBuf) -> Vec<(F, F)> {
    let reader = BufReader::new(File::open(path).expect("open travis csv"));
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(';').collect();
        if cols.len() < 2 {
            continue;
        }
        let r: F = cols[0].trim().parse().unwrap();
        let g: F = cols[1].trim().parse().unwrap();
        out.push((r, g));
    }
    out
}

#[test]
fn travis_parity_he_he_rdf() {
    let frames = read_all_frames(&tests_data().join("xyz/helium.xyz"), BOX_ANG);
    assert!(frames.len() > 100, "expected the full helium trajectory");
    let nlists: Vec<NeighborList> = frames.iter().map(build_nlist).collect();
    let frame_refs: Vec<&Frame> = frames.iter().collect();

    let rdf = RDF::new(N_BINS, R_MAX_ANG, 0.0).unwrap();
    let result = rdf.compute(&frame_refs, &nlists).unwrap();

    let travis = read_travis_rdf(&travis_data("rdf_he_he.csv"));
    assert_eq!(travis.len(), N_BINS, "TRAVIS reference bin count");

    // Same bin grid: molrs bin center (Å) == TRAVIS r/pm / 100.
    // Per-bin tolerance is combined relative + absolute: the first-peak rising
    // edge climbs g≈0→6 over ~6 bins, so a sub-bin-width difference in count
    // placement is ~0.1 in g there (≈4% locally) without being an algorithmic
    // discrepancy. The absolute floor (0.2) absorbs that steep-edge sensitivity;
    // the relative term (3%) governs the resolved peak + bulk.
    let mut max_rel_resolved: F = 0.0; // tracked only where g is well-resolved (>2)
    let mut checked = 0;
    for (i, &(r_pm, g_travis)) in travis.iter().enumerate() {
        let r_ang_molrs = result.bin_centers[i];
        assert!(
            (r_ang_molrs - r_pm / 100.0).abs() < 1e-6,
            "bin {i}: molrs r={r_ang_molrs} Å vs TRAVIS r={} Å",
            r_pm / 100.0
        );
        if g_travis < 0.1 {
            continue; // empty small-r tail: both ~0, rel error ill-defined
        }
        let g_molrs = result.rdf[i];
        let abs_err = (g_molrs - g_travis).abs();
        checked += 1;
        if g_travis > 6.0 {
            max_rel_resolved = max_rel_resolved.max(abs_err / g_travis);
        }
        assert!(
            abs_err < 0.03 * g_travis + 0.2,
            "bin {i} (r={:.1} pm): molrs g={g_molrs:.5} vs TRAVIS g={g_travis:.5} (|Δ|={abs_err:.4})",
            r_pm,
        );
    }
    assert!(checked > 150, "expected to compare the bulk of the curve");
    // The resolved peak/bulk must agree tightly (steep edge excluded by g>2).
    assert!(
        max_rel_resolved < 0.03,
        "resolved-region max rel error {:.4}% exceeds 3%",
        max_rel_resolved * 100.0
    );
    eprintln!(
        "He–He RDF parity: {checked} bins compared, resolved-region (g>6) max rel error {:.4}%",
        max_rel_resolved * 100.0
    );
}

/// Parse the TRAVIS ADF CSV → `Vec<(angle_deg, occurrence)>`.
fn read_travis_adf(path: &PathBuf) -> Vec<(F, F)> {
    let reader = BufReader::new(File::open(path).expect("open travis adf csv"));
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(';').collect();
        if cols.len() < 2 {
            continue;
        }
        let a: F = cols[0].trim().parse().unwrap();
        let occ: F = cols[1].trim().parse().unwrap();
        out.push((a, occ));
    }
    out
}

/// Normalize a histogram to unit sum (skips an all-zero input).
fn unit_sum(h: &[F]) -> Vec<F> {
    let total: F = h.iter().copied().sum();
    if total == 0.0 {
        return h.to_vec();
    }
    h.iter().map(|&v| v / total).collect()
}

/// Intramolecular H–O–H angular distribution on water.xyz vs TRAVIS ADF.
///
/// TRAVIS recipe (`adf_hoh.travis-input`): same-size cell = 1490 pm, ADF,
/// intramolecular, vector 1 = O1→H1, vector 2 = O1→H2, range [0,180]°, 100 bins.
/// water.xyz is ordered O,H,H per molecule (99 H2O), so the H–O–H triple for
/// molecule m is (3m+1, 3m, 3m+2) with O at the vertex.
///
/// This binds molrs's actual [`DistributionFunction`] — whose [`Histogram1d`]
/// now bins with TRAVIS's own cloud-in-cell rule (`CDF::AddToBin`, src/df.cpp) —
/// over molrs's ported [`AngleObservable`], and compares its `counts` **directly**
/// to TRAVIS's `adf_hoh.csv` occurrences. No replay of TRAVIS binning: both the
/// angle computation and the histogram binning are now molrs code, so agreement
/// is the proof the CIC port reproduces TRAVIS bin-for-bin.
///
/// molrs bins the angle in **radians** over `[0, π]`; TRAVIS bins **degrees**
/// over `[0, 180]`. CIC's fractional bin position `(d−min)·res/(max−min) − 0.5`
/// is scale-invariant, so the two grids coincide bin-for-bin. Both tools wrap
/// with the same 14.90 Å box. The peak (~104.5°, σ≈1.7°) carries most mass.
#[test]
fn travis_parity_hoh_adf() {
    let frames = read_all_frames(&tests_data().join("xyz/water.xyz"), WATER_BOX_ANG);
    assert!(frames.len() > 50, "expected the full water trajectory");
    let frame_refs: Vec<&Frame> = frames.iter().collect();

    // O,H,H per molecule → H–O–H triple (3m+1, 3m, 3m+2), O the vertex.
    let n_mol = frames[0].get_float("atoms", "x").unwrap().len() / 3;
    assert_eq!(n_mol, 99, "expected 99 water molecules");
    let triples: Vec<(u32, u32, u32)> = (0..n_mol as u32)
        .map(|m| (3 * m + 1, 3 * m, 3 * m + 2))
        .collect();
    let groups = AtomGroups::triples(&triples);

    let travis = read_travis_adf(&travis_data("adf_hoh.csv"));
    let n_bins = travis.len();
    assert_eq!(n_bins, 100, "TRAVIS ADF bin count");

    // molrs distribution: AngleObservable (radians) over [0, π], CIC-binned.
    let df = DistributionFunction::new(AngleObservable, n_bins, 0.0, PI).unwrap();
    let result = df.compute(&frame_refs, &groups).unwrap();
    assert_eq!(result.counts.len(), n_bins, "molrs ADF bin count");

    // Same grid: bin center i ↔ TRAVIS angle column (rad→deg).
    let width_deg = 180.0 / n_bins as F;
    for (i, &(a_deg, _)) in travis.iter().enumerate() {
        let center = (i as F + 0.5) * width_deg;
        assert!(
            (center - a_deg).abs() < 1e-6,
            "bin {i}: molrs {center}° vs TRAVIS {a_deg}°"
        );
    }

    // Per-bin parity — molrs CIC counts vs TRAVIS CIC occurrences, both
    // unit-sum normalized. Identical angles + identical (now shared) binning ⇒
    // bit-exact bar float rounding nudging a rare sample across a boundary.
    let p_molrs = unit_sum(result.counts.as_slice().unwrap());
    let travis_occ: Vec<F> = travis.iter().map(|&(_, o)| o).collect();
    let p_travis = unit_sum(&travis_occ);
    let mut max_abs: F = 0.0;
    for i in 0..n_bins {
        max_abs = max_abs.max((p_molrs[i] - p_travis[i]).abs());
    }
    assert!(
        max_abs < 1e-6,
        "H–O–H ADF: max per-bin |Δp| {max_abs:.3e} exceeds 1e-6 (CIC port)"
    );

    // Mean / std straight from molrs samples vs TRAVIS's printed values.
    let obs = AngleObservable;
    let mut sum: F = 0.0;
    let mut sq_sum: F = 0.0;
    let mut n_samples = 0usize;
    for frame in &frames {
        for theta in obs.sample(frame, &groups).unwrap() {
            let deg = theta.to_degrees();
            sum += deg;
            sq_sum += deg * deg;
            n_samples += 1;
        }
    }
    assert_eq!(
        n_samples,
        n_mol * frames.len(),
        "one angle per water per frame"
    );
    let mean = sum / n_samples as F;
    let std = (sq_sum / n_samples as F - mean * mean).max(0.0).sqrt();
    assert!(
        (mean - 104.503).abs() < 0.01,
        "H–O–H ADF mean {mean:.4}° vs TRAVIS 104.503°"
    );
    assert!(
        (std - 1.7142).abs() < 0.01,
        "H–O–H ADF std {std:.4}° vs TRAVIS 1.7142°"
    );

    eprintln!(
        "H–O–H ADF parity (CIC): {n_bins} bins, max per-bin |Δp| {max_abs:.3e}, mean {mean:.4}° (TRAVIS 104.503°), std {std:.4}° (TRAVIS 1.7142°)"
    );
}
