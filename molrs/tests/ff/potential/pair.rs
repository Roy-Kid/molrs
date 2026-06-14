//! Lennard-Jones 12-6 pair potential: E = 4*eps*((sigma/r)^12 - (sigma/r)^6).

use crate::helpers::{numerical_forces, pairs_block, typed_atoms_block};
use molrs::ff::ForceField;
use molrs::ff::potential::Potential;
use molrs::ff::potential::extract_coords;
use molrs::ff::potential::pair::lj_cut::PairLJCut;
use molrs::store::frame::Frame;
use molrs::types::F;

const EPS: F = 1.0;
const SIGMA: F = 1.0;

fn single_pair() -> PairLJCut {
    PairLJCut::new(vec![0], vec![1], vec![EPS], vec![SIGMA])
}

#[test]
fn energy_matches_closed_form() {
    let pot = single_pair();
    // r = 2.0, sigma = 1.0 -> 4*((1/2)^12 - (1/2)^6) = 4*(1/4096 - 1/64).
    let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let (e, _) = pot.calc_energy_forces(&coords);
    let expected = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
    assert!((e - expected).abs() < 1e-12, "energy {e}");
}

#[test]
fn energy_is_zero_at_sigma() {
    let pot = single_pair();
    // r = sigma -> (sigma/r)^12 - (sigma/r)^6 = 1 - 1 = 0.
    let coords: Vec<F> = vec![0.0, 0.0, 0.0, SIGMA, 0.0, 0.0];
    let (e, _) = pot.calc_energy_forces(&coords);
    assert!(e.abs() < 1e-12, "energy {e}");
}

#[test]
fn minimum_at_r_min() {
    // r_min = 2^(1/6) * sigma -> E = -eps.
    let pot = single_pair();
    let r_min = 2.0_f64.powf(1.0 / 6.0) * SIGMA;
    let coords: Vec<F> = vec![0.0, 0.0, 0.0, r_min, 0.0, 0.0];
    let (e, f) = pot.calc_energy_forces(&coords);
    assert!((e + EPS).abs() < 1e-9, "energy at minimum {e}");
    // Force along the bond axis vanishes at the minimum.
    assert!(f[0].abs() < 1e-6, "fx {}", f[0]);
}

#[test]
fn newtons_third_law() {
    let pot = single_pair();
    let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];
    let (_, f) = pot.calc_energy_forces(&coords);
    for dim in 0..3 {
        assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
    }
}

#[test]
fn forces_match_finite_difference() {
    let pot = PairLJCut::new(vec![0], vec![1], vec![0.5], vec![1.2]);
    let coords: Vec<F> = vec![0.1, -0.1, 0.0, 1.4, 0.5, -0.2];
    let (_, analytical) = pot.calc_energy_forces(&coords);
    let numerical = numerical_forces(|c| pot.calc_energy(c), &coords, 1e-7);
    for i in 0..coords.len() {
        assert!(
            (analytical[i] - numerical[i]).abs() < 1e-5,
            "i={i} analytical={} numerical={}",
            analytical[i],
            numerical[i]
        );
    }
}

#[test]
fn overlapping_atoms_are_skipped() {
    // Zero distance -> kernel skips the pair (no NaN / inf).
    let pot = single_pair();
    let coords: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (e, f) = pot.calc_energy_forces(&coords);
    assert!(e.abs() < 1e-12, "energy {e}");
    assert!(f.iter().all(|x| x.abs() < 1e-12));
}

#[test]
fn compile_path_resolves_self_pair_type() {
    let mut ff = ForceField::new("pair-only");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("Ar", &[("epsilon", EPS), ("sigma", SIGMA)]);
    // Per-atom convention: atoms carry `type`; the neighbour list is
    // `atomi/atomj/is_14`. Two Ar atoms Lorentz-Berthelot-combine to (EPS, SIGMA).
    let mut frame = Frame::new();
    frame.insert(
        "atoms",
        typed_atoms_block(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], &["Ar", "Ar"], None),
    );
    frame.insert("pairs", pairs_block(&[0], &[1], &[false]));
    let pots = ff.to_potentials(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    let expected = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
    assert!((pots.calc_energy(&coords) - expected).abs() < 1e-12);
}
