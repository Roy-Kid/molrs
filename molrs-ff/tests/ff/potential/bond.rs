//! Harmonic bond potential: E = 0.5 * k0 * (r - r0)^2.

use crate::helpers::{atoms_frame, numerical_forces, topo_block};
use molrs::types::F;
use molrs_ff::ForceField;
use molrs_ff::potential::Potential;
use molrs_ff::potential::bond::harmonic::BondHarmonic;
use molrs_ff::potential::extract_coords;

const K0: F = 300.0;
const R0: F = 1.5;

fn single_bond() -> BondHarmonic {
    BondHarmonic::new(vec![0], vec![1], vec![K0], vec![R0])
}

#[test]
fn energy_matches_closed_form() {
    let pot = single_bond();
    // r = 2.0 -> dr = 0.5 -> E = 0.5*300*0.25 = 37.5
    let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let (e, _) = pot.calc_energy_forces(&coords);
    assert!((e - 37.5).abs() < 1e-9, "energy {e}");
}

#[test]
fn at_equilibrium_energy_and_force_vanish() {
    let pot = single_bond();
    let coords: Vec<F> = vec![0.0, 0.0, 0.0, R0, 0.0, 0.0];
    let (e, f) = pot.calc_energy_forces(&coords);
    assert!(e.abs() < 1e-12, "energy {e}");
    for fi in f {
        assert!(fi.abs() < 1e-9);
    }
}

#[test]
fn newtons_third_law() {
    let pot = single_bond();
    // Off-axis geometry to exercise all three components.
    let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
    let (_, f) = pot.calc_energy_forces(&coords);
    for dim in 0..3 {
        assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
    }
}

#[test]
fn forces_match_finite_difference() {
    let pot = single_bond();
    let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
    let (_, analytical) = pot.calc_energy_forces(&coords);
    let numerical = numerical_forces(|c| pot.calc_energy(c), &coords, 1e-7);
    for i in 0..coords.len() {
        assert!(
            (analytical[i] - numerical[i]).abs() < 1e-6,
            "i={i} analytical={} numerical={}",
            analytical[i],
            numerical[i]
        );
    }
}

#[test]
fn compile_path_resolves_type_labels() {
    let mut ff = ForceField::new("bond-only");
    ff.def_bondstyle("harmonic")
        .def_type("CT-CT", &[("k0", K0), ("r0", R0)]);
    let mut frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["CT-CT"]),
    );
    let pots = ff.compile(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    assert!((pots.calc_energy(&coords) - 37.5).abs() < 1e-9);
}

#[test]
fn compile_unknown_bond_label_errors() {
    let mut ff = ForceField::new("bond-only");
    ff.def_bondstyle("harmonic")
        .def_type("CT-CT", &[("k0", K0), ("r0", R0)]);
    let mut frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["XX-YY"]),
    );
    let err = ff.compile(&frame).expect_err("unknown bond type");
    assert!(err.contains("unknown bond type"), "{err}");
}
