//! Harmonic angle potential: E = 0.5 * k0 * (theta - theta0)^2 (theta0 in radians).

use crate::helpers::{atoms_frame, numerical_forces, topo_block};
use molrs::ff::ForceField;
use molrs::ff::potential::Potential;
use molrs::ff::potential::angle::harmonic::AngleHarmonic;
use molrs::ff::potential::extract_coords;
use molrs::types::F;

const K0: F = 50.0;

#[test]
fn equilibrium_right_angle_has_zero_energy() {
    let theta0 = std::f64::consts::FRAC_PI_2;
    let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![K0], vec![theta0]);
    // i at +x, j at origin (vertex), k at +y -> angle = 90 deg.
    let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let (e, f) = pot.calc_energy_forces(&coords);
    assert!(e.abs() < 1e-9, "energy {e}");
    for fi in f {
        assert!(fi.abs() < 1e-6, "force {fi}");
    }
}

#[test]
fn displaced_angle_matches_closed_form() {
    // Actual angle = 90 deg, theta0 = 60 deg -> dtheta = pi/6.
    let theta0 = std::f64::consts::FRAC_PI_3; // 60 deg
    let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![K0], vec![theta0]);
    let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let (e, _) = pot.calc_energy_forces(&coords);
    let dtheta = std::f64::consts::FRAC_PI_2 - theta0; // pi/6
    let expected = 0.5 * K0 * dtheta * dtheta;
    assert!(
        (e - expected).abs() < 1e-9,
        "energy {e} expected {expected}"
    );
}

#[test]
fn forces_match_finite_difference() {
    let theta0 = std::f64::consts::FRAC_PI_3;
    let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![K0], vec![theta0]);
    // Generic non-collinear, non-right geometry.
    let coords: Vec<F> = vec![1.1, 0.2, -0.1, 0.0, 0.0, 0.0, -0.3, 0.9, 0.2];
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
fn total_force_is_balanced() {
    // Internal potential: net force on the three-atom system is ~zero.
    let theta0 = std::f64::consts::FRAC_PI_3;
    let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![K0], vec![theta0]);
    let coords: Vec<F> = vec![1.1, 0.2, -0.1, 0.0, 0.0, 0.0, -0.3, 0.9, 0.2];
    let (_, f) = pot.calc_energy_forces(&coords);
    for dim in 0..3 {
        let net = f[dim] + f[3 + dim] + f[6 + dim];
        assert!(net.abs() < 1e-9, "dim {dim} net {net}");
    }
}

#[test]
fn compile_path_consumes_radians() {
    // def_type stores theta0 in radians (molrs internal convention); the kernel
    // consumes it directly with no .to_radians() of its own.
    let mut ff = ForceField::new("angle-only");
    ff.def_anglestyle("harmonic").def_type(
        "H-O-H",
        &[("k", K0), ("theta0", std::f64::consts::FRAC_PI_2)],
    );
    // 90 deg geometry -> dtheta = 0 -> energy 0.
    let mut frame = atoms_frame(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    frame.insert(
        "angles",
        topo_block(
            &[("atomi", &[0]), ("atomj", &[1]), ("atomk", &[2])],
            &["H-O-H"],
        ),
    );
    let pots = ff.to_potentials(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    assert!(pots.calc_energy(&coords).abs() < 1e-9);
}
