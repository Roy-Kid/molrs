//! Integration: a LAMMPS `.ff` (LammpsFfReader) compiles through `to_potentials`
//! into an evaluable `Potentials`, and the energies are numerically right.
//!
//! This is the end-to-end contract the GAFF relaxation path depends on: the
//! reader must emit the exact kernel param keys/units `to_potentials` expects
//! (`k = 2·K_lammps`, angle/dihedral phases in **degrees**). A wrong key (`k0`)
//! or a pre-converted radian `theta0` would compile here to a missing-param error
//! or a wrong energy — so these tests, not the inline parse tests, are what pin
//! the convention.

use crate::helpers::{atoms_frame, topo_block};
use molrs::ff::potential::extract_coords;
use molrs::ff::{ForceFieldReader, LammpsFfReader};

#[test]
fn bond_roundtrips_through_to_potentials() {
    // bond_coeff K=150, r0=1.5 → molrs k = 2K = 300. At r = 2.0:
    // E = ½·300·(2−1.5)² = 37.5 kcal/mol.
    let ff = LammpsFfReader::new()
        .read_str("bond_style harmonic\nbond_coeff c3-c3 150.0 1.5\n")
        .unwrap();

    let mut frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["c3-c3"]),
    );

    let pots = ff.to_potentials(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    let (e, _) = pots.calc_energy_forces(&coords);
    assert!((e - 37.5).abs() < 1e-9, "bond energy {e}");
}

#[test]
fn angle_phase_stays_in_degrees() {
    // angle_coeff K=50, theta0=120° → molrs k = 100, theta0 = 120 (degrees; the
    // kernel converts). A right angle (90°) gives
    // E = ½·100·((90−120)·π/180)² . If the reader had pre-converted theta0 to
    // radians (double conversion), theta0 would be ~2.09° and the energy wrong.
    let ff = LammpsFfReader::new()
        .read_str("angle_style harmonic\nangle_coeff a-b-c 50.0 120.0\n")
        .unwrap();

    // i=(1,0,0), j=(0,0,0) vertex, k=(0,1,0) → theta = 90°.
    let mut frame = atoms_frame(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    frame.insert(
        "angles",
        topo_block(
            &[("atomi", &[0]), ("atomj", &[1]), ("atomk", &[2])],
            &["a-b-c"],
        ),
    );

    let pots = ff.to_potentials(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    let (e, _) = pots.calc_energy_forces(&coords);

    let dtheta = (90.0_f64 - 120.0).to_radians();
    let expect = 0.5 * 100.0 * dtheta * dtheta;
    assert!(
        (e - expect).abs() < 1e-9,
        "angle energy {e}, expected {expect}"
    );
}
