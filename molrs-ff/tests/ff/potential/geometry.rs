//! Shared geometry helpers used by all kernel families.

use molrs::types::F;
use molrs_ff::potential::geometry::{
    compute_angle, compute_dihedral, cross3, dot3, mag3, validate_coords,
};

const TOL: F = 1e-12;

#[test]
fn vector_primitives() {
    let a = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    // x cross y = z
    let c = cross3(a, b);
    assert!((c[0]).abs() < TOL && (c[1]).abs() < TOL && (c[2] - 1.0).abs() < TOL);
    assert!(dot3(a, b).abs() < TOL);
    assert!((dot3(a, a) - 1.0).abs() < TOL);
    assert!((mag3([3.0, 4.0, 0.0]) - 5.0).abs() < TOL);
}

#[test]
fn validate_coords_returns_atom_count() {
    assert_eq!(validate_coords(&[0.0; 9]), 3);
    assert_eq!(validate_coords(&[]), 0);
}

#[test]
#[should_panic(expected = "multiple of 3")]
fn validate_coords_rejects_non_multiple() {
    validate_coords(&[0.0, 1.0]);
}

#[test]
fn right_angle() {
    // i=(1,0,0), j=(0,0,0) vertex, k=(0,1,0) -> 90 deg.
    let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let theta = compute_angle(&coords, 0, 1, 2);
    assert!(
        (theta - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
        "{theta}"
    );
}

#[test]
fn collinear_angle_is_pi() {
    // i and k on opposite sides of j -> 180 deg.
    let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0];
    let theta = compute_angle(&coords, 0, 1, 2);
    assert!((theta - std::f64::consts::PI).abs() < 1e-9, "{theta}");
}

#[test]
fn dihedral_cis_and_trans() {
    // Planar trans dihedral -> pi; cis -> 0.
    // Atoms i-j-k-l along x with i,l displaced in y.
    // trans: i and l on opposite sides.
    let trans: Vec<F> = vec![
        0.0, 1.0, 0.0, // i
        0.0, 0.0, 0.0, // j
        1.0, 0.0, 0.0, // k
        1.0, -1.0, 0.0, // l
    ];
    let phi_trans = compute_dihedral(&trans, 0, 1, 2, 3);
    assert!(
        (phi_trans.abs() - std::f64::consts::PI).abs() < 1e-9,
        "{phi_trans}"
    );

    // cis: i and l on the same side.
    let cis: Vec<F> = vec![
        0.0, 1.0, 0.0, // i
        0.0, 0.0, 0.0, // j
        1.0, 0.0, 0.0, // k
        1.0, 1.0, 0.0, // l
    ];
    let phi_cis = compute_dihedral(&cis, 0, 1, 2, 3);
    assert!(phi_cis.abs() < 1e-9, "{phi_cis}");
}

#[test]
fn dihedral_ninety_degrees() {
    // l rotated 90 deg out of the i-j-k plane.
    let coords: Vec<F> = vec![
        0.0, 1.0, 0.0, // i
        0.0, 0.0, 0.0, // j
        1.0, 0.0, 0.0, // k
        1.0, 0.0, 1.0, // l (in xz)
    ];
    let phi = compute_dihedral(&coords, 0, 1, 2, 3);
    assert!(
        (phi.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
        "{phi}"
    );
}
