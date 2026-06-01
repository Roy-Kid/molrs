//! PME electrostatics via the public `PmePotential` / `PmeParams` API.
//!
//! The kernel ctor (`pme_ctor`) is driven through `ForceField::compile`
//! elsewhere; here we construct `PmePotential` directly with in-code charges
//! and a cubic box to check physical invariants.

use molrs::types::F;
use molrs_ff::potential::Potential;
use molrs_ff::potential::kspace::pme::{PmeParams, PmePotential};

fn cubic_box(l: F) -> [[F; 3]; 3] {
    [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]
}

fn params(alpha: F, cutoff: F, order: usize) -> PmeParams {
    PmeParams {
        alpha,
        cutoff,
        grid_size: [32, 32, 32],
        order,
        coulomb: 1.0,
    }
}

#[test]
fn two_ion_energy_approaches_vacuum_coulomb() {
    let box_l: F = 20.0;
    let r: F = 3.0;
    let pme = PmePotential::new(params(0.3, 9.0, 5), vec![1.0, -1.0], cubic_box(box_l), vec![]);
    let c = box_l / 2.0;
    let coords: Vec<F> = vec![c, c, c, c + r, c, c];
    let e = pme.energy(&coords);
    // q_i q_j / r with coulomb=1, opposite unit charges -> -1/r, plus small
    // periodic-image correction.
    let e_vacuum = -1.0 / r;
    assert!((e - e_vacuum).abs() < 0.05, "pme {e} vacuum {e_vacuum}");
}

#[test]
fn newtons_third_law_total_force_balances() {
    let box_l: F = 10.0;
    let pme = PmePotential::new(
        params(0.35, 4.5, 5),
        vec![0.5, -0.3, 0.4, -0.6],
        cubic_box(box_l),
        vec![[0, 1], [2, 3]],
    );
    let coords: Vec<F> = vec![1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 6.0, 7.0, 2.0, 8.0, 7.5, 2.5];
    let (_, forces) = pme.eval(&coords);
    for dim in 0..3 {
        let sum: F = (0..4).map(|a| forces[a * 3 + dim]).sum();
        assert!(sum.abs() < 0.1, "dim {dim} net {sum}");
    }
}

#[test]
fn forces_match_finite_difference() {
    let box_l: F = 10.0;
    let pme = PmePotential::new(
        params(0.4, 4.5, 4),
        vec![0.5, -0.3, 0.2],
        cubic_box(box_l),
        vec![[0, 1]],
    );
    let coords: Vec<F> = vec![2.0, 3.0, 4.0, 5.0, 3.5, 4.5, 7.0, 6.0, 5.0];
    let forces = pme.forces(&coords);
    let eps: F = 1e-3;
    for idx in 0..coords.len() {
        let mut cp = coords.clone();
        let mut cm = coords.clone();
        cp[idx] += eps;
        cm[idx] -= eps;
        let numerical = -(pme.energy(&cp) - pme.energy(&cm)) / (2.0 * eps);
        assert!(
            (forces[idx] - numerical).abs() < 1.0,
            "idx={idx} analytical={} numerical={}",
            forces[idx],
            numerical
        );
    }
}

#[test]
fn neutral_system_energy_is_finite() {
    // Single charge: only self energy + reciprocal; must be finite, no NaN.
    let pme = PmePotential::new(params(0.3, 4.5, 4), vec![1.0], cubic_box(10.0), vec![]);
    let e = pme.energy(&[5.0, 5.0, 5.0]);
    assert!(e.is_finite(), "energy {e}");
}
