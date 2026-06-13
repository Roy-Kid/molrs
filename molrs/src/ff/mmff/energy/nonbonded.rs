//! MMFF94 nonbonded terms: buffered-14-7 van der Waals + buffered Coulomb
//! electrostatics, with the 1-4 mask and 0.75 1-4 electrostatic scaling.
//!
//! Ported from RDKit `Code/ForceField/MMFF/Nonbonded.cpp`
//! (BSD-3, Paolo Tosco / RDKit contributors).
//!
//! Energy:
//! ```text
//! EvdW = eps * (1.07 R*/(R+0.07 R*))^7 * (1.12 R*^7/(R^7+0.12 R*^7) - 2)
//! EQ   = 332.0716 * qi qj / (D*(R + 0.05)) * (is1_4 ? 0.75 : 1.0)
//! ```
//! The dielectric model is the RDKit default CONSTANT (`dielModel == 1`).

use super::geom::{add_grad, norm, pt, scale, sub};
use super::params::{COULOMB, ELE_BUFFER};

const VDW1: f64 = 1.07;
const VDW1M1: f64 = VDW1 - 1.0;
const VDW2: f64 = 1.12;
const VDW2M1: f64 = VDW2 - 1.0;
const SC1_4: f64 = 0.75;

/// One enumerated nonbonded pair (van der Waals and/or electrostatic).
#[derive(Clone, Copy, Debug)]
pub(super) struct NonbondedTerm {
    pub i: usize,
    pub j: usize,
    /// van der Waals R* and well depth; `None` if the pair has no vdW params.
    pub vdw: Option<(f64, f64)>,
    /// `qi*qj / D` charge term; `None` if either partial charge vanishes.
    pub charge: Option<f64>,
    pub is_1_4: bool,
}

impl NonbondedTerm {
    pub(super) fn energy_grad(&self, coords: &[f64], grad: &mut [f64]) -> f64 {
        let rij = sub(pt(coords, self.i), pt(coords, self.j));
        let dist = norm(rij);
        let mut energy = 0.0;
        let mut vdw_grad = 0.0;
        let mut ele_grad = 0.0;

        if dist <= 0.0 {
            // RDKit's degenerate-distance kick.
            if let Some((r_star, _)) = self.vdw {
                let g = [r_star * 0.01; 3];
                add_grad(grad, self.i, g);
                add_grad(grad, self.j, scale(g, -1.0));
            }
            if self.charge.is_some() {
                let g = [0.02; 3];
                add_grad(grad, self.i, g);
                add_grad(grad, self.j, scale(g, -1.0));
            }
            return 0.0;
        }

        if let Some((r_star, well_depth)) = self.vdw {
            let dist7 = dist.powi(7);
            let a_term = VDW1 * r_star / (dist + VDW1M1 * r_star);
            let a_term7 = a_term.powi(7);
            let rs7 = r_star.powi(7);
            let b_term = VDW2 * rs7 / (dist7 + VDW2M1 * rs7) - 2.0;
            energy += well_depth * a_term7 * b_term;

            let q = dist / r_star;
            let q6 = q.powi(6);
            let q7 = q6 * q;
            let q7p = q7 + VDW2M1;
            let t = VDW1 / (q + VDW1M1);
            let t7 = t.powi(7);
            let de_dr = well_depth / r_star
                * t7
                * (-VDW2 * 7.0 * q6 / (q7p * q7p) + ((-VDW2 * 7.0 / q7p + 14.0) / (q + VDW1M1)));
            vdw_grad = de_dr / dist;
        }

        if let Some(charge_term) = self.charge {
            let scale_14 = if self.is_1_4 { SC1_4 } else { 1.0 };
            let corr = dist + ELE_BUFFER; // CONSTANT dielectric model
            energy += COULOMB * charge_term / corr * scale_14;
            // d/dr [1/(r+0.05)] = -1/(r+0.05)^2  (RDKit EleContrib::getGrad)
            let de_dr = -COULOMB * charge_term / (corr * corr) * scale_14;
            ele_grad = de_dr / dist;
        }

        let de_dr = vdw_grad + ele_grad;
        let g = scale(rij, de_dr);
        add_grad(grad, self.i, g);
        add_grad(grad, self.j, scale(g, -1.0));
        energy
    }
}
