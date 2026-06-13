//! MMFF94 bond-stretch term (quartic) + analytical gradient.
//!
//! Ported from RDKit `Code/ForceField/MMFF/BondStretch.cpp`
//! (BSD-3, Paolo Tosco / RDKit contributors).
//!
//! Energy (MMFF.I eq. 2):
//! ```text
//! EB = 0.5 * 143.9325 * kb * dr^2 * (1 + cs*dr + 7/12*cs^2*dr^2)
//! ```
//! with `cs = -2 Angstrom^-1`, `dr = r - r0`.

use super::geom::{add_grad, norm, pt, scale, sub};
use super::params::MDYNE_A_TO_KCAL;

const CS: f64 = -2.0;
const C3: f64 = 7.0 / 12.0;

/// One enumerated bond-stretch interaction.
#[derive(Clone, Copy, Debug)]
pub(super) struct BondTerm {
    pub i: usize,
    pub j: usize,
    pub r0: f64,
    pub kb: f64,
}

impl BondTerm {
    /// Accumulate this term's gradient into `grad` and return its energy.
    pub(super) fn energy_grad(&self, coords: &[f64], grad: &mut [f64]) -> f64 {
        let rij = sub(pt(coords, self.i), pt(coords, self.j));
        let dist = norm(rij);
        let dr = dist - self.r0;
        let dr2 = dr * dr;

        let energy = 0.5 * MDYNE_A_TO_KCAL * self.kb * dr2 * (1.0 + CS * dr + C3 * CS * CS * dr2);

        // dE/dr (RDKit BondStretchContrib::getGrad)
        let de_dr =
            MDYNE_A_TO_KCAL * self.kb * dr * (1.0 + 1.5 * CS * dr + 2.0 * C3 * CS * CS * dr2);
        if dist > 0.0 {
            let g = scale(rij, de_dr / dist);
            add_grad(grad, self.i, g);
            add_grad(grad, self.j, scale(g, -1.0));
        } else {
            let g = [self.kb * 0.01, self.kb * 0.01, self.kb * 0.01];
            add_grad(grad, self.i, g);
            add_grad(grad, self.j, scale(g, -1.0));
        }
        energy
    }
}
