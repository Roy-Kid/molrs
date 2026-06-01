//! MMFF94 stretch-bend coupling term + analytical gradient.
//!
//! Ported from RDKit `Code/ForceField/MMFF/StretchBend.cpp`
//! (BSD-3, Paolo Tosco / RDKit contributors).
//!
//! Energy (MMFF.I eq. 5):
//! ```text
//! ESB = 2.51210 * (kba_ijk*dr_ij + kba_kji*dr_kj) * dtheta
//! ```
//! where `2.51210 = 143.9325 * DEG2RAD`.

use super::geom::{add_grad, clip_to_one, dot, norm, pt, scale, sub};
use super::params::{DEG2RAD, MDYNE_A_TO_KCAL, RAD2DEG};

/// One enumerated stretch-bend interaction (`i`-`j`-`k`, `j` central).
#[derive(Clone, Copy, Debug)]
pub(super) struct StretchBendTerm {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub rest1: f64,
    pub rest2: f64,
    pub theta0: f64,
    pub fc1: f64,
    pub fc2: f64,
}

impl StretchBendTerm {
    pub(super) fn energy_grad(&self, coords: &[f64], grad: &mut [f64]) -> f64 {
        let p1 = pt(coords, self.i);
        let p2 = pt(coords, self.j);
        let p3 = pt(coords, self.k);
        let dist1 = norm(sub(p1, p2));
        let dist2 = norm(sub(p3, p2));
        if dist1 < 1.0e-12 || dist2 < 1.0e-12 {
            return 0.0;
        }
        let p12 = scale(sub(p1, p2), 1.0 / dist1);
        let p32 = scale(sub(p3, p2), 1.0 / dist2);
        let cos_theta = clip_to_one(dot(p12, p32));
        let theta = RAD2DEG * cos_theta.acos();
        let delta_theta = theta - self.theta0;

        let factor = MDYNE_A_TO_KCAL * DEG2RAD * delta_theta;
        let energy =
            factor * self.fc1 * (dist1 - self.rest1) + factor * self.fc2 * (dist2 - self.rest2);

        // gradient (RDKit StretchBendContrib::getGrad)
        let c5 = MDYNE_A_TO_KCAL * DEG2RAD;
        let sin_theta_sq = 1.0 - cos_theta * cos_theta;
        let sin_theta = sin_theta_sq.max(0.0).sqrt().max(1.0e-8);
        let angle_term = theta - self.theta0;
        let dist_term =
            RAD2DEG * (self.fc1 * (dist1 - self.rest1) + self.fc2 * (dist2 - self.rest2));
        let dc = [
            (p32[0] - cos_theta * p12[0]) / dist1,
            (p32[1] - cos_theta * p12[1]) / dist1,
            (p32[2] - cos_theta * p12[2]) / dist1,
            (p12[0] - cos_theta * p32[0]) / dist2,
            (p12[1] - cos_theta * p32[1]) / dist2,
            (p12[2] - cos_theta * p32[2]) / dist2,
        ];
        let inv = 1.0 / (-sin_theta);

        add_grad(
            grad,
            self.i,
            [
                c5 * (p12[0] * self.fc1 * angle_term + dc[0] * inv * dist_term),
                c5 * (p12[1] * self.fc1 * angle_term + dc[1] * inv * dist_term),
                c5 * (p12[2] * self.fc1 * angle_term + dc[2] * inv * dist_term),
            ],
        );
        add_grad(
            grad,
            self.j,
            [
                c5 * ((-p12[0] * self.fc1 - p32[0] * self.fc2) * angle_term
                    + (-dc[0] - dc[3]) * inv * dist_term),
                c5 * ((-p12[1] * self.fc1 - p32[1] * self.fc2) * angle_term
                    + (-dc[1] - dc[4]) * inv * dist_term),
                c5 * ((-p12[2] * self.fc1 - p32[2] * self.fc2) * angle_term
                    + (-dc[2] - dc[5]) * inv * dist_term),
            ],
        );
        add_grad(
            grad,
            self.k,
            [
                c5 * (p32[0] * self.fc2 * angle_term + dc[3] * inv * dist_term),
                c5 * (p32[1] * self.fc2 * angle_term + dc[4] * inv * dist_term),
                c5 * (p32[2] * self.fc2 * angle_term + dc[5] * inv * dist_term),
            ],
        );
        energy
    }
}
