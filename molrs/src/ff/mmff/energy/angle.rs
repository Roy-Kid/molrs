//! MMFF94 angle-bend term (cubic, with near-linear special case) + gradient.
//!
//! Ported from RDKit `Code/ForceField/MMFF/AngleBend.cpp`
//! (BSD-3, Paolo Tosco / RDKit contributors).
//!
//! Energy (MMFF.I eq. 3):
//! ```text
//! EA = 0.043844 * (ka/2) * dtheta^2 * (1 + cb*dtheta)      (cb = -0.007 deg^-1)
//! EA = 143.9325 * ka * (1 + cos(theta))                    (linear j)
//! ```

use super::geom::{add_grad, clip_to_one, dot, norm, pt, scale, sub};
use super::params::{DEG2RAD, MDYNE_A_TO_KCAL, RAD2DEG};

const CB: f64 = -0.006981317;

/// One enumerated angle-bend interaction (`i`-`j`-`k`, `j` central).
#[derive(Clone, Copy, Debug)]
pub(super) struct AngleTerm {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub theta0: f64,
    pub ka: f64,
    pub linear: bool,
}

impl AngleTerm {
    pub(super) fn energy_grad(&self, coords: &[f64], grad: &mut [f64]) -> f64 {
        let p1 = pt(coords, self.i);
        let p2 = pt(coords, self.j);
        let p3 = pt(coords, self.k);
        let r0v = sub(p1, p2);
        let r1v = sub(p3, p2);
        let dist0 = norm(r0v);
        let dist1 = norm(r1v);
        if dist0 < 1.0e-12 || dist1 < 1.0e-12 {
            return 0.0;
        }
        let r0 = scale(r0v, 1.0 / dist0);
        let r1 = scale(r1v, 1.0 / dist1);
        let mut cos_theta = clip_to_one(dot(r0, r1));

        let c2 = MDYNE_A_TO_KCAL * DEG2RAD * DEG2RAD;
        let energy = if self.linear {
            MDYNE_A_TO_KCAL * self.ka * (1.0 + cos_theta)
        } else {
            let angle = RAD2DEG * cos_theta.acos() - self.theta0;
            0.5 * c2 * self.ka * angle * angle * (1.0 + CB * angle)
        };

        // gradient (RDKit AngleBendContrib::getGrad + calcAngleBendGrad)
        let sin_theta_sq = 1.0 - cos_theta * cos_theta;
        let sin_theta = if sin_theta_sq > 0.0 {
            sin_theta_sq.sqrt()
        } else {
            0.0
        }
        .max(1.0e-8);

        let angle_term = RAD2DEG * cos_theta.acos() - self.theta0;
        let de_dtheta = if self.linear {
            -MDYNE_A_TO_KCAL * self.ka * sin_theta
        } else {
            RAD2DEG * c2 * self.ka * angle_term * (1.0 + 1.5 * CB * angle_term)
        };

        cos_theta = clip_to_one(cos_theta);
        let d_cos = [
            (r1[0] - cos_theta * r0[0]) / dist0,
            (r1[1] - cos_theta * r0[1]) / dist0,
            (r1[2] - cos_theta * r0[2]) / dist0,
            (r0[0] - cos_theta * r1[0]) / dist1,
            (r0[1] - cos_theta * r1[1]) / dist1,
            (r0[2] - cos_theta * r1[2]) / dist1,
        ];
        let f = de_dtheta / (-sin_theta);
        add_grad(grad, self.i, [f * d_cos[0], f * d_cos[1], f * d_cos[2]]);
        add_grad(
            grad,
            self.j,
            [
                f * (-d_cos[0] - d_cos[3]),
                f * (-d_cos[1] - d_cos[4]),
                f * (-d_cos[2] - d_cos[5]),
            ],
        );
        add_grad(grad, self.k, [f * d_cos[3], f * d_cos[4], f * d_cos[5]]);
        energy
    }
}
