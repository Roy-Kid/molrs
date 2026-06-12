//! MMFF94 out-of-plane (Wilson) bend term + analytical gradient.
//!
//! Ported from RDKit `Code/ForceField/MMFF/OopBend.cpp`
//! (BSD-3, Paolo Tosco / RDKit contributors).
//!
//! Energy (MMFF.I eq. 6):
//! ```text
//! EOOP = 0.043844 * (koop/2) * chi^2     (chi = Wilson out-of-plane angle)
//! ```

use super::geom::{add_grad, clip_to_one, cross, dot, norm, pt, scale, sub};
use super::params::{DEG2RAD, MDYNE_A_TO_KCAL, RAD2DEG};

/// One enumerated out-of-plane interaction. `j` is the central atom; the
/// out-of-plane angle is between bond `j`-`l` and the `i`-`j`-`k` plane.
#[derive(Clone, Copy, Debug)]
pub(super) struct OopTerm {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub koop: f64,
}

impl OopTerm {
    pub(super) fn energy_grad(&self, coords: &[f64], grad: &mut [f64]) -> f64 {
        let pi = pt(coords, self.i);
        let pj = pt(coords, self.j);
        let pk = pt(coords, self.k);
        let pl = pt(coords, self.l);

        // energy uses chi from n = rJI x rJK (RDKit calcOopChi)
        let energy = {
            let mut r_ji = sub(pi, pj);
            let mut r_jk = sub(pk, pj);
            let mut r_jl = sub(pl, pj);
            let (a, b, c) = (norm(r_ji), norm(r_jk), norm(r_jl));
            if a < 1.0e-12 || b < 1.0e-12 || c < 1.0e-12 {
                return 0.0;
            }
            r_ji = scale(r_ji, 1.0 / a);
            r_jk = scale(r_jk, 1.0 / b);
            r_jl = scale(r_jl, 1.0 / c);
            let mut n = cross(r_ji, r_jk);
            let nn = norm(n);
            if nn < 1.0e-12 {
                return 0.0;
            }
            n = scale(n, 1.0 / nn);
            let sin_chi = clip_to_one(dot(n, r_jl));
            let chi = RAD2DEG * sin_chi.asin();
            let c2 = MDYNE_A_TO_KCAL * DEG2RAD * DEG2RAD;
            0.5 * c2 * self.koop * chi * chi
        };

        // gradient (RDKit OopBendContrib::getSingleGrad; n = (-rJI) x rJK)
        let mut r_ji = sub(pi, pj);
        let mut r_jk = sub(pk, pj);
        let mut r_jl = sub(pl, pj);
        let d_ji = norm(r_ji);
        let d_jk = norm(r_jk);
        let d_jl = norm(r_jl);
        if d_ji < 1.0e-12 || d_jk < 1.0e-12 || d_jl < 1.0e-12 {
            return energy;
        }
        r_ji = scale(r_ji, 1.0 / d_ji);
        r_jk = scale(r_jk, 1.0 / d_jk);
        r_jl = scale(r_jl, 1.0 / d_jl);

        let mut n = cross(scale(r_ji, -1.0), r_jk);
        let nn = norm(n);
        if nn < 1.0e-12 {
            return energy;
        }
        n = scale(n, 1.0 / nn);
        let c2 = MDYNE_A_TO_KCAL * DEG2RAD * DEG2RAD;
        let sin_chi = clip_to_one(dot(r_jl, n));
        let cos_chi_sq = 1.0 - sin_chi * sin_chi;
        let cos_chi = if cos_chi_sq > 0.0 {
            cos_chi_sq.sqrt()
        } else {
            0.0
        }
        .max(1.0e-8);
        let chi = RAD2DEG * sin_chi.asin();
        let cos_theta = clip_to_one(dot(r_ji, r_jk));
        let sin_theta_sq = (1.0 - cos_theta * cos_theta).max(1.0e-8);
        let sin_theta = if sin_theta_sq > 0.0 {
            sin_theta_sq.sqrt()
        } else {
            0.0
        }
        .max(1.0e-8);

        let de_dchi = RAD2DEG * c2 * self.koop * chi;
        let t1 = cross(r_jl, r_jk);
        let t2 = cross(r_ji, r_jl);
        let t3 = cross(r_jk, r_ji);
        let term1 = cos_chi * sin_theta;
        let term2 = sin_chi / (cos_chi * sin_theta_sq);

        let tg1 = [
            (t1[0] / term1 - (r_ji[0] - r_jk[0] * cos_theta) * term2) / d_ji,
            (t1[1] / term1 - (r_ji[1] - r_jk[1] * cos_theta) * term2) / d_ji,
            (t1[2] / term1 - (r_ji[2] - r_jk[2] * cos_theta) * term2) / d_ji,
        ];
        let tg3 = [
            (t2[0] / term1 - (r_jk[0] - r_ji[0] * cos_theta) * term2) / d_jk,
            (t2[1] / term1 - (r_jk[1] - r_ji[1] * cos_theta) * term2) / d_jk,
            (t2[2] / term1 - (r_jk[2] - r_ji[2] * cos_theta) * term2) / d_jk,
        ];
        let tg4 = [
            (t3[0] / term1 - r_jl[0] * sin_chi / cos_chi) / d_jl,
            (t3[1] / term1 - r_jl[1] * sin_chi / cos_chi) / d_jl,
            (t3[2] / term1 - r_jl[2] * sin_chi / cos_chi) / d_jl,
        ];
        add_grad(grad, self.i, scale(tg1, de_dchi));
        add_grad(
            grad,
            self.j,
            [
                -de_dchi * (tg1[0] + tg3[0] + tg4[0]),
                -de_dchi * (tg1[1] + tg3[1] + tg4[1]),
                -de_dchi * (tg1[2] + tg3[2] + tg4[2]),
            ],
        );
        add_grad(grad, self.k, scale(tg3, de_dchi));
        add_grad(grad, self.l, scale(tg4, de_dchi));
        energy
    }
}
