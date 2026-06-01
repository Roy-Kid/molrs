//! MMFF94 torsion term (3-term Fourier) + analytical gradient.
//!
//! Ported from RDKit `Code/ForceField/MMFF/TorsionAngle.cpp` and the
//! `computeDihedral` helper in `Code/ForceField/ForceField.cpp`
//! (BSD-3, Paolo Tosco / RDKit contributors).
//!
//! Energy (MMFF.I eq. 4):
//! ```text
//! ET = 0.5 * (V1*(1+cosPhi) + V2*(1-cos2Phi) + V3*(1+cos3Phi))
//! ```

use super::geom::{V3, add_grad, clip_to_one, cross, dot, norm, pt, scale, sub};

/// One enumerated torsion interaction (`i`-`j`-`k`-`l`).
#[derive(Clone, Copy, Debug)]
pub(super) struct TorsionTerm {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

/// RDKit `computeDihedral`: fills `r[4]`, normalized cross products `t[2]`,
/// their unnormalized lengths `d[2]`, and `cosPhi`.
fn compute_dihedral(p1: V3, p2: V3, p3: V3, p4: V3) -> ([V3; 4], [V3; 2], [f64; 2], f64) {
    let mut r = [[0.0; 3]; 4];
    r[0] = sub(p1, p2);
    r[1] = sub(p3, p2);
    r[2] = scale(r[1], -1.0);
    r[3] = sub(p4, p3);

    let mut t = [[0.0; 3]; 2];
    let mut d = [0.0; 2];
    let mut t0 = cross(r[0], r[1]);
    d[0] = norm(t0).max(1.0e-5);
    t0 = scale(t0, 1.0 / d[0]);
    t[0] = t0;
    let mut t1 = cross(r[2], r[3]);
    d[1] = norm(t1).max(1.0e-5);
    t1 = scale(t1, 1.0 / d[1]);
    t[1] = t1;
    let cos_phi = clip_to_one(dot(t[0], t[1]));
    (r, t, d, cos_phi)
}

impl TorsionTerm {
    pub(super) fn energy_grad(&self, coords: &[f64], grad: &mut [f64]) -> f64 {
        let p1 = pt(coords, self.i);
        let p2 = pt(coords, self.j);
        let p3 = pt(coords, self.k);
        let p4 = pt(coords, self.l);
        let (r, t, d, cos_phi) = compute_dihedral(p1, p2, p3, p4);

        let cos2 = 2.0 * cos_phi * cos_phi - 1.0;
        let cos3 = cos_phi * (2.0 * cos2 - 1.0);
        let energy =
            0.5 * (self.v1 * (1.0 + cos_phi) + self.v2 * (1.0 - cos2) + self.v3 * (1.0 + cos3));

        // gradient (RDKit TorsionAngleContrib::getGrad + calcTorsionGrad)
        let sin_phi_sq = 1.0 - cos_phi * cos_phi;
        let sin_phi = if sin_phi_sq > 0.0 {
            sin_phi_sq.sqrt()
        } else {
            0.0
        };
        let sin2 = 2.0 * sin_phi * cos_phi;
        let sin3 = 3.0 * sin_phi - 4.0 * sin_phi * sin_phi_sq;
        let de_dphi = 0.5 * (-self.v1 * sin_phi + 2.0 * self.v2 * sin2 - 3.0 * self.v3 * sin3);
        let sin_term = -de_dphi
            * if sin_phi.abs() < 1.0e-8 {
                1.0 / cos_phi
            } else {
                1.0 / sin_phi
            };

        let dc = [
            (t[1][0] - cos_phi * t[0][0]) / d[0],
            (t[1][1] - cos_phi * t[0][1]) / d[0],
            (t[1][2] - cos_phi * t[0][2]) / d[0],
            (t[0][0] - cos_phi * t[1][0]) / d[1],
            (t[0][1] - cos_phi * t[1][1]) / d[1],
            (t[0][2] - cos_phi * t[1][2]) / d[1],
        ];
        let (r0, r1, r2, r3) = (r[0], r[1], r[2], r[3]);

        add_grad(
            grad,
            self.i,
            [
                sin_term * (dc[2] * r1[1] - dc[1] * r1[2]),
                sin_term * (dc[0] * r1[2] - dc[2] * r1[0]),
                sin_term * (dc[1] * r1[0] - dc[0] * r1[1]),
            ],
        );
        add_grad(
            grad,
            self.j,
            [
                sin_term
                    * (dc[1] * (r1[2] - r0[2])
                        + dc[2] * (r0[1] - r1[1])
                        + dc[4] * (-r3[2])
                        + dc[5] * r3[1]),
                sin_term
                    * (dc[0] * (r0[2] - r1[2])
                        + dc[2] * (r1[0] - r0[0])
                        + dc[3] * r3[2]
                        + dc[5] * (-r3[0])),
                sin_term
                    * (dc[0] * (r1[1] - r0[1])
                        + dc[1] * (r0[0] - r1[0])
                        + dc[3] * (-r3[1])
                        + dc[4] * r3[0]),
            ],
        );
        add_grad(
            grad,
            self.k,
            [
                sin_term
                    * (dc[1] * r0[2]
                        + dc[2] * (-r0[1])
                        + dc[4] * (r3[2] - r2[2])
                        + dc[5] * (r2[1] - r3[1])),
                sin_term
                    * (dc[0] * (-r0[2])
                        + dc[2] * r0[0]
                        + dc[3] * (r2[2] - r3[2])
                        + dc[5] * (r3[0] - r2[0])),
                sin_term
                    * (dc[0] * r0[1]
                        + dc[1] * (-r0[0])
                        + dc[3] * (r3[1] - r2[1])
                        + dc[4] * (r2[0] - r3[0])),
            ],
        );
        add_grad(
            grad,
            self.l,
            [
                sin_term * (dc[4] * r2[2] - dc[5] * r2[1]),
                sin_term * (dc[5] * r2[0] - dc[3] * r2[2]),
                sin_term * (dc[3] * r2[1] - dc[4] * r2[0]),
            ],
        );
        energy
    }
}
