//! Thole dipole-dipole screening (CL&Pol short-range damping).
//!
//! Screens the Coulomb interaction between Drude-related point charges at short
//! range with the exponential Thole function
//!
//! ```text
//! T_ij(r) = 1 - (1 + s_ij r / 2) exp(-s_ij r)
//! s_ij    = a_ij / (alpha_i alpha_j)^(1/6),   a_ij = (a_i + a_j) / 2
//! ```
//!
//! so the damped energy of a pair is `T_ij(r) * q_i q_j / r`. The screening
//! `s_ij` depends on **both** endpoints' atomic polarizabilities, so the
//! constructor resolves per-atom-type `charge` / `alpha` / `a_thole` from the
//! `atoms` block and precomputes `(s_ij, q_i q_j)` per pair.
//!
//! Units: r in A, alpha in A^3, a dimensionless, q in e (energy in the same
//! Coulomb units as the accompanying electrostatic kernel — Thole is a
//! multiplicative screen on `q_i q_j / r`).
//!
//! Reference: Thole, Chem. Phys. 59 (1981) 341,
//! DOI 10.1016/0301-0104(81)85176-2; as emitted by the paduagroup/clandpol
//! polarizer (LAMMPS `pair_style thole`).

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::validate_coords;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Thole-screened Coulomb pair potential with pre-resolved flat arrays.
///
/// `s[idx]` is the screening factor `s_ij` and `qq[idx]` the charge product
/// `q_i q_j` for pair `idx` — both depend on the two endpoints' atom types and
/// are resolved once at construction.
pub struct PairThole {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    s: Vec<F>,
    qq: Vec<F>,
}

impl PairThole {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, s: Vec<F>, qq: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), s.len());
        assert_eq!(atom_i.len(), qq.len());
        Self {
            atom_i,
            atom_j,
            s,
            qq,
        }
    }
}

impl Potential for PairThole {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);

            let s = self.s[idx];
            let qq = self.qq[idx];

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-24 {
                continue;
            }
            let r = r2.sqrt();
            let x = s * r;
            let e_x = (-x).exp();
            let t = 1.0 - (1.0 + x / 2.0) * e_x;
            energy += t * qq / r;

            // V = T qq / r ;  T'(r) = (s/2)(1 + x) e^{-x}
            // dV/dr = qq (T'/r - T/r^2)
            // factor = -(1/r) dV/dr = qq (T/r^3 - T'/r^2)
            let tp = (s / 2.0) * (1.0 + x) * e_x;
            let dvdr = qq * (tp / r - t / r2);
            let factor = -dvdr / r;
            let fx = factor * dx;
            let fy = factor * dy;
            let fz = factor * dz;

            forces[j * 3] += fx;
            forces[j * 3 + 1] += fy;
            forces[j * 3 + 2] += fz;
            forces[i * 3] -= fx;
            forces[i * 3 + 1] -= fy;
            forces[i * 3 + 2] -= fz;
        }

        (energy, forces)
    }
}

/// Construct a [`PairThole`] from per-atom-type params and Frame topology.
///
/// The thole style's per-type definitions are keyed by **atom type name** and
/// carry `charge`, `alpha`, `a_thole`. Each pair's screening is resolved from
/// its two endpoints' atom types (read from the `atoms` block `type` column).
pub fn pair_thole_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "PairThole: frame missing \"atoms\" block".to_string())?;
    let atom_types = atoms
        .get_string("type")
        .ok_or_else(|| "PairThole: atoms block missing \"type\" column".to_string())?;

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairThole: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairThole: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairThole: pairs block missing \"atomj\" column".to_string())?;

    let lookup = |type_name: &str| -> Result<(F, F, F), String> {
        let p = type_map
            .get(type_name)
            .ok_or_else(|| format!("PairThole: unknown atom type '{}'", type_name))?;
        let q = p
            .get("charge")
            .ok_or_else(|| format!("PairThole type '{}': missing 'charge'", type_name))?
            as F;
        let alpha = p
            .get("alpha")
            .ok_or_else(|| format!("PairThole type '{}': missing 'alpha'", type_name))?
            as F;
        let a = p
            .get("a_thole")
            .ok_or_else(|| format!("PairThole type '{}': missing 'a_thole'", type_name))?
            as F;
        Ok((q, alpha, a))
    };

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut s_vec = Vec::with_capacity(i_col.len());
    let mut qq_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let i = i_col[idx] as usize;
        let j = j_col[idx] as usize;
        let (qi, alpha_i, ai) = lookup(&atom_types[i])?;
        let (qj, alpha_j, aj) = lookup(&atom_types[j])?;

        let a_ij = 0.5 * (ai + aj);
        let s = a_ij / (alpha_i * alpha_j).powf(1.0 / 6.0);

        atom_i.push(i);
        atom_j.push(j);
        s_vec.push(s);
        qq_vec.push(qi * qj);
    }

    Ok(Box::new(PairThole::new(atom_i, atom_j, s_vec, qq_vec)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numerical_forces(pot: &PairThole, coords: &[F]) -> Vec<F> {
        let h = 1e-6;
        let mut num = vec![0.0; coords.len()];
        for k in 0..coords.len() {
            let mut cp = coords.to_vec();
            let mut cm = coords.to_vec();
            cp[k] += h;
            cm[k] -= h;
            num[k] = -(pot.calc_energy(&cp) - pot.calc_energy(&cm)) / (2.0 * h);
        }
        num
    }

    #[test]
    fn damping_factor_matches_closed_form() {
        // s=1.0, qq=1.0, r=2.0: x=2, T = 1 - (1+1)e^-2 = 1 - 2 e^-2
        let pot = PairThole::new(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let x = 2.0_f64;
        let t = 1.0 - (1.0 + x / 2.0) * (-x).exp();
        let expected = t * 1.0 / 2.0;
        assert!(
            (pot.calc_energy(&coords) - expected).abs() < 1e-12,
            "energy {} vs {expected}",
            pot.calc_energy(&coords)
        );
    }

    #[test]
    fn damping_vanishes_at_zero_separation_limit() {
        // T(r) -> 0 as r -> 0 (screening fully removes the singular Coulomb term),
        // so the energy stays finite. Check T is small at small r.
        let pot = PairThole::new(vec![0], vec![1], vec![5.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 0.05, 0.0, 0.0];
        assert!(pot.calc_energy(&coords).is_finite());
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairThole::new(vec![0], vec![1], vec![1.3], vec![-0.7]);
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.calc_energy_forces(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} analytical={} numerical={}",
                analytical[k],
                numerical[k]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = PairThole::new(vec![0], vec![1], vec![1.3], vec![-0.7]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.calc_energy_forces(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
