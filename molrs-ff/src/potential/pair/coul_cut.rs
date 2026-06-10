//! Coulomb pair potential with a hard distance cutoff:
//!
//! E(r) = k_e · q_i q_j / r   for r < r_cut, else 0
//!
//! `k_e = 332.063_71 kcal·Å·mol⁻¹·e⁻²` is the electrostatic constant in the
//! kcal/mol–Å–e unit system (matches molpy's Coulomb constant).
//!
//! The kernel is topology-blind: it consumes a pre-resolved pair list whose
//! per-pair charge products `qiqj` already include any exclusion / 1-4 scaling.
//! Geometric combining and OPLS 1-4 factors are the caller's responsibility.

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::potential::Potential;
use crate::potential::geometry::validate_coords;
use molrs::frame::Frame;
use molrs::types::F;

/// Electrostatic constant `k_e` in kcal·Å·mol⁻¹·e⁻².
pub const COULOMB_CONSTANT: F = 332.063_71;

/// Coulomb-with-cutoff pair potential with pre-resolved flat arrays.
pub struct PairCoulCut {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    /// Per-pair charge product `q_i q_j` (already scaled for 1-4 etc.).
    qiqj: Vec<F>,
    /// Distance cutoff in Å; `f64::INFINITY` disables it.
    cutoff: F,
}

impl PairCoulCut {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, qiqj: Vec<F>, cutoff: F) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), qiqj.len());
        Self {
            atom_i,
            atom_j,
            qiqj,
            cutoff,
        }
    }
}

impl Potential for PairCoulCut {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];
        let cut2 = self.cutoff * self.cutoff;

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-24 || r2 >= cut2 {
                continue;
            }
            let r = r2.sqrt();
            let qq = COULOMB_CONSTANT * self.qiqj[idx];
            energy += qq / r;

            // F = -dE/dx; factor·(r_j - r_i) is the force on j, -that on i.
            let factor = qq / (r2 * r);
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

/// Construct a [`PairCoulCut`] from style params (`cutoff`, default ∞), per-type
/// params (`qiqj`), and a Frame's `"pairs"` block (`atomi/atomj/type`).
pub fn pair_coul_cut_ctor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let cutoff = style_params
        .get("cutoff")
        .map(|c| c as F)
        .unwrap_or(F::INFINITY);

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairCoulCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairCoulCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairCoulCut: pairs block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "PairCoulCut: pairs block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut qiqj = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("PairCoulCut: unknown pair type '{}'", label))?;
        let qq = params
            .get("qiqj")
            .ok_or_else(|| format!("PairCoulCut type '{}': missing 'qiqj'", label))?
            as F;
        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        qiqj.push(qq);
    }

    Ok(Box::new(PairCoulCut::new(atom_i, atom_j, qiqj, cutoff)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_and_sign() {
        // Unit positive charges 2 Å apart: E = k_e / 2.
        let pot = PairCoulCut::new(vec![0], vec![1], vec![1.0], F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, forces) = pot.eval(&coords);
        assert!((e - COULOMB_CONSTANT / 2.0).abs() < 1e-9, "E got {e}");
        // Like charges repel: force on j (+x atom) points +x (away from i).
        assert!(
            forces[3] > 0.0,
            "like charges should repel, fxj={}",
            forces[3]
        );
        // Unlike charges attract.
        let pot2 = PairCoulCut::new(vec![0], vec![1], vec![-1.0], F::INFINITY);
        let (_, f2) = pot2.eval(&coords);
        assert!(f2[3] < 0.0, "unlike charges should attract, fxj={}", f2[3]);
    }

    #[test]
    fn cutoff_and_zero_distance() {
        let pot = PairCoulCut::new(vec![0], vec![1], vec![1.0], 1.5);
        // 2 Å apart, cutoff 1.5 -> excluded, E=0.
        let far: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert_eq!(pot.eval(&far).0, 0.0);
        // Coincident atoms -> skipped, finite.
        let zero: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (e, f) = pot.eval(&zero);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn numerical_gradient() {
        let pot = PairCoulCut::new(vec![0], vec![1], vec![0.8], F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.7, 0.3, -0.4];
        let (_, forces) = pot.eval(&coords);
        let h = 1e-6;
        for d in 0..coords.len() {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[d] += h;
            cm[d] -= h;
            let fd = -(pot.eval(&cp).0 - pot.eval(&cm).0) / (2.0 * h);
            assert!(
                (forces[d] - fd).abs() < 1e-5,
                "comp {d}: analytic {} vs fd {fd}",
                forces[d]
            );
        }
        // Newton's third law.
        for dim in 0..3 {
            assert!((forces[dim] + forces[3 + dim]).abs() < 1e-9);
        }
    }
}
