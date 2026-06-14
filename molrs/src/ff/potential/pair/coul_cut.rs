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

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::validate_coords;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Electrostatic constant `k_e` in kcal·Å·mol⁻¹·e⁻². Re-exported from
/// [`molrs::units::constants::COULOMB_REAL`] — the single CODATA-derived source
/// shared with the dielectric/conductivity analyses.
pub use molrs::units::constants::COULOMB_REAL as COULOMB_CONSTANT;

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
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
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

/// Construct a [`PairCoulCut`] from **per-atom charges** + a neighbour list.
///
/// Reads per-atom `charge` from the atoms block (the molecule carries its own
/// charges — RESP/AM1-BCC for GAFF, etc.), forms `qᵢqⱼ` for each interacting
/// pair, and scales 1-4-flagged pairs by the force field's 1-4 Coulomb weight
/// (projected into `style_params["coulomb14scale"]` by `Style::to_potential`
/// from the ForceField's `special_bonds`; default `1.0`).
///
/// The `pairs` block is the consumer-built neighbour list
/// (`atomi`/`atomj`/`is_14`) from `intramolecular_pairs`; 1-2/1-3 are already
/// excluded. `cutoff` comes from `style_params` (default ∞). Charge-free pair
/// types are not consulted — this kernel is per-atom, mirroring `mmff_ele`.
pub fn pair_coul_cut_ctor(
    style_params: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let cutoff = style_params
        .get("cutoff")
        .map(|c| c as F)
        .unwrap_or(F::INFINITY);
    let scale_14 = style_params.get("coulomb14scale").unwrap_or(1.0) as F;

    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "PairCoulCut: frame missing \"atoms\" block".to_string())?;
    let charges = atoms
        .get_float("charge")
        .ok_or_else(|| "PairCoulCut: atoms block missing \"charge\" column".to_string())?;
    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairCoulCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairCoulCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairCoulCut: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get_bool("is_14");

    let n = i_col.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut qiqj = Vec::with_capacity(n);

    for idx in 0..n {
        let i = i_col[idx] as usize;
        let j = j_col[idx] as usize;
        let mut qq = charges[i] as F * charges[j] as F;
        if is_14.is_some_and(|b| b[idx]) {
            qq *= scale_14;
        }
        atom_i.push(i);
        atom_j.push(j);
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
        let (e, forces) = pot.calc_energy_forces(&coords);
        assert!((e - COULOMB_CONSTANT / 2.0).abs() < 1e-9, "E got {e}");
        // Like charges repel: force on j (+x atom) points +x (away from i).
        assert!(
            forces[3] > 0.0,
            "like charges should repel, fxj={}",
            forces[3]
        );
        // Unlike charges attract.
        let pot2 = PairCoulCut::new(vec![0], vec![1], vec![-1.0], F::INFINITY);
        let (_, f2) = pot2.calc_energy_forces(&coords);
        assert!(f2[3] < 0.0, "unlike charges should attract, fxj={}", f2[3]);
    }

    #[test]
    fn cutoff_and_zero_distance() {
        let pot = PairCoulCut::new(vec![0], vec![1], vec![1.0], 1.5);
        // 2 Å apart, cutoff 1.5 -> excluded, E=0.
        let far: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert_eq!(pot.calc_energy_forces(&far).0, 0.0);
        // Coincident atoms -> skipped, finite.
        let zero: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (e, f) = pot.calc_energy_forces(&zero);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn numerical_gradient() {
        let pot = PairCoulCut::new(vec![0], vec![1], vec![0.8], F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.7, 0.3, -0.4];
        let (_, forces) = pot.calc_energy_forces(&coords);
        let h = 1e-6;
        for d in 0..coords.len() {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[d] += h;
            cm[d] -= h;
            let fd = -(pot.calc_energy_forces(&cp).0 - pot.calc_energy_forces(&cm).0) / (2.0 * h);
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
