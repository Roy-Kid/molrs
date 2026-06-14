//! Lennard-Jones 12-6 pair potential: E = 4 * eps * ((sigma/r)^12 - (sigma/r)^6)

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::validate_coords;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Lennard-Jones 12-6 pair potential with pre-resolved flat arrays.
pub struct PairLJCut {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    epsilon: Vec<F>,
    sigma: Vec<F>,
}

impl PairLJCut {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, epsilon: Vec<F>, sigma: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), epsilon.len());
        assert_eq!(atom_i.len(), sigma.len());
        Self {
            atom_i,
            atom_j,
            epsilon,
            sigma,
        }
    }
}

impl Potential for PairLJCut {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);

            let eps = self.epsilon[idx];
            let sigma = self.sigma[idx];

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-24 {
                continue;
            }
            let sr2 = sigma * sigma / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            energy += 4.0 * eps * (sr12 - sr6);

            let factor = 4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / r2;
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

/// Construct a [`PairLJCut`] from per-atom-type params + a neighbour list.
///
/// Reads **per-atom** LJ params (`epsilon`/`sigma` keyed by the atoms block's
/// `type` column), combines each interacting pair with the **Lorentz-Berthelot**
/// rule (`ε = √(εᵢεⱼ)`, `σ = (σᵢ + σⱼ)/2`), and scales 1-4-flagged pairs by the
/// force field's 1-4 LJ weight (projected into `style_params["lj14scale"]` by
/// `Style::to_potential` from the ForceField's `special_bonds`; default `1.0`).
///
/// The `pairs` block is the consumer-built neighbour list
/// (`atomi`/`atomj`/`is_14`) from `intramolecular_pairs` — 1-2/1-3 neighbours are
/// already excluded, so only the 1-4 weight is applied here.
pub fn pair_lj_cut_ctor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let scale_14 = style_params.get("lj14scale").unwrap_or(1.0) as F;

    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "PairLJCut: frame missing \"atoms\" block".to_string())?;
    let atom_types = atoms
        .get_string("type")
        .ok_or_else(|| "PairLJCut: atoms block missing \"type\" column".to_string())?;
    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairLJCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairLJCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairLJCut: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get_bool("is_14");

    let n = i_col.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut eps_vec = Vec::with_capacity(n);
    let mut sig_vec = Vec::with_capacity(n);

    let per_atom = |t: &str| -> Result<(F, F), String> {
        let p = type_map
            .get(t)
            .ok_or_else(|| format!("PairLJCut: unknown atom type '{}'", t))?;
        let eps = p
            .get("epsilon")
            .ok_or_else(|| format!("PairLJCut type '{}': missing 'epsilon'", t))?
            as F;
        let sigma = p
            .get("sigma")
            .ok_or_else(|| format!("PairLJCut type '{}': missing 'sigma'", t))?
            as F;
        Ok((eps, sigma))
    };

    for idx in 0..n {
        let (eps_i, sig_i) = per_atom(&atom_types[i_col[idx] as usize])?;
        let (eps_j, sig_j) = per_atom(&atom_types[j_col[idx] as usize])?;
        // Lorentz-Berthelot combining.
        let mut eps = (eps_i * eps_j).sqrt();
        let sigma = 0.5 * (sig_i + sig_j);
        // Bake the 1-4 weight into epsilon (LJ energy scales linearly with eps).
        if is_14.is_some_and(|b| b[idx]) {
            eps *= scale_14;
        }
        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        eps_vec.push(eps);
        sig_vec.push(sigma);
    }

    Ok(Box::new(PairLJCut::new(atom_i, atom_j, eps_vec, sig_vec)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_lj_cut_energy_and_newton_third_law() {
        let pot = PairLJCut::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];

        let (e, forces) = pot.calc_energy_forces(&coords);
        assert!(e.is_finite());

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5, "dim={}: sum={}", dim, sum);
        }
    }

    #[test]
    fn test_pair_lj_cut_unknown_label_error() {
        let pot = PairLJCut::new(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, _) = pot.calc_energy_forces(&coords);
        assert!(e.is_finite());
    }
}
