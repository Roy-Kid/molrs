//! Class2 (9-6) Lennard-Jones pair potential:
//! E = epsilon * (2*(sigma/r)^9 - 3*(sigma/r)^6)
//!
//! The COMPASS/class2 non-bonded form. Parameters per pair type: `epsilon`
//! (energy), `sigma` (length).

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::potential::Potential;
use crate::potential::geometry::validate_coords;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Class2 (9-6) LJ pair potential with pre-resolved flat arrays.
pub struct PairLJClass2 {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    epsilon: Vec<F>,
    sigma: Vec<F>,
}

impl PairLJClass2 {
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

impl Potential for PairLJClass2 {
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
            let r = r2.sqrt();
            let u = sigma / r; // sigma/r
            let u3 = u * u * u;
            let u6 = u3 * u3;
            let u9 = u6 * u3;
            energy += eps * (2.0 * u9 - 3.0 * u6);

            // E = eps(2 u^9 - 3 u^6), u = sigma/r, du/dr = -u/r
            // dE/dr = eps(18 u^8 - 18 u^5)(-u/r) = -18 eps (u^9 - u^6)/r
            // factor = -(1/r) dE/dr = 18 eps (u^9 - u^6)/r^2
            let factor = 18.0 * eps * (u9 - u6) / r2;
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

/// Construct a [`PairLJClass2`] from style params, type params, and Frame topology.
pub fn pair_lj_class2_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairLJClass2: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairLJClass2: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairLJClass2: pairs block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "PairLJClass2: pairs block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut eps_vec = Vec::with_capacity(i_col.len());
    let mut sig_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("PairLJClass2: unknown pair type '{}'", label))?;
        let eps = params
            .get("epsilon")
            .ok_or_else(|| format!("PairLJClass2 type '{}': missing 'epsilon'", label))?
            as F;
        let sigma = params
            .get("sigma")
            .ok_or_else(|| format!("PairLJClass2 type '{}': missing 'sigma'", label))?
            as F;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        eps_vec.push(eps);
        sig_vec.push(sigma);
    }

    Ok(Box::new(PairLJClass2::new(
        atom_i, atom_j, eps_vec, sig_vec,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numerical_forces(pot: &PairLJClass2, coords: &[F]) -> Vec<F> {
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
    fn energy_at_sigma_is_negative_eps() {
        // At r = sigma: E = eps(2 - 3) = -eps.
        let pot = PairLJClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!((pot.calc_energy(&coords) - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn force_vanishes_at_minimum() {
        // Minimum of 2u^9 - 3u^6 is at u=1 (r=sigma); force is zero there.
        let pot = PairLJClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (_, f) = pot.calc_energy_forces(&coords);
        for fi in f {
            assert!(fi.abs() < 1e-9, "force {fi}");
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairLJClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
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
        let pot = PairLJClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.calc_energy_forces(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
