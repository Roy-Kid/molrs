//! Morse pair potential: E = D0 * ((1 - exp(-alpha*(r-r0)))^2 - 1)
//!
//! Morse non-bonded form (note the `-1` offset vs the Morse bond, so the well
//! minimum is `-D0` at `r = r0`). Parameters per pair type: `D0`, `alpha`, `r0`.

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::potential::Potential;
use crate::potential::geometry::validate_coords;
use molrs::frame::Frame;
use molrs::types::F;

/// Morse pair potential with pre-resolved flat arrays.
pub struct PairMorse {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    d0: Vec<F>,
    alpha: Vec<F>,
    r0: Vec<F>,
}

impl PairMorse {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        d0: Vec<F>,
        alpha: Vec<F>,
        r0: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(d0.len(), n);
        assert_eq!(alpha.len(), n);
        assert_eq!(r0.len(), n);
        Self {
            atom_i,
            atom_j,
            d0,
            alpha,
            r0,
        }
    }
}

impl Potential for PairMorse {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);

            let (d0, alpha, r0) = (self.d0[idx], self.alpha[idx], self.r0[idx]);

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-24 {
                continue;
            }
            let r = r2.sqrt();
            let y = (-alpha * (r - r0)).exp();
            let one_my = 1.0 - y;
            energy += d0 * (one_my * one_my - 1.0);

            // d/dr[(1-y)^2 - 1] = 2 alpha y (1 - y)  -> dE/dr = 2 D0 alpha y (1-y)
            let dedr = 2.0 * d0 * alpha * y * one_my;
            let factor = -dedr / r;
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

/// Construct a [`PairMorse`] from style params, type params, and Frame topology.
pub fn pair_morse_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairMorse: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairMorse: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairMorse: pairs block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "PairMorse: pairs block missing \"type\" column".to_string())?;

    let (mut ai, mut aj) = (Vec::new(), Vec::new());
    let (mut dv, mut av, mut rv) = (Vec::new(), Vec::new(), Vec::new());
    let need = |p: &Params, key: &str, label: &str| -> Result<F, String> {
        p.get(key)
            .ok_or_else(|| format!("PairMorse type '{}': missing '{}'", label, key))
            .map(|v| v as F)
    };
    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let p = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("PairMorse: unknown pair type '{}'", label))?;
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        dv.push(need(p, "D0", label)?);
        av.push(need(p, "alpha", label)?);
        rv.push(need(p, "r0", label)?);
    }

    Ok(Box::new(PairMorse::new(ai, aj, dv, av, rv)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numerical_forces(pot: &PairMorse, coords: &[F]) -> Vec<F> {
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
    fn well_minimum_is_minus_d0() {
        // At r = r0: E = D0 (0 - 1) = -D0, force zero.
        let pot = PairMorse::new(vec![0], vec![1], vec![5.0], vec![1.5], vec![3.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (e, f) = pot.calc_energy_forces(&coords);
        assert!((e - (-5.0)).abs() < 1e-12, "energy {e}");
        for fi in f {
            assert!(fi.abs() < 1e-9);
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairMorse::new(vec![0], vec![1], vec![5.0], vec![1.5], vec![3.0]);
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.calc_energy_forces(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} a={} n={}",
                analytical[k],
                numerical[k]
            );
        }
    }
}
