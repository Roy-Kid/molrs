//! Built-in potential kernel implementations.
//!
//! Each kernel implements [`Potential`] and stores pre-bound atom indices in SoA
//! layout. Constructors resolve type labels → parameters at construction time,
//! so the hot loop does zero HashMap lookups.

use std::collections::HashMap;

use super::forcefield::Params;
use super::frame::Frame;
use super::potential::Potential;
use crate::core::types::F;

// ---------------------------------------------------------------------------
// BondHarmonic: E = 0.5 * k * (r - r0)^2
// ---------------------------------------------------------------------------

/// Harmonic bond potential with pre-bound SoA data.
pub struct BondHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    k: Vec<F>,
    r0: Vec<F>,
}

impl BondHarmonic {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, k: Vec<F>, r0: Vec<F>) -> Self {
        Self {
            atom_i,
            atom_j,
            k,
            r0,
        }
    }
}

impl Potential for BondHarmonic {
    fn energy(&self, coords: &[F]) -> F {
        let mut e: F = 0.0;
        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            let k = self.k[idx];
            let r0 = self.r0[idx];
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let dr = r - r0;
            e += 0.5 * k * dr * dr;
        }
        e
    }

    fn forces(&self, coords: &[F]) -> Vec<F> {
        let mut forces = vec![0.0; coords.len()];
        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            let k = self.k[idx];
            let r0 = self.r0[idx];
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-12 {
                continue;
            }
            // force = -gradient: factor sign is flipped vs old gradient
            let factor = -k * (r - r0) / r;
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
        forces
    }
}

pub fn bond_harmonic_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let mut param_map = HashMap::new();
    for &(name, p) in type_params {
        let k = p
            .get("k")
            .ok_or_else(|| format!("BondHarmonic type '{}': missing 'k'", name))?
            as F;
        let r0 = p
            .get("r0")
            .ok_or_else(|| format!("BondHarmonic type '{}': missing 'r0'", name))?
            as F;
        param_map.insert(name.to_owned(), (k, r0));
    }

    let mut atom_i = Vec::new();
    let mut atom_j = Vec::new();
    let mut k_vec = Vec::new();
    let mut r0_vec = Vec::new();

    if let Some(block) = frame.get("bonds")
        && let (Some(i_col), Some(j_col), Some(type_col)) = (
            block.get_u32("i"),
            block.get_u32("j"),
            block.get_string("type"),
        )
    {
        for idx in 0..i_col.len() {
            let label = &type_col[idx];
            if let Some(&(k, r0)) = param_map.get(label.as_str()) {
                atom_i.push(i_col[idx] as usize);
                atom_j.push(j_col[idx] as usize);
                k_vec.push(k);
                r0_vec.push(r0);
            }
        }
    }

    Ok(Box::new(BondHarmonic::new(atom_i, atom_j, k_vec, r0_vec)))
}

// ---------------------------------------------------------------------------
// AngleHarmonic: E = 0.5 * k * (theta - theta0)^2
// ---------------------------------------------------------------------------

/// Harmonic angle potential with pre-bound SoA data.
/// `theta0` is stored in **radians**.
pub struct AngleHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    k: Vec<F>,
    theta0: Vec<F>,
}

impl AngleHarmonic {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        atom_k: Vec<usize>,
        k: Vec<F>,
        theta0: Vec<F>,
    ) -> Self {
        Self {
            atom_i,
            atom_j,
            atom_k,
            k,
            theta0,
        }
    }
}

impl Potential for AngleHarmonic {
    fn energy(&self, coords: &[F]) -> F {
        let mut e: F = 0.0;
        for idx in 0..self.atom_i.len() {
            let theta = compute_angle(coords, self.atom_i[idx], self.atom_j[idx], self.atom_k[idx]);
            let dtheta = theta - self.theta0[idx];
            e += 0.5 * self.k[idx] * dtheta * dtheta;
        }
        e
    }

    fn forces(&self, coords: &[F]) -> Vec<F> {
        let mut forces = vec![0.0; coords.len()];
        for idx in 0..self.atom_i.len() {
            angle_forces(
                coords,
                self.atom_i[idx],
                self.atom_j[idx],
                self.atom_k[idx],
                self.k[idx],
                self.theta0[idx],
                &mut forces,
            );
        }
        forces
    }
}

/// Compute angle i-j-k in radians.
fn compute_angle(coords: &[F], i: usize, j: usize, k: usize) -> F {
    let rji = [
        coords[i * 3] - coords[j * 3],
        coords[i * 3 + 1] - coords[j * 3 + 1],
        coords[i * 3 + 2] - coords[j * 3 + 2],
    ];
    let rjk = [
        coords[k * 3] - coords[j * 3],
        coords[k * 3 + 1] - coords[j * 3 + 1],
        coords[k * 3 + 2] - coords[j * 3 + 2],
    ];
    let dot = rji[0] * rjk[0] + rji[1] * rjk[1] + rji[2] * rjk[2];
    let mag_ji = (rji[0] * rji[0] + rji[1] * rji[1] + rji[2] * rji[2]).sqrt();
    let mag_jk = (rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2]).sqrt();
    let cos_theta = (dot / (mag_ji * mag_jk)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

/// Accumulate angle forces for angle i-j-k with harmonic potential.
/// Forces = −gradient, so the sign of the prefactor is flipped.
fn angle_forces(
    coords: &[F],
    i: usize,
    j: usize,
    k: usize,
    k_spring: F,
    theta0: F,
    forces: &mut [F],
) {
    let rji = [
        coords[i * 3] - coords[j * 3],
        coords[i * 3 + 1] - coords[j * 3 + 1],
        coords[i * 3 + 2] - coords[j * 3 + 2],
    ];
    let rjk = [
        coords[k * 3] - coords[j * 3],
        coords[k * 3 + 1] - coords[j * 3 + 1],
        coords[k * 3 + 2] - coords[j * 3 + 2],
    ];

    let d_ji_sq = rji[0] * rji[0] + rji[1] * rji[1] + rji[2] * rji[2];
    let d_jk_sq = rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2];
    let d_ji = d_ji_sq.sqrt();
    let d_jk = d_jk_sq.sqrt();

    if d_ji < 1e-12 || d_jk < 1e-12 {
        return;
    }

    let dot = rji[0] * rjk[0] + rji[1] * rjk[1] + rji[2] * rjk[2];
    let cos_theta = (dot / (d_ji * d_jk)).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1e-12);

    let de_dtheta = k_spring * (theta - theta0);
    // force prefactor = +de_dtheta / sin_theta (sign flipped from gradient)
    let prefactor = de_dtheta / sin_theta;

    for dim in 0..3 {
        let dcos_dri = rjk[dim] / (d_ji * d_jk) - cos_theta * rji[dim] / d_ji_sq;
        let dcos_drk = rji[dim] / (d_ji * d_jk) - cos_theta * rjk[dim] / d_jk_sq;
        let dcos_drj = -dcos_dri - dcos_drk;

        forces[i * 3 + dim] += prefactor * dcos_dri;
        forces[k * 3 + dim] += prefactor * dcos_drk;
        forces[j * 3 + dim] += prefactor * dcos_drj;
    }
}

pub fn angle_harmonic_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let mut param_map = HashMap::new();
    for &(name, p) in type_params {
        let k = p
            .get("k")
            .ok_or_else(|| format!("AngleHarmonic type '{}': missing 'k'", name))?
            as F;
        let theta0_rad = p
            .get("theta0")
            .ok_or_else(|| format!("AngleHarmonic type '{}': missing 'theta0'", name))?
            .to_radians() as F;
        param_map.insert(name.to_owned(), (k, theta0_rad));
    }

    let mut atom_i = Vec::new();
    let mut atom_j = Vec::new();
    let mut atom_k = Vec::new();
    let mut k_vec = Vec::new();
    let mut theta0_vec = Vec::new();

    if let Some(block) = frame.get("angles")
        && let (Some(i_col), Some(j_col), Some(k_col), Some(type_col)) = (
            block.get_u32("i"),
            block.get_u32("j"),
            block.get_u32("k"),
            block.get_string("type"),
        )
    {
        for idx in 0..i_col.len() {
            let label = &type_col[idx];
            if let Some(&(k, theta0)) = param_map.get(label.as_str()) {
                atom_i.push(i_col[idx] as usize);
                atom_j.push(j_col[idx] as usize);
                atom_k.push(k_col[idx] as usize);
                k_vec.push(k);
                theta0_vec.push(theta0);
            }
        }
    }

    Ok(Box::new(AngleHarmonic::new(
        atom_i, atom_j, atom_k, k_vec, theta0_vec,
    )))
}

// ---------------------------------------------------------------------------
// PairLJ126: E = 4 * eps * ((sigma/r)^12 - (sigma/r)^6)
// ---------------------------------------------------------------------------

/// Lennard-Jones 12-6 pair potential with pre-bound SoA data.
pub struct PairLJ126 {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    epsilon: Vec<F>,
    sigma: Vec<F>,
}

impl PairLJ126 {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, epsilon: Vec<F>, sigma: Vec<F>) -> Self {
        Self {
            atom_i,
            atom_j,
            epsilon,
            sigma,
        }
    }
}

impl Potential for PairLJ126 {
    fn energy(&self, coords: &[F]) -> F {
        let mut e: F = 0.0;
        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            let eps = self.epsilon[idx];
            let sigma = self.sigma[idx];
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let sr2 = sigma * sigma / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            e += 4.0 * eps * (sr12 - sr6);
        }
        e
    }

    fn forces(&self, coords: &[F]) -> Vec<F> {
        let n = coords.len();
        let n_pairs = self.atom_i.len();

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let chunk_size = (n_pairs / rayon::current_num_threads()).max(64);
            let indices: Vec<usize> = (0..n_pairs).collect();

            let partials: Vec<Vec<F>> = indices
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut f = vec![0.0; n];
                    for &idx in chunk {
                        Self::lj126_force_pair(
                            coords,
                            self.atom_i[idx],
                            self.atom_j[idx],
                            self.epsilon[idx],
                            self.sigma[idx],
                            &mut f,
                        );
                    }
                    f
                })
                .collect();

            let mut forces = vec![0.0; n];
            for pf in &partials {
                for (f, p) in forces.iter_mut().zip(pf.iter()) {
                    *f += p;
                }
            }
            forces
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut forces = vec![0.0; n];
            for idx in 0..n_pairs {
                Self::lj126_force_pair(
                    coords,
                    self.atom_i[idx],
                    self.atom_j[idx],
                    self.epsilon[idx],
                    self.sigma[idx],
                    &mut forces,
                );
            }
            forces
        }
    }
}

impl PairLJ126 {
    /// Compute and accumulate force for a single i-j pair.
    #[inline]
    fn lj126_force_pair(coords: &[F], i: usize, j: usize, eps: F, sigma: F, forces: &mut [F]) {
        let dx = coords[j * 3] - coords[i * 3];
        let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
        let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 < 1e-24 {
            return;
        }
        let sr2 = sigma * sigma / r2;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
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
}

pub fn pair_lj126_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let mut param_map = HashMap::new();
    for &(name, p) in type_params {
        let eps = p
            .get("epsilon")
            .ok_or_else(|| format!("PairLJ126 type '{}': missing 'epsilon'", name))?
            as F;
        let sigma = p
            .get("sigma")
            .ok_or_else(|| format!("PairLJ126 type '{}': missing 'sigma'", name))?
            as F;
        param_map.insert(name.to_owned(), (eps, sigma));
    }

    let mut atom_i = Vec::new();
    let mut atom_j = Vec::new();
    let mut eps_vec = Vec::new();
    let mut sigma_vec = Vec::new();

    if let Some(block) = frame.get("pairs")
        && let (Some(i_col), Some(j_col), Some(type_col)) = (
            block.get_u32("i"),
            block.get_u32("j"),
            block.get_string("type"),
        )
    {
        for idx in 0..i_col.len() {
            let label = &type_col[idx];
            if let Some(&(eps, sigma)) = param_map.get(label.as_str()) {
                atom_i.push(i_col[idx] as usize);
                atom_j.push(j_col[idx] as usize);
                eps_vec.push(eps);
                sigma_vec.push(sigma);
            }
        }
    }

    Ok(Box::new(PairLJ126::new(atom_i, atom_j, eps_vec, sigma_vec)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- BondHarmonic tests ---

    #[test]
    fn test_bond_harmonic_energy() {
        let pot = BondHarmonic::new(vec![0], vec![1], vec![300.0], vec![1.5]);

        // Two atoms at distance 2.0, r0=1.5, k=300
        // E = 0.5 * 300 * (2.0 - 1.5)^2 = 37.5
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let e = pot.energy(&coords);
        assert!((e - 37.5).abs() < 1e-3, "energy = {}", e);
    }

    #[test]
    fn test_bond_harmonic_forces() {
        let pot = BondHarmonic::new(vec![0], vec![1], vec![300.0], vec![1.5]);

        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let forces = pot.forces(&coords);

        // forces = -gradient: atom 0 pulled toward atom 1, atom 1 pulled toward atom 0
        assert!(
            (forces[0] - 150.0).abs() < 1e-3,
            "forces[0] = {}",
            forces[0]
        );
        assert!(
            (forces[3] - (-150.0)).abs() < 1e-3,
            "forces[3] = {}",
            forces[3]
        );

        // Verify numerically: forces = -dE/dx
        let eps: F = 1e-4;
        let mut coords_p = coords.clone();
        coords_p[3] += eps;
        let e_p = pot.energy(&coords_p);
        let mut coords_m = coords.clone();
        coords_m[3] -= eps;
        let e_m = pot.energy(&coords_m);
        let numerical_force = -(e_p - e_m) / (2.0 * eps);
        assert!(
            (forces[3] - numerical_force).abs() < 1.0,
            "analytical={}, numerical={}",
            forces[3],
            numerical_force
        );
    }

    #[test]
    fn test_bond_harmonic_at_equilibrium() {
        let pot = BondHarmonic::new(vec![0], vec![1], vec![300.0], vec![1.5]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        assert!(pot.energy(&coords).abs() < 1e-5);

        let forces = pot.forces(&coords);
        for f in &forces {
            assert!(f.abs() < 1e-5);
        }
    }

    #[test]
    fn test_bond_harmonic_multiple_bonds() {
        // 3 atoms, 2 bonds: 0-1 (k=300, r0=1.5) and 1-2 (k=200, r0=1.0)
        let pot = BondHarmonic::new(vec![0, 1], vec![1, 2], vec![300.0, 200.0], vec![1.5, 1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.5, 0.0, 0.0];
        let energy = pot.energy(&coords);
        // E1 = 0.5*300*(2.0-1.5)^2 = 37.5
        // E2 = 0.5*200*(1.5-1.0)^2 = 25.0
        assert!((energy - 62.5).abs() < 1e-3, "expected 62.5, got {energy}",);
    }

    // --- AngleHarmonic tests ---

    #[test]
    fn test_angle_harmonic_energy() {
        let theta0: F = std::f64::consts::FRAC_PI_2 as F;
        let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![50.0], vec![theta0]);

        // Three atoms forming a 90-degree angle at j: i=(1,0,0), j=(0,0,0), k=(0,1,0)
        let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let e = pot.energy(&coords);
        assert!(e.abs() < 1e-4, "energy = {} (expected 0)", e);

        // Still 90 degrees with k=(0,0,1)
        let coords2: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let e2 = pot.energy(&coords2);
        assert!(e2.abs() < 1e-4);

        // Atom 2 at (1, 1, 0): angle = 45 degrees = pi/4
        // E = 0.5 * 50 * (pi/4 - pi/2)^2 = 25 * (pi/4)^2
        let coords3: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let e3 = pot.energy(&coords3);
        let expected: F =
            25.0 * (std::f64::consts::FRAC_PI_4 as F) * (std::f64::consts::FRAC_PI_4 as F);
        assert!(
            (e3 - expected).abs() < 1e-3,
            "energy = {} (expected {})",
            e3,
            expected
        );
    }

    #[test]
    fn test_angle_harmonic_forces_numerical() {
        let theta0: F = (100.0_f64).to_radians() as F;
        let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![60.0], vec![theta0]);

        let coords: Vec<F> = vec![1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.1];
        let forces = pot.forces(&coords);

        let eps: F = 1e-3;
        for idx in 0..9 {
            let mut cp = coords.clone();
            cp[idx] += eps;
            let ep = pot.energy(&cp);
            let mut cm = coords.clone();
            cm[idx] -= eps;
            let em = pot.energy(&cm);
            let numerical_force = -(ep - em) / (2.0 * eps);
            assert!(
                (forces[idx] - numerical_force).abs() < 1.0,
                "idx={}: analytical={}, numerical={}",
                idx,
                forces[idx],
                numerical_force
            );
        }
    }

    #[test]
    fn test_angle_harmonic_forces_translational_invariance() {
        let theta0: F = (100.0_f64).to_radians() as F;
        let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![60.0], vec![theta0]);
        let coords: Vec<F> = vec![1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.1];
        let forces = pot.forces(&coords);

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim] + forces[6 + dim];
            assert!(sum.abs() < 1e-3, "dim={dim}: total force = {sum}",);
        }
    }

    #[test]
    fn test_angle_harmonic_forces_at_equilibrium() {
        let theta0: F = std::f64::consts::FRAC_PI_2 as F;
        let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![50.0], vec![theta0]);
        let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let forces = pot.forces(&coords);

        for (i, f) in forces.iter().enumerate() {
            assert!(
                f.abs() < 1e-4,
                "forces[{i}]={f}, expected ~0 at equilibrium"
            );
        }
    }

    // --- PairLJ126 tests ---

    #[test]
    fn test_pair_lj126_energy() {
        let pot = PairLJ126::new(vec![0], vec![1], vec![0.5], vec![1.0]);

        // At r = sigma: E = 4 * eps * (1 - 1) = 0
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let e = pot.energy(&coords);
        assert!(e.abs() < 1e-5, "energy at r=sigma should be 0, got {}", e);

        // At r = 2^(1/6) * sigma (minimum): E = -eps = -0.5
        let r_min: F = (2.0_f64).powf(1.0 / 6.0) as F;
        let coords2: Vec<F> = vec![0.0, 0.0, 0.0, r_min, 0.0, 0.0];
        let e2 = pot.energy(&coords2);
        assert!(
            (e2 - (-0.5)).abs() < 1e-4,
            "energy at r_min should be -eps=-0.5, got {}",
            e2
        );
    }

    #[test]
    fn test_pair_lj126_forces_numerical() {
        let pot = PairLJ126::new(vec![0], vec![1], vec![0.5], vec![1.0]);

        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];
        let forces = pot.forces(&coords);

        let eps: F = 1e-3;
        for idx in 0..6 {
            let mut cp = coords.clone();
            cp[idx] += eps;
            let ep = pot.energy(&cp);
            let mut cm = coords.clone();
            cm[idx] -= eps;
            let em = pot.energy(&cm);
            let numerical_force = -(ep - em) / (2.0 * eps);
            assert!(
                (forces[idx] - numerical_force).abs() < 1.0,
                "idx={}: analytical={}, numerical={}",
                idx,
                forces[idx],
                numerical_force
            );
        }
    }

    #[test]
    fn test_pair_lj126_newton_third_law() {
        let pot = PairLJ126::new(vec![0], vec![1], vec![0.5], vec![1.0]);

        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];
        let forces = pot.forces(&coords);

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5, "dim={}: sum={}", dim, sum);
        }
    }

    #[test]
    fn test_pair_lj126_forces_zero_at_minimum() {
        let r_min: F = (2.0_f64).powf(1.0 / 6.0) as F;
        let pot = PairLJ126::new(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, r_min, 0.0, 0.0];
        let forces = pot.forces(&coords);

        for (i, f) in forces.iter().enumerate() {
            assert!(f.abs() < 1e-3, "forces[{i}]={f}, expected ~0 at minimum");
        }
    }

    #[test]
    fn test_pair_lj126_multiple_pairs() {
        // 3 atoms, 2 pairs: 0-1 and 0-2
        let pot = PairLJ126::new(vec![0, 0], vec![1, 2], vec![1.0, 0.5], vec![1.0, 1.5]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let energy = pot.energy(&coords);

        let p1 = PairLJ126::new(vec![0], vec![1], vec![1.0], vec![1.0]);
        let p2 = PairLJ126::new(vec![0], vec![1], vec![0.5], vec![1.5]);
        let e1 = p1.energy(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let e2 = p2.energy(&[0.0, 0.0, 0.0, 0.0, 3.0, 0.0]);

        assert!(
            (energy - (e1 + e2)).abs() < 1e-4,
            "total={energy}, sum={}",
            e1 + e2,
        );
    }

    // --- ForceField integration ---

    #[test]
    fn test_forcefield_to_potentials() {
        use super::super::block::Block;
        use super::super::frame::Frame;
        use ndarray::Array1;

        let mut ff = super::super::forcefield::ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style.def_bondtype("CT", "CT", &[("k", 300.0), ("r0", 1.5)]);

        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0_f64, 2.0]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![0.0_f64, 0.0]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![0.0_f64, 0.0]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        let mut bonds = Block::new();
        bonds
            .insert("i", Array1::from_vec(vec![0_u32]).into_dyn())
            .unwrap();
        bonds
            .insert("j", Array1::from_vec(vec![1_u32]).into_dyn())
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from_vec(vec!["CT-CT".to_string()]).into_dyn(),
            )
            .unwrap();
        frame.insert("bonds", bonds);

        let pots = ff.to_potentials_from_frame(&frame).unwrap();
        assert_eq!(pots.len(), 1);

        let coords = super::super::potential::extract_coords(&frame).unwrap();
        let e = pots.energy(&coords);
        assert!((e - 37.5).abs() < 1e-3);
    }
}
