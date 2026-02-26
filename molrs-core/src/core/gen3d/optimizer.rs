//! Lightweight geometry optimizer used by the Gen3D pipeline.

use std::collections::{HashMap, HashSet};

use super::geom::{add, dot, norm, normalize, scale, sub};
use crate::core::element::Element;
use crate::core::molgraph::{AtomId, MolGraph};
use crate::error::MolRsError;

#[derive(Debug, Clone, Copy)]
struct SpringTerm {
    i: usize,
    j: usize,
    r0: f64,
    k: f64,
}

#[derive(Debug, Clone, Copy)]
struct PairRepulsion {
    i: usize,
    j: usize,
    sigma: f64,
    k: f64,
}

/// Internal energy model for Gen3D minimization.
#[derive(Debug, Clone)]
pub(crate) struct EnergyModel {
    atom_ids: Vec<AtomId>,
    bond_terms: Vec<SpringTerm>,
    angle_terms: Vec<SpringTerm>,
    pair_terms: Vec<PairRepulsion>,
}

/// Result of a minimization run.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MinResult {
    pub steps: usize,
    pub converged: bool,
    pub energy: f64,
}

impl EnergyModel {
    /// Build an optimization model from molecular topology and elements.
    pub(crate) fn from_mol(mol: &MolGraph) -> Self {
        let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
        let id_to_idx: HashMap<AtomId, usize> = atom_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let mut bond_terms = Vec::new();
        let mut excluded_pairs: HashSet<(usize, usize)> = HashSet::new();
        let mut vdw = vec![1.7_f64; atom_ids.len()];

        for (i, atom_id) in atom_ids.iter().copied().enumerate() {
            let radius = mol
                .get_atom(atom_id)
                .ok()
                .and_then(|a| a.get_str("symbol"))
                .and_then(Element::by_symbol)
                .map(|e| e.vdw_radius() as f64)
                .unwrap_or(1.7);
            vdw[i] = radius;
        }

        for (_, bond) in mol.bonds() {
            let ai = bond.atoms[0];
            let aj = bond.atoms[1];
            let Some(&i) = id_to_idx.get(&ai) else {
                continue;
            };
            let Some(&j) = id_to_idx.get(&aj) else {
                continue;
            };
            let order = bond_order_between(mol, ai, aj);
            let r0 = ideal_bond_length(mol, ai, aj);
            let k = 160.0 * order.clamp(1.0, 3.0);
            bond_terms.push(SpringTerm { i, j, r0, k });
            excluded_pairs.insert(minmax(i, j));
        }

        let mut angle_terms = Vec::new();
        for center in atom_ids.iter().copied() {
            let mut neigh: Vec<usize> = mol
                .neighbors(center)
                .filter_map(|nid| id_to_idx.get(&nid).copied())
                .collect();
            neigh.sort_unstable();
            if neigh.len() < 2 {
                continue;
            }
            let theta0 = ideal_angle_for_center(mol, center);
            for a in 0..neigh.len() {
                for b in (a + 1)..neigh.len() {
                    let i = neigh[a];
                    let k = neigh[b];
                    excluded_pairs.insert(minmax(i, k));
                    let r1 = ideal_bond_length(mol, atom_ids[i], center);
                    let r2 = ideal_bond_length(mol, center, atom_ids[k]);
                    let d0 = (r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * theta0.cos())
                        .max(0.1)
                        .sqrt();
                    angle_terms.push(SpringTerm {
                        i,
                        j: k,
                        r0: d0,
                        k: 18.0,
                    });
                }
            }
        }

        let mut pair_terms = Vec::new();
        for i in 0..atom_ids.len() {
            for j in (i + 1)..atom_ids.len() {
                if excluded_pairs.contains(&minmax(i, j)) {
                    continue;
                }
                let sigma = 0.70 * (vdw[i] + vdw[j]);
                pair_terms.push(PairRepulsion {
                    i,
                    j,
                    sigma,
                    k: 0.015,
                });
            }
        }

        Self {
            atom_ids,
            bond_terms,
            angle_terms,
            pair_terms,
        }
    }

    pub(crate) fn read_coords_from_mol(&self, mol: &MolGraph) -> Vec<[f64; 3]> {
        let mut coords = vec![[0.0; 3]; self.atom_ids.len()];
        for (i, atom_id) in self.atom_ids.iter().copied().enumerate() {
            if let Ok(atom) = mol.get_atom(atom_id) {
                coords[i] = [
                    atom.get_f64("x").unwrap_or(0.0),
                    atom.get_f64("y").unwrap_or(0.0),
                    atom.get_f64("z").unwrap_or(0.0),
                ];
            }
        }
        coords
    }

    pub(crate) fn write_coords_to_mol(
        &self,
        mol: &mut MolGraph,
        coords: &[[f64; 3]],
    ) -> Result<(), MolRsError> {
        for (i, atom_id) in self.atom_ids.iter().copied().enumerate() {
            let atom = mol.get_atom_mut(atom_id)?;
            atom.set("x", coords[i][0]);
            atom.set("y", coords[i][1]);
            atom.set("z", coords[i][2]);
        }
        Ok(())
    }

    pub(crate) fn energy(&self, coords: &[[f64; 3]]) -> f64 {
        let mut e = 0.0;
        for t in &self.bond_terms {
            e += spring_energy(*t, coords);
        }
        for t in &self.angle_terms {
            e += spring_energy(*t, coords);
        }
        for t in &self.pair_terms {
            e += repulsion_energy(*t, coords);
        }
        e
    }

    pub(crate) fn energy_and_gradient(&self, coords: &[[f64; 3]], grad: &mut [[f64; 3]]) -> f64 {
        for g in grad.iter_mut() {
            *g = [0.0, 0.0, 0.0];
        }
        let mut e = 0.0;

        for t in &self.bond_terms {
            e += spring_energy_grad(*t, coords, grad);
        }
        for t in &self.angle_terms {
            e += spring_energy_grad(*t, coords, grad);
        }
        for t in &self.pair_terms {
            e += repulsion_energy_grad(*t, coords, grad);
        }

        e
    }
}

pub(crate) fn steepest_descent(
    model: &EnergyModel,
    coords: &mut [[f64; 3]],
    max_steps: usize,
    init_step: f64,
    grad_tol: f64,
) -> MinResult {
    if coords.is_empty() {
        return MinResult {
            steps: 0,
            converged: true,
            energy: 0.0,
        };
    }

    let mut grad = vec![[0.0; 3]; coords.len()];
    let mut direction = vec![[0.0; 3]; coords.len()];
    let mut trial = coords.to_vec();
    let mut energy = model.energy_and_gradient(coords, &mut grad);

    for step in 0..max_steps {
        let rms = grad_rms(&grad);
        if rms < grad_tol {
            return MinResult {
                steps: step,
                converged: true,
                energy,
            };
        }

        for i in 0..coords.len() {
            direction[i] = scale(grad[i], -1.0);
        }

        let (accepted, e_next) =
            line_search(model, coords, &direction, &mut trial, energy, init_step, 10);
        if !accepted {
            return MinResult {
                steps: step + 1,
                converged: false,
                energy,
            };
        }
        energy = e_next;
        model.energy_and_gradient(coords, &mut grad);
    }

    MinResult {
        steps: max_steps,
        converged: grad_rms(&grad) < grad_tol,
        energy,
    }
}

pub(crate) fn conjugate_gradients(
    model: &EnergyModel,
    coords: &mut [[f64; 3]],
    max_steps: usize,
    init_step: f64,
    grad_tol: f64,
) -> MinResult {
    if coords.is_empty() {
        return MinResult {
            steps: 0,
            converged: true,
            energy: 0.0,
        };
    }

    let mut grad = vec![[0.0; 3]; coords.len()];
    let mut grad_prev = vec![[0.0; 3]; coords.len()];
    let mut direction = vec![[0.0; 3]; coords.len()];
    let mut trial = coords.to_vec();

    let mut energy = model.energy_and_gradient(coords, &mut grad);
    for i in 0..coords.len() {
        direction[i] = scale(grad[i], -1.0);
    }

    for step in 0..max_steps {
        let rms = grad_rms(&grad);
        if rms < grad_tol {
            return MinResult {
                steps: step,
                converged: true,
                energy,
            };
        }

        let (accepted, e_next) =
            line_search(model, coords, &direction, &mut trial, energy, init_step, 10);
        if !accepted {
            return MinResult {
                steps: step + 1,
                converged: false,
                energy,
            };
        }
        energy = e_next;

        grad_prev.copy_from_slice(&grad);
        model.energy_and_gradient(coords, &mut grad);

        let mut numer = 0.0;
        let mut denom = 0.0;
        for i in 0..coords.len() {
            let g = grad[i];
            let gp = grad_prev[i];
            numer += dot(g, sub(g, gp));
            denom += dot(gp, gp);
        }
        let beta = if denom > 1e-20 {
            (numer / denom).max(0.0)
        } else {
            0.0
        };

        for i in 0..coords.len() {
            direction[i] = add(scale(grad[i], -1.0), scale(direction[i], beta));
        }

        // Restart if direction is no longer downhill.
        let mut dg = 0.0;
        for i in 0..coords.len() {
            dg += dot(direction[i], grad[i]);
        }
        if dg >= 0.0 {
            for i in 0..coords.len() {
                direction[i] = scale(grad[i], -1.0);
            }
        }
    }

    MinResult {
        steps: max_steps,
        converged: grad_rms(&grad) < grad_tol,
        energy,
    }
}

fn line_search(
    model: &EnergyModel,
    coords: &mut [[f64; 3]],
    direction: &[[f64; 3]],
    trial: &mut [[f64; 3]],
    energy: f64,
    init_step: f64,
    max_backtracks: usize,
) -> (bool, f64) {
    let mut step = init_step;
    for _ in 0..max_backtracks {
        for i in 0..coords.len() {
            trial[i] = add(coords[i], scale(direction[i], step));
        }
        let e_trial = model.energy(trial);
        if e_trial < energy {
            coords.copy_from_slice(trial);
            return (true, e_trial);
        }
        step *= 0.5;
    }
    (false, energy)
}

fn grad_rms(grad: &[[f64; 3]]) -> f64 {
    if grad.is_empty() {
        return 0.0;
    }
    let s2: f64 = grad.iter().map(|g| dot(*g, *g)).sum();
    (s2 / grad.len() as f64).sqrt()
}

fn spring_energy(term: SpringTerm, coords: &[[f64; 3]]) -> f64 {
    let d = sub(coords[term.i], coords[term.j]);
    let r = norm(d).max(1e-10);
    let dr = r - term.r0;
    term.k * dr * dr
}

fn spring_energy_grad(term: SpringTerm, coords: &[[f64; 3]], grad: &mut [[f64; 3]]) -> f64 {
    let d = sub(coords[term.i], coords[term.j]);
    let r = norm(d).max(1e-10);
    let dr = r - term.r0;
    let e = term.k * dr * dr;
    let gscale = 2.0 * term.k * dr / r;
    let g = scale(d, gscale);
    grad[term.i] = add(grad[term.i], g);
    grad[term.j] = sub(grad[term.j], g);
    e
}

fn repulsion_energy(term: PairRepulsion, coords: &[[f64; 3]]) -> f64 {
    let d = sub(coords[term.i], coords[term.j]);
    let r = norm(d).max(1e-10);
    if r >= term.sigma {
        return 0.0;
    }
    let x = term.sigma / r;
    let x6 = x.powi(6);
    term.k * x6 * x6
}

fn repulsion_energy_grad(term: PairRepulsion, coords: &[[f64; 3]], grad: &mut [[f64; 3]]) -> f64 {
    let d = sub(coords[term.i], coords[term.j]);
    let r = norm(d).max(1e-10);
    if r >= term.sigma {
        return 0.0;
    }
    let x = term.sigma / r;
    let x6 = x.powi(6);
    let x12 = x6 * x6;
    let e = term.k * x12;
    let d_e_dr = -12.0 * term.k * x12 / r;
    let g = scale(normalize(d), d_e_dr);
    grad[term.i] = add(grad[term.i], g);
    grad[term.j] = sub(grad[term.j], g);
    e
}

fn minmax(i: usize, j: usize) -> (usize, usize) {
    if i < j { (i, j) } else { (j, i) }
}

fn ideal_angle_for_center(mol: &MolGraph, center: AtomId) -> f64 {
    let degree = mol.neighbors(center).count();
    let max_order = mol
        .neighbor_bonds(center)
        .map(|(_, order)| order)
        .fold(1.0_f64, f64::max);

    if max_order >= 2.5 || (degree == 2 && max_order >= 2.0) {
        std::f64::consts::PI
    } else if max_order >= 1.5 || degree == 3 {
        120_f64.to_radians()
    } else {
        109.47_f64.to_radians()
    }
}

fn ideal_bond_length(mol: &MolGraph, a: AtomId, b: AtomId) -> f64 {
    let sym_a = mol.get_atom(a).ok().and_then(|x| x.get_str("symbol"));
    let sym_b = mol.get_atom(b).ok().and_then(|x| x.get_str("symbol"));
    let r_a = sym_a
        .and_then(Element::by_symbol)
        .map(|e| e.covalent_radius() as f64)
        .unwrap_or(0.77);
    let r_b = sym_b
        .and_then(Element::by_symbol)
        .map(|e| e.covalent_radius() as f64)
        .unwrap_or(0.77);
    let order = bond_order_between(mol, a, b);

    let mut length = r_a + r_b;
    if order >= 2.5 {
        length *= 0.86;
    } else if order >= 1.5 {
        length *= 0.92;
    }
    length.clamp(0.9, 2.2)
}

fn bond_order_between(mol: &MolGraph, a: AtomId, b: AtomId) -> f64 {
    mol.neighbor_bonds(a)
        .find_map(|(nbr, order)| (nbr == b).then_some(order))
        .unwrap_or(1.0)
}
