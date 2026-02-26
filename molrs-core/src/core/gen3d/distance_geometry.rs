//! Distance-geometry embedding backend (stage-1).

use std::collections::{HashMap, VecDeque};

use rand::Rng;

use super::builder::BuildSummary;
use super::geom::{add, dot, norm, random_unit, scale, sub};
use crate::core::element::Element;
use crate::core::molgraph::{AtomId, MolGraph};
use crate::error::MolRsError;

/// Embed initial 3D coordinates with a distance-geometry style workflow:
/// bounds matrix -> bounds smoothing -> random distance realization ->
/// stress minimization in 3D.
pub(crate) fn embed_distance_geometry(
    mol: &mut MolGraph,
    rng: &mut impl Rng,
) -> Result<BuildSummary, MolRsError> {
    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    if atom_ids.is_empty() {
        return Err(MolRsError::validation(
            "cannot build 3D coordinates for empty molecule",
        ));
    }
    let n = atom_ids.len();
    let id_to_idx: HashMap<AtomId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let adjacency = build_adjacency(mol, &atom_ids, &id_to_idx);
    let topo_dist = topological_distances(&adjacency);
    let cov_r = atom_ids
        .iter()
        .map(|&id| covalent_radius(mol, id))
        .collect::<Vec<_>>();
    let vdw_r = atom_ids
        .iter()
        .map(|&id| vdw_radius(mol, id))
        .collect::<Vec<_>>();

    let mut lower = vec![vec![0.0_f64; n]; n];
    let mut upper = vec![vec![1.0e6_f64; n]; n];
    for i in 0..n {
        upper[i][i] = 0.0;
    }

    // Seed bounds from topology and element-based heuristics.
    for i in 0..n {
        for j in (i + 1)..n {
            let ai = atom_ids[i];
            let aj = atom_ids[j];
            let dij = topo_dist[i][j];
            let base = cov_r[i] + cov_r[j];
            let bonded = bond_order_between(mol, ai, aj);

            if bonded > 0.0 {
                let r0 = ideal_bond_length(mol, ai, aj);
                lower[i][j] = (r0 - 0.06).max(0.4);
                upper[i][j] = r0 + 0.06;
            } else if dij == 2 {
                lower[i][j] = (base + 0.35).max(1.1);
                upper[i][j] = (base + 1.55).max(lower[i][j] + 0.3);
            } else if dij == 3 {
                lower[i][j] = (base + 0.55).max(1.4);
                upper[i][j] = (base + 2.35).max(lower[i][j] + 0.4);
            } else if dij < usize::MAX {
                lower[i][j] = (0.35 * (vdw_r[i] + vdw_r[j])).max(1.2);
                upper[i][j] = (1.7 * dij as f64 + 0.9).max(lower[i][j] + 0.5);
            } else {
                // Disconnected components.
                lower[i][j] = (0.30 * (vdw_r[i] + vdw_r[j])).max(1.0);
                upper[i][j] = (2.0 * (n as f64).sqrt() + 6.0).max(lower[i][j] + 1.0);
            }

            lower[j][i] = lower[i][j];
            upper[j][i] = upper[i][j];
        }
    }

    smooth_bounds(&mut lower, &mut upper, 3);

    let mut warnings = Vec::new();
    let mut inconsistent = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if upper[i][j] < lower[i][j] + 1.0e-6 {
                inconsistent += 1;
                let mid = 0.5 * (upper[i][j] + lower[i][j]);
                lower[i][j] = (mid - 0.005).max(0.0);
                upper[i][j] = mid + 0.005;
                lower[j][i] = lower[i][j];
                upper[j][i] = upper[i][j];
            }
        }
    }
    if inconsistent > 0 {
        warnings.push(format!(
            "distance-geometry bounds had {} inconsistent pair(s); softened during smoothing",
            inconsistent
        ));
    }

    let target = sample_distance_matrix(&lower, &upper, rng);
    let mut coords = random_init_coords(n, rng);
    optimize_stress(&mut coords, &target, &lower);
    recenter(&mut coords);

    for (i, atom_id) in atom_ids.iter().copied().enumerate() {
        let atom = mol.get_atom_mut(atom_id)?;
        atom.set("x", coords[i][0]);
        atom.set("y", coords[i][1]);
        atom.set("z", coords[i][2]);
    }

    Ok(BuildSummary {
        placed_atoms: n,
        warnings,
    })
}

fn build_adjacency(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); atom_ids.len()];
    for (i, &aid) in atom_ids.iter().enumerate() {
        let mut nlist = mol
            .neighbors(aid)
            .filter_map(|nid| id_to_idx.get(&nid).copied())
            .collect::<Vec<_>>();
        nlist.sort_unstable();
        adj[i] = nlist;
    }
    adj
}

fn topological_distances(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut dist = vec![vec![usize::MAX; n]; n];
    for start in 0..n {
        let mut q = VecDeque::new();
        dist[start][start] = 0;
        q.push_back(start);
        while let Some(i) = q.pop_front() {
            let d = dist[start][i];
            for &j in &adjacency[i] {
                if dist[start][j] == usize::MAX {
                    dist[start][j] = d + 1;
                    q.push_back(j);
                }
            }
        }
    }
    dist
}

fn smooth_bounds(lower: &mut [Vec<f64>], upper: &mut [Vec<f64>], rounds: usize) {
    let n = lower.len();
    for _ in 0..rounds {
        for k in 0..n {
            for i in 0..n {
                if i == k {
                    continue;
                }
                for j in (i + 1)..n {
                    if j == k {
                        continue;
                    }
                    let u = upper[i][k] + upper[k][j];
                    if u < upper[i][j] {
                        upper[i][j] = u;
                        upper[j][i] = u;
                    }
                }
            }
        }
        for k in 0..n {
            for i in 0..n {
                if i == k {
                    continue;
                }
                for j in (i + 1)..n {
                    if j == k {
                        continue;
                    }
                    let l = (lower[i][k] - upper[k][j]).max(lower[k][j] - upper[i][k]);
                    if l > lower[i][j] {
                        lower[i][j] = l;
                        lower[j][i] = l;
                    }
                }
            }
        }
    }
}

fn sample_distance_matrix(
    lower: &[Vec<f64>],
    upper: &[Vec<f64>],
    rng: &mut impl Rng,
) -> Vec<Vec<f64>> {
    let n = lower.len();
    let mut d = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let lo = lower[i][j].max(1.0e-5);
            let hi = upper[i][j].max(lo + 1.0e-5);
            let v = if (hi - lo) < 1.0e-8 {
                lo
            } else {
                rng.random_range(lo..hi)
            };
            d[i][j] = v;
            d[j][i] = v;
        }
    }
    d
}

fn random_init_coords(n: usize, rng: &mut impl Rng) -> Vec<[f64; 3]> {
    let mut coords = Vec::with_capacity(n);
    for _ in 0..n {
        let r = rng.random_range(0.6..1.8);
        coords.push(scale(random_unit(rng), r));
    }
    coords
}

fn optimize_stress(coords: &mut [[f64; 3]], target: &[Vec<f64>], lower: &[Vec<f64>]) {
    let n = coords.len();
    if n < 2 {
        return;
    }

    let mut grad = vec![[0.0_f64; 3]; n];
    let mut trial = coords.to_vec();
    let mut step = 0.04_f64;
    let mut energy = stress_energy_grad(coords, target, lower, &mut grad);

    for _ in 0..900 {
        let rms = (grad.iter().map(|g| dot(*g, *g)).sum::<f64>() / n as f64).sqrt();
        if rms < 1.0e-4 {
            break;
        }

        for i in 0..n {
            trial[i] = sub(coords[i], scale(grad[i], step));
        }

        let e_trial = stress_energy(&trial, target, lower);
        if e_trial < energy {
            coords.copy_from_slice(&trial);
            energy = e_trial;
            let _ = stress_energy_grad(coords, target, lower, &mut grad);
            step = (step * 1.05).min(0.12);
        } else {
            step *= 0.5;
            if step < 1.0e-5 {
                break;
            }
        }
    }
}

fn stress_energy(coords: &[[f64; 3]], target: &[Vec<f64>], lower: &[Vec<f64>]) -> f64 {
    let n = coords.len();
    let mut e = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let rij = sub(coords[i], coords[j]);
            let d = norm(rij).max(1.0e-8);
            let t = target[i][j].max(1.0e-6);
            let w = 1.0 / (t * t);
            let err = d - t;
            e += w * err * err;

            // Keep a soft lower-bound penalty to avoid collapse in hard cases.
            let lo = lower[i][j];
            if d < lo {
                let o = lo - d;
                e += 0.2 * o * o;
            }
        }
    }
    e
}

fn stress_energy_grad(
    coords: &[[f64; 3]],
    target: &[Vec<f64>],
    lower: &[Vec<f64>],
    grad: &mut [[f64; 3]],
) -> f64 {
    for g in grad.iter_mut() {
        *g = [0.0, 0.0, 0.0];
    }

    let n = coords.len();
    let mut e = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let rij = sub(coords[i], coords[j]);
            let d = norm(rij).max(1.0e-8);
            let inv_d = 1.0 / d;
            let t = target[i][j].max(1.0e-6);
            let w = 1.0 / (t * t);
            let err = d - t;
            e += w * err * err;
            let gscale = 2.0 * w * err * inv_d;
            let g = scale(rij, gscale);
            grad[i] = add(grad[i], g);
            grad[j] = sub(grad[j], g);

            let lo = lower[i][j];
            if d < lo {
                let o = lo - d;
                e += 0.2 * o * o;
                let gs2 = -0.4 * o * inv_d;
                let g2 = scale(rij, gs2);
                grad[i] = add(grad[i], g2);
                grad[j] = sub(grad[j], g2);
            }
        }
    }
    e
}

fn recenter(coords: &mut [[f64; 3]]) {
    if coords.is_empty() {
        return;
    }
    let n = coords.len() as f64;
    let c = [
        coords.iter().map(|p| p[0]).sum::<f64>() / n,
        coords.iter().map(|p| p[1]).sum::<f64>() / n,
        coords.iter().map(|p| p[2]).sum::<f64>() / n,
    ];
    for p in coords {
        p[0] -= c[0];
        p[1] -= c[1];
        p[2] -= c[2];
    }
}

fn covalent_radius(mol: &MolGraph, atom_id: AtomId) -> f64 {
    mol.get_atom(atom_id)
        .ok()
        .and_then(|a| a.get_str("symbol"))
        .and_then(Element::by_symbol)
        .map(|e| e.covalent_radius() as f64)
        .unwrap_or(0.77)
}

fn vdw_radius(mol: &MolGraph, atom_id: AtomId) -> f64 {
    mol.get_atom(atom_id)
        .ok()
        .and_then(|a| a.get_str("symbol"))
        .and_then(Element::by_symbol)
        .map(|e| e.vdw_radius() as f64)
        .unwrap_or(1.8)
}

fn ideal_bond_length(mol: &MolGraph, a: AtomId, b: AtomId) -> f64 {
    let r_a = covalent_radius(mol, a);
    let r_b = covalent_radius(mol, b);
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
        .unwrap_or(0.0)
}
