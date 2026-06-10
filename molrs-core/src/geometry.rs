//! Geometry *systems* — free functions that transform a [`MolGraph`]'s node
//! coordinates in place.
//!
//! Under the ECS model the graph is pure data; spatial transforms are systems
//! that operate over the world, so they live here as free functions rather than
//! as methods on the data structure. Coordinates are read and written through
//! the canonical [`crate::keys`] coordinate convention — no field-name literals.

use crate::keys;
use crate::molgraph::MolGraph;

/// Translate every node that has coordinates by `delta` (nodes without a full
/// coordinate set are left untouched).
///
/// Operates directly on each dense coordinate column rather than per-node handle
/// lookups, so cost is linear in the number of atoms with one pass per axis.
pub fn translate(mol: &mut MolGraph, delta: [f64; 3]) {
    let table = mol.node_table_mut();
    for (i, key) in keys::COORDS.iter().enumerate() {
        if let Ok((data, valid)) = table.column_f64_mut(key) {
            for (row, val) in data.iter_mut().enumerate() {
                if valid.get(row) {
                    *val += delta[i];
                }
            }
        }
    }
}

/// Scale every node that has coordinates by a per-axis `factor` about an
/// optional center (defaults to the origin). Pass `[s, s, s]` for a uniform
/// scale. Nodes missing any coordinate are left untouched.
///
/// Operates directly on each dense coordinate column (one pass per axis), so
/// cost is linear in the number of atoms.
pub fn scale(mol: &mut MolGraph, factor: [f64; 3], about: Option<[f64; 3]>) {
    let origin = about.unwrap_or([0.0, 0.0, 0.0]);
    let table = mol.node_table_mut();
    for (i, key) in keys::COORDS.iter().enumerate() {
        if let Ok((data, valid)) = table.column_f64_mut(key) {
            for (row, val) in data.iter_mut().enumerate() {
                if valid.get(row) {
                    *val = (*val - origin[i]) * factor[i] + origin[i];
                }
            }
        }
    }
}

/// Rotate every node that has coordinates around `axis` by `angle` radians,
/// optionally about a center point (defaults to the origin). Nodes missing any
/// coordinate are left untouched.
pub fn rotate(mol: &mut MolGraph, axis: [f64; 3], angle: f64, about: Option<[f64; 3]>) {
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if len < 1e-15 {
        return;
    }
    let k = [axis[0] / len, axis[1] / len, axis[2] / len];
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let origin = about.unwrap_or([0.0, 0.0, 0.0]);

    let table = mol.node_table_mut();

    // Read the three dense coordinate columns once, rotate only rows whose
    // x/y/z are all present, and write each column back in a single pass.
    // Rows missing any coordinate keep their original value (identity write).
    let (nx, ny, nz) = {
        let (x, vx) = match table.column_f64(keys::X) {
            Ok(t) => t,
            Err(_) => return,
        };
        let (y, vy) = match table.column_f64(keys::Y) {
            Ok(t) => t,
            Err(_) => return,
        };
        let (z, vz) = match table.column_f64(keys::Z) {
            Ok(t) => t,
            Err(_) => return,
        };
        let mut nx = x.to_vec();
        let mut ny = y.to_vec();
        let mut nz = z.to_vec();
        for row in 0..x.len() {
            if !(vx.get(row) && vy.get(row) && vz.get(row)) {
                continue;
            }
            let p = [x[row] - origin[0], y[row] - origin[1], z[row] - origin[2]];

            // Rodrigues' rotation formula.
            let kdotp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
            let cross = [
                k[1] * p[2] - k[2] * p[1],
                k[2] * p[0] - k[0] * p[2],
                k[0] * p[1] - k[1] * p[0],
            ];
            nx[row] = p[0] * cos_a + cross[0] * sin_a + k[0] * kdotp * (1.0 - cos_a) + origin[0];
            ny[row] = p[1] * cos_a + cross[1] * sin_a + k[1] * kdotp * (1.0 - cos_a) + origin[1];
            nz[row] = p[2] * cos_a + cross[2] * sin_a + k[2] * kdotp * (1.0 - cos_a) + origin[2];
        }
        (nx, ny, nz)
    };

    // Safe: columns exist (checked above) and lengths are unchanged.
    table
        .column_f64_mut(keys::X)
        .unwrap()
        .0
        .copy_from_slice(&nx);
    table
        .column_f64_mut(keys::Y)
        .unwrap()
        .0
        .copy_from_slice(&ny);
    table
        .column_f64_mut(keys::Z)
        .unwrap()
        .0
        .copy_from_slice(&nz);
}
