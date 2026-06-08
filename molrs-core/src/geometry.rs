//! Geometry *systems* — free functions that transform a [`MolGraph`]'s node
//! coordinates in place.
//!
//! Under the ECS model the graph is pure data; spatial transforms are systems
//! that operate over the world, so they live here as free functions rather than
//! as methods on the data structure. Coordinates are read and written through
//! the canonical [`crate::keys`] coordinate convention — no field-name literals.

use crate::keys;
use crate::molgraph::{MolGraph, NodeId};

/// Translate every node that has coordinates by `delta` (nodes without a full
/// coordinate set are left untouched).
pub fn translate(mol: &mut MolGraph, delta: [f64; 3]) {
    let ids: Vec<NodeId> = mol.node_ids().collect();
    let table = mol.node_table_mut();
    for id in ids {
        for (i, key) in keys::COORDS.iter().enumerate() {
            if let Ok(v) = table.get_f64(id, key) {
                let _ = table.set_f64(id, key, v + delta[i]);
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

    let ids: Vec<NodeId> = mol.node_ids().collect();
    let table = mol.node_table_mut();
    for id in ids {
        let (Ok(x), Ok(y), Ok(z)) = (
            table.get_f64(id, keys::X),
            table.get_f64(id, keys::Y),
            table.get_f64(id, keys::Z),
        ) else {
            continue;
        };
        let p = [x - origin[0], y - origin[1], z - origin[2]];

        // Rodrigues' rotation formula.
        let kdotp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
        let cross = [
            k[1] * p[2] - k[2] * p[1],
            k[2] * p[0] - k[0] * p[2],
            k[0] * p[1] - k[1] * p[0],
        ];
        let rotated = [
            p[0] * cos_a + cross[0] * sin_a + k[0] * kdotp * (1.0 - cos_a) + origin[0],
            p[1] * cos_a + cross[1] * sin_a + k[1] * kdotp * (1.0 - cos_a) + origin[1],
            p[2] * cos_a + cross[2] * sin_a + k[2] * kdotp * (1.0 - cos_a) + origin[2],
        ];
        let _ = table.set_f64(id, keys::X, rotated[0]);
        let _ = table.set_f64(id, keys::Y, rotated[1]);
        let _ = table.set_f64(id, keys::Z, rotated[2]);
    }
}
