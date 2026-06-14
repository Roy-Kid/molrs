//! Shared in-code builders for `molrs-ff` integration tests.
//!
//! Everything here builds inputs from scratch (no file-format strings). Frames
//! are assembled with the public `molrs::store::block::Block` / `molrs::store::frame::Frame`
//! API exactly as a real consumer would after typification.

#![allow(dead_code)]

use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::{F, U};
use ndarray::Array1;

/// Build an `"atoms"` block from a slice of `[x, y, z]` positions.
pub fn atoms_block(coords: &[[F; 3]]) -> Block {
    let mut atoms = Block::new();
    atoms
        .insert(
            "x",
            Array1::from_vec(coords.iter().map(|p| p[0]).collect()).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "y",
            Array1::from_vec(coords.iter().map(|p| p[1]).collect()).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "z",
            Array1::from_vec(coords.iter().map(|p| p[2]).collect()).into_dyn(),
        )
        .unwrap();
    atoms
}

/// Build a topology block (`bonds` / `angles` / `pairs` / ...) from index columns.
///
/// `idx_cols` is `(column_name, indices)`; `types` populates the `"type"` column.
pub fn topo_block(idx_cols: &[(&str, &[U])], types: &[&str]) -> Block {
    let mut block = Block::new();
    for (name, vals) in idx_cols {
        block
            .insert(*name, Array1::from_vec(vals.to_vec()).into_dyn())
            .unwrap();
    }
    block
        .insert(
            "type",
            Array1::from_vec(types.iter().map(|s| s.to_string()).collect()).into_dyn(),
        )
        .unwrap();
    block
}

/// `"atoms"` block with x/y/z + per-atom `type` (and optional `charge`) — the
/// per-atom convention the `lj/cut` / `coul/cut` kernels read.
pub fn typed_atoms_block(coords: &[[F; 3]], types: &[&str], charges: Option<&[F]>) -> Block {
    let mut atoms = atoms_block(coords);
    atoms
        .insert(
            "type",
            Array1::from_vec(types.iter().map(|s| s.to_string()).collect()).into_dyn(),
        )
        .unwrap();
    if let Some(q) = charges {
        atoms
            .insert("charge", Array1::from_vec(q.to_vec()).into_dyn())
            .unwrap();
    }
    atoms
}

/// `"pairs"` neighbour-list block in the per-atom convention (`atomi/atomj/is_14`).
pub fn pairs_block(atomi: &[U], atomj: &[U], is_14: &[bool]) -> Block {
    let mut block = Block::new();
    block
        .insert("atomi", Array1::from_vec(atomi.to_vec()).into_dyn())
        .unwrap();
    block
        .insert("atomj", Array1::from_vec(atomj.to_vec()).into_dyn())
        .unwrap();
    block
        .insert("is_14", Array1::from_vec(is_14.to_vec()).into_dyn())
        .unwrap();
    block
}

/// Flat `[x0,y0,z0, ...]` coordinate vector from `[x,y,z]` positions.
pub fn flat_coords(coords: &[[F; 3]]) -> Vec<F> {
    coords.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
}

/// Frame with only an `"atoms"` block.
pub fn atoms_frame(coords: &[[F; 3]]) -> Frame {
    let mut frame = Frame::new();
    frame.insert("atoms", atoms_block(coords));
    frame
}

/// Central finite-difference gradient of `energy` at `coords`.
///
/// Returns the analytical force's negation comparison value: `-dE/dx_i` for
/// each coordinate. Callers compare against the reported forces (= -gradient).
pub fn numerical_forces<E: Fn(&[F]) -> F>(energy: E, coords: &[F], h: F) -> Vec<F> {
    let mut grad = vec![0.0; coords.len()];
    for i in 0..coords.len() {
        let mut cp = coords.to_vec();
        let mut cm = coords.to_vec();
        cp[i] += h;
        cm[i] -= h;
        let ep = energy(&cp);
        let em = energy(&cm);
        // force = -dE/dx
        grad[i] = -(ep - em) / (2.0 * h);
    }
    grad
}
