//! File I/O for molecular data, organized by content kind:
//!
//! - [`data`] — single-structure formats (PDB, XYZ, GRO, mol2, SDF, CIF,
//!   LAMMPS data, CHGCAR/POSCAR, Cube)
//! - [`trajectory`] — multi-frame formats (DCD, LAMMPS dump)
//! - [`store`] — persistence backends (Zarr V3, feature `zarr`)
//! - [`reader`] / [`writer`] / [`streaming`] — shared traits and the
//!   chunk-based frame-indexing infrastructure
//! - [`smiles`] — SMILES/SMARTS notation parsing (feature `smiles`)

pub mod data;
pub mod store;
pub mod trajectory;

pub mod reader;
pub mod streaming;
pub mod writer;

#[cfg(feature = "smiles")]
pub mod smiles;
