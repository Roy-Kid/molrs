//! Multi-frame trajectory formats: DCD and LAMMPS dump.
//!
//! Single-structure formats that also support multi-frame streaming (XYZ,
//! PDB) live in [`crate::data`]; the chunk-based indexing infrastructure
//! shared by all of them is in [`crate::streaming`].

pub mod dcd;
pub mod lammps_dump;
