//! Multi-frame trajectory formats: DCD and LAMMPS dump.
//!
//! Single-structure formats that also support multi-frame streaming (XYZ,
//! PDB) live in [`crate::io::data`]; the chunk-based indexing infrastructure
//! shared by all of them is in [`crate::io::streaming`].

pub mod dcd;
pub mod lammps_dump;
