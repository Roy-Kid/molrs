//! Integration tests for `molrs-io`.
//!
//! The module tree below mirrors the crate's `src/` layout: one test module per
//! source format module. Every format reader/writer is exercised against **all**
//! real files in `tests-data/<format>/` (located by the local `common` module),
//! never a synthetic happy-path string — see the IO Testing Rules in CLAUDE.md.

#[path = "io/common.rs"]
mod common;

#[path = "io/chgcar.rs"]
mod chgcar;
#[path = "io/cif.rs"]
mod cif;
#[path = "io/cube.rs"]
mod cube;
#[path = "io/dcd.rs"]
mod dcd;
#[path = "io/gro.rs"]
mod gro;
#[path = "io/lammps_data.rs"]
mod lammps_data;
#[path = "io/lammps_dump.rs"]
mod lammps_dump;
#[path = "io/mol2.rs"]
mod mol2;
#[path = "io/pdb.rs"]
mod pdb;
#[path = "io/poscar.rs"]
mod poscar;
#[path = "io/sdf.rs"]
mod sdf;
#[path = "io/streaming.rs"]
mod streaming;
#[path = "io/xyz.rs"]
mod xyz;

// SMILES/SMARTS are string parsers (no `tests-data/` files); they mirror
// `src/smiles/` and are gated by the `smiles` feature.
#[path = "io/smiles/mod.rs"]
mod smiles;
