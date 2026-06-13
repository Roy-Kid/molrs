//! Integration tests for `molrs-embed`.
//!
//! The module tree below mirrors the crate's `src/` layout. The public surface
//! is the `generate_3d` pipeline plus the `options`/`report` types, so the
//! end-to-end tests live in `pipeline.rs`. The `pipeline` tests build their
//! molecules (`Atomistic`) in code; the validation suites (`distgeom`,
//! `etkdg`, `torsions`, `success_rate`) load V2000 SDF / JSON fixtures from
//! `tests/embed/fixtures/` to compare against RDKit reference data.

#[path = "embed/pipeline.rs"]
mod pipeline;

#[path = "embed/distgeom.rs"]
mod distgeom;

#[path = "embed/etkdg.rs"]
mod etkdg;

#[path = "embed/torsions.rs"]
mod torsions;

#[path = "embed/success_rate.rs"]
mod success_rate;
