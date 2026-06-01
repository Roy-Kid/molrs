//! Integration tests for `molrs-embed`.
//!
//! The module tree below mirrors the crate's `src/` layout. The public surface
//! is the `generate_3d` pipeline plus the `options`/`report` types, so the
//! end-to-end tests live in `pipeline.rs`. Every test builds its molecule
//! (`Atomistic`) in code and exercises the public API — this crate does no file
//! I/O, so no real-file fixtures are involved.

#[path = "embed/pipeline.rs"]
mod pipeline;

#[path = "embed/distgeom.rs"]
mod distgeom;
