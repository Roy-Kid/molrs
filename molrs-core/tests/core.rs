//! Integration tests for `molrs-core`.
//!
//! The module tree below mirrors the crate's `src/` layout: one test module per
//! source module (filename = source module name, no `test_` prefix). Tests
//! exercise the public API with realistic molecular inputs built in code —
//! core does no file I/O, so no `tests-data/` files are involved.

#[path = "core/aromaticity.rs"]
mod aromaticity;
#[path = "core/hydrogens.rs"]
mod hydrogens;
#[path = "core/rings.rs"]
mod rings;
#[path = "core/smarts.rs"]
mod smarts;
#[path = "core/stereo.rs"]
mod stereo;
