//! Integration tests for `molrs-signal`.
//!
//! The module tree below mirrors the crate's `src/` layout: one test module per
//! source module. Every test exercises the **public** API (`acf_fft`,
//! `apply_window`, `frequency_grid`) with inputs built in code and compares the
//! results against analytical references (ACF of a cosine, window symmetry/sum,
//! grid spacing = 2π/(N·dt)). This crate does no file I/O, so no real-file
//! fixtures are involved.

#[path = "signal/acf.rs"]
mod acf;
#[path = "signal/grid.rs"]
mod grid;
#[path = "signal/window.rs"]
mod window;
