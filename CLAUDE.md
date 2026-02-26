# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

molrs is a Rust workspace for molecular simulation: core data structures, molecular dynamics, molecular packing, FFI layer, and WebAssembly bindings.

## Build & Test Commands

```bash
# Build the entire workspace
cargo build

# Run all tests (requires test data)
bash scripts/fetch-test-data.sh   # one-time: clones test data to molrs-core/target/tests-data/
cargo test --all-features

# Run tests for a single crate
cargo test -p molrs-core
cargo test -p molrs-md

# Run a single test by name
cargo test -p molrs-core test_name

# Format and lint (enforced by pre-commit hooks and CI)
cargo fmt --all
cargo clippy -- -D warnings

# Build WASM
cd molrs-wasm && wasm-pack build --release --target web

# Run benchmarks (criterion)
cargo bench -p molrs-core
cargo bench -p molrs-md
```

## Workspace Crates

| Crate | Purpose |
|---|---|
| `molrs-core` | Core library: data structures (`Frame`, `Block`, `MolGraph`), I/O (PDB, XYZ, LAMMPS, Zarr), neighbor lists, potentials, PME |
| `molrs-md` | MD engine with LAMMPS-style fix/dump plugin architecture. CPU backend (pure Rust) + optional CUDA |
| `molrs-pack` | Molecular packing (Packmol-style) with GENCAN optimizer and geometric constraints |
| `molrs-ffi` | Handle-based FFI layer using `SlotMap`. Shared by WASM and Python bindings |
| `molrs-wasm` | `wasm-bindgen` bindings for browser use. Built with `wasm-pack` |

**Dependency flow**: `molrs-core` ← `molrs-ffi` ← `molrs-wasm`, `molrs-core` ← `molrs-md`, `molrs-core` ← `molrs-pack` ← `molrs-md`

## Key Architecture Decisions

- **Float precision**: `type F = f32` by default. Enable feature `f64` for double precision. All numeric code uses the `F` alias from `molrs-core/src/core/types.rs`.
- **ndarray everywhere**: Coordinates, matrices, and block columns use `ndarray` arrays (`Vec3 = Array1<F>`, `Mat3 = Array2<F>`).
- **Handle-based FFI**: `molrs-ffi` owns `Frame` objects in a `SlotMap` and hands out `FrameId`/`BlockHandle` — no raw pointers cross the FFI boundary. Handles carry a version counter for invalidation detection.
- **Trait-based extensibility**: Neighbor algorithms (`NbListAlgo`), MD integrators/thermostats (`Fix`), output formats (`Dump`), packing callbacks (`Handler`), and constraints (`Constraint`) are all traits.
- **Backend abstraction**: `molrs-md` uses `Backend` trait with `CPU` and `CUDA` marker types. CUDA code lives behind the `cuda` feature and uses CMake + bindgen at build time.
- **Zarr trajectory format**: Per-component arrays (`x`, `y`, `z` separately, not `positions [F,N,3]`). Spec in `docs/zarr-traj-spec.md`.

## Feature Flags

- `rayon` (default) — parallel neighbor lists and potentials
- `igraph` (default) — graph algorithms for molecular topology
- `cuda` — CUDA acceleration (requires CMake, CUDA toolkit; triggers `build.rs`)
- `zarr` / `filesystem` — Zarr V3 trajectory I/O
- `blas` — BLAS-backed linear algebra via `ndarray-linalg`
- `f64` — double-precision floats
- `slow-tests` — include expensive integration tests

## Rust Edition

The workspace uses **Rust edition 2024** (`resolver = "3"`).
