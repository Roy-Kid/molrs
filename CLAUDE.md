---
mol_project:
  name: molrs
  language: rust
  stage: experimental
  build:
    install: "cargo build && bash scripts/fetch-test-data.sh"
    check: "cargo fmt --all --check && cargo clippy -- -D warnings && cargo check"
    test: "cargo test --all-features"
    test_single: "cargo test {path}"
  arch:
    style: crate-graph
    rules_section: "## Workspace Crates & Dependency Flow"
  doc:
    style: rustdoc
  science:
    required: true
  notes_path: .claude/notes/notes.md
  specs_path: .claude/specs/
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

molrs is a Rust workspace for molecular simulation: core data structures, file I/O, trajectory analysis, signal processing, force fields, 3D coordinate generation, and a CXX bridge to Atomiverse C++. Rust edition 2024, resolver "3".

## IO Testing Rules (MANDATORY)

**NEVER write synthetic/hand-crafted test data for IO tests.**

Every file-format reader/writer MUST be tested against **all** real files in
`tests-data/<format>/` — a binding-neutral directory at the **workspace root**
(gitignored; cloned by `scripts/fetch-test-data.sh`), shared by every Rust crate
and by the Python / C / WASM bindings. Rules:

1. When adding a new format reader (e.g. CHGCAR), add matching real files to
   the `tests-data` repo (`https://github.com/MolCrafts/tests-data`) under a
   new `<format>/` subdirectory before writing tests.
2. Tests iterate over **every** file in that directory — not a hardcoded subset.
   Use the small local `common` helper in the io test target
   (`common::format_files("<format>")`) and run assertions on each.
3. **Inline `#[cfg(test)]` tests in `src/` are pure function unit tests only** —
   logic, edge cases, error paths. They must NOT read real files from
   `tests-data/`. A minimal `include_str!` fixture is permitted ONLY to cover a
   parser edge-case hard to produce from real data (e.g. malformed input →
   expected error); keep it tiny and document its origin.
4. Data-driven integration tests live in each crate's `tests/` tree, mirroring
   that crate's `src/` module layout (e.g. `molrs-io/tests/io/<format>.rs`), and
   resolve files via the io test target's local `common` module
   (`common::{tests_data_dir, data_path, format_files}`), which simply reads
   `../tests-data` (or `$MOLRS_TESTS_DATA`). No helper crate.

Violation: writing `let content = "..."; read_from_str(content)` for happy-path
format tests instead of reading a real file is **forbidden**.

## Build & Test Commands

```bash
# Build
cargo build
# Test (requires test data on first run)
bash scripts/fetch-test-data.sh      # clones to <root>/tests-data/ (binding-neutral)
cargo test --all-features
cargo test -p molcrafts-molrs-core              # single crate (-p takes the package name)
cargo test -p molcrafts-molrs-core test_name    # single test
cargo test --features slow-tests                # expensive integration tests
cargo test -p molcrafts-molrs-io                # IO format tests (iterate every file in tests-data/)

# Lint & Format
cargo fmt --all
cargo clippy -- -D warnings

# Benchmarks (criterion)
cargo bench -p molcrafts-molrs-core
```

## Workspace Crates & Dependency Flow

8 workspace members (root `Cargo.toml` `[workspace] members`). `molrs-core` is the
foundation; everything else depends on it (plus, in a few cases, on each other as listed):

```
molrs-core ── molrs-io ── molrs-cxxapi
            ── molrs-signal ── molrs-compute
            ── molrs-ff (may also depend on molrs-io for parameter files)
            ── molrs-conformer
            └─ molrs (umbrella façade re-exporting all sub-crates)
```

| Crate (dir) | Package name | Purpose |
|---|---|---|
| `molrs-core` | `molcrafts-molrs-core` | Frame/Block/Grid/MolGraph/MolRec/Topology/Element, neighbors, math, region (SimBox), stereochemistry, rings, Gasteiger charges, hydrogen perception, atom-type mapping |
| `molrs-io` | `molcrafts-molrs-io` | File I/O: PDB, XYZ, LAMMPS data/dump, CHGCAR/POSCAR, Gaussian Cube, CIF, mol2, SDF, GRO, DCD, Zarr V3 trajectories; SMILES/SMARTS parsing in `src/smiles/` (gated by the `smiles` feature) |
| `molrs-signal` | `molcrafts-molrs-signal` | Signal processing: FFT-based autocorrelation, window functions, frequency grids |
| `molrs-compute` | `molcrafts-molrs-compute` | Trajectory analysis: RDF, MSD, clustering, gyration/inertia tensors (depends on `molrs-signal`) |
| `molrs-ff` | `molcrafts-molrs-ff` | Force fields, potentials (KernelRegistry), atom typifier |
| `molrs-conformer` | `molcrafts-molrs-conformer` | 3D conformer generation: distance geometry, fragment assembly, optimizer, rotor search |
| `molrs-cxxapi` | `molcrafts-molrs-cxxapi` | CXX bridge to Atomiverse C++ (zero-copy I/O via `FrameView`) |
| `molrs` | `molcrafts-molrs` | Umbrella façade crate re-exporting the sub-crates behind feature flags (`io`, `compute`, `smiles`, `ff`, `conformer`, `signal`, `full`) |

Molecular packing (Packmol port) used to live here as `molrs-pack`; it now lives in the
standalone repo `MolCrafts/molpack` (crates.io: `molcrafts-molpack`, PyPI:
`molcrafts-molpack`). Add it as a separate dependency when needed.

Dirs `molrs-ffi/`, `molrs-wasm/`, `molrs-capi/`, `molrs-python/` exist on disk but are NOT
workspace members; treat as inactive / future work.

## Feature Flags

`molrs-core` features:
- `rayon` (default) — parallel neighbor lists and potentials
- `zarr` — enables `From<zarrs::*Error>` conversions for `MolRsError` (used by `molrs-io`)
- `blas` — BLAS-backed linear algebra via `ndarray-linalg`
- `f64` / `i64` / `u64` — **deprecated / no-op** (F/I/U are hardcoded to f64/i32/u32)
- `slow-tests` — expensive integration tests

`molrs-io` features:
- `smiles` — SMILES/SMARTS parsing (pulls in `petgraph`)
- `zarr` / `filesystem` — Zarr V3 trajectory I/O

The `molrs` umbrella crate gates each sub-crate behind a feature (`io`, `compute`,
`smiles`, `ff`, `conformer`, `signal`) with `full` enabling all of them.

## Core Data Model

### Type Precision Principle

**Scientific algorithms use high precision; estimation / general code uses natural types.**

- `F = f64` always. The `f64` feature flag is deprecated and ignored.
- `I = i32` always. The `i64` feature flag is deprecated and ignored.
- `U = u32` always. The `u64` feature flag is deprecated and ignored.
- Neighbor list internals use `u32` directly — that's the natural type, no alias needed.
- The CXX bridge to Atomiverse still uses a `{F}` template that is resolved at build time by `cfg!(feature = "f64")` — this feature is set by CMake/corrosion to match Atomiverse's `ATV_REAL`.

Key type aliases: `F3 = Array1<F>`, `F3x3 = Array2<F>`, `FN = Array1<F>`, `FNx3 = Array2<F>`. Since `F = f64`, these are all double precision.

### Block (heterogeneous column store)

`Block` maps string keys to typed ndarray columns (f32, f64, i64, bool). Enforces consistent `nrows` across all columns. Type-safe access via `get_float()`, `get_int()`, `get_bool()`, `get_uint()`, `get_u8()`, `get_string()`. (`molrs-core/src/block/`).

### Frame (hierarchical data container)

`Frame` maps string keys (e.g. "atoms", "bonds", "angles") to `Block`s. Contains optional `SimBox` for periodic boundaries and a metadata hashmap. No forced cross-block row consistency — caller responsibility.

### MolGraph (molecular topology)

Graph-based molecular structure with atoms, bonds, stereochemistry, ring detection. Uses petgraph. (`molrs-core/src/molgraph.rs`).

## Trait-Based Extensibility

| Trait | Crate | Purpose | Key Implementations |
|---|---|---|---|
| `NbListAlgo` | `molrs-core::neighbors` | Neighbor search | `LinkCell` (O(N), default), `BruteForce` (O(N²), testing), `NeighborQuery` (high-level wrapper) |
| `Potential` | `molrs-ff::potential` | Energy/force evaluation | Bond harmonic, MMFF bond/angle/torsion/oop/vdw/ele, LJ/cut, PME |
| `Typifier` | `molrs-ff::typifier` | MolGraph → typed Frame | MMFFTypifier |

Pack-related traits (`Restraint`, `Region`, `Relaxer`, `Handler`, `Objective`) now live
in the standalone `molcrafts-molpack` crate.

## Key Subsystems

### Potential System (molrs-ff/src/potential/)

`KernelRegistry` maps `(category, style_name)` → `KernelConstructor`. Categories: bonds, angles, dihedrals, impropers, pairs, kspace. `ForceField::compile(frame)` resolves topology and constructs `Potentials` (aggregate sum). Coordinate format: flat `[x0,y0,z0, x1,y1,z1, ...]` (3N elements). MMFF94/MMFF94s parameters are embedded at compile time in `molrs-core` (`molrs-core/data/mmff94.xml`, exposed as `molrs::data::MMFF94_XML`).

### Free-Boundary Support

`SimBox::free(points, padding)` creates a non-periodic bounding box from atom positions. `NeighborQuery::free(points, cutoff)` auto-generates this box when no SimBox is present. RDF normalization (`molrs-compute`) falls back to bounding-box volume for free-boundary systems.

### Conformer Pipeline (molrs-conformer/)

Multi-stage 3D coordinate generation: distance geometry → fragment assembly → coarse minimization → rotor search → final minimization → stereo guards. Public API: `Conformer::new(opts).generate(mol) -> Result<(Atomistic, ConformerReport)>`.

### Packing

Lives in the standalone `MolCrafts/molpack` repository (crate
`molcrafts-molpack`, Python package `molcrafts-molpack` exposing `import
molpack`). Depends on `molcrafts-molrs-core` + `molcrafts-molrs-io`. See that
repo's docs for the Packmol-alignment workflow and associated conventions.

### FFI Layer (molrs-cxxapi/)

CXX bridge to Atomiverse C++. Zero-copy I/O via `FrameView` (borrowed) into existing `write_xyz_frame`; owned `Frame` only built when persisting to MolRec (Zarr). Bridge generated from `#[cxx::bridge]` in `bridge.rs`. No raw pointers cross the boundary.

## Critical Conventions

- **Coordinate format**: Potentials use flat `[x0,y0,z0, x1,y1,z1, ...]` vectors (3N elements), not Nx3 matrices.
- **`Cell<f64>` is NOT Sync**: Use `AtomicU64` with `f64::to_bits()`/`f64::from_bits()` for interior mutability in Sync contexts.

Packmol-port specific conventions (gradient sign, two-scale contract, LEFT
rotation multiplication) now live in the molpack repo's CLAUDE.md.

## Development Workflow (mol plugin)

Process workflows come from the molcrafts `mol` plugin; project-specific
standards live in `.claude/notes/` topic pages (see next section).

Workflow skills (user-invocable):

- `/mol:spec <requirement>` — NL requirement → spec + acceptance contract in `.claude/specs/`.
- `/mol:impl <spec>` — orchestrator for new feature work; spec → TDD → implement → verify → simplify → close.
- `/mol:review [path]` — parallel multi-axis review (architecture, performance, science, FFI via `--axis=ffi`, …).
- `/mol:fix <bug>` — minimal-diff bug fix, regression test first.
- `/mol:refactor <scope>` — in-place restructure without behavior change; enforces hot-path extraction discipline (see `.claude/notes/performance.md`).
- `/mol:debug <symptom>` — read-only diagnosis; never edits.
- `/mol:docs <kind>` — docs work (rustdoc, Zensical site, `.pyi`, READMEs, `docs.yml`); site rules in `.claude/notes/docs.md`.
- `/mol:note <decision> | sweep | promote <slug>` — capture evolving decisions into `.claude/notes/notes.md`; promote stable entries into this CLAUDE.md.

Agent mapping by domain (plugin agents read the notes pages below):

| Domain | Standard (notes page) | Agent |
|---|---|---|
| Architecture | `.claude/notes/architecture-rules.md` | `mol:architect` |
| Performance | `.claude/notes/performance.md` | `mol:optimizer` |
| Documentation (rustdoc) | `.claude/notes/docs.md` (Part A) | `mol:documenter` |
| Documentation (docs system) | `.claude/notes/docs.md` (Part B) | `mol:documenter` |
| Testing | `.claude/notes/testing.md` | `mol:tester` |
| Scientific correctness | `.claude/notes/science.md` | `mol:scientist` |
| FFI safety | `.claude/notes/ffi.md` | `mol:ffi-guard` (`/mol:review --axis=ffi`) |

Evolving decisions live in `.claude/notes/notes.md`; specs live in
`.claude/specs/` indexed by `.claude/specs/INDEX.md`.
