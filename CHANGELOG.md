# Changelog

All notable changes to molrs are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

## [0.1.3] - 2026-06-14

### Added

- **Force-field FFI handle.** `molrs_ffi::ForceFieldRef` — a stable, zero-copy
  handle for `molrs::ff::ForceField` (the force-field analogue of `FrameRef`),
  gated behind a new `ff` feature on `molcrafts-molrs-ffi`. Lets force-field
  consumers (e.g. molpack) borrow a `ForceField` across the FFI boundary.
- **`molrs::ff::potential::intramolecular_pairs`** — builds the intramolecular
  neighbour-pair `Block` for a frame, consumed by relaxation / energy callers.
- LAMMPS force-field reader and per-instance force-field parameter support.

### Changed

- `molrs::ff::potential` is now a directory module (`potential/`) rather than a
  single file; MMFF typifier internals reorganized. The WASM / TypeScript
  public surface (`@molcrafts/molrs` npm) is unchanged.

## [0.1.0] - 2026-06-10

### Added

- `Frame.from_dict` on the native PyO3 core — accepts either the
  `{"blocks": {...}, "metadata": {...}}` envelope or a direct
  `{name: {column: array}}` mapping, completing the `to_dict` / `from_dict`
  round-trip. Column values use the same accepted types as `Block.insert`.
- `molrs.BlockDtypeError` — public exception (subclasses `TypeError`) raised by
  `Block.insert` on a non-numpy-representable column. Importable and stable so
  downstream code can `except molrs.BlockDtypeError` precisely.

### Changed

- **Block column dtype contract is now numpy-only, fail-fast (behavior change).**
  `Block.insert` (and therefore every `Frame` column write) now accepts only
  numpy-representable dtypes — float, int, bool, and str. Object-dtype,
  None-bearing, and ragged/mixed arrays were previously rejected with a generic
  `TypeError`; they now raise the new public `molrs.BlockDtypeError` with a
  message naming the offending column and the detected dtype. There is no
  Python-side object-column overflow — columns the Rust Store cannot represent
  must be coerced to a supported dtype or dropped by the caller.
