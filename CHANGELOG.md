# Changelog

All notable changes to molrs are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

### Changed

- **Block column dtype contract is now numpy-only, fail-fast (behavior change).**
  `Block.insert` (and therefore every `Frame` column write) now accepts only
  numpy-representable dtypes — float, int, bool, and str. Object-dtype,
  None-bearing, and ragged/mixed arrays were previously rejected with a generic
  `TypeError`; they now raise the new public `molrs.BlockDtypeError` with a
  message naming the offending column and the detected dtype. There is no
  Python-side object-column overflow — columns the Rust Store cannot represent
  must be coerced to a supported dtype or dropped by the caller.

### Added

- `molrs.BlockDtypeError` — public exception (subclasses `TypeError`) raised by
  `Block.insert` on a non-numpy-representable column. Importable and stable so
  downstream code can `except molrs.BlockDtypeError` precisely.
