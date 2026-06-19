# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

| 2026-06-19 | [gaff-typifier-redesign](gaff-typifier-redesign.md) | approved | molrs (ff/typifier/gaff, ff/forcefield), molpy-wrapper (charges) | Replace stale gaff-typifier-01…05 (pre-merge `molrs-ff/` addressing; built on old MMFFTypifier + a from-scratch assignment engine). Re-ground GAFF on the now-mature **`opls/` typifier**: native `gaff.dat` parser (single crate, openmmforcefields MIT ground truth), clean-room predicate atom typing for antechamber parity (mirror `opls/typing.rs`, no SMARTS, no GPL `atomtype.c`), **reuse** `assign_bonded`/`CandidateTables`/`ParameterEstimator`/`build()` (no GAFF-private assignment), AM1-BCC charge delegation. Decide conjugated/heteroaromatic scope now that `perceive_aromaticity` exists |
| 2026-06-19 | [interop-docs](interop-docs.md) | approved | molrs (docs), molrs-ffi | Retire obsolete interop-wrapup (its `molrs-ffi::bridge`/`prelude` ergonomics were never built and have no consumer — molpack depends on `molcrafts-molrs` directly, never touches molrs-ffi). Reduce to the one non-stale deliverable: `docs/interop.md` documenting the two **as-built** paths — native Rust (`molcrafts-molrs` dep) and Python/WASM handle API (`SharedStore`/`FrameRef`/`ForceFieldRef`/…) — plus the data contract (uint indices, `atomi/atomj/is_14`, `special_bonds`, `intramolecular_pairs`) + CLAUDE.md pointer |
| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
