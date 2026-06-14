# interop-wrapup — end-to-end review + FFI ergonomics & API-naming polish so downstream projects build on molrs/molpy with low friction

Status: **draft**
Scope: molrs-ffi, molrs-python, molrs-ff, molrs-capi, docs; reference consumer: molpack
Depends on: `interop-ffi-bridge` and `ff-special-bonds-nblist` (both code-complete first)

## Summary

The two implementation specs deliver the **mechanism** — `interop-ffi-bridge` (the
zero-copy capsule path) and `ff-special-bonds-nblist` (the unified potential path).
This wrapup makes the **result a product other teams can build on**: a consolidating
review of the whole implementation, an FFI/bridge layer that is genuinely
**ergonomic and consistently named**, a one-call adoption path, and a blessed,
documented recipe + worked example. Run **after** both implementations are
code-complete; naming changes here may feed back into them before their public API
is depended upon externally.

## Goal / why

The mechanism alone is not enough — a new downstream Rust/PyO3 project must be able
to adopt molrs/molpy data objects **without re-discovering the capsule plumbing or
copying molpack's internals**. This spec locks: cross-spec correctness/consistency
(the seams compose), an ergonomic + discoverable API, one naming scheme, and the
docs/example so adoption is ~10 lines.

## Scope of the wrapup

1. **End-to-end review** (both specs code-complete):
   - Correctness: bridge is truly zero-copy (no hidden deep copies / shared backing
     proven); the **uint atom-index contract holds across all readers** (the molpy
     reader sweep done — not just amber); pair-kernel energy parity; MMFF parity
     then `frame_builder` deletion; **no leftover** `frame_marshal`,
     `from_lammps_ff`, or MMFF private machinery.
   - Consistency: **one** pairs-block convention (`atomi/atomj/is_14`), **one** nblist
     utility (`intramolecular_pairs`), **one** capsule pattern for Frame + ForceField
     + Block.
   - Seams compose: relaxer ← bridge ← `molrs.ForceField`; relaxer →
     `intramolecular_pairs` → `to_potentials`.
2. **FFI ergonomics + API naming**:
   - Audit `molrs-ffi::bridge` + the `_ffi_*_capsule()` surface for discoverability
     and naming; pick **one** scheme (e.g. `frame_from_py`/`frame_to_py`,
     `forcefield_from_py`/`forcefield_to_py`, `block_from_py` — `verb_noun`, `py↔ref`
     symmetric; capsule names `molrs.FrameRef.vN`).
   - A **one-call adoption path**: link `molrs-ffi` (`pyo3` feature) + `use
     molrs_ffi::prelude::*`, then `obj.as_frame_ref()?` / `frame_ref.into_py(py)?`
     "just work" on any capsule-bearing object — the addr/capsule mechanics hidden.
   - Errors: one error type, clear messages (wrong capsule name, version mismatch,
     missing capsule).
   - Re-exports: a flat, documented public surface — canonical
     `molrs::io::forcefield::{read_lammps, read_opls, ForceFieldReader}`, no leaky
     deep paths.
3. **Docs + worked example**:
   - `docs/interop.md` — the blessed recipe (link → resolve capsule → operate via
     handle → return) + the data contract (uint indices, pairs-block schema,
     special_bonds) + naming conventions.
   - A minimal runnable downstream example (~10 lines) adopting molrs frames using
     only the prelude; molpack called out as the canonical reference.
4. **Conventions codified** so future kernels/consumers follow without asking.

## Files

- `molrs-ffi/src/bridge.rs` + `molrs-ffi/src/prelude.rs` — ergonomic trait + prelude
  + consistent names + one error type.
- `docs/interop.md` (new) — recipe, example, data contract, naming conventions.
- `molrs/CLAUDE.md` — pointer to `docs/interop.md` + the data contract.
- Naming sweeps across the two implementation specs' public surfaces (rename for
  consistency before external projects pin them).

## Testing / acceptance

- A NEW throwaway downstream consumer adopts molrs frames in ~10 lines using only
  the documented prelude — without copying molpack internals.
- API-name consistency review passes (`verb_noun`, `py↔ref` symmetry, no leaky deep
  module paths; capsule names versioned).
- The end-to-end review checklist is fully green (no leftover marshalling / private
  machinery; uint everywhere; bridge zero-copy proven; parity tests pass).
- `docs/interop.md`'s worked example runs.

## Dependencies / order

`interop-ffi-bridge` → `ff-special-bonds-nblist` → **this**. The wrapup runs last
(review + polish), but its naming decisions should land **before** external projects
pin the public surface.

## Out of scope

- The implementations themselves — this spec is the review + ergonomics layer on top.
