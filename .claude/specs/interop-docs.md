---
title: docs/interop.md for the as-built molrs consumption paths (retires interop-wrapup)
status: approved
created: 2026-06-19
supersedes: interop-wrapup (premise overtaken; reduced to the docs deliverable)
---

# docs/interop.md for the as-built molrs consumption paths

## Summary

`interop-wrapup` set out to make downstream **Rust** adoption of molrs frictionless
via a `molrs-ffi::bridge` + `prelude` (`use molrs_ffi::prelude::*; obj.as_frame_ref()?`)
and a naming unification "before external projects pin the public surface." The
current codebase makes most of that **obsolete**:

- `molrs-ffi` shipped a **handle-based** API (`FrameRef`, `BlockRef`, `ForceFieldRef`,
  `SharedStore`, `new_shared`, `FrameId`, `BlockHandle`, one `FfiError` —
  `molrs-ffi/src/lib.rs`), explicitly *"for Python and WASM bindings."* No `bridge`
  module, no `prelude`, no `as_frame_ref()`/`into_py()` — the prescribed ergonomic
  shape was never built.
- The named reference consumer **bypasses the FFI entirely**: `molpack/Cargo.toml`
  depends on `molcrafts-molrs` **directly** (`molrs = { path = "../molrs/molrs",
  package = "molcrafts-molrs" }`); nothing in `molpack/src` touches `molrs_ffi`,
  `FrameRef`, or `frame_marshal`. So the "ergonomic Rust adoption via the molrs-ffi
  prelude" goal has **no consumer**, and the "rename before external pin" window is
  closed.

What remains genuinely valuable and non-stale is the **documentation**: a blessed
`docs/interop.md` describing the *as-built* consumption paths and the data contract.
This spec reduces `interop-wrapup` to that docs deliverable and drops the
prelude/naming-rename scope.

## Domain basis

Two real consumption paths exist today:

1. **Rust → Rust (native, zero-copy by construction)**: depend on `molcrafts-molrs`
   directly and use `molrs::Frame` / `molrs::ff::ForceField` natively (what molpack
   does). No FFI, no marshalling.
2. **Python / WASM → molrs (handle-based FFI)**: go through the `molrs-ffi` handle
   API (`SharedStore`, `FrameRef`, `BlockRef`, `ForceFieldRef`, `FrameId`,
   `BlockHandle`, `FfiError`) — the stable cross-language surface.

The data contract both paths share: **uint atom indices**, the pairs-block schema
(`atomi/atomj/is_14`), and `special_bonds` weights carried on the `ForceField`
(nblist built by the consumer via `intramolecular_pairs`, already used by
`OplsTypifier::build`).

## Design

Write `docs/interop.md` documenting, for each of the two paths above: the minimal
adoption recipe, a short runnable snippet, and the shared data contract. Verify the
snippets compile/run. Add a one-line pointer from `molrs/CLAUDE.md` to the doc.

Do **not** introduce a `bridge` module / `prelude` / `as_frame_ref()` retrofit
unless a concrete downstream Rust-via-FFI consumer materializes — there is none
today.

## Files to create or modify

- `docs/interop.md` (new) — the two as-built paths + data contract + runnable snippets.
- `molrs/CLAUDE.md` — one-line pointer to `docs/interop.md` + the data contract.

## Tasks

- [ ] Document the native Rust path (depend on `molcrafts-molrs`; use `molrs::Frame`/`ForceField` directly) with a runnable snippet mirroring molpack's usage
- [ ] Document the Python/WASM handle path (`SharedStore`/`FrameRef`/`ForceFieldRef`/…) with a runnable snippet
- [ ] Document the shared data contract (uint indices, `atomi/atomj/is_14` pairs block, `special_bonds`, consumer-built `intramolecular_pairs`)
- [ ] Add the `molrs/CLAUDE.md` pointer
- [ ] Verify both snippets build/run

## Testing strategy

- The native-Rust snippet compiles against `molcrafts-molrs`.
- The handle-API snippet compiles against `molrs-ffi`.
- A `docs`-type acceptance check confirms the doc covers both paths + the data contract.

## Out of scope

- The `molrs-ffi::bridge`/`prelude`/`as_frame_ref()` ergonomic layer and the
  cross-spec naming unification (obsolete — no consumer; API already shipped + pinned).
- Any change to the `molrs-ffi` public surface.
- Re-reviewing the already-closed `interop-ffi-bridge` / `ff-special-bonds-nblist`
  mechanism (separate; see ff-perinstance-mmff-kernels for the MMFF frame_builder tail).
