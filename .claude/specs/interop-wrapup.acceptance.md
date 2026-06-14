---
slug: interop-wrapup
criteria:
  - id: ac-001
    summary: End-to-end review checklist is green
    type: code
    evaluator_hint: "Run after interop-ffi-bridge + ff-special-bonds-nblist are code-complete."
    pass_when: |
      No leftover frame_marshal, from_lammps_ff, or MMFF frame_builder private machinery;
      uint atom-index contract holds across ALL molpy readers (not just amber); one
      pairs-block convention (atomi/atomj/is_14), one nblist utility, one capsule pattern
      for Frame+ForceField+Block; the relaxer/bridge/to_potentials seams compose.
    status: todo
    last_checked: ""
  - id: ac-002
    summary: One-call adoption path with a prelude
    type: code
    evaluator_hint: "A new throwaway downstream Rust/PyO3 crate."
    pass_when: |
      `use molrs_ffi::prelude::*` exposes ergonomic helpers so `obj.as_frame_ref()?` and
      `frame_ref.into_py(py)?` work on any capsule-bearing object; a new consumer adopts
      molrs frames in ~10 lines without copying molpack internals or touching addr/capsule
      mechanics.
    status: todo
    last_checked: ""
  - id: ac-003
    summary: API naming is consistent and discoverable
    type: code
    evaluator_hint: ""
    pass_when: |
      bridge fns follow one verb_noun scheme with py<->ref symmetry (frame_from_py/into_py,
      forcefield_from_py/into_py, block_from_py); capsule names are versioned
      (molrs.FrameRef.vN); one error type with clear messages; canonical flat re-exports
      (molrs::io::forcefield::{read_lammps, read_opls, ForceFieldReader}), no leaky deep paths.
    status: todo
    last_checked: ""
  - id: ac-004
    summary: Blessed recipe + worked example documented and runnable
    type: code
    evaluator_hint: ""
    pass_when: |
      docs/interop.md documents the recipe (link -> resolve capsule -> operate via handle ->
      return), the data contract (uint indices, pairs-block schema, special_bonds), and the
      naming conventions; its minimal worked example runs; molrs CLAUDE.md points to it.
    status: todo
    last_checked: ""
---
