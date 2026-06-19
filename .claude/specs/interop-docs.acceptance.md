---
slug: interop-docs
criteria:
  - id: ac-001
    summary: docs/interop.md documents both as-built consumption paths
    type: docs
    evaluator_hint: "Human/doc review: does docs/interop.md cover the native-Rust path and the Python/WASM handle path, each with a recipe + runnable snippet?"
    pass_when: |
      docs/interop.md exists and documents (a) the native Rust path — depend on molcrafts-molrs directly, use molrs::Frame / molrs::ff::ForceField (the molpack pattern) — and (b) the Python/WASM handle path — SharedStore / FrameRef / BlockRef / ForceFieldRef / FrameId / BlockHandle / FfiError — each with a minimal adoption recipe and a short snippet.
    status: pending
    last_checked: ""
  - id: ac-002
    summary: The shared data contract is documented
    type: docs
    evaluator_hint: "Doc review: uint indices, pairs-block schema, special_bonds, intramolecular_pairs."
    pass_when: |
      docs/interop.md states the data contract: uint atom indices, the pairs block schema (atomi/atomj/is_14), special_bonds weights carried on the ForceField, and the consumer-built intramolecular_pairs neighbour list.
    status: pending
    last_checked: ""
  - id: ac-003
    summary: Snippets build and CLAUDE.md points to the doc
    type: code
    evaluator_hint: ""
    pass_when: |
      The native-Rust snippet compiles against molcrafts-molrs and the handle-API snippet compiles against molrs-ffi (e.g. as doctests or a small example), and molrs/CLAUDE.md contains a pointer to docs/interop.md.
    status: pending
    last_checked: ""
---
