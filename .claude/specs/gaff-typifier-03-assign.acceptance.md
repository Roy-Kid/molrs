---
slug: gaff-typifier-03-assign
criteria:
  - id: ac-001
    summary: Exact bond/angle tuples on typed ethanol resolve to stored GAFF type-names
    type: code
    evaluator_hint: "molrs-ff/src/typifier/gaff/frame_builder.rs tests"
    pass_when: |
      A unit test typing a non-conjugated molecule (e.g. ethanol) builds a Frame
      whose bonds and angles `type` columns contain only resolved stored GAFF
      type-names, each matched by exact tuple; test passes under
      `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-002
    summary: Exact dihedral quartet is chosen over a competing X-X-a-b wildcard term
    type: code
    evaluator_hint: "classify.rs precedence RED test"
    pass_when: |
      A test seeding the store with BOTH an exact dihedral quartet and a matching
      general X-X-a-b term asserts the resolver returns the exact quartet's
      type-name (never the wildcard); passes under `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-003
    summary: Dihedral with only a general X-term falls back to that wildcard term
    type: code
    pass_when: |
      A test where the quartet has no exact term but matches a single X-term
      asserts the resolver returns that general term's type-name; passes under
      `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-004
    summary: Most-specific wildcard match wins among competing wildcard terms
    type: code
    pass_when: |
      A test with two matching wildcard terms of differing specificity
      (e.g. X-a-b-c vs X-X-b-c) asserts the resolver returns the one with fewer
      X positions; passes under `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-005
    summary: Bond and angle tuple lookup is order-independent
    type: code
    pass_when: |
      A test asserts bond a-b resolves identically to b-a, and angle a-b-c
      resolves identically to c-b-a (central atom fixed); passes under
      `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-006
    summary: Multi-term dihedral assembles all Fourier terms from the matched quartet
    type: code
    pass_when: |
      A test where the matched quartet's 01-parser entry holds multiple Fourier
      terms (negative-PN continuation) asserts all terms are carried onto the
      assigned dihedral; passes under `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-007
    summary: Improper assignment applies PK directly with no IDIVF division
    type: code
    pass_when: |
      A test asserts the assigned improper parameter equals the store PK with no
      IDIVF division applied (store entry carries a non-unit IDIVF that must be
      ignored); passes under `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-008
    summary: Missing required parameter surfaces a clear Err, not a silent skip
    type: code
    pass_when: |
      A test feeding a topology tuple with neither exact nor wildcard match
      asserts `build_gaff_frame` returns Err whose message names the category
      and the offending atom-type tuple; passes under
      `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-009
    summary: typify on an in-scope molecule yields a fully parameterized Frame that compiles to Potentials
    type: code
    evaluator_hint: "end-to-end test via ff.to_potentials"
    pass_when: |
      A test calls GaffTypifier::typify on an in-scope molecule and asserts the
      returned Frame's bonds/angles/dihedrals/impropers blocks each carry a
      non-empty `type` column, then asserts `ff.to_potentials(&frame)` returns
      Ok(Potentials); passes under `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-010
    summary: Crate builds clean under fmt, clippy -D warnings, and full test suite
    type: code
    pass_when: |
      `cargo fmt --all --check && cargo clippy -- -D warnings && cargo check`
      and `cargo test --all-features` all succeed.
    status: pending
---

# Acceptance criteria

- **ac-001** anchors the exact-tuple happy path (Tasks: frame_builder impl + its tests).
- **ac-002 / ac-003 / ac-004** are the wildcard-precedence trio (Tasks: classify resolver + its tests); ac-002 is the binding RED test that exact beats wildcard.
- **ac-005** order-independence of bond/angle lookup (Tasks: classify resolver).
- **ac-006** multi-term dihedral assembly, tying to 01-parser's negative-PN handling (Tasks: frame_builder/params).
- **ac-007** improper PK-direct / no-IDIVF domain rule (Tasks: frame_builder + classify improper resolver).
- **ac-008** no-match error surfacing (Tasks: frame_builder no-match path + tests).
- **ac-009** end-to-end typify → to_potentials (Tasks: GaffTypifier + end-to-end test).
- **ac-010** subsumes the "Run full check + test suite" task and the rustdoc task's compile.
