---
slug: opls-typifier-02-assign
criteria:
  - id: ac-001
    summary: _end_score specificity 3/1/0/None
    type: code
    evaluator_hint: "test: ff::typifier::opls::assign end_score"
    pass_when: |
      _end_score returns 3 for exact type match, 1 for class match, 0 for
      wildcard X/*, and None for no match.
    status: pending
  - id: ac-002
    summary: _sequence_score is end-for-end symmetric and additive
    type: code
    evaluator_hint: "test: ff::typifier::opls::assign sequence_score"
    pass_when: |
      _sequence_score sums per-end specificities, tries forward and reversed
      orientations, returns the best, and returns None if any end is None.
    status: pending
  - id: ac-003
    summary: specificity + layer ranking picks the right bonded type
    type: scientific
    evaluator_hint: "test: ff::typifier::opls::assign ranking"
    pass_when: |
      Given a fully-resolved candidate and a wildcard candidate both matching,
      the higher-score (fully-resolved) one wins; with equal score, the higher
      layer wins (CL&P/CL&Pol overlay over OPLS-AA).
    status: pending
  - id: ac-004
    summary: assigned params match molpy OplsTypifier on real molecules
    type: scientific
    evaluator_hint: "test: ff::typifier::opls assign on tests-data; vs molpy reference"
    pass_when: |
      For molecules in tests-data/, every bond/angle/dihedral assigned by
      assign_bonded carries params equal to molpy OplsTypifier's within
      bond r0 atol 0.02 Å / angle θ0 atol 3° / force constants rtol 0.10.
    status: pending
  - id: ac-005
    summary: no-match seam respects strict flag
    type: code
    evaluator_hint: "test: ff::typifier::opls::assign no-match"
    pass_when: |
      A term with no specificity match yields Err when strict=true, and (with
      no estimator attached) returns the term without params when strict=false.
    status: pending
  - id: ac-006
    summary: build() closes typify→assign→to_frame→to_potentials
    type: code
    evaluator_hint: "test: ff::typifier::opls build"
    pass_when: |
      OplsTypifier::build(mol) returns Potentials whose energy on a known
      conformer matches the molpy OPLS reference within rtol 1e-4 (reusing the
      opls-ef-01 kernel parity harness).
    status: pending
  - id: ac-007
    summary: lint, type check, and test suite clean
    type: runtime
    pass_when: |
      `cargo fmt --all --check`, `cargo clippy --all-targets --all-features
      -- -D warnings`, and `cargo test --all-features` all exit 0.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003**: 特异性排序匹配器逐位复刻 molpy `_end_score`/`_sequence_score` + `(score, layer)` 排序。
- **ac-004**: 真实分子上成键参数与 molpy `OplsTypifier` 在容差内一致（全量 parity = 链 3/3）。
- **ac-005**: no-match 接缝遵守 strict（与 GAFF-03 同语义）。
- **ac-006**: `build()` 闭环到 potentials，能量对照 opls-ef-01 kernel parity harness。
- **ac-007**: cargo 质量闸。
