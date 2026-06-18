---
slug: opls-typifier-03-parity
criteria:
  - id: ac-001
    summary: per-atom OPLS type 100% identical to molpy OplsTypifier
    type: scientific
    evaluator_hint: "test: ff::typifier::opls_parity; tests-data/opls/; gated"
    pass_when: |
      For every molecule in tests-data/opls/, molrs OplsTypifier assigns the
      exact same opls_NNN atom type to every atom as the molpy ground-truth
      JSON. Skips cleanly when fixtures are absent.
    status: pending
  - id: ac-002
    summary: per-term bonded params within physical tolerance
    type: scientific
    evaluator_hint: "test: ff::typifier::opls_parity params"
    pass_when: |
      For every bond/angle/dihedral, molrs params equal molpy's within bond r0
      atol 0.02 Å, angle θ0 atol 3°, and force constants rtol 0.10.
    status: pending
  - id: ac-003
    summary: fixture set covers wildcard dihedrals + overrides/layer typing
    type: code
    evaluator_hint: "test: fixture coverage assertion"
    pass_when: |
      The tests-data/opls/ set includes at least one molecule exercising a
      wildcard-end dihedral (X-CT-CT-X class) and at least one atom whose type is
      decided by an overrides/layer priority rule.
    status: pending
  - id: ac-004
    summary: parity holds with estimator disabled
    type: scientific
    evaluator_hint: "test: estimator-off parity"
    pass_when: |
      With the estimator not injected, parity (ac-001/ac-002) holds; any term
      molpy leaves unparameterized is also unparameterized in molrs (no silent
      estimation).
    status: pending
  - id: ac-005
    summary: lint, type check, and test suite clean
    type: runtime
    pass_when: |
      `cargo fmt --all --check`, `cargo clippy --all-targets --all-features
      -- -D warnings`, and `cargo test --all-features` all exit 0.
    status: pending
---

# Acceptance criteria

- **ac-001**: 逐原子 type 100% 一致（离散，必须精确）——删 molpy Python 分型的硬门。
- **ac-002**: 成键参数物理容差内（吸收单位换算浮点差）。
- **ac-003**: fixture 覆盖通配端二面角 + overrides/layer 分型分支。
- **ac-004**: estimator 关闭时 parity 成立、无静默估计。
- **ac-005**: cargo 质量闸（含 gated fixtures 干净跳过）。
