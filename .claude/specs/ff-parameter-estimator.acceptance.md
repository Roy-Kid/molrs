---
slug: ff-parameter-estimator
criteria:
  - id: ac-001
    summary: empirical bond k (Badger) matches transcribed GAFF reference
    type: scientific
    evaluator_hint: "test: ff::typifier::estimate empirical bond"
    pass_when: |
      _empirical_bond_k for ≥2 element pairs with known GAFF reference k
      reproduces the reference within rtol 1e-3 (formula pinned).
    status: pending
  - id: ac-002
    summary: empirical angle θ0 is mean of shared-center existing angles
    type: scientific
    evaluator_hint: "test: ff::typifier::estimate empirical angle theta0"
    pass_when: |
      Given angles A-B-A (θ1) and C-B-C (θ2) sharing center B,
      _empirical_angle_theta0 for A-B-C returns (θ1+θ2)/2 within rtol 1e-6.
    status: pending
  - id: ac-003
    summary: empirical angle K_theta (GAFF Eq.5) matches transcribed reference
    type: scientific
    evaluator_hint: "test: ff::typifier::estimate empirical angle k"
    pass_when: |
      _empirical_angle_k for ≥1 angle with known GAFF reference K_θ (143.9 +
      Z/C factors) reproduces the reference within rtol 1e-3.
    status: pending
  - id: ac-004
    summary: leave-one-out analogy recovers a deleted term within tolerance
    type: scientific
    evaluator_hint: "test: ff::typifier::estimate leave-one-out"
    pass_when: |
      After removing one known bond/angle/dihedral from a loaded force field,
      ParameterEstimator recovers params within bond r0 atol 0.02 Å, angle θ0
      atol 3°, force constants rtol 0.10.
    status: pending
  - id: ac-005
    summary: nearest-analog copied verbatim (not averaged) + provenance
    type: code
    evaluator_hint: "test: ff::typifier::estimate analogy copy"
    pass_when: |
      With an exact-equivalence analog present, the estimated params equal that
      analog's params value-for-value; estimate_method="analogy",
      estimate_analog=analog name, estimate_penalty a float.
    status: pending
  - id: ac-006
    summary: penalty tiers <10 / 10-50 / >50 classified correctly
    type: code
    evaluator_hint: "test: ff::typifier::estimate penalty tiers"
    pass_when: |
      For constructed substitution cases with known total penalty (inner-atom
      ×10 applied), the tier label matches CGenFF bands at boundary values.
    status: pending
  - id: ac-007
    summary: dihedral never fabricates a barrier; multi-periodicity copied as group
    type: scientific
    evaluator_hint: "test: ff::typifier::estimate dihedral"
    pass_when: |
      No analog + no generic wildcard → near-zero |k| (≤ test epsilon) with
      high penalty; a multi-periodicity generic term copies all its
      periodicity/k/phase entries as one group.
    status: pending
  - id: ac-008
    summary: no-match seam opt-in; strict=true unaffected
    type: code
    evaluator_hint: "test: ff::typifier::{opls,gaff} estimator seam"
    pass_when: |
      With no estimator injected, OPLS/GAFF assign behavior is byte-identical to
      pre-estimator; with strict=true a missing term still returns Err even when
      an estimator is attached.
    status: pending
  - id: ac-009
    summary: parmchk2 gold-standard cross-validation (gated)
    type: scientific
    evaluator_hint: "test: ff::typifier::estimate_parity; skips without fixtures"
    pass_when: |
      For a molecule with GAFF missing terms, estimated params agree with
      parmchk2 frcmod within bond r0 0.02 Å / angle θ0 3° / force const rtol
      0.10. Skips cleanly when AmberTools fixtures are absent.
    status: pending
  - id: ac-010
    summary: lint, type check, and test suite clean
    type: runtime
    pass_when: |
      `cargo fmt --all --check`, `cargo clippy --all-targets --all-features
      -- -D warnings`, and `cargo test --all-features` all exit 0.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003**: GAFF 经验公式正确性，常数逐字转录、单测钉死（公式 pin，非拟合容差）。
- **ac-004**: 留一法端到端还原。
- **ac-005 / ac-006**: 方法学约束——最近类比直接复制（不平均）+ 溯源齐全 + CGenFF penalty 分级（内原子 ×10）。
- **ac-007**: 二面角"绝不伪造势垒" + 多重周期整组复制。
- **ac-008**: opt-in 接缝 + strict=true 零介入（与 OPLS/GAFF assign 共享语义）。
- **ac-009**: parmchk2 金标准对照（gated，缺 fixtures 干净跳过）。
- **ac-010**: cargo 质量闸。
