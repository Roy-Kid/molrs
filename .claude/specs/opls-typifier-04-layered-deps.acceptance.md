---
slug: opls-typifier-04-layered-deps
criteria:
  - id: ac-001
    summary: SMARTS %LABEL parses to a context-label atom query
    type: code
    evaluator_hint: "test: core::chem::smarts %label parse"
    pass_when: |
      Parsing a def containing `[C;%opls_145]` yields an AtomQuery variant
      carrying the label "opls_145" (HasContextLabel), and a def with no `%`
      parses unchanged.
    status: pending
  - id: ac-002
    summary: context-label matcher predicate consults the assignment map
    type: code
    evaluator_hint: "test: core::chem::smarts find_matches_with_labels"
    pass_when: |
      find_matches_with_labels(query, mol, &labels) matches an atom carrying a
      HasContextLabel(L) constraint iff labels[atom] == L; the legacy
      find_matches (empty labels) is byte-for-byte unaffected on non-%-patterns.
    status: pending
  - id: ac-003
    summary: dependency extraction + Kahn levels + Tarjan SCC
    type: code
    evaluator_hint: "test: ff::typifier::opls::deps"
    pass_when: |
      OplsDependencyAnalyzer extracts %opls_NNN deps per def; assigns level 0 to
      no-dep defs, level k to defs depending only on <k; a constructed A→B→A
      cycle is detected as one circular group placed at max_level+1.
    status: pending
  - id: ac-004
    summary: LayeredTypingEngine level ordering + fixed-point convergence
    type: code
    evaluator_hint: "test: ff::typifier::opls::layered"
    pass_when: |
      The engine resolves levels in order; a circular group iterates to a fixed
      point (assignment equality) within max_iterations (10) and terminates.
    status: pending
  - id: ac-005
    summary: real molecules requiring %-deps type correctly (coverage of the 133 defs)
    type: scientific
    evaluator_hint: "test: ff::typifier::opls layered on tests-data; vs molpy"
    pass_when: |
      Over tests-data/ molecules needing layered typing — benzene
      (opls_145→opls_146), toluene (→opls_148), an alcohol (opls_154→opls_155),
      a triol (→opls_171) — every atom gets the same opls_NNN type as molpy
      OplsTypifier; the previously-skipped %opls_NNN defs are now covered.
    status: pending
  - id: ac-006
    summary: chain-1 standalone path preserved as level-0
    type: code
    evaluator_hint: "test: ff::typifier::opls regression"
    pass_when: |
      Molecules typeable by standalone SMARTS alone (chain-1) still type
      identically after integration (annotate_opls level-0 path unchanged in
      result), with no regression in the 32 existing OPLS tests.
    status: pending
  - id: ac-007
    summary: lint, type check, and test suite clean
    type: runtime
    pass_when: |
      `cargo fmt --all --check`, `cargo clippy --features "io,signal,smiles,ff,
      conformer" --lib --tests -- -D warnings`, and the ff-feature `cargo test`
      all exit 0. (Literal --all-features may remain blocked by unrelated
      compute WIP — note it, as in chain-1/2.)
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002**: 通用 context-label SMARTS 扩展（`%LABEL` → `HasContextLabel` + 查赋型表谓词），旧入口零回归。
- **ac-003 / ac-004**: 依赖分析（Kahn + Tarjan）+ 分层定点驱动，复刻 molpy。
- **ac-005**: 真实分子上覆盖 133 个 `%opls_NNN` def，per-atom 与 molpy 一致（chain-3 全量 parity 的前置覆盖）。
- **ac-006**: chain-1 独立 SMARTS 路径作为 level-0 保留、零回归。
- **ac-007**: cargo 质量闸（ff feature set；`--all-features` 受无关 compute WIP 阻塞时注明）。
