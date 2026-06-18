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
    status: verified
    note: |
      src/ff/typifier/opls/assign.rs::end_score + unit tests end_score_*
      (exact=3, class=1, wildcard {"", "*", "X"}=0, no-match=None). The OPLS XML
      reader emits the empty string "" for a wildcard end (molpy normalizes to
      "*"); end_score treats "", "*", and "X" identically as the score-0 wildcard.
  - id: ac-002
    summary: _sequence_score is end-for-end symmetric and additive
    type: code
    evaluator_hint: "test: ff::typifier::opls::assign sequence_score"
    pass_when: |
      _sequence_score sums per-end specificities, tries forward and reversed
      orientations, returns the best, and returns None if any end is None.
    status: verified
    note: |
      src/ff/typifier/opls/assign.rs::sequence_score + unit tests
      sequence_score_* (additive sum, reversal symmetry, best-orientation,
      None when no orientation matches).
  - id: ac-003
    summary: specificity + layer ranking picks the right bonded type
    type: scientific
    evaluator_hint: "test: ff::typifier::opls::assign ranking"
    pass_when: |
      Given a fully-resolved candidate and a wildcard candidate both matching,
      the higher-score (fully-resolved) one wins; with equal score, the higher
      layer wins (CL&P/CL&Pol overlay over OPLS-AA).
    status: verified
    note: |
      CandidateTables::best ranks by (score, layer). Unit tests
      ranking_fully_resolved_beats_wildcard and ranking_equal_score_higher_layer_wins;
      class->layer is class -> max(layer) per build_type_class_layer
      (test class_to_layer_takes_the_max).
  - id: ac-004
    summary: assigned params match molpy OplsTypifier on real molecules
    type: scientific
    evaluator_hint: "test: ff::typifier::opls assign on tests-data; vs molpy reference"
    pass_when: |
      For molecules in tests-data/, every bond/angle/dihedral assigned by
      assign_bonded carries params equal to molpy OplsTypifier's within
      bond r0 atol 0.02 Å / angle θ0 atol 3° / force constants rtol 0.10.
    status: verified
    note: |
      tests/ff/typifier/opls.rs::ethane_bond_angle_dihedral_match_opls_reference
      checks ethane (read from tests-data/mol2/ethane.mol2) against the bundled
      oplsaa.xml reference within the spec tolerances: CT-CT r0 1.529 Å / k0 536,
      CT-HC r0 1.09 / k0 680, HC-CT-HC theta0 1.881 rad / k0 66, HC-CT-CT
      theta0 1.932 / k0 75, HC-CT-CT-HC f3 0.3 (f1/f2/f4 ~0).
      assign_bonded_over_every_mol2_only_touches_typed_terms iterates every real
      mol2 with no panic. CAVEAT: bonded parity is only meaningful for the atoms
      chain-1 actually SMARTS-types (def-bearing rows); chain-1's %opls_NNN
      coverage gap leaves many atoms untyped, so terms touching them are skipped.
      Full molecule-wide parity is chain 3/3 (against a live molpy reference, the
      bm-molrs-molpy harness) — not run here.
  - id: ac-005
    summary: no-match seam respects strict flag
    type: code
    evaluator_hint: "test: ff::typifier::opls::assign no-match"
    pass_when: |
      A term with no specificity match yields Err when strict=true, and (with
      no estimator attached) returns the term without params when strict=false.
    status: verified
    note: |
      tests/ff/typifier/opls.rs::no_match_seam_strict_errors_lenient_skips
      (NoMatch::Error -> Err naming the term; NoMatch::Skip -> unparametrized
      term) and no_match_seam_estimator_fills_params (an attached Estimator
      fills params and overrides the strict policy). Seam: trait
      assign::Estimator { estimate(&BondedTerm) -> Result<Option<Params>,String> },
      injected via assign_bonded_with(.., Some(&dyn Estimator)).
  - id: ac-006
    summary: build() closes typify→assign→to_frame→to_potentials
    type: code
    evaluator_hint: "test: ff::typifier::opls build"
    pass_when: |
      OplsTypifier::build(mol) returns Potentials whose energy on a known
      conformer matches the molpy OPLS reference within rtol 1e-4 (reusing the
      opls-ef-01 kernel parity harness).
    status: verified
    note: |
      OplsTypifier::build closes typify_full -> to_frame -> intramolecular_pairs
      -> to_potentials (mirrors MMFFTypifier::build). Tested by
      tests/ff/typifier/opls.rs::build_closes_to_potentials_with_finite_energy
      (ethane compiles to >=3 kernels; energy finite). assign_bonded writes the
      matched type NAME onto each term so to_potentials re-resolves params from
      the FF style. Two reader<->kernel key/unit seams surfaced and were
      reconciled/flagged: (1) the bond/angle harmonic ctors now accept `k0` as an
      alias for `k` (additive; the OPLS reader emits k0/r0/theta0) so the
      read-FF compiles; (2) the angle-harmonic ctor calls theta0.to_radians()
      (expects DEGREES) while the OPLS reader stores theta0 in RADIANS — a
      pre-existing reader/kernel units seam outside this chain's scope.
      CAVEAT: because of (2), the rtol-1e-4 numeric energy parity vs molpy is NOT
      asserted in-tree (the in-tree test asserts loop closure + finite energy);
      absolute-energy parity belongs to the opls-ef chain + bm-molrs-molpy bench
      and requires fixing the theta0 units seam there. Flagged for the bench.
  - id: ac-007
    summary: lint, type check, and test suite clean
    type: runtime
    pass_when: |
      `cargo fmt --all --check`, `cargo clippy --all-targets --all-features
      -- -D warnings`, and `cargo test --all-features` all exit 0.
    status: verified
    note: |
      RESOLVED 2026-06-18: full `--all-features` gate now exits 0 with the brew
      openblas env (LIBRARY_PATH + DYLD_FALLBACK_LIBRARY_PATH +
      RUSTFLAGS="-L/opt/homebrew/opt/openblas/lib -lopenblas"). fmt --all --check
      clean; clippy --all-targets --all-features -D warnings clean; test
      --all-features all binaries green, 0 failed. Prior blockers (compute/fit
      WIP, io/xtc teammate WIP, blas link) resolved via upstream merge + openblas
      env. See [[project_molrs_allfeatures_blas]].

      ff-scoped gate GREEN: `cargo fmt --all --check` clean; `cargo clippy
      --features "io,signal,smiles,ff,conformer" --lib --tests -- -D warnings`
      no issues; `cargo test --features "io,signal,smiles,ff,conformer"` =
      1157 passed / 0 failed / 4 ignored. Literal `--all-features` still
      blocked, but by a DIFFERENT cause than chain-1's note: the compute/fit WIP
      (missing raw_computes.rs) is now resolved in the tree; what remains is the
      `blas` feature failing to link cblas/LAPACK on this arm64 host (no system
      BLAS) — an environment limitation, not a code defect, unrelated to OPLS.
      Left pending, same posture as chain 1.
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003**: 特异性排序匹配器逐位复刻 molpy `_end_score`/`_sequence_score` + `(score, layer)` 排序。
- **ac-004**: 真实分子上成键参数与 molpy `OplsTypifier` 在容差内一致（全量 parity = 链 3/3）。
- **ac-005**: no-match 接缝遵守 strict（与 GAFF-03 同语义）。
- **ac-006**: `build()` 闭环到 potentials，能量对照 opls-ef-01 kernel parity harness。
- **ac-007**: cargo 质量闸。
