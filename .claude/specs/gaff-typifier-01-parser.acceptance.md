---
slug: gaff-typifier-01-parser
criteria:
  - id: ac-001
    summary: Parse embedded gaff.dat with expected non-empty section counts
    type: code
    evaluator_hint: "test: amber_dat embedded happy path"
    pass_when: |
      gaff_forcefield() (parsing the include_str!-embedded
      molrs-ff/data/gaff.dat) returns Ok, and the resulting ForceField has
      non-empty atom, bond, angle, dihedral, improper, and pair type sets, each
      matching the count expected for the bundled GAFF 1.x file.
    status: pending
  - id: ac-002
    summary: MASS line parsed with mass, polarizability, and tolerated comment
    type: code
    pass_when: |
      A spot-checked atom-type from the MASS section has params "mass" and
      "polarizability" equal to the file values, parsed correctly even though the
      card carries a trailing free-text comment.
    status: pending
  - id: ac-003
    summary: BOND and ANGLE numeric fields parse correctly
    type: code
    pass_when: |
      A spot-checked bond yields params k and r0 (order-independent lookup) and a
      spot-checked angle yields params k and theta0 equal to the gaff.dat values.
    status: pending
  - id: ac-004
    summary: Negative-PN dihedral folds into one multi-term torsion, all terms kept
    type: scientific
    evaluator_hint: "test: dihedral multi-term X-c2-c2-X"
    pass_when: |
      A known multi-term quartet (e.g. X-c2-c2-X) parses to a single DihedralType
      whose stored term count equals the number of continuation cards; every term
      stores its |PN| (positive) with the matching pk/phase, and no continuation
      term is dropped.
    status: pending
  - id: ac-005
    summary: IMPROPER parsed with no IDIVF, PK used directly
    type: code
    pass_when: |
      A spot-checked improper yields params pk, phase, pn with PK taken directly
      from the card and no idivf param present.
    status: pending
  - id: ac-006
    summary: NONBON MOD4/RE R* and epsilon parsed
    type: code
    pass_when: |
      A spot-checked symbol from the MOD4/RE block has pair params r_star and
      epsilon equal to the gaff.dat values.
    status: pending
  - id: ac-007
    summary: Hydrophilic-atoms line position tolerance
    type: code
    pass_when: |
      A fixture placing the hydrophilic-atoms line at a non-canonical position
      parses without error and all sections after it are populated correctly.
    status: pending
  - id: ac-008
    summary: Parsed parameters round-trip through ForceField query API
    type: code
    pass_when: |
      Entries parsed from the embedded gaff.dat are retrievable via ForceField
      query methods (get_atomtypes / get_bondtype order-independent /
      get_pairtype / get_impropertypes), returning the populated Params.
    status: pending
  - id: ac-009
    summary: Malformed required field returns Err, never silent zero
    type: code
    pass_when: |
      A tiny fixture with a malformed required numeric field causes read_str to
      return Err, with no parameter silently defaulting to zero.
    status: pending
  - id: ac-010
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      cargo fmt --all --check && cargo clippy -- -D warnings && cargo check &&
      cargo test --all-features all succeed for the molrs-ff crate.
    status: pending
  - id: ac-011
    summary: Parsed parameters match openmmforcefields converted ffxml (AmberTools-free ground truth)
    type: scientific
    evaluator_hint: "test: gaff.dat vs openmmforcefields ffxml cross-check"
    pass_when: |
      For the pinned GAFF version, the natively-parsed ForceField's bond, angle,
      dihedral (including all multi-term Fourier terms), and vdW parameters equal
      the values in openmmforcefields' converted OpenMM ffxml for that same
      version (within float tolerance), confirming the native .dat parser
      reproduces the trusted MIT ground truth without an AmberTools install.
    status: pending
  - id: ac-012
    summary: Embedded gaff.dat carries provenance + license header
    type: code
    pass_when: |
      molrs-ff/data/gaff.dat (or the embedding module docstring) records the
      upstream source (openmm/openmmforcefields), commit/version, GAFF version,
      and MIT license note.
    status: pending
---

# Acceptance criteria

- **ac-001** — Embedded-file happy path; the binding proof the real `gaff.dat` is bundled and parses end-to-end.
- **ac-002 / ac-003 / ac-005 / ac-006** — Per-section numeric correctness on spot-checked entries (MASS, BOND/ANGLE, IMPROPER, NONBON).
- **ac-004** — The load-bearing domain invariant: negative-PN multi-term Fourier torsion accumulation. `scientific` type because a dropped term is a silent physics bug, not a code-shape error. Explicit RED test required before implementation.
- **ac-007 / ac-009** — Robustness invariants: position-tolerant hydrophilic line; total (fail-fast) parsing of malformed required fields.
- **ac-008** — Confirms data lands in the real store types, not just counted.
- **ac-010** — Crate-wide check + test gate.
