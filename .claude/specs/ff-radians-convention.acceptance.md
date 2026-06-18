---
slug: ff-radians-convention
criteria:
  - id: ac-001
    summary: OPLS angle energy is ~0 at equilibrium (the bug, now fixed)
    type: scientific
    evaluator_hint: "test: ff::potential::opls equilibrium angle energy"
    pass_when: |
      For an OPLS-typified molecule held at its reference angle θ0, the
      harmonic-angle energy contribution is ~0 (|E_angle| < 1e-6 kcal/mol),
      not the ~264 kcal/mol spurious value the double-conversion produced.
    status: pending
  - id: ac-002
    summary: kernels consume radians; no .to_radians() in kernel ctors
    type: code
    evaluator_hint: "test: grep + angle/dihedral/improper unit tests"
    pass_when: |
      angle (harmonic/class2/mmff), dihedral (periodic/charmm/class2), and
      improper (periodic/harmonic) kernels no longer call .to_radians() on
      theta0/phase/chi0; their doc-blocks state radians-in.
    status: pending
  - id: ac-003
    summary: LAMMPS + MMFF readers normalize angle/phase/chi0 to radians at read
    type: code
    evaluator_hint: "test: ff::readers::lammps stores radians; mmff theta0 radians"
    pass_when: |
      The LAMMPS reader stores angle theta0, dihedral phase, and improper chi0 in
      radians (deg→rad at read); the MMFF typifier stores theta0 in radians. The
      OPLS reader is unchanged (already radians). Convention-pinning tests
      (angle_phase_stays_in_degrees, compile_path_converts_degrees_to_radians)
      are inverted to assert radians-at-storage.
    status: pending
  - id: ac-004
    summary: LAMMPS angle energies unchanged (no regression)
    type: scientific
    evaluator_hint: "test: ff::potential angle (LAMMPS-sourced)"
    pass_when: |
      LAMMPS-sourced angle-harmonic energies/forces are numerically identical
      before and after the migration (the deg→rad moved from kernel to reader,
      net behavior preserved). Existing angle energy tests pass.
    status: pending
  - id: ac-005
    summary: MMFF RDKit parity unchanged (no regression)
    type: scientific
    evaluator_hint: "test: ff::typifier::mmff + mmff energy parity"
    pass_when: |
      MMFF angle-bend + stretch-bend energies retain RDKit parity (e_ethane
      ~2.3e-5); the standalone mmff/energy/* path is audited and unaffected.
    status: pending
  - id: ac-006
    summary: lint, type check, and test suite clean (atomic landing)
    type: runtime
    pass_when: |
      `cargo fmt --all --check`, `cargo clippy --features "io,signal,smiles,ff,
      conformer" --lib --tests -- -D warnings`, and the ff-feature `cargo test`
      all exit 0, with all changes landed in one commit (no half-migration).
      (--all-features may stay blocked by unrelated blas/teammate WIP.)
    status: pending
---

# Acceptance criteria

- **ac-001**: the actual bug — OPLS equilibrium angle energy ≈ 0 (RED-first; currently ~264 kcal/mol).
- **ac-002 / ac-003**: convention migration — kernels consume radians, readers normalize at boundary (LAMMPS/MMFF deg→rad; OPLS unchanged); guard tests inverted.
- **ac-004 / ac-005**: no regression — LAMMPS angle + MMFF RDKit parity preserved (the migration must net-preserve their energies). These are the atomic-landing guards.
- **ac-006**: cargo gate + single-commit atomicity.
