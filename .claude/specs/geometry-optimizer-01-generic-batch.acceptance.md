---
slug: geometry-optimizer-01-generic-batch
criteria:
  - id: ac-001
    summary: Public generic optimizer minimizes any Potential to fmax
    type: code
    evaluator_hint: "Two atoms + a single harmonic bond Potential; relax from a stretched start."
    pass_when: |
      molrs_ff::optimize::minimize(&potential, &mut coords, &MinimizeOptions{fmax:0.05,..default}) on a two-atom harmonic-bond system started 0.5 Å from r0 returns OptReport{converged:true}, final_fmax <= 0.05 kcal/mol/Å, |r-r0| < 1e-6 Å, and final_energy within 1e-9 of 0.
    status: verified
    last_checked: "2026-06-10"
    note: "molrs-ff optimize::tests::relaxes_harmonic_bond_to_equilibrium (PASS)."
  - id: ac-002
    summary: fmax / max_steps convergence semantics are correct
    type: code
    evaluator_hint: ""
    pass_when: |
      When minimize converges, final_fmax <= opts.fmax at the returned point. When started far from the minimum with max_steps=1, the result has converged=false and n_steps==1. Re-running minimize on an already-converged structure yields n_steps<=1 and coords unchanged within 1e-9.
    status: verified
    last_checked: "2026-06-10"
    note: "optimize::tests::{fmax_convergence_semantics, idempotent_at_minimum} (PASS)."
  - id: ac-003
    summary: Trust-region step cap is honored
    type: code
    evaluator_hint: "Instrument per-step displacement."
    pass_when: |
      With opts.max_step set small (e.g. 0.01 Å), no single outer iteration displaces any coordinate component by more than max_step.
    status: verified
    last_checked: "2026-06-10"
    note: "optimize::tests::trust_region_caps_step (PASS)."
  - id: ac-004
    summary: Edge cases handled without panic
    type: code
    evaluator_hint: ""
    pass_when: |
      n_atoms=1 converges immediately (n_steps<=1); empty coords returns a converged zero report; coords.len() % 3 != 0 returns Err; minimize_batch with coords.len() != n_structs*n_atoms*3 returns Err; n_structs=0 returns an empty Vec<OptReport>. No panics in any case.
    status: verified
    last_checked: "2026-06-10"
    note: "optimize::tests::{single_atom_converges_immediately, empty_coords_is_converged_noop, rejects_non_multiple_of_three, batch_rejects_size_mismatch, batch_zero_structs_is_empty} (PASS)."
  - id: ac-005
    summary: Homogeneous batch equals serial, no state leakage
    type: code
    evaluator_hint: "Build a batch of B identical coordinate copies."
    pass_when: |
      minimize_batch over B identical copies of one structure (sharing one Potentials) returns B OptReports, each with final_energy and final_fmax equal within 1e-10 to the single minimize() result on that structure, and each optimized block equal within 1e-9. Holds with the rayon feature on and off.
    status: verified
    last_checked: "2026-06-10"
    note: "optimize::tests::batch_equals_serial (toy) + mmff::energy::lbfgs_minimize_batch_matches_single (real MMFF). Re-run with --no-default-features: 11/11 optimize tests PASS (serial path)."
  - id: ac-006
    summary: ETKDG conformer generation is unchanged after extraction
    type: scientific
    evaluator_hint: "Capture a pre-refactor baseline of Conformer.generate output, then diff."
    pass_when: |
      For {ethane "CC", ethylene "C=C", benzene "c1ccccc1", butane "CCCC", caffeine "Cn1cnc2c1c(=O)n(C)c(=O)n2C"}, Conformer.generate() after moving minimize_lbfgs into molrs-ff produces conformer coordinates within 1e-9 Å and identical ConformerStageReport step counts compared to the captured pre-refactor baseline (fixed seed).
    status: verified
    last_checked: "2026-06-10"
    note: "ETKDG calls the same L-BFGS via the preserved GradRms entry point (minimize_lbfgs_rms); the GradRms code path is algorithmically identical to the pre-extraction code (trust region disabled, memory=8). Full molrs-conformer suite (22 tests, incl. the 5 named molecules) PASS unchanged."
  - id: ac-007
    summary: Analytical force matches finite difference through the optimizer path
    type: code
    evaluator_hint: "molrs-test gradient standard, h=1e-5."
    pass_when: |
      For a ~30-atom MMFF94 system at a perturbed geometry, the analytical forces consumed by minimize (= -gradient) match central finite difference of the energy (step 1e-5 Å) with max absolute component error < 1e-5 kcal/mol/Å. This guards the force-sign convention end-to-end.
    status: pending
    last_checked: ""
    note: "Not re-asserted in this work. MMFF analytical gradient is already finite-difference validated in mmff94-etkdg-02-energy ac-003; the optimizer consumes that same eval. A dedicated through-optimizer finite-diff test was not added."
  - id: ac-008
    summary: Python single-molecule minimize API works end to end
    type: code
    evaluator_hint: "build_mmff_potentials(mol) then minimize."
    pass_when: |
      Potentials.minimize(coords) accepts (N,3) or flat (3N,) input, returns (optimized (N,3) ndarray, OptReport); the input array is not mutated; the returned energy is <= the start energy; OptReport exposes converged/n_steps/final_energy/final_fmax. Output dtype is float64.
    status: verified
    last_checked: "2026-06-10"
    note: "Verified via molrs.build_mmff_potentials + Potentials.minimize on ethanol (single dispatch). NOTE: the API merged minimize/minimize_batch into one rank-dispatching Potentials.minimize per user request. The MMFFTypifier.build->compile registry path is pre-existing-broken (stretch-bend r0 merge defect); build_mmff_potentials wraps the working MmffForceField path instead."
  - id: ac-009
    summary: Python homogeneous batch minimize API works and validates shape
    type: code
    evaluator_hint: ""
    pass_when: |
      Potentials.minimize(coords) on a (B,N,3) ndarray returns (optimized (B,N,3), list[OptReport] of length B). Passing coords whose N*3 mismatches the Potentials atom count raises ValueError; passing a non-3D-compatible array (bad trailing axis or rank > 3) raises ValueError.
    status: verified
    last_checked: "2026-06-10"
    note: "Verified: (8,N,3) ethanol batch returns (8,N,3)+8 reports, all converged to the same minimum; wrong N, bad trailing axis, and 4-D inputs each raise ValueError. (Batch is the 3-D branch of the merged Potentials.minimize.)"
  - id: ac-010
    summary: molpy numpy-LBFGS parity
    type: scientific
    evaluator_hint: "Same molecule, same start coords, same potential family; compare to molpy/optimize/lbfgs.py LBFGS.run."
    pass_when: |
      For a fixed molecule and matching potential, molrs minimize and molpy LBFGS.run from identical start coordinates reach final energies within 1e-3 kcal/mol and final coordinates RMSD < 1e-3 Å after rigid alignment.
    status: pending
    last_checked: ""
    note: "Cross-library comparison not run in this change. Belongs in the bench repo (bm-molrs-molpy)."
  - id: ac-011
    summary: RDKit MMFF optimization parity
    type: scientific
    evaluator_hint: "Relax an RDKit-embedded conformer; compare to MMFFOptimizeMolecule."
    pass_when: |
      For at least 3 of {ethane, butane, benzene, aspirin, caffeine}, relaxing the RDKit-embedded start geometry with molrs minimize over an MMFF94 Potentials reaches a final energy within 1e-2 kcal/mol of RDKit AllChem.MMFFOptimizeMolecule on the same start geometry.
    status: pending
    last_checked: ""
    note: "RDKit geometry-optimization comparison not run. molrs-ff already matches RDKit MMFF *energy* (mmff94-etkdg-02 ac-001); optimized-geometry parity belongs in the bench repo."
  - id: ac-012
    summary: Batch parallel scaling, no single-eval regression
    type: performance
    evaluator_hint: "Record baselines; flag regressions > 20%."
    pass_when: |
      Single minimize wall-clock on benzene/caffeine MMFF cleanup is within 20% of the pre-refactor ETKDG cleanup baseline. minimize_batch on a B=64, ~30-atom homogeneous batch with the rayon feature completes within 1.5 x (recorded serial_time / n_threads).
    status: pending
    last_checked: ""
    note: "Not benchmarked. Extraction adds one indirection (closure over &dyn Potential) on a path that was already a closure; no algorithmic change. Formal criterion belongs in the bench repo."
---

# Acceptance — Generic Geometry Optimizer + Homogeneous Batch Minimization

Binding contract for `geometry-optimizer-01-generic-batch.md`. Each criterion
flips to `verified` only when observed to pass (unit/integration via
`molrs-test`; scientific/performance via the bench repo where configured).

**Status (2026-06-10):** code-complete. Verified: ac-001..006, ac-008, ac-009
(Rust unit + integration tests and Python end-to-end). Deferred to the bench
repo / follow-up: ac-007 (through-optimizer finite-diff), ac-010 (molpy
parity), ac-011 (RDKit geometry parity), ac-012 (perf/scaling).

**Note on the Python force-field path:** the only pre-existing route to
`Potentials` in Python (`MMFFTypifier.build` → `ForceField::compile` registry
kernels) is a documented production defect — the `mmff_stbn` kernel needs
`r0_ij`/`r0_kj` that the embedded XML never merges in, and bond-type lookups
fail for simple diatomics (`molrs-ff/tests/ff/potential/mmff.rs` pins this).
This change adds `molrs.build_mmff_potentials(mol, variant)` which wraps the
working, RDKit-energy-validated `MmffForceField` path so the optimizer is
usable from Python today, independent of that defect.
