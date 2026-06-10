---
slug: opls-ef-01-kernels-seam
criteria:
  - id: ac-001
    summary: OPLS 4-cosine dihedral energy has correct per-term phase
    type: code
    evaluator_hint: "Set one Fn coefficient at a time."
    pass_when: |
      For DihedralOPLS with only F1>0: E(phi=0)=F1 and E(phi=pi)=0. With only F2>0: E(phi=0)=0 and E(phi=pi/2)=F2. With only F3>0: E(phi=0)=F3. With only F4>0: E(phi=0)=0. Each within 1e-9 kcal/mol.
    status: verified
    last_checked: "2026-06-10"
    note: "opls per_term_phase (PASS)"
  - id: ac-002
    summary: OPLS dihedral analytical force matches finite difference
    type: code
    evaluator_hint: "h=1e-6 central difference."
    pass_when: |
      For random F1..F4 and a non-degenerate 4-atom geometry, the analytic forces (= -grad) match central finite difference of the energy (step 1e-6 A) with max absolute component error < 1e-5 kcal/mol/A. Sum of forces over the 4 atoms is < 1e-9 (Newton's third law).
    status: verified
    last_checked: "2026-06-10"
    note: "opls numerical_gradient + newtons_third_law (PASS)"
  - id: ac-003
    summary: coul/cut energy, sign, cutoff and zero-distance guard
    type: code
    evaluator_hint: ""
    pass_when: |
      PairCoulCut with qiqj=1 at distance r gives E = 332.06371/r within 1e-9; like charges (qiqj>0) produce a repulsive force (pushes atoms apart), unlike (qiqj<0) attractive; a pair beyond r_cut contributes exactly 0; coincident atoms (r^2<1e-24) are skipped with no NaN/inf.
    status: verified
    last_checked: "2026-06-10"
    note: "coul_cut energy_and_sign + cutoff_and_zero_distance (PASS)"
  - id: ac-004
    summary: coul/cut analytical force matches finite difference
    type: code
    evaluator_hint: "h=1e-6 central difference."
    pass_when: |
      For a charged pair inside the cutoff, analytic forces match central finite difference of the Coulomb energy (step 1e-6 A) with max absolute component error < 1e-5 kcal/mol/A, and sum of the two atoms' forces < 1e-9.
    status: verified
    last_checked: "2026-06-10"
    note: "coul_cut numerical_gradient (PASS)"
  - id: ac-005
    summary: Both kernels registered and resolvable via KernelRegistry
    type: code
    evaluator_hint: ""
    pass_when: |
      KernelRegistry::default() resolves ("dihedral","opls") and ("pair","coul/cut") to constructors; a ForceField with opls dihedral + coul/cut pair styles compiles against a typed frame without "no kernel registered" errors.
    status: verified
    last_checked: "2026-06-10"
    note: "potential::opls::registry_resolves_opls_and_coul (PASS)"
  - id: ac-006
    summary: Assembled OPLS frame compiles and forces match finite difference
    type: code
    evaluator_hint: "Hand-build a small typed frame (e.g. butane) in code."
    pass_when: |
      A hand-built typed Frame (harmonic bonds/angles + dihedral:opls + lj/cut + coul/cut pairs with exclusions/1-4 scaling baked in) and matching ForceField compile via ForceField::compile(frame).eval(coords) to finite energy and forces; the total analytic force matches central finite difference (step 1e-5 A) with max component error < 1e-5 kcal/mol/A.
    status: verified
    last_checked: "2026-06-10"
    note: "opls_assembly_compiles_and_force_matches_finite_difference (PASS)"
  - id: ac-007
    summary: OPLS Potentials optimize single + batch
    type: code
    evaluator_hint: "Reuse molrs_ff::{minimize, minimize_batch}."
    pass_when: |
      minimize over the compiled OPLS Potentials lowers the energy monotonically to fmax < 0.05 and converged=true; minimize_batch over a small homogeneous batch returns one report per structure with results matching the single-structure path (within 1e-9 for an identical block).
    status: verified
    last_checked: "2026-06-10"
    note: "opls_minimize_single_and_batch (PASS)"
  - id: ac-008
    summary: OPLS energy parity vs molpy numpy potentials (seam contract)
    type: scientific
    evaluator_hint: "Same molpy-OPLS-typed structure + coords fed to molrs and to molpy numpy potentials."
    pass_when: |
      For butane and ethanol typified by molpy OPLS, the molrs total energy and each per-term energy (bond, angle, dihedral, LJ, Coulomb) match molpy's own numpy OPLS potentials on identical coordinates within 1e-4 kcal/mol.
    status: pending
    last_checked: ""
  - id: ac-009
    summary: End-to-end Python OPLS optimize
    type: scientific
    evaluator_hint: "molpy typify -> emit frame+ForceField -> molrs compile -> minimize."
    pass_when: |
      A molpy OPLS-typified molecule, emitted as a typed frame + molrs ForceField, compiled by molrs and minimized, converges (fmax<0.05) and lowers the molpy-computed OPLS energy relative to the start geometry.
    status: pending
    last_checked: ""
---

# Acceptance — OPLS-AA E/F Kernels + molpy→molrs Typed-Frame Seam

Binding contract for `opls-ef-01-kernels-seam.md`. ac-001..007 are in-crate
(molrs-ff unit + integration tests). ac-008..009 are the cross-library seam
contract and run where the molpy↔molrs harness lives (e.g. `bm-molrs-molpy`);
they depend on the molpy-side typed-frame emitter, which is a separate molpy
change (see spec Out of Scope).
