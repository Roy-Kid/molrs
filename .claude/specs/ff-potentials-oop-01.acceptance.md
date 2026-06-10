---
slug: ff-potentials-oop-01
criteria:
  - id: ac-001
    summary: Potential trait is frame-driven (calc_energy/calc_forces)
    type: code
    evaluator_hint: ""
    pass_when: |
      The Potential trait exposes calc_energy(&Frame)->Result<F> and calc_forces(&Frame)->Result<Vec<F>> (and calc_energy_forces); no eval(&[F]) remains. Kernels hold type->param tables, not pre-resolved atom indices.
    status: pending
    last_checked: ""
  - id: ac-002
    summary: Kernel math preserved (finite-difference gradients)
    type: code
    evaluator_hint: "Rewrite existing fd tests to the frame API."
    pass_when: |
      DihedralOPLS, PairCoulCut, harmonic bond/angle: on a hand-built typed frame, calc_energy matches the pre-rewrite value and calc_forces matches central finite difference (h=1e-6) with max component error < 1e-5; Newton's third law holds per term.
    status: pending
    last_checked: ""
  - id: ac-003
    summary: Style.to_potential builds the kernel class; no free ctors
    type: code
    evaluator_hint: ""
    pass_when: |
      Each Style yields its Potential via Style::to_potential(); the free *_ctor functions and the KernelRegistry free-fn map are gone. ForceField::to_potentials() collects one Potential per style with no frame argument.
    status: pending
    last_checked: ""
  - id: ac-004
    summary: Potentials is molecule-independent
    type: code
    evaluator_hint: ""
    pass_when: |
      One Potentials built from a ForceField, evaluated via calc_energy/calc_forces on two different frames (same type vocabulary, different coords and atom counts), yields each frame's correct energy/forces. No compile(frame) exists.
    status: pending
    last_checked: ""
  - id: ac-005
    summary: LBFGS optimizer class relaxes single + batch
    type: code
    evaluator_hint: "Mirror molpy LBFGS(potential).run(structure)."
    pass_when: |
      LBFGS::new(potential, cfg).run(frame) relaxes a harmonic system to fmax<cfg.fmax with converged=true (energy non-increasing); run_batch over a homogeneous set returns one OptReport per frame matching per-frame run within 1e-9; rayon and serial agree. No free minimize/minimize_batch remain.
    status: pending
    last_checked: ""
  - id: ac-006
    summary: Real MMFF relax via the class API
    type: scientific
    evaluator_hint: ""
    pass_when: |
      Ethane relaxes via LBFGS over an MMFF Potentials (built through to_potentials) to fmax<0.05, energy non-increasing — parity with the prior lbfgs_minimize_relaxes_mmff_ethane result under the new API.
    status: pending
    last_checked: ""
  - id: ac-007
    summary: ETKDG conformer generation unchanged
    type: scientific
    evaluator_hint: ""
    pass_when: |
      The molrs-conformer suite passes unchanged: ETKDG still uses the private flat-coords L-BFGS engine (minimize_lbfgs_rms), not the public class, so conformer output is bit-for-bit identical.
    status: pending
    last_checked: ""
  - id: ac-008
    summary: PyO3 surface is the OOP one
    type: code
    evaluator_hint: ""
    pass_when: |
      Python exposes molrs.LBFGS(potentials, ...).run(frame), ForceField.to_potentials(), Potentials.calc_energy(frame)/calc_forces(frame). ForceField.compile, Potentials.eval, Potentials.minimize/minimize_batch are removed.
    status: pending
    last_checked: ""
---

# Acceptance — molrs-ff OOP rewrite

Binding contract for `ff-potentials-oop-01.md`. Kernel math (ac-002, ac-006) is
preserved from this session's validated kernels; only the calling shape changes.
ac-007 guards the ETKDG engine reuse.
