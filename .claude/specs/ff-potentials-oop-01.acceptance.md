---
slug: ff-potentials-oop-01
criteria:
  - id: ac-001
    summary: Potential trait is coords-driven (calc_energy_forces); params baked at build
    type: code
    evaluator_hint: "molpy model: to_potentials(frame) bakes per-element param arrays; calc_* takes coords only."
    pass_when: |
      The Potential trait exposes calc_energy_forces(&[F])->(F,Vec<F>) (required) + default calc_energy/calc_forces; no eval(&[F]) remains. Per the molpy model the molecule binding (string type-label -> per-element/per-bond param arrays) is baked once at to_potentials(frame) build time, so calc_* takes only flat coords — no per-call frame/topology re-extraction.
    status: pass
    last_checked: "2026-06-10"
  - id: ac-002
    summary: Kernel math preserved (finite-difference gradients)
    type: code
    evaluator_hint: "Rewrite existing fd tests to the frame API."
    pass_when: |
      DihedralOPLS, PairCoulCut, harmonic bond/angle: on a hand-built typed frame, calc_energy matches the pre-rewrite value and calc_forces matches central finite difference (h=1e-6) with max component error < 1e-5; Newton's third law holds per term.
    status: pass
    last_checked: "2026-06-10"
  - id: ac-003
    summary: Style.to_potential builds the kernel class; no free ctors
    type: code
    evaluator_hint: ""
    pass_when: |
      Each Style yields its Potential via Style::to_potential(frame); the KernelRegistry free-fn map and ForceField::compile are gone. ForceField::to_potentials(frame) collects one Potential per style (atom styles -> None, unknown -> Err), baking per-element params from the frame's type labels.
    status: pass
    last_checked: "2026-06-10"
  - id: ac-004
    summary: Potentials is molecule-bound (built per frame), then coords-only
    type: code
    evaluator_hint: "Mirrors molpy: ForceField.to_potentials(frame) bakes the molecule; calc_* then varies only coords."
    pass_when: |
      Potentials is built once via ForceField::to_potentials(frame) (baking that molecule's per-element params); calc_energy_forces then takes only coords, so relaxation/finite-diff over many coordinate sets reuse the same Potentials with no per-call frame work. ForceField::compile(frame) no longer exists.
    status: pass
    last_checked: "2026-06-10"
  - id: ac-005
    summary: LBFGS optimizer class relaxes single + batch
    type: code
    evaluator_hint: "Mirror molpy LBFGS(potential).run(structure)."
    pass_when: |
      LBFGS::new(&potential, cfg).run(coords) relaxes a harmonic system to fmax<cfg.fmax with converged=true (energy non-increasing); run_batch(coords, n_atoms, n_structs) over a homogeneous set returns one OptReport per structure matching per-structure run_one within 1e-9; rayon and serial agree. No free minimize/minimize_batch remain.
    status: pass
    last_checked: "2026-06-10"
  - id: ac-006
    summary: Real MMFF relax via the class API
    type: scientific
    evaluator_hint: ""
    pass_when: |
      Ethane relaxes via LBFGS over an MMFF potential to fmax<0.05, energy non-increasing — parity with the prior lbfgs_minimize_relaxes_mmff_ethane result under the new class API.
    status: pass
    last_checked: "2026-06-10"
  - id: ac-007
    summary: ETKDG conformer generation unchanged
    type: scientific
    evaluator_hint: ""
    pass_when: |
      The molrs-conformer suite passes unchanged: ETKDG still uses the private flat-coords L-BFGS engine (minimize_lbfgs_rms) over MmffForceField::calc_energy_forces, not the public class, so conformer output is bit-for-bit identical.
    status: pass
    last_checked: "2026-06-10"
  - id: ac-008
    summary: PyO3 surface is the OOP one
    type: code
    evaluator_hint: ""
    pass_when: |
      Python exposes molrs.LBFGS(potentials, fmax=...).run(coords) (single (N,3)/(3N,) or batch (B,N,3)), ForceField.to_potentials(frame), Potentials.calc_energy_forces(coords)/calc_energy/calc_forces. ForceField.compile, Potentials.eval/energy, Potentials.minimize/minimize_batch are removed; .pyi + __init__ updated.
    status: pass
    last_checked: "2026-06-10"
---

# Acceptance — molrs-ff OOP rewrite

Binding contract for `ff-potentials-oop-01.md`. Kernel math (ac-002, ac-006) is
preserved from this session's validated kernels; only the calling shape changes.
ac-007 guards the ETKDG engine reuse.
