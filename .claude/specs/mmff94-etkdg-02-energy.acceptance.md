---
slug: mmff94-etkdg-02-energy
criteria:
  - id: ac-001
    summary: Total MMFF94 energy matches RDKit within 1e-3 kcal/mol
    type: scientific
    evaluator_hint: "Embed each molecule with RDKit, read its coords, feed identical coords to molrs MmffForceField and to RDKit MMFFGetMoleculeForceField; compare CalcEnergy."
    pass_when: |
      For each molecule in {ethane "CC", ethylene "C=C", benzene "c1ccccc1", butane "CCCC", caffeine "Cn1cnc2c1c(=O)n(C)c(=O)n2C"} embedded by RDKit ETKDG (seed 0xC0FFEE), feeding the exact RDKit conformer coordinates to MmffForceField::build(...).eval(coords).0 yields a total energy within 1e-3 kcal/mol of RDKit AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy().
    status: verified
    last_checked: \"2026-06-01\"
  - id: ac-002
    summary: Per-term energy breakdown matches RDKit
    type: scientific
    evaluator_hint: "Use RDKit per-contrib energies (or small isolated systems) to validate each term independently."
    pass_when: |
      For the molecules in ac-001, MmffEnergyBreakdown fields {bond, angle, stretch_bend, oop, torsion, vdw, electrostatic} each match the corresponding RDKit MMFF term energy within 1e-3 kcal/mol, and total equals the sum of terms within 1e-9.
    status: verified
    last_checked: \"2026-06-01\"
  - id: ac-003
    summary: Analytical gradient matches finite difference
    type: code
    evaluator_hint: ""
    pass_when: |
      For each molecule in ac-001 at its RDKit conformer plus a random perturbation (max 0.1 Å per coord, fixed RNG seed), the analytical gradient returned by MmffForceField::eval matches central finite difference (step 1e-5 Å) with maximum absolute component error < 1e-5 kcal/mol/Å.
    status: verified
    last_checked: \"2026-06-01\"
  - id: ac-004
    summary: Energy is translation/rotation invariant
    type: code
    evaluator_hint: ""
    pass_when: |
      For benzene, translating all coordinates by an arbitrary vector and applying an arbitrary rigid rotation leaves MmffForceField::eval energy unchanged within 1e-9 kcal/mol.
    status: verified
    last_checked: \"2026-06-01\"
  - id: ac-005
    summary: MMFF94s variant energy matches RDKit MMFF94s
    type: scientific
    evaluator_hint: "Compare with RDKit mmffVariant='MMFF94s'."
    pass_when: |
      For aniline "Nc1ccccc1" at a fixed RDKit conformer, building MmffForceField with the MMFF94s variant gives total energy within 1e-3 kcal/mol of RDKit's MMFF94s force-field energy, and differs from the MMFF94 energy by the same sign/order as RDKit's two variants differ.
    status: pending
    last_checked: ""
  - id: ac-006
    summary: Single eval performance baseline (no O(N^2) regression)
    type: performance
    evaluator_hint: "Record a baseline; flag regressions > 20%."
    pass_when: |
      For a ~50-atom molecule, one MmffForceField::eval (energy + gradient) completes within a recorded baseline wall-clock budget on the dev machine; the measured time is stored and a regression > 20% over the stored baseline fails the criterion.
    status: verified
    last_checked: \"2026-06-01\"
---
