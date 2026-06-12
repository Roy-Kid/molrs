---
slug: etkdg-smarts-02-torsions
criteria:
  - id: ac-001
    summary: Per-bond experimental torsion params match RDKit
    type: scientific
    evaluator_hint: "Compare per-rotatable-bond assigned SMARTS + (V[6],s[6]) to RDKit getExperimentalTorsions. Needs RDKit (installed)."
    pass_when: |
      For {butane "CCCC", biphenyl, alanine "C[C@@H](N)C(=O)O", methyl acetate "COC(C)=O", acetamide "CC(N)=O", a 12-membered carbocycle "C1CCCCCCCCCCC1"}, each rotatable bond's assigned torsion params (the six V_k and sign_k, and the winning pattern) equal RDKit's ETKDGv3 getExperimentalTorsions output for the same bond within 1e-3 on the V_k and exact on signs/multiplicity; first-match-wins ordering matches RDKit.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-002
    summary: Macrocycle vs small-ring layering matches RDKit
    type: code
    evaluator_hint: ""
    pass_when: |
      Ring bonds of a >=9-membered ring receive macrocycle-table torsions; small-ring bonds receive small-ring-table torsions; acyclic bonds receive v2 torsions — the table-of-origin per bond matches RDKit's selection for the test molecules.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-003
    summary: No representative-subset fallback remains
    type: code
    evaluator_hint: ""
    pass_when: |
      The prior representative-subset code path in torsion_prefs.rs is removed; every torsion assignment flows through the SMARTS-driven full tables (grep shows no hardcoded subset patterns; build_constraints signature unchanged).
    status: verified
    last_checked: "2026-06-01"
  - id: ac-004
    summary: ETKDG conformer RMSD vs RDKit tightened incl. alanine
    type: scientific
    evaluator_hint: "Re-run generate_3d vs RDKit ETKDGv3 best-fit RMSD; closes mmff94-etkdg-04 ac-001."
    pass_when: |
      For {ethanol, butane, benzene, alanine}, symmetry-aware best-fit heavy-atom RMSD of the molrs conformer vs RDKit ETKDGv3 is < 0.5 Å for ALL four (alanine specifically improves from the prior 0.755 Å to < 0.5 Å), and each conformer's MMFF94 energy is within 10% relative of RDKit's.
    status: verified
    last_checked: "2026-06-01"
---
