---
slug: core-perception-01-aromaticity
criteria:
  - id: ac-001
    summary: Per-atom aromatic flag matches RDKit default model
    type: scientific
    evaluator_hint: "Compare is_aromatic per atom to RDKit Atom.GetIsAromatic() (default AROMATICITY_RDKIT). Needs RDKit 2026.03.2 (installed)."
    pass_when: |
      For each molecule in {benzene "c1ccccc1", pyridine "c1ccncc1", pyrrole "c1cc[nH]c1", imidazole "c1cnc[nH]1", furan "c1ccoc1", naphthalene "c1ccc2ccccc2c1", indole "c1ccc2[nH]ccc2c1", caffeine "Cn1cnc2c1c(=O)n(C)c(=O)n2C", biphenyl "c1ccc(-c2ccccc2)cc1", cyclohexane "C1CCCCC1", cyclopentadiene "C1=CCC=C1", phenol "Oc1ccccc1"}, after add_hydrogens and Atomistic::perceive_aromaticity, every atom's is_aromatic flag equals RDKit mol.GetAtomWithIdx(i).GetIsAromatic() (same SDF atom order).
    status: verified
    last_checked: "2026-06-01"
  - id: ac-002
    summary: Per-bond aromatic flag matches RDKit
    type: scientific
    evaluator_hint: ""
    pass_when: |
      For the ac-001 set, every bond flagged aromatic (order set to 1.5) corresponds exactly to RDKit bonds with GetIsAromatic()==true, and non-aromatic bonds are not given order 1.5.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-003
    summary: Non-aromatic rings are not mislabeled
    type: code
    evaluator_hint: ""
    pass_when: |
      cyclohexane and 1,3-cyclopentadiene have zero atoms flagged aromatic after perceive_aromaticity; the return count is 0 for both.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-004
    summary: SMARTS aromatic primitives match RDKit end-to-end after perception
    type: scientific
    evaluator_hint: "After native perception (not transplanted flags), SMARTS a/c queries must still match RDKit."
    pass_when: |
      After Atomistic::perceive_aromaticity (native, NOT RDKit-transplanted flags), SmartsPattern matches for patterns {"[c:1]", "[a:1][a:2]", "[cX3H1:1]", "[n:1]"} on the ac-001 molecules equal RDKit GetSubstructMatches(uniquify=False) sets.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-005
    summary: Perception is idempotent
    type: code
    evaluator_hint: ""
    pass_when: |
      Calling perceive_aromaticity twice in succession yields identical is_aromatic flags and bond orders, and the second call returns the same count as the first.
    status: verified
    last_checked: "2026-06-01"
---
