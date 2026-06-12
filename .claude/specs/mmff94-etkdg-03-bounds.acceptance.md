---
slug: mmff94-etkdg-03-bounds
criteria:
  - id: ac-001
    summary: Smoothed bounds matrix matches RDKit within 1e-3
    type: scientific
    evaluator_hint: "Compare molrs smoothed BoundsMatrix to RDKit rdDistGeom.GetMoleculeBoundsMatrix per element. Needs RDKit (installed)."
    pass_when: |
      For each molecule in {butane "CCCC", cyclohexane "C1CCCCC1", benzene "c1ccccc1", biphenyl "c1ccc(-c2ccccc2)cc1", glycine "NCC(=O)O"} after add_hydrogens, every element of the molrs smoothed bounds matrix (lower[i][j], upper[i][j]) is within 1e-3 Å of RDKit GetMoleculeBoundsMatrix(mol) with the same atom ordering.
    status: verified
    last_checked: \"2026-06-01\"
  - id: ac-002
    summary: Smoothing yields a triangle-consistent matrix
    type: code
    evaluator_hint: ""
    pass_when: |
      After smooth_bounds on each ac-001 molecule, for all triples (i,j,k): upper[i][j] <= upper[i][k] + upper[k][j] + 1e-9 and lower[i][j] >= (lower[i][k] - upper[k][j]) - 1e-9, and lower[i][j] <= upper[i][j] for all i,j. Injecting an inconsistent bound (lower>upper) makes smooth_bounds return Err or report the inconsistency.
    status: verified
    last_checked: \"2026-06-01\"
  - id: ac-003
    summary: ETKDGv3 experimental torsion constraints present and correct
    type: scientific
    evaluator_hint: "Compare experimental torsion terms (atoms + preferred angles/V) to RDKit getExperimentalTorsions for the same bond."
    pass_when: |
      For butane "CCCC" and biphenyl, build_constraints(mol, Etkdgv3) returns experimental_torsions covering the central rotatable bond(s); the matched atom quartets and their preferred-angle parameters agree with RDKit's ETKDGv3 experimental-torsion assignment (same multiplicity and preferred minima within 5 degrees).
    status: verified
    last_checked: "2026-06-01"
  - id: ac-004
    summary: Macrocycle gets ETKDGv3 ring-specific terms
    type: code
    evaluator_hint: ""
    pass_when: |
      For a 12-membered carbocycle "C1CCCCCCCCCCC1", build_constraints(mol, Etkdgv3) produces ring/torsion constraints flagged as macrocycle-specific (the ETKDGv3 large-ring path), distinct from the constraint set produced for a 6-membered ring of the same atom type.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-005
    summary: Chiral constraint sign matches CIP assignment
    type: scientific
    evaluator_hint: ""
    pass_when: |
      For (R)-bromochlorofluoromethane "[C@H](F)(Cl)Br", the generated chiral constraint volume sign is consistent with the R configuration (matches the sign RDKit DistGeom uses for the same center), and flipping to the S SMILES flips the constraint sign.
    status: verified
    last_checked: \"2026-06-01\"
---
