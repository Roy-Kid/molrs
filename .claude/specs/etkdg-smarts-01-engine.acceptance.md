---
slug: etkdg-smarts-01-engine
criteria:
  - id: ac-001
    summary: SMARTS matches equal RDKit across representative patterns x molecules
    type: scientific
    evaluator_hint: "For each (pattern, molecule), compare molrs find_matches to RDKit mol.GetSubstructMatches(MolFromSmarts(p), uniquify=False) as sets of atom-index tuples. Needs RDKit 2026.03.2 (installed)."
    pass_when: |
      For a fixture set of >=15 SMARTS patterns drawn from RDKit torsionPreferences_v2/smallrings/macrocycles (including at least 5 with recursive $(), plus patterns using !@;-, H<n>, X<n>, aromatic/aliphatic, and charge) crossed with molecules {butane "CCCC", benzene "c1ccccc1", pyridine "c1ccncc1", biphenyl "c1ccc(-c2ccccc2)cc1", alanine "C[C@@H](N)C(=O)O", caffeine, methyl acetate "COC(C)=O", acetamide "CC(N)=O", cyclooctane "C1CCCCCCC1"}, the set of matching atom-index tuples from SmartsPattern::find_matches (mapped through :n labels to the same ordering RDKit reports) equals RDKit's GetSubstructMatches(..., uniquify=False) set for every (pattern, molecule) pair.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-002
    summary: Recursive SMARTS evaluated correctly
    type: code
    evaluator_hint: ""
    pass_when: |
      For a carbonyl-recursive pattern such as "[$([CX3]=[OX1]):1]~[*:2]" (or a representative recursive pattern from the tables), find_matches on acetamide and methyl acetate returns exactly the carbonyl-carbon-rooted matches RDKit returns, and a molecule with no carbonyl (butane) returns none.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-003
    summary: Atom-map labels preserved and addressable
    type: code
    evaluator_hint: ""
    pass_when: |
      For a 4-atom torsion SMARTS with maps :1 :2 :3 :4, each returned match exposes which MolGraph AtomId corresponds to map label 1..4 via map_label(), and the four are mutually distinct bonded atoms forming the torsion path.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-004
    summary: Invalid SMARTS returns Err without panic
    type: code
    evaluator_hint: ""
    pass_when: |
      SmartsPattern::parse on malformed inputs (unbalanced brackets "[C", dangling bond "CC=", bad primitive "[Zq]") returns Err for each and never panics.
    status: verified
    last_checked: "2026-06-01"
---
