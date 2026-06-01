---
slug: core-perception-02-smarts-rings
criteria:
  - id: ac-001
    summary: Ring-size-range r{lo-hi} matches RDKit
    type: scientific
    evaluator_hint: "Compare find_matches to RDKit GetSubstructMatches for patterns with r{lo-hi}. Needs RDKit (installed)."
    pass_when: |
      For patterns including a real ETKDG macrocycle pattern using r{9-} (and r{3-6}, r{-8}), crossed with {cyclohexane "C1CCCCC1", 12-ring "C1CCCCCCCCCCC1", naphthalene "c1ccc2ccccc2c1", butane "CCCC"}, SmartsPattern::find_matches equals RDKit GetSubstructMatches(MolFromSmarts(p), uniquify=False) as sets; specifically r{9-} matches no atom in cyclohexane/benzene and matches ring atoms in the 12-membered ring.
    status: pending
    last_checked: ""
  - id: ac-002
    summary: Ring-bond connectivity x<n> matches RDKit
    type: scientific
    evaluator_hint: ""
    pass_when: |
      For patterns "[x2:1]" and "[x3:1]" on {cyclohexane, naphthalene, spiro[4.4]nonane "C1CCCC12CCCC2", biphenyl}, find_matches equals RDKit's set; x2 selects atoms in exactly two ring bonds (simple ring members) and x3 selects fusion/spiro atoms (three ring bonds).
    status: pending
    last_checked: ""
  - id: ac-003
    summary: Existing SMARTS suite still fully passes
    type: code
    evaluator_hint: ""
    pass_when: |
      The full molrs-core smarts test suite (the prior 189-pair validation plus recursive/atom-map/invalid tests) still passes unchanged after adding the two new primitives.
    status: pending
    last_checked: ""
  - id: ac-004
    summary: torsion_prefs shim removed, ETKDG torsions still match RDKit
    type: code
    evaluator_hint: "After removing the strip+post-check shim in molrs-embed, torsions.rs must still pass."
    pass_when: |
      The strip-and-post-check shim for r{}/x<n> in molrs-embed/src/distgeom/torsion_prefs.rs is removed (patterns parsed directly by SmartsPattern); molrs-embed tests/embed/torsions.rs (per-bond vs RDKit) and tests/embed/etkdg.rs (RMSD incl. alanine <0.5) still pass.
    status: pending
    last_checked: ""
---
