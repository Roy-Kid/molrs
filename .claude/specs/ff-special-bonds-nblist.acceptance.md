---
slug: ff-special-bonds-nblist
criteria:
  - id: ac-001
    summary: ForceField carries special_bonds weights; readers set them
    type: code
    evaluator_hint: "LammpsFfReader on a GAFF .ff; OPLS reader."
    pass_when: |
      ForceField exposes special_bonds [1-2,1-3,1-4] per nonbonded kind (lj, coul).
      LammpsFfReader sets amber defaults (lj 0/0/0.5, coul 0/0/0.8333); the legacy
      lj14scale/coulomb14scale are folded in. OPLS reader sets its 0.5/0.5.
    status: todo
    last_checked: ""
  - id: ac-002
    summary: intramolecular_pairs builds the consumer nblist; ForceField never builds pairs
    type: code
    evaluator_hint: "Free fn already exists in potential/mod.rs."
    pass_when: |
      intramolecular_pairs(frame) returns a pairs block atomi/atomj/is_14 = every i<j
      minus 1-2 (bond ends) and 1-3 (angle i,k), is_14 flagged on dihedral (i,l).
      to_potentials does NOT call it; absent a pairs block it still skips pair styles.
    status: todo
    last_checked: ""
  - id: ac-003
    summary: Generic pair ctors are per-atom + is_14, one convention
    type: code
    evaluator_hint: ""
    pass_when: |
      lj/cut and coul/cut read per-atom params (atoms block type + pair style per-type),
      combine via Lorentz-Berthelot, and apply the ff special_bonds 1-4 weight on is_14
      pairs. The pre-combined per-pair "type" requirement is gone. Their unit tests pass
      against a hand-combined reference; flagged 1-4 pairs scaled correctly.
    status: todo
    last_checked: ""
  - id: ac-004
    summary: GAFF/LAMMPS LJ+Coulomb energy parity through the generic path
    type: scientific
    evaluator_hint: "Single chain + intramolecular_pairs + to_potentials vs LAMMPS/molpy single-point."
    pass_when: |
      For a GAFF molecule, frame + intramolecular_pairs + special_bonds -> to_potentials
      gives LJ + Coulomb energy matching a LAMMPS single-point (or molpy) reference within
      tolerance, with 1-4 scaling correct.
    status: todo
    last_checked: ""
  - id: ac-005
    summary: MMFF demoted — generic path reproduces the old frame_builder energy, then it is deleted — SUPERSEDED, now owned by ff-perinstance-params
    type: scientific
    evaluator_hint: "Pin parity before deleting frame_builder."
    pass_when: |
      MMFFTypifier::build typifies -> ForceField only; the molecule's energy via the generic
      path (consumer-built pairs -> to_potentials) equals the pre-refactor frame_builder
      energy within tolerance; typifier/mmff/frame_builder.rs's frame/pairs assembly is removed;
      mmff_* kernels still resolve via the registry.
    status: todo
    last_checked: ""
  - id: ac-006
    summary: molpack relaxer takes a ForceField and relaxes with non-bonded
    type: code
    evaluator_hint: "Reference consumer."
    pass_when: |
      molpack LBFGSRelaxer::new(ff, frame) builds the nblist, to_potentials, and minimizes;
      a relaxation-assisted pack with LJ+Coulomb runs end to end. No from_lammps_ff on the
      relaxer; the LAMMPS input is read by molrs::io::forcefield.
    status: todo
    last_checked: ""
---
