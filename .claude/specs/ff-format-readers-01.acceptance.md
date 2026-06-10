---
slug: ff-format-readers-01
criteria:
  - id: ac-001
    summary: OplsXmlReader parses to a molrs ForceField in molrs units
    type: code
    evaluator_hint: "Small OPLS XML fixture with known rows."
    pass_when: |
      OplsXmlReader.read_str(fixture) yields a ForceField whose bond/angle/dihedral/pair styles + types carry molrs-unit values: a known bond r0 nm->A (x10) and k kJ/nm^2->kcal/A^2 (/4.184/100), a known epsilon kJ->kcal (/4.184) and sigma nm->A, and a known RB c0..c5 dihedral mapped to OPLS f1..f4 (kcal). Each within 1e-9.
    status: pending
    last_checked: ""
  - id: ac-002
    summary: Reader is total (no silent parameter loss)
    type: code
    evaluator_hint: ""
    pass_when: |
      Malformed XML or a missing required attribute returns Err with a clear message; the reader never silently drops a parameter that would later read as 0.
    status: pending
    last_checked: ""
  - id: ac-003
    summary: Reads molpy's bundled oplsaa.xml
    type: code
    evaluator_hint: ""
    pass_when: |
      Reading the bundled oplsaa.xml produces a non-empty ForceField with bond, angle, dihedral and pair styles.
    status: pending
    last_checked: ""
  - id: ac-004
    summary: OPLS energy parity vs molpy numpy potentials
    type: scientific
    evaluator_hint: "Runs where the molpy<->molrs harness lives (bm-molrs-molpy)."
    pass_when: |
      For butane and ethanol typified by molpy OPLS, read_opls_xml(...).to_potentials().calc_energy(frame) total and each per-term energy (bond/angle/dihedral/LJ/Coulomb) match molpy's own numpy OPLS potentials on identical coordinates within 1e-4 kcal/mol.
    status: pending
    last_checked: ""
---

# Acceptance — Force-field format readers in molrs

Binding contract for `ff-format-readers-01.md`. ac-001..003 are in-crate
(molrs-ff). ac-004 is the cross-library parity (depends on `ff-potentials-oop-01`
for `to_potentials`/`calc_energy`) and runs in the bench harness.
