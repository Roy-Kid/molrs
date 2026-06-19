---
slug: gaff-typifier-redesign
criteria:
  - id: ac-001
    summary: Native gaff.dat parser → molrs ForceField, cross-checked vs openmmforcefields ffxml
    type: scientific
    evaluator_hint: "Parse the embedded gaff.dat; compare bond/angle/dihedral/vdW params to openmmforcefields' converted ffxml for the same pinned version."
    pass_when: |
      GaffDatReader parses the embedded version-pinned gaff.dat (MASS/BOND/ANGLE/DIHEDRAL/IMPROPER/NONBON) into a molrs ForceField in molrs units; negative-PN multi-term dihedrals accumulate all terms; impropers carry no idivf (PK direct); the parsed bond/angle/dihedral/vdW parameters match openmmforcefields' converted ffxml within numeric tolerance.
    status: pending
    last_checked: ""
  - id: ac-002
    summary: Clean-room predicate atom typing matches ATOMTYPE_GFF.DEF for the in-scope set
    type: scientific
    evaluator_hint: "Assert per-atom GAFF types on representative molecules vs ATOMTYPE_GFF.DEF / gaff.html. No SMARTS; no transcription of GPL atomtype.c."
    pass_when: |
      For the representative non-conjugated molecules (ethanol, diethyl ether, acetone, acetonitrile, methylamine, tetramethylammonium, benzene, fluorobenzene, halomethanes, thiols/sulfides/sulfone, phosphate), assign_atom_types returns the per-atom GAFF type map documented in ATOMTYPE_GFF.DEF; inputs the in-scope rules cannot type faithfully (e.g. pyridine, furan, 1,3-butadiene) error / flag out-of-contract rather than silently mis-type. Typing is predicate-based (no SMARTS) and clean-room (no atomtype.c transcription).
    status: pending
    last_checked: ""
  - id: ac-003
    summary: GaffTypifier reuses the opls/ bonded-assignment + build infrastructure
    type: code
    evaluator_hint: ""
    pass_when: |
      GaffTypifier mirrors OplsTypifier: bonded-parameter assignment goes through the shared assign_bonded / CandidateTables / ParameterEstimator (GAFF X-wildcard via specificity ranking) — no GAFF-private assignment engine — and build() = typify → to_frame → to_potentials with intramolecular_pairs + special_bonds, reusing ff::potential. All paths are the single crate (molrs/src/ff/...).
    status: pending
    last_checked: ""
  - id: ac-004
    summary: antechamber per-atom parity (gated)
    type: scientific
    evaluator_hint: "Run antechamber -at gaff over the fixture corpus; require 100% per-atom type agreement. Clean-skip when AmberTools/fixtures absent."
    pass_when: |
      Over the parity fixture corpus, GAFF atom typing agrees with antechamber `-at gaff` per-atom 100% for the in-scope types; the harness clean-skips when AmberTools or fixtures are unavailable.
    status: pending
    last_checked: ""
  - id: ac-005
    summary: AM1-BCC charges delegated, net charge conserved (gated)
    type: scientific
    evaluator_hint: "Charges via antechamber -c bcc / openff am1bcc in the wrapper layer; compare per-atom vs omff/antechamber; assert net-charge conservation; reject Gasteiger substitution."
    pass_when: |
      GAFF charges are AM1-BCC delegated (antechamber -c bcc / openff am1bcc) in the wrapper layer and written to the frame charge column; per-atom charges match omff/antechamber within tolerance (gated), net charge is conserved, and no Gasteiger substitute is used. molrs only consumes the charges.
    status: pending
    last_checked: ""
---
