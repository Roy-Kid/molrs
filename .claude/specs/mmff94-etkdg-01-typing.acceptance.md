---
slug: mmff94-etkdg-01-typing
criteria:
  - id: ac-001
    summary: MMFF atom types match RDKit per-atom across a coverage set
    type: scientific
    evaluator_hint: "Run molrs (via molrs-python) and RDKit on the same SMILES; compare GetMMFFAtomType per atom. Needs RDKit 2026.03.2 (installed)."
    pass_when: |
      For each SMILES in {methane "C", ethylene "C=C", benzene "c1ccccc1", pyridine "c1ccncc1", imidazole "c1cnc[nH]1", aniline "Nc1ccccc1", acetamide "CC(N)=O", nitrobenzene "[O-][N+](=O)c1ccccc1", benzenesulfonic acid "OS(=O)(=O)c1ccccc1", caffeine "Cn1cnc2c1c(=O)n(C)c(=O)n2C"}, after add_hydrogens, the MMFF numeric atom type assigned by molrs_ff::mmff::MmffMolProperties::compute(mol, Mmff94) equals RDKit AllChem.MMFFGetMoleculeProperties(mol).GetMMFFAtomType(i) for every atom i (heavy + H), with explicit-H atom ordering aligned.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-002
    summary: MMFF partial charges match RDKit within 1e-4
    type: scientific
    evaluator_hint: "Compare GetMMFFPartialCharge per atom against molrs BCI charges."
    pass_when: |
      For the same coverage set as ac-001, every per-atom MMFF partial charge from MmffMolProperties::partial_charge(i) is within 1e-4 of RDKit GetMMFFPartialCharge(i).
    status: verified
    last_checked: "2026-06-01"
  - id: ac-003
    summary: MMFF aromaticity reassigns ring atoms to aromatic MMFF types
    type: code
    evaluator_hint: ""
    pass_when: |
      For benzene the 6 ring carbons receive MMFF type 37 (CB); for pyridine the ring N receives an aromatic nitrogen type and ring carbons aromatic carbon types; for imidazole both ring nitrogens receive MMFF 5-ring aromatic nitrogen types. All assertions match the RDKit type for the same atoms.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-004
    summary: Unsupported atoms fail loudly, charge sum equals net charge
    type: code
    evaluator_hint: ""
    pass_when: |
      MmffMolProperties::compute on a molecule containing a transition metal (e.g. ferrocene Fe) returns Err and is_setup_complete() is false; for ammonium [NH4+] and acetate "CC(=O)[O-]" the sum of partial_charge over all atoms equals the molecule net formal charge to within 1e-6.
    status: verified
    last_checked: "2026-06-01"
  - id: ac-005
    summary: MMFF94 vs MMFF94s variant difference matches RDKit
    type: scientific
    evaluator_hint: "Compare both variants against RDKit mmffVariant='MMFF94' and 'MMFF94s'."
    pass_when: |
      For aniline "Nc1ccccc1", computing with Mmff94 and Mmff94s yields atom-type/charge sets that each match RDKit's corresponding mmffVariant output per atom (types exact; charges within 1e-4), and the two variants differ in at least the amine-nitrogen-related entries exactly where RDKit differs.
    status: verified
    last_checked: "2026-06-01"
---
