---
slug: mmff94-etkdg-04-embed
criteria:
  - id: ac-001
    summary: ETKDGv3 conformer best-fit RMSD vs RDKit < 0.5 A
    type: scientific
    evaluator_hint: "Generate with molrs (fixed seed) and RDKit EmbedMolecule(ETKDGv3, randomSeed matched); align and compute best-fit heavy-atom RMSD allowing symmetry/mirror. Needs RDKit (installed)."
    pass_when: |
      For each molecule in {ethanol "CCO", butane "CCCC", benzene "c1ccccc1", caffeine "Cn1cnc2c1c(=O)n(C)c(=O)n2C", (R)-alanine "C[C@@H](N)C(=O)O", 12-ring "C1CCCCCCCCCCC1"} with rng_seed fixed, the molrs ETKDGv3 conformer aligns to a RDKit ETKDGv3 conformer with symmetry-aware best-fit heavy-atom RMSD < 0.5 Å.
    status: pending
    last_checked: ""
  - id: ac-002
    summary: Generated conformer MMFF energy is same order as RDKit
    type: scientific
    evaluator_hint: "Compare MmffForceField energy of molrs conformer to RDKit ETKDG+MMFF-optimized energy."
    pass_when: |
      For the ac-001 molecules, the MMFF94 energy (from mmff94-etkdg-02 MmffForceField) of the molrs-generated conformer is within 10% relative of the MMFF94 energy of the RDKit-generated-and-MMFF-optimized conformer for the same molecule.
    status: pending
    last_checked: ""
  - id: ac-003
    summary: Success rate on rdkit_problems.smi >= RDKit and >= old impl
    type: runtime
    evaluator_hint: "Run generate_3d over every SMILES in tests-data/smi/rdkit_problems.smi; count successes. Compare to RDKit EmbedMolecule on the same set."
    pass_when: |
      Over all parseable SMILES in tests-data/smi/rdkit_problems.smi, the count of molecules for which molrs generate_3d returns Ok with a valid (non-degenerate, no severe clashes) conformer is >= the count for RDKit EmbedMolecule(ETKDGv3) on the same set, and strictly >= the recorded count for the retired FragmentRules implementation. All three counts are reported.
    status: pending
    last_checked: ""
  - id: ac-004
    summary: Fixed seed is fully reproducible
    type: code
    evaluator_hint: ""
    pass_when: |
      For butane with EmbedOptions.rng_seed = Some(42), two successive generate_3d calls produce coordinates equal element-wise to within 1e-9 Å.
    status: verified
    last_checked: \"2026-06-01\"
  - id: ac-005
    summary: Chirality preserved, no stereo inversion
    type: code
    evaluator_hint: ""
    pass_when: |
      For (R)-alanine "C[C@@H](N)C(=O)O" and (S)-alanine, generate_3d produces a conformer whose 3D chirality at the alpha carbon matches the input CIP label (EmbedReport.warnings contains no tetrahedral-inversion warning), and the two enantiomers yield mirror-image geometries.
    status: verified
    last_checked: \"2026-06-01\"
  - id: ac-006
    summary: Retired modules deleted, EmbedAlgorithm no longer exported
    type: code
    evaluator_hint: ""
    pass_when: |
      cargo build succeeds with no references to builder.rs/optimizer.rs/distance_geometry.rs/rotor_search.rs/fragment_data.rs (the files are deleted); molrs_embed::EmbedAlgorithm and EmbedSpeed are no longer part of the public API; generate_3d retains its (&Atomistic, &EmbedOptions) -> Result<(Atomistic, EmbedReport), MolRsError> signature.
    status: pending
    last_checked: ""
  - id: ac-007
    summary: Degenerate inputs behave like RDKit
    type: code
    evaluator_hint: ""
    pass_when: |
      generate_3d on an empty molecule returns Err; on a single atom returns Ok with one placed coordinate; on a disconnected two-component system "C.C" returns Ok with both fragments placed without NaN coordinates.
    status: verified
    last_checked: \"2026-06-01\"
---
