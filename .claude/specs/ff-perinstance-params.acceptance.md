---
slug: ff-perinstance-params
criteria:
  - id: ac-001
    summary: MMFF kernels evaluate from per-instance parameter columns, not a type-label table lookup
    type: code
    evaluator_hint: "ff/potential/{bond,angle,dihedral,improper,pair}/mmff.rs ctors read get_float columns (like mmff_stbn reads r0_ij), not type_map.get(label)."
    pass_when: |
      Every MMFF kernel constructor (mmff_bond/angle/stbn/torsion/oop/vdw/ele) sources its numeric
      parameters from per-instance columns on the frame's topology blocks via get_float (bonds: kb/r0;
      angles: ka/theta0/r0_ij/r0_kj; dihedrals: v1/v2/v3; impropers: koop; pair: per-atom charge +
      vdw params), NOT from a string type-label looked up in a shared style table. No
      resolve_*_label string round-trip remains in the MMFF kernel parameter path. The kernel contains
      no MMFF-specific resolution (equivalence fallback / empirical rules).
    status: todo
    last_checked: ""
  - id: ac-002
    summary: The MMFF typifier assigns per-instance numeric params by reusing the RDKit-validated resolution
    type: code
    evaluator_hint: "Relocate ff/mmff/energy/params.rs resolution to a kept module; typifier calls it per interaction and sets Atomistic relation props; to_frame carries columns."
    pass_when: |
      The MMFF parameter resolution (bond_params/angle_params/torsion/oop_koop/stbn + empirical rules +
      eq_level + bond_type/angle_type) is relocated out of ff/mmff/energy/ into a kept module
      (pub(crate)). The MMFF typifier builds Topo + validated types/charges and, for every
      bond/angle/stretch-bend/torsion/oop, calls that resolution to obtain final numeric parameters,
      attaching them as Atomistic relation properties so Atomistic::to_frame projects them into
      per-instance columns. The stretch-bend r0 merge currently done post-to_frame (merge_stbn_r0) is
      folded into the typifier. No type-label -> shared-table parameter lookup remains for MMFF.
    status: todo
    last_checked: ""
  - id: ac-003
    summary: Generic-path RDKit energy + gradient parity across the full fixture set (incl. aromatics/sp2)
    type: code
    evaluator_hint: "Generic typify -> nonbonded_pairs -> to_potentials -> calc_energy vs tests/ff/mmff/fixtures/*.energy.json; currently only e_ethane passes."
    pass_when: |
      For the full RDKit fixture coverage set in tests/ff/mmff/fixtures (including the empirical-rule
      molecules: e_ethylene, e_benzene, e_butane, e_caffeine, s_aniline, s_acetamide, ...), the
      generic-path total energy (typify -> nonbonded_pairs -> to_potentials -> calc_energy) matches the
      RDKit reference (mmff94_total_energy) within 1e-3 kcal/mol and equals the (still-present,
      pre-deletion) MmffForceField total within tolerance; analytical gradient vs central finite
      difference (h=1e-5) max component error < 1e-5. (Supersedes ff-mmff-unify-generic-path ac-004.)
    status: todo
    last_checked: ""
  - id: ac-004
    summary: ETKDG decoupled from MMFF (generation != optimization)
    type: code
    evaluator_hint: "conformer/etkdg: no MmffForceField/MmffMolProperties imports, no mmff_cleanup/mmff_min/mmff_cleanup_internal. (= ff-mmff-unify-generic-path ac-005.)"
    pass_when: |
      conformer/etkdg no longer references MMFF: generate_3d returns the embedded + ET-refined geometry;
      MMFF cleanup is a caller-composed step (MMFFTypifier::build -> Potentials + generic ff::optimize).
      The full pipeline (generate + external MMFF optimize) reproduces the prior conformer RMSD
      distribution / success rate within tolerance; existing ETKDG RMSD tests pass under the decoupled
      flow; no Mmff* reference remains in conformer/.
    status: todo
    last_checked: ""
  - id: ac-005
    summary: nonbonded_pairs renamed from intramolecular_pairs and exposed to Python
    type: code
    evaluator_hint: "Rename across potential/mod.rs + typifier + tests; mirror extract_coords in molrs-python; molrs.pyi entry."
    pass_when: |
      The pub fn currently named intramolecular_pairs (potential/mod.rs) is renamed to nonbonded_pairs
      across the crate and tests. molrs.nonbonded_pairs(frame) -> Block is exposed in Python (mirrors
      extract_coords), round-trips a frame to an atomi/atomj/is_14 block, and is documented in
      molrs.pyi. MMFFTypifier().build(mol) remains the blessed one-call path.
    status: todo
    last_checked: ""
  - id: ac-006
    summary: Standalone MmffForceField evaluator deleted; validated front-end + relocated resolution kept
    type: code
    evaluator_hint: "Delete only AFTER ac-003 is green. Keep mmff::{topo,aromaticity,atomtype,charges}, MmffMolProperties, and the relocated params resolution."
    pass_when: |
      ff/mmff/energy/ (MmffForceField, impl Potential, MmffEnergyBreakdown, term-math structs) and its
      mmff/mod.rs energy re-exports are removed; build_mmff_potentials_py + its lib.rs registration +
      pyi entry are removed; tests/ff/mmff/energy.rs RDKit parity is relocated onto the generic path.
      The validated front-end (mmff::{topo,aromaticity,atomtype,charges}, MmffMolProperties) and the
      relocated parameter resolution are RETAINED. grep for MmffForceField / build_mmff_potentials is
      clean; mmff_* kernels still resolve via the registry; build + tests pass; clippy -D warnings clean.
    status: todo
    last_checked: ""
  - id: ac-007
    summary: The per-instance-parameter contract is the documented, force-field-agnostic kernel interface
    type: code
    evaluator_hint: "Doc the contract so future force fields plug in as new assigners; kernels stay generic."
    pass_when: |
      The kernel contract -- "a kernel evaluates one functional form from per-instance numeric
      parameters supplied on the frame; parameter assignment is the force field's responsibility" -- is
      documented (rustdoc on the Potential / kernel registry and/or .claude/notes). Adding a new force
      field requires only a new parameter assigner (+ reusing existing kernels or adding new functional
      forms); no kernel embeds force-field-specific resolution or assumes a shared parameter table.
    status: todo
    last_checked: ""
---

# Acceptance — ff-perinstance-params

Binding contract for `ff-perinstance-params.md`. The MMFF parameter math is **reused** from the
RDKit-validated `MmffForceField` resolution (relocated, not re-derived), so `ac-003` parity holds by
construction once the assigner writes per-instance columns and the kernels read them. `ac-006`
(deletion) is gated on `ac-003` being green. `ac-001/002/003` replace the type-label kernel mechanism of the retired `ff-mmff-unify-generic-path`;
`ac-004/005/006` carry over its ETKDG-decouple / Python-`nonbonded_pairs` / delete-monolith criteria.
