---
slug: gaff-typifier-04-parity
criteria:
  - id: ac-001
    summary: gen_gaff_fixtures.py committed; drives antechamber -at gaff to SDF+JSON
    type: code
    pass_when: |
      molrs-ff/tests/gen_gaff_fixtures.py exists, invokes
      `antechamber -at gaff`, and writes per-molecule <name>.sdf plus
      <name>.json carrying a per-atom "gaff_type" field.
    status: pending
  - id: ac-002
    summary: curated molecule list excludes all conjugated/fused/heteroaromatic species
    type: code
    pass_when: |
      The curated molecule list in gen_gaff_fixtures.py is a greppable
      flat list of names + SMILES that contains none of: fused/polycyclic
      aromatics, pyridine/pyrrole/imidazole/furan, dienes/enones, or
      conjugated hypervalent S/P; and a doc note states this M1 limitation.
    status: pending
  - id: ac-003
    summary: committed fixtures cover every M1 non-conjugated category
    type: code
    pass_when: |
      molrs-ff/tests/ff/gaff/fixtures/ contains committed <name>.sdf +
      <name>.json pairs with at least one molecule per M1 category
      (alkane, alcohol, ether, ester, amine, amide, isolated
      C=C/C=N/C=O/C=S, nitrile, nitro, thiol/sulfide/sulfone,
      phosphate/phosphine, halide, 3/4-ring, isolated benzene).
    status: pending
  - id: ac-004
    summary: gated test skips cleanly (passes) when fixtures dir absent/empty
    type: code
    pass_when: |
      With molrs-ff/tests/ff/gaff/fixtures/ removed or containing no *.json,
      `cargo test -p molcrafts-molrs-ff gaff_parity` passes by early-return,
      and antechamber is never invoked at test time.
    status: pending
  - id: ac-005
    summary: harness loads fixtures and compares per-atom types element-by-element
    type: code
    pass_when: |
      typing.rs resolves fixtures via CARGO_MANIFEST_DIR, loads each
      <name>.sdf preserving atom order and each <name>.json gaff_type
      vector, runs the GaffTypifier, and compares the produced per-atom
      type vector index-by-index against the reference vector.
    status: pending
  - id: ac-006
    summary: any per-atom mismatch fails with a readable molecule/index/expected/got diff
    type: code
    pass_when: |
      A per-atom type mismatch causes the test to panic with a message
      listing molecule name, atom index, expected type, and got type,
      mirroring the MMFF tests/ff/mmff/typing.rs mismatch-message pattern.
    status: pending
  - id: ac-007
    summary: per-atom GAFF type parity == 100% over the committed non-conjugated set
    type: scientific
    evaluator_hint: "marker: gaff_parity; oracle: antechamber -at gaff"
    pass_when: |
      With the committed fixtures present,
      `cargo test -p molcrafts-molrs-ff gaff_parity` passes, i.e. every
      atom of every fixture molecule has molrs GaffTypifier type ==
      antechamber -at gaff reference type (exact string equality, 0
      mismatches). Regenerable via gen_gaff_fixtures.py against AmberTools.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003 (code)** — The generator and its committed outputs.
  ac-001 fixes the generator's contract (antechamber-driven, SDF+JSON with a
  per-atom `gaff_type`). ac-002 makes the M1 non-conjugated exclusion machine-
  auditable in the committed molecule list. ac-003 requires committed coverage
  of every in-scope category so the parity gate is meaningful.
- **ac-004 (code)** — The gate. The test must pass by skipping when fixtures are
  absent/empty; antechamber is a fixture-generation-time dependency only. This is
  what makes ac-007 non-blocking in AmberTools-free environments.
- **ac-005 / ac-006 (code)** — The comparison harness. ac-005 fixes the load +
  element-by-element compare path (mirroring MMFF `typing.rs`); ac-006 fixes the
  readable mismatch diagnostic (molecule, index, expected, got).
- **ac-007 (scientific)** — The headline parity gate: 100% per-atom GAFF atom-type
  parity against `antechamber -at gaff` over the committed non-conjugated fixture
  set. Classified scientific because the oracle is an external AmberTools tool and
  the fixtures are (re)generable only with it; verified via the committed-fixture
  test path. Evaluator hint `marker: gaff_parity` selects the parity test.
