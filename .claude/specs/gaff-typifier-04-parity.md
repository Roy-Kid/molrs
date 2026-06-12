---
title: GAFF antechamber-parity validation harness (M1 capstone)
status: approved
created: 2026-06-12
---

# GAFF antechamber-parity validation harness (M1 capstone)

## Summary
This sub-spec adds the validation capstone for Milestone 1 of the GAFF typifier chain: a committed, reproducible parity harness that proves the molrs `GaffTypifier` (built by sub-specs 01-parser, 02-typing, 03-assign) reproduces AmberTools `antechamber -at gaff` atom typing exactly over a curated set of non-conjugated small molecules. It ships two artifacts — a committed `gen_gaff_fixtures.py` generator that drives antechamber to produce per-atom reference GAFF types as structure + JSON fixture pairs, and a Rust integration test under `molrs-ff/tests/ff/gaff/` that asserts 100% per-atom type parity against those checked-in fixtures. The test is gated so it skips cleanly (and passes) in environments without the fixtures present, so AmberTools is only a fixture-generation-time dependency, never a test-time one. No Python binding to the typing engine is introduced.

## Domain basis
The conformance oracle is AmberTools `antechamber -at gaff`, whose perception logic is driven by `ATOMTYPE_GFF.DEF`; the canonical GAFF atom-type table is documented at ambermd.org/antechamber/gaff.html.

- GAFF force field and atom typing: Wang, Wolf, Caldwell, Kollman & Case, "Development and testing of a general amber force field", J. Comput. Chem. 25(9):1157-1174 (2004), DOI 10.1002/jcc.20035.
- antechamber perception program: Wang, Wang, Kollman & Case, "Automatic atom type and bond type perception in molecular mechanical calculations", J. Mol. Graph. Model. 25(2):247-260 (2006), DOI 10.1016/j.jmgm.2005.12.005.

The parity metric is exact per-atom GAFF atom-type string equality (no tolerance): for each fixture molecule, `molrs_type[i] == antechamber_type[i]` for every atom index `i`. There is no numeric tolerance because GAFF types are discrete labels.

**Oracle equivalence + generation path.** `openmm/openmmforcefields` (MIT, the project's parameter ground truth) delegates GAFF typing to antechamber, so "consistent with openmmforcefields" ≡ "consistent with antechamber" for atom types. `gen_gaff_fixtures.py` may therefore generate references either by driving `antechamber -at gaff` directly or via the openmmforcefields pipeline (openff `Molecule` → `GAFFTemplateGenerator`); both yield the identical oracle, and the openmmforcefields path is the cleaner driver. **Licensing guard:** these fixtures are the *only* sanctioned channel for antechamber equivalence — the molrs typing engine (02) must stay clean-room from `ATOMTYPE_GFF.DEF` and must never embed transcribed antechamber GPL C.

The curated fixture set is deliberately restricted to the **non-conjugated** chemistry covered by M1: alkanes, alcohols, ethers, esters, amines, amides, isolated C=C / C=N / C=O / C=S, nitriles, nitro groups, thiols / sulfides / sulfones, phosphates / phosphines, halides, 3- and 4-membered rings, and isolated benzene rings. The set explicitly **excludes** chemistry deferred to M2: fused/polycyclic aromatics, heteroaromatics (pyridine, pyrrole, imidazole, furan), conjugated chains (dienes, enones at the conjugated-label level), and conjugated hypervalent S/P. This exclusion is a contract boundary, not an oversight, and is recorded in both the generator and the test doc comments.

## Design
The harness mirrors the existing MMFF per-atom validation pattern (`molrs-ff/tests/ff/mmff/typing.rs`), which loads `<name>.sdf` + `<name>.json` fixture pairs resolved via `CARGO_MANIFEST_DIR` and asserts per-atom equality with a readable mismatch diff. The GAFF analogue lives under `molrs-ff/tests/ff/gaff/`.

Entities and lifecycle:

- `gen_gaff_fixtures.py` (committed): a standalone, auditable Python script. It holds the curated molecule list (name → SMILES or inline structure), and for each molecule runs `antechamber -at gaff` to obtain reference per-atom GAFF types, then writes a structure file `<name>.sdf` (atom order == the order the Rust harness will load) plus `<name>.json` carrying `{ "atoms": [ { "gaff_type": "<str>" }, ... ] }`. The script requires AmberTools at generation time only. Its curated list is the single source of truth for the exclusion contract and must be machine-auditable (a flat, greppable list of names + SMILES with no conjugated/fused/heteroaromatic species).
- `molrs-ff/tests/ff/gaff/fixtures/<name>.{sdf,json}` (committed): the generated outputs, committed so the test runs without re-running antechamber.
- `molrs-ff/tests/ff/gaff/typing.rs` (new): the integration test. It defines `fixtures_dir()` via `CARGO_MANIFEST_DIR` (mirroring MMFF), an SDF loader preserving atom order, a JSON loader for `gaff_type`, a gate that early-returns (skips, test passes) when the fixtures dir is absent or empty, and a parity check that runs the `GaffTypifier` over each loaded molecule and compares the produced per-atom type vector element-by-element against the reference. Any mismatch accumulates a readable per-atom diff (molecule name, atom index, expected vs got) and `panic!`s, mirroring the MMFF mismatch-message pattern.
- `molrs-ff/tests/ff/gaff/mod.rs` (new): wires `typing.rs` into the test target, mirroring `molrs-ff/tests/ff/mmff/mod.rs`.
- `molrs-ff/tests/ff.rs` (modify): add the `#[path = "ff/gaff/mod.rs"] mod gaff;` declaration alongside the existing `mmff` module.

The gate is the mechanism that makes the 100%-parity criterion non-blocking where AmberTools/fixtures are absent: the test reads only checked-in fixtures; if `fixtures_dir()` does not exist or contains no `*.json`, the test logs a skip note and returns `Ok`/passes. Only `gen_gaff_fixtures.py` ever invokes antechamber.

This sub-spec consumes the `GaffTypifier` public API produced by the forward dependencies (`gaff-typifier-01-parser`, `gaff-typifier-02-typing`, `gaff-typifier-03-assign`); those are upstream in the M1 chain and must land before this harness compiles. The exact constructor/entry-point symbol is owned by 03-assign — the test calls whatever per-atom typing entry point 03 exposes and collects its per-atom GAFF type strings.

## Files to create or modify
- `molrs-ff/tests/ff/gaff/typing.rs` (new)
- `molrs-ff/tests/ff/gaff/mod.rs` (new)
- `molrs-ff/tests/ff/gaff/fixtures/` (new — committed generated `<name>.sdf` / `<name>.json` pairs for the curated non-conjugated set)
- `molrs-ff/tests/gen_gaff_fixtures.py` (new — committed antechamber-driving generator + curated molecule list)
- `molrs-ff/tests/ff.rs` (modify — register the `gaff` test module)

## Tasks
- [ ] Write `gen_gaff_fixtures.py` (molrs-ff/tests/gen_gaff_fixtures.py) with the curated non-conjugated molecule list (names + SMILES) and an auditable exclusion note; drive `antechamber -at gaff` to emit `<name>.sdf` + `<name>.json` (per-atom `gaff_type`)
- [ ] Generate and commit the fixture pairs under molrs-ff/tests/ff/gaff/fixtures/ covering every category in the M1 contract (alkanes, alcohols, ethers, esters, amines, amides, isolated C=C/C=N/C=O/C=S, nitriles, nitro, thiols/sulfides/sulfones, phosphates/phosphines, halides, 3/4-rings, isolated benzene)
- [ ] Write failing parity test for GaffTypifier (molrs-ff/tests/ff/gaff/typing.rs): fixtures_dir() via CARGO_MANIFEST_DIR, order-preserving SDF loader, JSON gaff_type loader
- [ ] Implement the gated-skip guard in typing.rs: early-return passing when fixtures dir is absent or contains no fixtures
- [ ] Implement the per-atom parity check in typing.rs: run GaffTypifier over each fixture molecule, compare type vectors element-by-element, accumulate (molecule, atom index, expected, got) diffs and panic with a readable message on any mismatch
- [ ] Add molrs-ff/tests/ff/gaff/mod.rs and register `mod gaff;` in molrs-ff/tests/ff.rs
- [ ] Add doc comments to typing.rs and gen_gaff_fixtures.py stating the non-conjugated scope and the M2-deferred exclusions (fused/heteroaromatic/conjugated)
- [ ] Run full check + test suite (cargo fmt --all --check && cargo clippy -- -D warnings && cargo check && cargo test --all-features)

## Testing strategy
- Happy path (parity present): with the committed fixtures in place, `cargo test -p molcrafts-molrs-ff gaff_parity` runs the `GaffTypifier` over every fixture molecule and passes only when per-atom GAFF types match the antechamber reference for all atoms of all molecules.
- Gate / skip path (fixtures absent): when `fixtures_dir()` does not exist or has no `*.json`, the test early-returns and passes; AmberTools is never invoked at test time. This is exercised by reasoning about the guard and confirming `cargo test` is green on a checkout where the fixtures dir is empty/removed.
- Mismatch path (readable diff): a single per-atom type mismatch causes the test to fail with a message listing molecule name, atom index, expected vs got — mirroring the MMFF `typing.rs` mismatch-message format — so failures are diagnosable without rereading the spec.
- Domain validation: the headline scientific gate is per-atom GAFF atom-type parity == 100% over the committed non-conjugated fixture set, regenerable via `gen_gaff_fixtures.py` against AmberTools `antechamber -at gaff`. The curated list is auditable to confirm it excludes all conjugated/fused/heteroaromatic species per the M1 contract.

## Out of scope
- The GAFF typing engine itself (parser, typing rules, assignment) — owned by `gaff-typifier-01-parser`, `gaff-typifier-02-typing`, `gaff-typifier-03-assign`. This sub-spec only validates them.
- Any Python binding to the molrs typing engine — none is introduced; `gen_gaff_fixtures.py` is a standalone antechamber driver, not a molrs binding.
- M2 chemistry: fused/polycyclic aromatics, heteroaromatics (pyridine, pyrrole, imidazole, furan), conjugated chains (dienes, enones), and conjugated hypervalent S/P. These are deliberately excluded from the fixture set and validated in a later milestone.
- Bond-type (`bondtype`) parity, charge parity, and full parameter assignment — only per-atom GAFF atom-type parity is asserted here.
