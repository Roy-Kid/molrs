---
title: GAFF typifier sunk into molrs, built on the opls/ typifier infrastructure
status: approved
created: 2026-06-19
supersedes: gaff-typifier-01-parser, gaff-typifier-02-typing, gaff-typifier-03-assign, gaff-typifier-04-parity, gaff-typifier-05-charges
---

# GAFF typifier sunk into molrs, built on the opls/ typifier infrastructure

## Summary

Replace the stale 2026-06-12 GAFF chain (`gaff-typifier-01…05`). The design
**intent** of that chain is correct and current — a native `gaff.dat` parser plus a
clean-room atom-typing engine sunk into molrs, with `openmmforcefields` (MIT) as the
AmberTools-free ground truth and antechamber's GPL `atomtype.c` deliberately avoided
— but the specs are **stale in two ways**: (1) every path is addressed to the
pre-merge multi-crate workspace (`molrs-ff/`, `molrs-core/`), which no longer exists
(molrs is one crate, `molrs/src/ff/…`); and (2) they were written to mirror the
older `MMFFTypifier` and to build a **from-scratch** bonded-parameter assignment
engine — before the far richer, **FF-agnostic** `opls/` typifier landed.

This redesign re-grounds GAFF on the current codebase: it **mirrors and reuses the
`opls/` typifier** (`molrs/src/ff/typifier/opls/`) — `typing.rs` (atom typing),
`assign.rs` (`assign_bonded` + `CandidateTables` + the `ParameterEstimator` no-match
seam, whose doc already says *"GAFF can attach the same estimator"*), and `build()`
(`typify → to_frame → to_potentials` with `intramolecular_pairs` + `special_bonds`).
GAFF therefore does **not** reinvent assignment, the neighbour-list seam, or the
build pipeline — it adds only what is GAFF-specific: the `gaff.dat` reader, the
clean-room typing predicates (kept predicate-based for antechamber parity), the
parity harness, and AM1-BCC charge delegation.

## Domain basis

- Wang et al., "Development and testing of a general AMBER force field," *J. Comput.
  Chem.* 25(9):1157-1174 (2004), DOI 10.1002/jcc.20035.
- Wang et al., "Automatic atom type and bond type perception…," *J. Mol. Graph.
  Model.* 25(2):247-260 (2006), DOI 10.1016/j.jmgm.2005.12.005.
- **Ground truth**: antechamber `ATOMTYPE_GFF.DEF` (the typing ruleset, a data file)
  + `ambermd.org/antechamber/gaff.html`; `openmm/openmmforcefields` (MIT) for the
  `gaff.dat` file and its converted ffxml as an AmberTools-free **parameter**
  cross-check. The `.dat`→ffxml converter in openmmforcefields is an MIT reference
  algorithm for the parser. **Do not** reuse molpy's lossy `gaff.xml`, and **do not**
  transcribe the GPL `atomtype.c` — typing is clean-room, equivalence to antechamber
  established empirically by the parity harness.

## Design

Place GAFF as `molrs/src/ff/typifier/gaff/` and `molrs/src/ff/forcefield/amber_dat.rs`,
mirroring `opls/`:

1. **`gaff.dat` native parser** (`forcefield/amber_dat.rs`) — a `ForceFieldReader`
   producing a pure-molrs-units `ForceField`/`Params` from a version-pinned
   `gaff.dat` (vendored from `openmm/openmmforcefields`, MIT, into `molrs/data/`,
   compiled in via `include_str!` like `MMFF94_XML`/`OPLSAA_XML`). Sections:
   MASS/BOND/ANGLE/DIHEDRAL/IMPROPER/NONBON; load-bearing invariants: negative-PN
   multi-term dihedral continuation accumulation, and IMPROPER having no IDIVF
   column (PK used directly). (Fortran column specs per ambermd.org FileFormats; the
   openmmforcefields converter is the reference.)
2. **Clean-room atom typing** (`gaff/typing.rs`) — the GAFF analog of
   `opls/typing.rs::annotate_opls`: per-element predicate routines keyed on local
   inputs (element, coordination, attached-H, EW-neighbour count, ring size,
   benzene-type aromaticity via `perceive_aromaticity`, hybridization via the
   shared `Hyb`). **Predicate code, not SMARTS** — SMARTS cannot reproduce
   antechamber exactly (that is precisely why molpy's SMARTS `gaff.xml` is rejected),
   so parity demands predicate rules auditable line-by-line against
   `ATOMTYPE_GFF.DEF`. Conjugated/heteroaromatic atoms that the local rule set cannot
   type faithfully must error / flag out-of-contract, never silently mis-type.
   **Scope decision to make explicitly**: the old chain deferred all conjugated
   types (`cc/cd/ce/nb/…`) to an unspecced "Milestone 2". Re-evaluate now that
   `perceive_aromaticity` + ring/conjugation perception exist in core — either cover
   them here or state the deferral in the module doc so nobody ships a GAFF that
   silently rejects pyridine.
3. **Bonded assignment via the shared infra** (`gaff/mod.rs`) — `GaffTypifier`
   mirrors `OplsTypifier`: it owns typing metadata + the parsed `ForceField`, and its
   bonded assignment **reuses** `assign_bonded` / `CandidateTables` /
   `ParameterEstimator`. GAFF's `X`-wildcard fallback maps onto the existing
   specificity-ranked candidate matching (exact > most-specific-wildcard); multi-term
   dihedrals assemble from the parser's accumulated terms; impropers use PK directly.
   No separate assignment engine.
4. **`build()`** = `typify → to_frame → to_potentials` with `intramolecular_pairs`
   + `special_bonds`, identical to `OplsTypifier::build` — GAFF gets the full
   evaluable-potentials pipeline for free.
5. **AM1-BCC charges** (delegated, molpy-wrapper layer) — charges consistent with
   openmm = AM1-BCC via `antechamber -c bcc` / `openff am1bcc`, computed in the
   delegation layer and written to the frame `charge` column; molrs only consumes.
   No semi-empirical QM in molrs, no Gasteiger substitute; net-charge conservation
   enforced. (Future QM-free path: NAGL GNN.)

## Files to create or modify

- `molrs/data/gaff.dat` (new) — version-pinned, vendored from openmmforcefields (MIT), provenance+license header.
- `molrs/src/ff/forcefield/amber_dat.rs` (new) — `GaffDatReader` (`ForceFieldReader`), embedded const + `gaff_forcefield()`.
- `molrs/src/ff/typifier/gaff/mod.rs` (new) — `GaffTypifier` (owns meta + ff), `Typifier` impl, `typify_full`/`build` reusing `assign_bonded`/`ParameterEstimator`.
- `molrs/src/ff/typifier/gaff/typing.rs` (new) — clean-room predicate atom typing.
- `molrs/src/ff/typifier/gaff/tests.rs` (new) — typing + assignment unit tests.
- `molrs/src/ff/typifier/mod.rs` — register `pub mod gaff;`.
- `molrs/src/ff/forcefield/mod.rs` — register `pub mod amber_dat;`.
- parity harness + fixtures under the `tests/` tree (gated; clean-skip when antechamber/fixtures absent).
- AM1-BCC delegation in the molpy-wrapper layer (separate package).

## Tasks

- [ ] Vendor version-pinned `gaff.dat` (openmmforcefields, MIT) to `molrs/data/`; add the `include_str!` const + `gaff_forcefield()`
- [ ] Implement `GaffDatReader` (MASS/BOND/ANGLE/DIHEDRAL/IMPROPER/NONBON; negative-PN multi-term; no-IDIVF improper); cross-check params against the converted ffxml
- [ ] Implement clean-room predicate atom typing (`gaff/typing.rs`) for the in-scope type set; out-of-contract guard for inputs the rules can't type
- [ ] Decide + document the conjugated/heteroaromatic scope (cover now vs. explicit deferral) given core's aromaticity/conjugation perception
- [ ] Implement `GaffTypifier` reusing `assign_bonded`/`CandidateTables`/`ParameterEstimator` for bonded assignment (X-wildcard via specificity ranking); wire `build()`
- [ ] antechamber parity harness + gated fixtures (per-atom 100% type agreement)
- [ ] AM1-BCC charge delegation in the wrapper layer; net-charge conservation; gated comparison vs omff/antechamber
- [ ] `cargo fmt --all --check && cargo clippy -- -D warnings && cargo test --all-features`

## Testing strategy

- **Parser**: parse the embedded `gaff.dat` without error; section counts non-empty;
  numeric spot-checks; negative-PN multi-term and no-IDIVF improper invariants;
  parameter cross-check vs openmmforcefields' converted ffxml.
- **Typing**: hand-built representative non-conjugated molecules (ethanol, acetone,
  benzene, methylamine, halomethanes, …) assert the full per-atom GAFF type map vs
  `ATOMTYPE_GFF.DEF`; out-of-contract inputs error rather than mis-type.
- **Assignment/build**: a typed molecule routes through `assign_bonded` + `build()`
  to evaluable `Potentials`; the shared `ParameterEstimator` covers no-match terms
  when lenient.
- **Parity (scientific, gated)**: per-atom 100% agreement with antechamber `-at gaff`
  over the fixture corpus; clean-skip when AmberTools/fixtures absent.
- **Charges (gated)**: AM1-BCC charges match omff/antechamber per-atom within
  tolerance; net charge conserved; Gasteiger substitution rejected.

## Out of scope

- GAFF2 / `gaff2.dat` (GAFF 1.x only).
- Reimplementing assignment, the neighbour-list seam, or the build pipeline — these
  are **reused** from `opls/`/`ff::potential`.
- Native semi-empirical QM for charges (delegated; NAGL GNN is a future path).
- Modifying the generic potential kernels (see ff-perinstance-mmff-kernels for the
  MMFF-only per-instance work).
