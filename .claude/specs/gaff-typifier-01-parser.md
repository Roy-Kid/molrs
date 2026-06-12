---
title: GAFF gaff.dat native parser into molrs ForceField/Params (sub-spec 01)
status: approved
created: 2026-06-12
---

# GAFF gaff.dat native parser into molrs ForceField/Params (sub-spec 01)

## Summary
molrs-ff gains a native reader for the Amber GAFF 1.x `gaff.dat` parameter file, translating every section of the authoritative Fortran-format file (atom masses, bonds, angles, dihedrals — including multi-term torsions — impropers, and the MOD4/RE nonbonded block) into the existing molrs `ForceField`/`Params` store in molrs units (Å, kcal/mol, radians, e). A real GAFF 1.x `gaff.dat` (from AmberTools `dat/leap/parm/gaff.dat`) is embedded into the crate at compile time so callers can obtain the full force field with no file on disk. This sub-spec delivers only the parser and embedding; atom typing, parameter assignment, and antechamber parity are deferred to later sub-specs in the chain.

## Domain basis
Parses the AMBER `parm.dat`/`gaff.dat` format per the authoritative Fortran column specs (ambermd.org/FileFormats.php, AMBER parameter file format), reading these sections in order:

1. **Title card** — single free-text line, consumed and ignored.
2. **MASS** `FORMAT(A2,2X,F10.2,F10.2)` → atom symbol, mass (amu), atomic polarizability (Å³). Blank-card terminated. Trailing free-text comment tolerated.
3. **Hydrophilic-atoms line** `FORMAT(20(A2,2X))` — a single line of 2-char symbols; **position-tolerant** because real files place it inconsistently. Consumed and ignored (not stored in this sub-spec).
4. **BOND** `FORMAT(A2,1X,A2,2F10.2)` → atoms a1,a2; force constant k (kcal/mol/Å²); equilibrium length req (Å). Blank-card terminated.
5. **ANGLE** `FORMAT(A2,1X,A2,1X,A2,2F10.2)` → 3 atoms; k (kcal/mol/rad²); theta_eq (deg). Blank-card terminated.
6. **DIHEDRAL** `FORMAT(A2,1X,A2,1X,A2,1X,A2,I4,3F15.2)` → 4 atoms; IDIVF (integer divisor); PK (barrier, kcal/mol); PHASE (deg); PN (signed periodicity). Energy convention `E = (PK/IDIVF)(1 + cos(PN·phi − PHASE))`.
7. **IMPROPER** — same atom/field layout **but no IDIVF column**: 4 atoms; PK (used directly, not divided); PHASE (deg); PN. Blank-card terminated.
8. **H-bond 10-12** block — parse-and-skip (usually empty in GAFF). Blank-card terminated.
9. **Nonbonded equivalencing** (optional) — parse-and-skip.
10. **NONBON MOD4/RE** — header line `MOD4   RE`, then per-symbol: R* (vdW radius, Å), epsilon (kcal/mol). Blank-card / EOF terminated.

Critical format invariants (each a test target):
- **(a) Negative-periodicity multi-term dihedrals**: when `PN < 0` the same atom-type quartet carries additional Fourier terms on the following card(s); store `|PN|`, accumulate each term, and keep reading continuation cards for that quartet until a card with `PN > 0` (the final term). Dropping continuation terms is a silent correctness bug.
- **(b) IMPROPER has no IDIVF column** — column offsets differ from DIHEDRAL; PK is used directly.
- **(c)** Free-text comments after the numeric fields must be tolerated on any card.
- **(d)** Blank-card section termination.
- **(e)** Position-tolerant hydrophilic-atoms line.

**Bundled file source + ground truth: `openmm/openmmforcefields` (MIT).** The embedded `gaff.dat` is vendored from openmmforcefields' `amber/` tree (which itself mirrors AmberTools `dat/leap/parm/gaff.dat`) rather than from an AmberTools install, so the source is MIT-licensed and redistributable in a crates.io-published crate. **Pin an exact version** (e.g. GAFF 1.81 — the last GAFF1 line) and record it + the upstream commit in the embedded module docstring; the pinned version MUST match the antechamber/openmmforcefields version used to generate the `gaff-typifier-04-parity` fixtures, or parity will chase version drift. openmmforcefields' converted OpenMM ffxml for that same version is a second, **AmberTools-free parameter ground truth**: this sub-spec cross-checks the natively-parsed `ForceField` against it (bond/angle/dihedral/vdW values), and openmmforcefields' `amber/` `.dat`→ffxml conversion script is a **directly copyable (MIT) reference algorithm** for this parser. Do NOT reuse molpy's existing `gaff.xml` (a lossy OpenMM conversion of the old GAFF 1.4 set with no provenance header).

## Design
New module `crate::forcefield::amber_dat`, sibling to `crate::forcefield::xml` and the external-format readers under `crate::forcefield::readers`. It implements the established `ForceFieldReader` trait (`read_str`/`read`) so GAFF joins OPLS as a foreign-format reader producing a pure-molrs-units `ForceField`.

Section → store mapping (all via existing `ForceField` builder methods, no new store types):
- MASS → `def_atomstyle("gaff")` + `def_atomtype(symbol, [("mass", m), ("polarizability", p)])`.
- BOND → `def_bondstyle("harmonic")` + `def_bondtype(a1, a2, [("k", k), ("r0", req)])`. Order-independent lookup already exists.
- ANGLE → `def_anglestyle("harmonic")` + `def_angletype(a1, a2, a3, [("k", k), ("theta0", theta_eq)])`.
- DIHEDRAL → `def_dihedralstyle("fourier")` + `def_dihedraltype`. Multi-term quartets fold into a single `Params` carrying indexed terms (`idivf_1`, `pk_1`, `phase_1`, `pn_1`, `idivf_2`, …) plus `n_terms`, so a negative-PN chain yields one `DihedralType` with all `|PN|` terms retained.
- IMPROPER → `def_improperstyle("cvff")` + `def_impropertype` with `[("pk", pk), ("phase", phase), ("pn", pn)]` (no `idivf`).
- NONBON → `def_pairstyle("lj/cut")` + `def_pairtype(symbol, None, [("r_star", r), ("epsilon", eps)])`.

`X` (the GAFF wildcard) is passed through verbatim as an ordinary atom-type name; the store already treats it as a plain string. The embedded file is exposed as a crate const via `include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/gaff.dat"))` in `amber_dat.rs`, mirroring the `molrs-core` `MMFF94_XML` precedent but kept entirely within molrs-ff (molrs-core untouched). A convenience `gaff_forcefield()` parses that embedded const.

Reading is total: a malformed required field is an `Err(String)`, never a silently-zeroed parameter.

## Files to create or modify
- `molrs-ff/src/forcefield/amber_dat.rs` (new) — parser, `GaffDatReader` (`ForceFieldReader` impl), embedded-const + `gaff_forcefield()`, inline unit tests.
- `molrs-ff/data/gaff.dat` (new) — GAFF 1.x parameter file vendored from `openmm/openmmforcefields` (MIT), version-pinned, with a provenance + license header (source repo, commit, GAFF version).
- `molrs-ff/tests/ff/fixtures/gaff_<version>.xml` (new, optional) — openmmforcefields' converted ffxml for the same version, used only as the parameter cross-check ground truth in tests.
- `molrs-ff/src/forcefield/mod.rs` — register `pub mod amber_dat;` alongside `xml`/`readers`.
- `molrs-ff/src/lib.rs` — re-export `GaffDatReader` and `gaff_forcefield`.
- `molrs-ff/tests/ff/amber_dat.rs` (new) — data-driven happy-path test asserting against the embedded real `gaff.dat`.

## Tasks
- [ ] Vendor a version-pinned GAFF 1.x `gaff.dat` from `openmm/openmmforcefields` (MIT) at `molrs-ff/data/gaff.dat` with a provenance+license header (repo, commit, GAFF version); add the `include_str!` embedded const + `gaff_forcefield()` in `molrs-ff/src/forcefield/amber_dat.rs` (new)
- [ ] Cross-check the parsed `ForceField` parameters (bond/angle/dihedral/vdW) against openmmforcefields' converted ffxml for the same version as an AmberTools-free ground truth (`molrs-ff/tests/ff/amber_dat.rs`)
- [ ] Write failing tests for section parsing and store mapping against the embedded `gaff.dat` (`molrs-ff/tests/ff/amber_dat.rs`)
- [ ] Write failing inline unit tests for format edge cases — negative-PN multi-term dihedral (e.g. `X-c2-c2-X`), IMPROPER without IDIVF, trailing comments, blank-card termination, position-tolerant hydrophilic line — using tiny `include_str!` fixtures (`molrs-ff/src/forcefield/amber_dat.rs`)
- [ ] Implement `GaffDatReader` section parser (`ForceFieldReader` impl) in `molrs-ff/src/forcefield/amber_dat.rs`, covering MASS/BOND/ANGLE/DIHEDRAL/IMPROPER/NONBON → `ForceField`/`Params`
- [ ] Implement negative-PN multi-term accumulation folding a quartet's continuation cards into one `DihedralType` with all `|PN|` terms in `molrs-ff/src/forcefield/amber_dat.rs`
- [ ] Register `pub mod amber_dat;` in `molrs-ff/src/forcefield/mod.rs` and re-export `GaffDatReader` + `gaff_forcefield` in `molrs-ff/src/lib.rs`
- [ ] Add rustdoc per crate style with units (Å, kcal/mol, deg→rad note) and cite the AmberTools source for the embedded file
- [ ] Run full check + test suite (`cargo fmt --all --check && cargo clippy -- -D warnings && cargo check && cargo test --all-features`)

## Testing strategy
- **Happy path (against embedded real `gaff.dat`)**: parse without error; assert each section yields its expected populated count (atom types, bond types, angle types, dihedral types, improper types, pair types) and is non-empty.
- **MASS spot-check**: a known symbol parses to its correct mass and polarizability, with a trailing comment present on the card.
- **BOND / ANGLE numeric spot-check**: a known entry's `k`/`r0` (bond) and `k`/`theta0` (angle) match the file values.
- **Negative-PN multi-term (RED-critical)**: a known multi-term quartet (e.g. `X-c2-c2-X`) yields a single dihedral type whose term count equals the number of continuation cards, with every `|PN|` retained and no term dropped.
- **IMPROPER**: a known improper parses with no `idivf` param and `pk` used directly.
- **NONBON**: a known symbol's `r_star`/`epsilon` parse correctly from the MOD4/RE block.
- **Position-tolerance**: a fixture placing the hydrophilic-atoms line at a non-canonical position still parses all downstream sections.
- **Edge cases**: blank-card termination ends a section; trailing free-text comments tolerated; malformed required field returns `Err`.
- **Round-trip**: parsed entries are retrievable through `ForceField` query methods (`get_atomtypes`, `get_bondtype` order-independent, `get_pairtype`, etc.), confirming the store types are populated, not just counted.
- **Domain validation**: the multi-term dihedral and no-IDIVF improper invariants above are the binding correctness checks for this sub-spec; antechamber/energy parity is explicitly out of scope (later sub-specs).

## Out of scope
- Atom typing / SMARTS perception of GAFF types (sub-spec `gaff-typifier-02-typing`).
- Parameter assignment / wildcard (`X`) matching onto a typed structure (sub-spec `gaff-typifier-03-assign`).
- antechamber / AmberTools numerical parity validation (sub-spec `gaff-typifier-04-parity`).
- Any Python binding for the reader (kept Rust-only this sub-spec).
- Storing the hydrophilic-atoms set and the H-bond 10-12 / nonbonded-equivalencing blocks as data (parsed-and-skipped only; not needed until typing/assignment).
- GAFF2 / `gaff2.dat` (this sub-spec is GAFF 1.x only).
- Modifying `molrs-core` (embedding stays in molrs-ff).
