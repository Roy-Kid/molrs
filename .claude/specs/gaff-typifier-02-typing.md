---
title: GAFF atom-type decision engine (non-conjugated type set)
status: approved
created: 2026-06-12
---

# GAFF atom-type decision engine (non-conjugated type set)

## Summary

Add a GAFF atom-type decision engine to `molrs-ff` that, given a perceived `Atomistic` molecule, assigns the **non-conjugated** subset of GAFF atom types (the H/C/N/O/S/P/halogen types listed below) using only **local** per-atom decision inputs — element, coordination number, attached-H count, electron-withdrawing (EW) neighbor count, ring membership and ring size, single-ring benzene-type aromaticity, and local hybridization/bond-order. This is sub-spec 2 of 4 in Milestone 1 of the GAFF-typifier chain: it delivers the typing logic (`atom_typing.rs`) plus a `GaffTypifier` skeleton mirroring `MMFFTypifier`, while deferring `gaff.dat` parsing (01-parser, **forward dependency**), bonded-parameter assignment (03-assign), and antechamber parity (04-parity). Conjugated, fused, and heteroaromatic systems are explicitly out of contract and must be rejected or flagged out-of-contract, never silently mis-typed.

## Domain basis

GAFF atom types and their semantics:

- Wang, Wolf, Caldwell, Kollman, Case. "Development and testing of a general AMBER force field." *J. Comput. Chem.* 25(9):1157-1174 (2004). DOI 10.1002/jcc.20035.
- Wang, Wang, Kollman, Case. "Automatic atom type and bond type perception in molecular mechanical calculations." *J. Mol. Graph. Model.* 25(2):247-260 (2006). DOI 10.1016/j.jmgm.2005.12.005.

**Executable conformance oracle (ground truth):** antechamber `ATOMTYPE_GFF.DEF` (AmberTools `dat/antechamber/`) and the type table at `ambermd.org/antechamber/gaff.html`. The two papers give semantics only; where prose and `.DEF` differ, the `.DEF` wins. The decision rules in this sub-spec are encoded to match `ATOMTYPE_GFF.DEF` for the in-scope types and must remain defensible against it.

**Licensing guard (clean-room).** No copyable typing algorithm exists upstream: `openmm/openmmforcefields` (MIT) — the project's chosen ground truth for parameters — does **not** reimplement GAFF typing; it shells out to antechamber. The only "algorithm" is antechamber's `atomtype.c`, which is **GPL-3**. Therefore this engine MUST be a clean-room reimplementation derived from the `ATOMTYPE_GFF.DEF` ruleset (a data/definition file) + the 2006 paper — **do NOT transcribe antechamber's GPL C** into this MIT/Apache crate. Equivalence with antechamber is established empirically by the `gaff-typifier-04-parity` fixtures, not by copying its code.

Per-atom decision inputs (all local for Milestone 1 — no global conjugation walk):
element; coordination number (heavy + H); attached-H count; attached electron-withdrawing (EW) neighbor count (N, O, F, Cl, Br, I, S per antechamber EW convention); ring membership and ring size (3/4/5/6 via `RingInfo`); single-ring benzene-type aromaticity (`perceive_aromaticity`); local hybridization / bond-order (via `mmff::hybrid::Hyb`).

## Design

`GaffTypifier` mirrors `MMFFTypifier` (`molrs-ff/src/typifier/mmff/mod.rs:46-118`): a struct that will own typing metadata + force field once 01-parser/03-assign land. For this sub-spec the struct exposes a public `assign_atom_types(&self, mol, ring_info) -> HashMap<AtomId, GaffAtomType>` and implements the `Typifier` trait, with `typify` deferring bonded-parameter assembly to 03 (it may return a `Frame` carrying the atom-type block, or a documented `Err("bonded assignment deferred to gaff-typifier-03-assign")` until 03 lands — chosen and documented in code).

Core entities:

- `GaffAtomType` — an enum (or newtype over a static `&str` label) for the in-scope GAFF types, with `as_str()` yielding the canonical lowercase GAFF label (`hc`, `ca`, `n3`, …). Lives in `atom_typing.rs`. New symbol.
- `assign_atom_types(mol: &Atomistic, ring_info: &RingInfo, /* params placeholder */) -> Result<HashMap<AtomId, GaffAtomType>, String>` — the free function in `atom_typing.rs` that `GaffTypifier::assign_atom_types` delegates to, mirroring `mmff::atom_typing::assign_atom_types`.
- Per-element decision routines (private fns: `type_hydrogen`, `type_carbon`, `type_nitrogen`, `type_oxygen`, `type_sulfur`, `type_phosphorus`, `type_halogen`) keyed off local inputs.
- An **out-of-contract guard**: before assigning, detect inputs the non-conjugated set cannot represent faithfully — any aromatic ring that is not a single benzene-like all-carbon 6-ring (i.e. heteroaromatic, fused, or bridged aromatic), and conjugated/alternating C=C–C=C chains that antechamber would route to `cc/cd/ce/cf` machinery. Such atoms cause `assign_atom_types` to return `Err(...)` (or mark the atom with an explicit out-of-contract sentinel) rather than emit a plausible-but-wrong local type.

**Rule encoding choice:** use **local predicate Rust code** (per-element decision routines), not the native SMARTS engine. Justification: the in-scope §types set is decided entirely by integer-valued local inputs (coordination, H-count, EW-count, ring size, aromatic flag) already exposed by `RingInfo` / `Hyb` / `perceive_aromaticity`; predicate code keeps the decision boundary against `ATOMTYPE_GFF.DEF` auditable line-by-line and avoids a SMARTS dependency for what are arithmetic comparisons. (SMARTS at `molrs-core/src/chem/smarts/` remains available; if 04-parity later shows a rule is cleaner as SMARTS, it can be swapped without changing the public surface.)

**Documented deliberate asymmetry:** `ca` (benzene aromatic carbon) is **in** Milestone 1 because a single all-carbon benzene ring is decidable from purely local inputs (6-ring + aromatic flag + sp2 C). `nb` (pyridine aromatic N) is **deferred** to a later milestone because antechamber assigns `nb` through the same conjugation/ring machinery as the alternating `cc/cd/ce/...` types, which requires the global walk that is out of scope here. This asymmetry is intentional and must be documented in the module doc-comment so it does not read as an inconsistency.

## Files to create or modify

- `molrs-ff/src/typifier/gaff/mod.rs` (new) — `GaffTypifier` struct + `Typifier` impl + `pub mod atom_typing;` + module doc documenting the `ca`-in / `nb`-out asymmetry and the out-of-contract policy.
- `molrs-ff/src/typifier/gaff/atom_typing.rs` (new) — `GaffAtomType`, `assign_atom_types`, per-element decision routines, EW-neighbor + out-of-contract helpers.
- `molrs-ff/src/typifier/gaff/tests.rs` (new) — `#[cfg(test)]` unit tests over hand-built representative molecules (per § Testing strategy).
- `molrs-ff/src/typifier/mod.rs` — register `pub mod gaff;` alongside the existing `pub mod mmff;`.

## Tasks

- [ ] Write failing tests for GaffAtomType labels and EW-neighbor + out-of-contract helpers (molrs-ff/src/typifier/gaff/tests.rs)
- [ ] Implement GaffAtomType enum with as_str() canonical labels in molrs-ff/src/typifier/gaff/atom_typing.rs
- [ ] Implement EW-neighbor counting and the out-of-contract guard (heteroaromatic/fused/conjugated detection) in molrs-ff/src/typifier/gaff/atom_typing.rs
- [ ] Write failing tests for per-element typing over representative non-conjugated molecules (molrs-ff/src/typifier/gaff/tests.rs)
- [ ] Implement assign_atom_types with per-element decision routines (H/C/N/O/S/P/halogen) in molrs-ff/src/typifier/gaff/atom_typing.rs
- [ ] Implement GaffTypifier skeleton and Typifier impl (typify defers bonded assembly to 03) in molrs-ff/src/typifier/gaff/mod.rs, documenting the ca-in/nb-out asymmetry
- [ ] Register pub mod gaff; in molrs-ff/src/typifier/mod.rs
- [ ] Write failing test asserting a conjugated/heteroaromatic input is rejected or flagged out-of-contract (molrs-ff/src/typifier/gaff/tests.rs)
- [ ] Run full check + test suite

## Testing strategy

Happy-path representative non-conjugated molecules (hand-built `Atomistic`; permitted as pure-function unit fixtures since this is typing logic, not IO), each asserting the full per-atom GAFF type map:

- ethanol → `c3, c3, oh, ho, hc(×3 on CH3), h1(on CH2 next to O)`
- diethyl ether → `os` on the bridging O, `c3`/`hc`/`h1` as appropriate
- acetone → carbonyl `c` + `o`, methyl `c3`/`hc`
- acetonitrile → `c1`, `n1`, methyl `c3`/`hc`
- nitromethane → `no` (nitro N), `o` (×2), `c3`, `h3` (CH3 with the nitro EW neighbor)
- methylamine → `n3`, `hn`, `c3`, `hc`
- tetramethylammonium → `n4`, four `c3`
- benzene → `ca` (×6), `ha` (×6)
- fluorobenzene → `ca`, `f`, with `h4` on the ring carbon ortho to F (aromatic C-H with 1 EW neighbor)
- cyclopropane → `cx` (sp3 3-ring); cyclobutane → `cy` (sp3 4-ring)
- methanethiol → `sh`, `hs`; dimethyl sulfide → `ss`
- a sulfone (e.g. dimethyl sulfone) → `s6`
- a phosphate (e.g. trimethyl phosphate) → `p5`, `o`, `os`
- halomethanes (CH3F, CH2Cl2, CHBr3, CH3I) → `f/cl/br/i` with `h1`/`h2`/`h3` reflecting the EW count

Edge cases: one-connected sulfur → `s`; aromatic ring carbon bearing two EW ring neighbors → `h5` on its H; sp2 aliphatic non-conjugated C=C (e.g. isobutene central C) → `c2`; amide N → `n` vs amine `n3` vs aromatic-substituent `nh`.

Out-of-contract: pyridine, furan, and 1,3-butadiene each cause `assign_atom_types` to return `Err`/out-of-contract sentinel — verified never to silently produce a plausible local type (e.g. pyridine N must not come back as `n2`/`na`).

Domain validation: each asserted type map is the type set documented in `ATOMTYPE_GFF.DEF` / `gaff.html` for that molecule; deviations are bugs in this engine, not in the test.

## Out of scope

- `gaff.dat` parameter-file parsing — delivered by **gaff-typifier-01-parser** (forward dependency: `GaffTypifier`'s eventual params field consumes its output).
- Bond/angle/dihedral/improper parameter assignment and full `Frame` compilation — **gaff-typifier-03-assign**.
- antechamber executable parity / regression corpus — **gaff-typifier-04-parity** (no AmberTools needed in this sub-spec).
- Conjugated/alternating, fused/bridged, and aromatic-heteroatom types: `cc, cd, ce, cf, cg, ch, cp, cq, nb, nc, nd, ne, nf, pb, pc, pd, pe, pf, px, py, sx, sy`. Deliberate asymmetry: `ca` (benzene C) is in Milestone 1 but `nb` (pyridine aromatic N) is deferred, because antechamber assigns `nb` via the same conjugation/ring machinery as the alternating types; documented in the module doc-comment so it does not read as inconsistent.
- GAFF2, parmchk2, AM1-BCC charge assignment.
- pyo3 / Python binding (no binding in this sub-spec).
