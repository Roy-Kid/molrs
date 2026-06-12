---
title: GAFF parameter assignment with X-wildcard precedence
status: approved
created: 2026-06-12
---

# GAFF parameter assignment with X-wildcard precedence

## Summary
This sub-spec (3 of 4 in Milestone 1 of the GAFF-typifier chain) wires the parsed GAFF parameter store from `gaff-typifier-01-parser` and the per-atom GAFF atom types from `gaff-typifier-02-typing` into a fully parameterized `Frame`. It implements antechamber bond/angle/dihedral/improper parameter assignment by atom-type tuple, including the X-wildcard fallback (try the exact tuple first, then the most-specific wildcard match) that the generic `ForceField` store does not provide. After this lands, `GaffTypifier::typify` returns a `Frame` whose every bond, angle, dihedral, and improper carries resolved GAFF parameters, and that `Frame` compiles to `Potentials` through the existing `ForceField::to_potentials` path. Parsing remains owned by 01, atom typing by 02, and bit-exact antechamber parity is deferred to 04-parity.

## Domain basis
Antechamber matching semantics (Wang, Wang, Kollman, Case, *J. Mol. Graph. Model.* 25(2):247-260, 2006, DOI 10.1016/j.jmgm.2005.12.005) layered on the GAFF force field (Wang, Wolf, Caldwell, Kollman, Case, *J. Comput. Chem.* 25(9):1157-1174, 2004, DOI 10.1002/jcc.20035). Assignment rules implemented here:

- **Exact precedence.** An exact atom-type tuple match takes precedence over any wildcard (`X`) match. Among wildcard matches, the most-specific (fewest `X` positions) wins.
- **Wildcard fallback.** Dihedrals and impropers commonly carry general `X-X-a-b` terms used only when no specific quartet exists.
- **Order independence.** Bond and angle tuples match regardless of endpoint ordering (`a-b` ≡ `b-a`; `a-b-c` ≡ `c-b-a`), with the central atom fixed for angles.
- **Multi-term dihedrals.** A matched quartet accumulates all Fourier terms emitted by 01-parser's negative-PN accumulation; the torsion energy is `E = Σ (PK/IDIVF)(1 + cos(PN·φ − PHASE))` over those terms.
- **Improper convention.** Improper PK is applied directly with **no IDIVF division** (unlike proper torsions).

Units follow the GAFF/AMBER convention as carried verbatim by 01-parser (kcal/mol for force constants, degrees for phases, Å for lengths). No unit conversion is performed in this sub-spec.

## Design
The crate mirrors the existing `molrs-ff/src/typifier/mmff/{classify,frame_builder,params}.rs` triad under `molrs-ff/src/typifier/gaff/`. As in MMFF, the typed `Frame` carries, in each topology block's `type` column, the **resolved stored type-name** — the name under which the matched parameter lives in the `ForceField` store — so that the existing `ForceField::subset` / `ForceField::to_potentials` pipeline resolves parameters with no GAFF-specific changes downstream.

- **`params.rs`** — owns `GaffParams` (the assignment-facing view over the parsed store from 01) and the `GaffTypifier` fields needed by assignment. It exposes the parameter store and the per-atom type map produced by 02 to the frame builder. It does **not** parse (01) and does **not** assign atom types (02); it adapts their outputs into the lookups assignment needs.
- **`classify.rs`** — owns the new wildcard-precedence resolver. For a topology tuple of GAFF atom-type names, it returns the resolved stored type-name (or a structured "no match" result) by: (1) trying the exact tuple in both canonical orders; (2) on miss, enumerating wildcard candidates, scoring each by specificity (count of non-`X` positions), and selecting the unique most-specific match. This logic lives here and is **not** bolted onto the generic `Style::get_bondtype` path. Bond/angle/dihedral/improper each get a resolver entry point sharing one specificity-scoring core.
- **`frame_builder.rs`** — mirrors `build_mmff_frame`: enumerates topology via `Topology` (bonds/angles/dihedrals/impropers), looks up each tuple through `classify.rs`, writes the resolved stored type-name into each block's `type` column, and assembles atoms/bonds/angles/dihedrals/impropers/pairs blocks. A tuple with neither an exact nor a wildcard match surfaces a clear `Err` naming the category and the offending atom-type tuple, rather than silently dropping the term.
- **`mod.rs`** — defines `GaffTypifier` and its `Typifier::typify` impl delegating to `frame_builder` (the `mmff/mod.rs:114-118` pattern). `GaffTypifier::typify(mol) -> Result<Frame, String>`; the existing `ff.to_potentials(&frame)` compiles it.

Ownership: `GaffTypifier` owns the parsed store (from 01) and the typing metadata (from 02); the builder borrows both. The `Frame` is the single product handed back; no parameter state is retained beyond the typifier's owned stores.

## Files to create or modify
- `molrs-ff/src/typifier/gaff/mod.rs` (new) — `GaffTypifier` + `Typifier` impl.
- `molrs-ff/src/typifier/gaff/classify.rs` (new) — wildcard-precedence resolvers (bond/angle/dihedral/improper) + specificity core.
- `molrs-ff/src/typifier/gaff/frame_builder.rs` (new) — `build_gaff_frame` assembling the typed `Frame`.
- `molrs-ff/src/typifier/gaff/params.rs` (new) — `GaffParams` assignment-facing view adapting 01/02 outputs.
- `molrs-ff/src/typifier/mod.rs` — register `pub mod gaff;` alongside `pub mod mmff;`.

## Tasks
- [ ] Write failing tests for wildcard-precedence resolver (molrs-ff/src/typifier/gaff/classify.rs `#[cfg(test)]`)
- [ ] Implement specificity-scoring core + bond/angle/dihedral/improper resolvers in molrs-ff/src/typifier/gaff/classify.rs
- [ ] Implement GaffParams assignment view adapting 01-parser store + 02-typing map in molrs-ff/src/typifier/gaff/params.rs
- [ ] Write failing tests for build_gaff_frame assignment + no-match error (molrs-ff/src/typifier/gaff/frame_builder.rs `#[cfg(test)]`)
- [ ] Implement build_gaff_frame in molrs-ff/src/typifier/gaff/frame_builder.rs writing resolved type-names into block `type` columns
- [ ] Implement GaffTypifier + Typifier::typify delegating to frame_builder in molrs-ff/src/typifier/gaff/mod.rs and register `pub mod gaff;` in molrs-ff/src/typifier/mod.rs
- [ ] Write failing end-to-end test: typed in-scope molecule → typify → to_potentials yields Potentials
- [ ] Add rustdoc per crate style with units on GaffTypifier, GaffParams, and the resolver entry points
- [ ] Run full check + test suite

## Testing strategy
- **Happy path (exact tuples).** A typed non-conjugated molecule (e.g. typed ethanol, all-`c3`/`oh`/`ho`/`hc`) resolves every bond and angle to an exact stored GAFF type-name; the resulting block `type` columns contain only resolved names.
- **Wildcard precedence (RED).** Given a store holding both an exact dihedral quartet and a general `X-X-a-b` term that also matches, the resolver selects the exact quartet. Explicit RED test asserting the wildcard is *not* chosen when an exact term exists.
- **Wildcard fallback.** A dihedral whose quartet has no exact term but matches a single `X`-term resolves to that general term.
- **Most-specific wildcard wins.** When two wildcard terms match at different specificity (e.g. `X-a-b-c` vs `X-X-b-c`), the one with fewer `X` positions is chosen.
- **Order independence.** Bond `a-b` resolves identically to `b-a`; angle `a-b-c` resolves identically to `c-b-a` (central atom fixed).
- **Multi-term dihedral.** A quartet whose 01-parser entry accumulated multiple Fourier terms (negative-PN continuation) assembles all terms onto the matched dihedral.
- **Improper PK direct.** Improper assignment applies PK with no IDIVF division (asserted against a store entry with a known PK and a non-unit IDIVF that must be ignored).
- **No-match error.** A topology tuple with neither exact nor wildcard match returns an `Err` naming the category and offending atom-type tuple — not a silent skip.
- **End-to-end compile.** `GaffTypifier::typify` on an in-scope molecule returns a `Frame` whose bonds/angles/dihedrals/impropers all carry parameters, and `ff.to_potentials(&frame)` succeeds.
- **Domain validation.** The exact-over-wildcard precedence and improper-no-IDIVF behaviors are checked against the antechamber rules cited in Domain basis.

## Out of scope
- GAFF parameter file parsing and the negative-PN Fourier accumulation itself — owned by `gaff-typifier-01-parser`; this sub-spec consumes its store.
- GAFF atom-type perception — owned by `gaff-typifier-02-typing`; this sub-spec consumes its per-atom type map.
- Bit-exact antechamber parity (golden-file comparison against AmberTools output) — owned by `gaff-typifier-04-parity`. No AmberTools dependency is introduced here.
- Any Python binding for `GaffTypifier` — not in this Rust-only sub-spec.
- Changes to the generic `ForceField` / `Style` lookup paths — wildcard precedence is implemented in `gaff/classify.rs` only.

## Out of scope alternatives considered
Bolting wildcard precedence onto `Style::get_bondtype` was considered and rejected (per load-bearing repo facts): it would entangle GAFF-specific antechamber semantics into the generic store used by MMFF/OPLS. Keeping resolution in `gaff/classify.rs` preserves the generic store's order-independent-but-literal lookup contract.
