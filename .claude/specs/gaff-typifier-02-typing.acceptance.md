---
slug: gaff-typifier-02-typing
criteria:
  - id: ac-001
    summary: GaffAtomType exposes canonical lowercase GAFF labels via as_str()
    type: code
    pass_when: |
      A unit test in molrs-ff/src/typifier/gaff/atom_typing.rs (or tests.rs)
      asserts GaffAtomType::*.as_str() returns the exact lowercase GAFF label
      for at least hc, ha, h1, h4, h5, c, c1, c2, c3, ca, cx, cy, n, n1, n3,
      n4, no, o, oh, os, s, ss, s6, p5, f, cl, br, i — and `cargo test
      -p molcrafts-molrs-ff` passes it.
    status: pending
  - id: ac-002
    summary: pub mod gaff registered alongside pub mod mmff in typifier/mod.rs
    type: code
    pass_when: |
      molrs-ff/src/typifier/mod.rs contains `pub mod gaff;` and `cargo check
      -p molcrafts-molrs-ff` compiles with the new module in the tree.
    status: pending
  - id: ac-003
    summary: GaffTypifier mirrors MMFFTypifier surface and implements Typifier
    type: code
    pass_when: |
      molrs-ff/src/typifier/gaff/mod.rs defines `struct GaffTypifier` with a
      public `assign_atom_types(&self, mol: &Atomistic, ring_info: &RingInfo)`
      returning the per-atom GaffAtomType map, and `impl Typifier for
      GaffTypifier`; compiles under `cargo check -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-004
    summary: Aliphatic oxygenates type correctly (ethanol, ether, acetone)
    type: code
    pass_when: |
      Tests build ethanol, diethyl ether, and acetone as Atomistic and assert
      assign_atom_types yields ethanol {c3,c3,oh,ho,hc,h1}, ether bridging O =
      os, acetone carbonyl C = c with its O = o; `cargo test
      -p molcrafts-molrs-ff` passes.
    status: pending
  - id: ac-005
    summary: sp and nitro/amine nitrogen cases type correctly
    type: code
    pass_when: |
      Tests assert acetonitrile -> {c1,n1}, nitromethane -> {no, o(×2), c3,
      h3}, methylamine -> {n3,hn}, tetramethylammonium -> n4; `cargo test
      -p molcrafts-molrs-ff` passes.
    status: pending
  - id: ac-006
    summary: Benzene and fluorobenzene aromatic typing including h4
    type: code
    pass_when: |
      Test asserts benzene -> ca(×6)/ha(×6), and fluorobenzene -> ca + f with
      the ring C-H ortho to F typed h4 (aromatic C-H, 1 EW neighbor);
      `cargo test -p molcrafts-molrs-ff` passes.
    status: pending
  - id: ac-007
    summary: Ring-strain carbons cx/cy and h5 aromatic-double-EW case
    type: code
    pass_when: |
      Test asserts cyclopropane -> cx, cyclobutane -> cy, and an aromatic C-H
      with two EW ring neighbors -> h5; `cargo test -p molcrafts-molrs-ff`
      passes.
    status: pending
  - id: ac-008
    summary: Sulfur and phosphorus hypervalent/connectivity cases
    type: code
    pass_when: |
      Test asserts methanethiol -> {sh,hs}, dimethyl sulfide -> ss, a sulfone
      -> s6, one-connected S -> s, and a phosphate -> {p5, o, os}; `cargo test
      -p molcrafts-molrs-ff` passes.
    status: pending
  - id: ac-009
    summary: Halomethanes map halogens and h1/h2/h3 by EW count
    type: code
    pass_when: |
      Test asserts CH3F/CH2Cl2/CHBr3/CH3I yield f/cl/br/i with the methyl H
      typed h1/h2/h3 matching the EW-neighbor count; `cargo test
      -p molcrafts-molrs-ff` passes.
    status: pending
  - id: ac-010
    summary: Conjugated/heteroaromatic input rejected, never silently typed
    type: code
    pass_when: |
      Test feeds pyridine, furan, and 1,3-butadiene and asserts
      assign_atom_types returns Err / out-of-contract sentinel for the
      offending atoms (pyridine N is NOT returned as n2/na/n3); `cargo test
      -p molcrafts-molrs-ff` passes.
    status: pending
  - id: ac-011
    summary: ca-in / nb-out asymmetry documented and enforced
    type: code
    pass_when: |
      molrs-ff/src/typifier/gaff/mod.rs module doc-comment explains that ca
      (benzene C) is in Milestone 1 while nb (pyridine aromatic N) is
      deferred to the conjugation machinery, AND a test confirms benzene C ->
      ca succeeds while pyridine aromatic N hits the out-of-contract path
      (ac-010); both observable under `cargo test -p molcrafts-molrs-ff`.
    status: pending
  - id: ac-012
    summary: Full check and test suite pass
    type: runtime
    evaluator_hint: "cargo fmt --all --check && cargo clippy -- -D warnings && cargo check && cargo test --all-features"
    pass_when: |
      cargo fmt --all --check, cargo clippy -- -D warnings, cargo check, and
      cargo test --all-features (or `-p molcrafts-molrs-ff` where BLAS backend
      is unavailable) all succeed with the gaff module in the tree.
    status: pending
---

# Acceptance criteria

- **ac-001..ac-003** — structural: the type enum, the module registration, and the `GaffTypifier` surface mirror the MMFF pattern at `molrs-ff/src/typifier/mmff/mod.rs:46-118`.
- **ac-004..ac-009** — the §types coverage matrix, asserted on hand-built representative molecules spanning H/C/N/O/S/P/halogen non-conjugated types, each map defensible against `ATOMTYPE_GFF.DEF`.
- **ac-010** — the safety bar: conjugated/heteroaromatic inputs must be rejected or flagged out-of-contract, never silently mis-typed.
- **ac-011** — documents and enforces the deliberate `ca`-in / `nb`-out asymmetry so the milestone boundary is explicit.
- **ac-012** — the standard molrs check + test gate.
