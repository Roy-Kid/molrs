# Spec: OPLS-AA Energy/Force Kernels (+ superseded molpy seam framing)

> **2026-06-10 architecture correction.** The "molpy→molrs typed-frame *seam*"
> framing below (a molpy-side emitter that builds a ForceField and hands molrs a
> typed frame) is **superseded** and must not be implemented. Per the durable
> directive *core sinks to molrs* (memory: `feedback-core-sinks-to-molrs`,
> roadmap: `project-molrs-core-sink-roadmap`), the FF and its IO live in molrs:
> molrs reads the OPLS format itself, the typifier sinks to molrs, and
> `molrs.ForceField.compile(frame)` runs entirely in molrs. **The two kernels in
> this spec (OPLS dihedral + `coul/cut`) are correct and shipped** — only the
> molpy-emitter seam (ac-008/ac-009 as written) is retired in favor of the
> sink-down tracks.

## Summary
Add the two force-field kernels molrs-ff is missing to evaluate an OPLS-AA
system — an OPLS 4-cosine (Fourier) dihedral and a generic cut Coulomb pair —
and define the typed-`Frame` contract by which molpy (which keeps the SMARTS
typifier) hands an OPLS-typed molecule to molrs for energy/force evaluation and
geometry optimization via the already-shipped `molrs_ff::{minimize,
minimize_batch}`.

## Motivation
The geometry optimizer (`geometry-optimizer-01-generic-batch`) relaxes any
`Potential`, but molrs-ff can only assemble a *complete* force field for MMFF94.
OPLS-AA — molpy's most-developed force field, with a mature SMARTS typifier
(`molpy/typifier/atomistic.py`, `oplsaa.xml`) — has no E/F path in molrs, so
OPLS molecules can't be optimized by the Rust engine. The typifier stays in
molpy (SMARTS-driven, override/dependency resolution — a separate B-line port);
molrs only needs the **kernels + the typed-frame seam**. The E/F path is already
typifier-agnostic: `ForceField::compile(typed_frame) -> Potentials ->
eval(coords)`. bond/angle reuse the existing `harmonic` kernels; only the
dihedral and Coulomb terms are missing.

## Scope
- **Crates affected**: `molrs-ff` (two new kernels + registry entries). No
  molpy code is written here — molpy's side of the seam is a follow-up tracked
  in Out of Scope.
- **Traits extended**: `Potential` (two new implementors via `KernelRegistry`).
- **Traits created**: none.
- **Data structures**: `DihedralOPLS` kernel struct, `PairCoulCut` kernel
  struct. No change to `ForceField` / `Frame` / `Potentials`.
- **Feature flags**: none.

## Technical Design

### API Surface

Two kernels registered in `KernelRegistry::register_builtins`
(`molrs-ff/src/potential/mod.rs`):

```rust
// molrs-ff/src/potential/dihedral/opls.rs
self.register("dihedral", "opls", dihedral::opls::dihedral_opls_ctor);

// molrs-ff/src/potential/pair/coul_cut.rs
self.register("pair", "coul/cut", pair::coul_cut::pair_coul_cut_ctor);
```

```rust
/// OPLS 4-cosine (Fourier) proper dihedral.
///
/// E(φ) = ½[ F1(1+cos φ) + F2(1−cos 2φ) + F3(1+cos 3φ) + F4(1−cos 4φ) ]
///
/// Pre-resolved per-dihedral atom quadruples and (F1,F2,F3,F4) coefficients
/// (kcal/mol). Reuses `geometry::compute_dihedral` /
/// `geometry::accumulate_dihedral_forces`.
pub struct DihedralOPLS {
    atoms: Vec<[usize; 4]>,
    f1: Vec<F>, f2: Vec<F>, f3: Vec<F>, f4: Vec<F>,
}

/// Coulomb pair interaction with a hard distance cutoff.
///
/// E(r) = k_e · q_i q_j / r   for r < r_cut, else 0.
/// k_e = 332.063_71 kcal·Å·mol⁻¹·e⁻² (CODATA-derived; matches molpy).
///
/// Per-pair `qiqj` (charge product) is supplied already scaled, so OPLS 1-4
/// electrostatic scaling is baked into the pair list (see Frame schema).
pub struct PairCoulCut {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    qiqj: Vec<F>,
    cutoff: F,   // f64::INFINITY allowed (no cutoff)
}
```

Both implement `fn eval(&self, coords: &[F]) -> (F, Vec<F>)` (forces = −grad,
length `coords.len()`), `Send + Sync` like every kernel.

### Frame schema (the molpy↔molrs seam)

molpy's OPLS typifier assigns atom types, then builds a typed `Frame` plus a
matching `ForceField` (Styles/Types/Params). `ForceField::compile(frame)`
resolves it into `Potentials`. The blocks and the type-label → `Params` mapping
each style consumes:

| Block | Columns | Style → Params keys |
|---|---|---|
| `atoms` | `x,y,z` (f64), `type` (str), `charge` (f64) | — |
| `bonds` | `atomi,atomj` (u32), `type` (str) | `bond:harmonic` → `k`, `r0` |
| `angles` | `atomi,atomj,atomk` (u32), `type` (str) | `angle:harmonic` → `k`, `theta0` |
| `dihedrals` | `atomi,atomj,atomk,atoml` (u32), `type` (str) | `dihedral:opls` → `F1,F2,F3,F4` |
| `impropers` | `atomi..atoml` (u32), `type` (str) | `improper:periodic` *(see below)* |
| `pairs` | `atomi,atomj` (u32), `type` (str) | `pair:lj/cut` → `epsilon,sigma`; `pair:coul/cut` → `qiqj` |

**Exclusions, 1-4 scaling, and combining rules live entirely on molpy's side,
baked into the `pairs` block** (this spec does not add topology awareness to the
kernels):
- 1-2 and 1-3 pairs are simply absent from `pairs`.
- 1-4 pairs are present with **pre-scaled** params: OPLS uses 0.5 for both LJ
  and Coulomb, so molpy emits a pair-type whose `epsilon` is 0.5·√(εᵢεⱼ) (and
  `sigma` = √(σᵢσⱼ)) and whose `qiqj` is 0.5·qᵢqⱼ. Full (≥1-5) pairs use the
  unscaled geometric-combined params.
- A single `pairs` block feeds **both** `lj/cut` and `coul/cut` (one row list,
  each kernel reads its own param keys), so a pair appears once with all of
  `epsilon`, `sigma`, `qiqj` available on its pair-type.

The existing `pair_lj_cut_ctor` already reads exactly this `pairs` schema
(`atomi/atomj/type` → `Params{epsilon, sigma}`); `coul/cut` mirrors it
(`… → Params{qiqj}`), and the cutoff comes from the pair **style** params
(`r_cut`, default ∞).

**OPLS impropers.** OPLS-AA represents impropers as a single periodic term
`E = K(1 + cos(nψ − ψ0))` (typically n=2, ψ0=180°, only for specific sp²
centers). If molpy emits an `impropers` block, register
`improper:periodic` → `improper::periodic` with params `K`, `n`, `psi0`.
Reuse the dihedral geometry (improper angle = dihedral over the four atoms in
the convention molpy emits). If molpy folds OPLS impropers into the
`dihedrals` block instead, no improper kernel is needed — the spec implements
`improper:periodic` only if the molpy emitter produces an `impropers` block
(decided during impl by inspecting molpy's OPLS frame output).

### Data Flow
```
molpy.Atomistic ─OPLS SMARTS typifier→ typed Atomistic (atom/bond/.../pair types + params)
            │  molpy builds: pairs with exclusions + 1-4 scaling + geometric combining baked in
            ▼
        typed Frame  ─────────────────────────┐
        molrs ForceField (Styles/Types/Params)─┤
                                               ▼
                       ForceField::compile(frame) → Potentials
                                               ▼
                       Potentials.eval(coords) / minimize / minimize_batch
```

### Algorithm

**OPLS dihedral** — for each dihedral (i,j,k,l):
1. `φ = compute_dihedral(coords, i, j, k, l)` (existing helper).
2. `E += 0.5·[F1(1+cos φ) + F2(1−cos 2φ) + F3(1+cos 3φ) + F4(1−cos 4φ)]`.
3. `dE/dφ = 0.5·[−F1 sin φ + 2F2 sin 2φ − 3F3 sin 3φ + 4F4 sin 4φ]`.
4. `accumulate_dihedral_forces(coords, i,j,k,l, dE/dφ, &mut forces)` (existing
   helper distributes the torsional force to the four atoms).
Complexity O(n_dihedrals).

**Coulomb cut** — for each pair (i,j) in `pairs`:
1. `r2 = |r_j − r_i|²`; if `r2 ≥ cutoff²` or `r2 < 1e-24`, skip.
2. `r = √r2`; `E += k_e · qiqj / r`.
3. `dE/dr = −k_e · qiqj / r²`; `F = −(dE/dr)·(r̂)` distributed ±to i/j
   (Newton's third law). In force form: `factor = k_e·qiqj / (r2·r)`;
   `F_j += factor·(r_j−r_i)`, `F_i −= …`.
Complexity O(n_pairs).

### Integration Points
- `molrs-ff/src/potential/dihedral/opls.rs` (new) + `mod.rs` re-export;
  `molrs-ff/src/potential/pair/coul_cut.rs` (new) + `mod.rs` re-export.
- Two `self.register(...)` lines in `KernelRegistry::register_builtins`
  (`molrs-ff/src/potential/mod.rs`).
- No change to the optimizer, `Potentials`, `ForceField`, or `Frame`.
- Consumes existing `geometry::{compute_dihedral, accumulate_dihedral_forces}`.

## Constraints & Invariants
- `forces = −gradient`; Newton's third law per term (Σ forces ≈ 0 to machine
  precision for an isolated dihedral / pair).
- `F = f64` throughout; coords flat `[x0,y0,z0,…]`.
- Units: energy kcal/mol, force kcal/mol/Å, length Å, charge e;
  `k_e = 332.063_71` kcal·Å·mol⁻¹·e⁻² (single named constant, matches molpy's
  Coulomb constant).
- Kernels are **topology-blind**: they never compute exclusions or combining
  rules. Any 1-4 scaling / geometric combination is the caller's (molpy's)
  responsibility, baked into the `pairs` block. This invariant is documented on
  both kernels.
- Dihedral at `φ` where `sin φ → 0`: the torque helper must stay finite
  (existing `accumulate_dihedral_forces` already guards near-collinear j-k).

## Test Criteria

### Unit Tests (`molrs-ff`, inline `#[cfg(test)]`)
1. **OPLS dihedral energy shape**: with only `F1>0` (others 0), `E(φ=0)=F1`,
   `E(φ=π)=0`; with only `F2>0`, `E(φ=0)=0`, `E(φ=π/2)=F2`. Verifies each
   cosine term's phase.
2. **OPLS dihedral numerical gradient**: random F1..F4 and a non-degenerate
   4-atom geometry; central finite difference (h=1e-6) matches analytic forces,
   max component error < 1e-5 kcal/mol/Å.
3. **OPLS dihedral Newton's third law**: Σ forces over the 4 atoms ≈ 0 (<1e-9).
4. **Coulomb energy + sign**: two unit charges at r: `E = k_e/r` for `qiqj=1`;
   like charges repel (force pushes apart), unlike attract; `E=0` beyond cutoff.
5. **Coulomb numerical gradient**: finite difference matches analytic, max
   component error < 1e-5.
6. **Coulomb cutoff & zero-distance guard**: pair beyond `r_cut` contributes 0;
   coincident atoms (r²<1e-24) skipped without NaN.

### Integration Tests (`molrs-ff/tests/ff/…`)
7. **compile + eval an OPLS-style frame built in code**: hand-build a small
   typed `Frame` (e.g. butane: harmonic bonds/angles + one `dihedral:opls` +
   `lj/cut`+`coul/cut` pairs) and a matching `ForceField`; `compile().eval()`
   returns finite energy and forces, and analytic forces match finite
   difference over the whole assembled potential (h=1e-5, <1e-5).
8. **minimize an OPLS frame**: `minimize` over the compiled OPLS `Potentials`
   lowers the energy monotonically to `fmax<0.05`; `minimize_batch` over a small
   homogeneous batch reproduces the single result (reuses the optimizer, no new
   optimizer code).

### Numerical Validation (cross-library; bench repo or gated test)
9. **OPLS parity vs molpy**: for butane and ethanol, OPLS-typed by molpy, the
   molrs total energy and each per-term energy (bond/angle/dihedral/LJ/Coulomb)
   match molpy's own numpy OPLS potentials on identical coordinates within
   1e-4 kcal/mol. This is the binding seam contract.
10. **End-to-end (Python)**: molpy OPLS-typifies a molecule → emits the typed
    frame + ForceField → molrs `compile` → `minimize` converges and lowers the
    molpy-computed OPLS energy.

## Performance Requirements
- Both kernels O(n_terms) per eval, no allocation beyond the output force
  buffer; consistent with existing harmonic/LJ kernels. No specific benchmark
  target beyond "no O(N²) in the per-term loop" (pair list is supplied, not
  rebuilt).

## Migration & Compatibility
- Purely additive: two new kernel modules + two registry entries. No existing
  kernel, signature, or schema changes. MMFF and harmonic paths untouched.

## Out of Scope
- **OPLS/GAFF typifier sink-down** to molrs (SMARTS-based; molrs-core already
  has the SMARTS matcher but override/priority/dependency resolution is a
  substantial port) — separate B-line spec. molpy remains the OPLS typifier.
- **molpy-side emitter** that writes the typed `Frame` + molrs `ForceField`
  from an OPLS-typed structure — a molpy change, specced/implemented in molpy
  (this spec defines the contract it must satisfy; tests 9–10 live where the
  cross-library harness runs, e.g. `bm-molrs-molpy`).
- **Long-range electrostatics** (PME/Ewald) for OPLS — `coul/cut` only here.
- **Other force fields** (GAFF, CHARMM, AMBER) kernels.
- **Vectorized batched kernels** — the optimizer's rayon batch path is reused
  as-is.
