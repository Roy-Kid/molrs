# Spec: Generic Geometry Optimizer + Homogeneous Batch Minimization

## Summary

Expose a general-purpose energy-minimization (geometry optimization) API over
any `molrs_ff::potential::Potential` — single structure and homogeneous batch.
The L-BFGS minimizer already exists inside the ETKDG conformer pipeline
(`molrs-conformer/src/etkdg/mmff_min.rs::minimize_lbfgs`); this spec **sinks it
down** into `molrs-ff` as a public, force-field-agnostic optimizer (the same
ownership pattern `molgraph` uses: Rust owns the engine, Python wraps it), adds
a rayon-parallel batch path for homogeneous systems, and binds both to the
Python wheel so a user can go *molecule + force field → optimized structure(s)*.

## Motivation

- molrs already has the full machinery — `molrs-ff` (MMFF94 + harmonic/LJ/PME,
  `Potential::eval(coords) -> (energy, forces)`) and a tested L-BFGS
  (two-loop recursion + Armijo/weak-Wolfe backtracking line search, an
  acknowledged RDKit `BFGSOpt::minimize` port). But the optimizer is **private
  to the ETKDG pipeline** and only reachable via `Conformer.generate()`. There
  is no way to relax an arbitrary system at a fixed force field.
- The Python wheel exposes `Potentials.eval` / `MMFFTypifier.build` but **no
  `minimize`** — users must hand-roll a Python optimizer (as molpy does today
  with a pure-numpy L-BFGS in `molpy/optimize/lbfgs.py`, single-structure only,
  no batch).
- High-throughput workflows (conformer ensembles, dataset relaxation, FF
  parameter fitting) need **batch optimization** of many structures sharing one
  topology. This is mandatory here, scoped to **homogeneous systems** (identical
  atom count and topology ⇒ one `Potentials`, stacked `(B, N, 3)` coordinates).

## Scope

- **Crates affected**: `molrs-ff` (new optimizer module + rayon dep),
  `molrs-conformer` (delete private L-BFGS copy, re-import from `molrs-ff`;
  observable ETKDG behavior unchanged), `molrs-python` (PyO3 bindings + `.pyi`).
- **Traits extended**: none — consumes the existing `Potential` trait.
- **Traits created**: none.
- **Data structures**: `MinimizeOptions`, `OptReport` (Rust);
  `PyOptReport` (Python `OptReport` class).
- **Feature flags**: `molrs-ff` gains a default-on `rayon` feature (mirrors the
  `molrs-core` pattern: `default = ["rayon"]`, serial fallback when disabled).

## Technical Design

### API Surface — Rust (`molrs-ff/src/optimize/`)

```rust
//! Force-field-agnostic geometry optimization over `Potential`.

use crate::potential::Potential;
use molrs::F; // = f64

/// Convergence + step controls for L-BFGS minimization.
///
/// `fmax` is the ASE/molpy convention: the largest per-atom force-vector
/// magnitude max_i ‖F_i‖ (kcal/mol/Å). Optimization stops when it drops below
/// `fmax`, or after `max_steps` outer iterations, whichever comes first.
#[derive(Clone, Copy, Debug)]
pub struct MinimizeOptions {
    /// Max per-atom force magnitude for convergence (kcal/mol/Å). Default 0.05.
    pub fmax: F,
    /// Outer-iteration cap. Default 500.
    pub max_steps: usize,
    /// Per-step displacement cap in Å (trust region). Default 0.2.
    pub max_step: F,
    /// L-BFGS correction-pair history size. Default 8.
    pub memory: usize,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self { fmax: 0.05, max_steps: 500, max_step: 0.2, memory: 8 }
    }
}

/// Outcome of one minimization. `final_fmax` is max_i ‖F_i‖ at the returned
/// point; `final_energy` is the potential energy there (kcal/mol).
#[derive(Clone, Copy, Debug)]
pub struct OptReport {
    pub converged: bool,
    pub n_steps: usize,
    pub final_energy: F,
    pub final_fmax: F,
}

/// Minimize a single structure in place.
///
/// `coords` is the flat `3·n_atoms` buffer `[x0,y0,z0, x1,y1,z1, ...]`,
/// updated in place to the located minimizer. Returns the convergence report.
/// `n_atoms` is inferred from `coords.len() / 3` (errors if not divisible).
pub fn minimize(
    potential: &dyn Potential,
    coords: &mut [F],
    opts: &MinimizeOptions,
) -> Result<OptReport, String>;

/// Minimize a homogeneous batch in place, one report per structure.
///
/// All structures share `potential` (identical topology). `coords` is the
/// concatenation of `n_structs` flat blocks each of length `3 * n_atoms`
/// (row-major over the conceptual `(B, N, 3)` array). Each block is optimized
/// independently; with the `rayon` feature the blocks run in parallel via
/// `par_chunks_mut`, otherwise serially. Errors if
/// `coords.len() != n_structs * n_atoms * 3`.
pub fn minimize_batch(
    potential: &dyn Potential,
    coords: &mut [F],
    n_atoms: usize,
    n_structs: usize,
    opts: &MinimizeOptions,
) -> Result<Vec<OptReport>, String>;
```

The two-loop recursion, line search, and history bookkeeping move verbatim from
`molrs-conformer/src/etkdg/mmff_min.rs` into `molrs-ff/src/optimize/lbfgs.rs`.
The only functional change vs the ported core is the **convergence criterion**:
ETKDG used RMS-gradient (`‖g‖/√n`); the public optimizer converges on `fmax`
(max per-atom force magnitude) to match ASE/molpy. ETKDG's existing RMS-gradient
entry point is preserved (see Integration Points) so its numerics are unchanged.

### API Surface — Python (`molrs-python`)

A **single rank-dispatching** `Potentials.minimize` (single and batch unified
per review feedback — `(N,3)`→single, `(B,N,3)`→batch — rather than two
methods), plus an `OptReport` result class and a `build_mmff_potentials`
helper.

```python
class OptReport:
    converged: bool
    n_steps: int
    final_energy: float   # kcal/mol
    final_fmax: float     # kcal/mol/Å

class Potentials:
    def __len__(self) -> int: ...
    def eval(self, coords: ArrayF) -> tuple[float, ArrayF]: ...
    def energy(self, coords: ArrayF) -> float: ...

    # Rank-dispatched: (N,3) or (3N,) -> single; (B,N,3) -> homogeneous batch.
    @overload
    def minimize(self, coords: ArrayF, *, fmax: float = 0.05, max_steps: int = 500,
                 max_step: float = 0.2, memory: int = 8) -> tuple[ArrayF, OptReport]: ...
    @overload
    def minimize(self, coords: ArrayF, *, fmax: float = 0.05, max_steps: int = 500,
                 max_step: float = 0.2, memory: int = 8) -> tuple[ArrayF, list[OptReport]]: ...

# Recommended constructor: wraps the working MmffForceField path as Potentials.
def build_mmff_potentials(mol: Atomistic, variant: str = "MMFF94") -> Potentials: ...
```

`minimize` accepts `(N, 3)`, flat `(3N,)`, or `(B, N, 3)`; it never mutates the
caller's input (immutable convention) and returns coordinates with the same
rank as a fresh `(N, 3)` / `(B, N, 3)` array. For a `(B, N, 3)` batch it
validates `N == Potentials.n_atoms()` and raises `ValueError` on mismatch, a
trailing axis ≠ 3, or rank > 3. (The Rust layer keeps the two typed functions
`minimize` / `minimize_batch`; only the Python user-facing surface is unified.)

> **Force-field source.** The pre-existing `MMFFTypifier.build` →
> `ForceField::compile` registry path is a documented production defect (the
> `mmff_stbn` kernel's `r0_ij`/`r0_kj` are never merged from the bond table; even
> diatomic bond-type lookups fail). To make the optimizer usable from Python
> today, `build_mmff_potentials` wraps the working, RDKit-energy-validated
> `MmffForceField` model as a single-kernel `Potentials`. Fixing the registry
> `compile` path is a separate follow-up.

#### End-to-end user example (single molecule)

```python
import molrs

mol = molrs.parse_smiles("CCO").to_atomistic()   # Atomistic
mol, _ = molrs.Conformer(seed=7).generate(mol)    # 3D start geometry
potentials = molrs.build_mmff_potentials(mol)     # MMFF94 Potentials for THIS topology
frame = molrs.MMFFTypifier().typify(mol)
coords = molrs.extract_coords(frame).reshape(-1, 3)   # (N, 3)

opt_coords, report = potentials.minimize(coords, fmax=0.05, max_steps=500)
print(report.converged, report.n_steps, report.final_energy)  # True 1 -1.337
```

#### End-to-end user example (homogeneous batch)

```python
import numpy as np, molrs

# B conformers of the SAME molecule (same topology) -> one Potentials object.
potentials = molrs.build_mmff_potentials(mol)
batch = np.stack([c.reshape(-1, 3) for c in conformer_coords])   # (B, N, 3)

opt_batch, reports = potentials.minimize(batch, fmax=0.05)        # 3-D -> batch
energies = np.array([r.final_energy for r in reports])
best = opt_batch[energies.argmin()]          # lowest-energy relaxed conformer
```

**Heterogeneous systems are out of scope for batch.** Different topologies have
different `Potentials` and different `N`, so they cannot be stacked; the user
loops over single `minimize` calls (one `Potentials` each). This is documented
on `minimize_batch` and enforced by the `(B, N, 3)` shape + atom-count check.

### Data Flow

```
Atomistic ─MMFFTypifier.build()→ Potentials ─┐
                                             ├─ minimize(coords, opts) ─→ (coords', report)
flat/stacked coords (from extract_coords) ───┘
                                             └─ minimize_batch(coords(B,N,3)) ─→ (coords'(B,N,3), [report])
```

Python layer: validate shape → reshape to flat `f64` buffer(s) → call Rust
`minimize` / `minimize_batch` → reshape result back to `(N,3)` / `(B,N,3)`
ndarray → wrap reports as `OptReport` objects.

### Algorithm

L-BFGS, unchanged from the existing port (Nocedal & Wright Alg. 7.4 two-loop
recursion + Alg. 3.1 backtracking line search with Armijo `c1=1e-4` and weak
Wolfe `c2=0.9`), history size = `memory`:

1. `(E, F) = potential.eval(x)`; `g = -F`.
2. Convergence test: `fmax = max_i sqrt(F[3i]² + F[3i+1]² + F[3i+2]²)`;
   if `fmax < opts.fmax` → converged.
3. Two-loop recursion gives descent direction `p = -H·g` from the stored
   `(s_k, y_k)` history (initial `H0 = (s·y)/(y·y)·I`).
4. Trust region: scale `p` so `max|Δx_component| ≤ max_step` (Å).
5. Backtracking line search along `p` (Armijo + weak Wolfe), step contraction
   0.5 / expansion 2.1, ≤ 40 trial evals.
6. Update `x`, push `(s,y)` to history (drop oldest beyond `memory`), loop until
   converged or `max_steps`.

- **Single**: complexity per step = one `eval` (cost of the force field) +
  O(`memory`·`n`) vector ops. Steps to convergence are problem-dependent.
- **Batch (homogeneous)**: `n_structs` independent single minimizations.
  Wall-clock with rayon ≈ `ceil(n_structs / n_threads) ·` (per-structure cost);
  embarrassingly parallel via `par_chunks_mut(3 * n_atoms)`. `potential: &dyn
  Potential` is `Send + Sync`, so it is shared by reference across threads (no
  clone). Per-structure state (history buffers) is thread-local.

### Integration Points

- **`molrs-ff`**: new `pub mod optimize;` in `lib.rs`, re-exporting `minimize`,
  `minimize_batch`, `MinimizeOptions`, `OptReport`. Add `rayon` (default-on
  feature) to `molrs-ff/Cargo.toml`, gated like `molrs-core`.
- **`molrs-conformer`**: `mmff_min.rs` currently *owns* `minimize_lbfgs`
  (signature `<F: FnMut(&[f64]) -> (f64, Vec<f64>)>`). Move the core down to
  `molrs-ff::optimize::lbfgs`, keeping a thin RMS-gradient-tolerance entry
  point (`minimize_lbfgs_rms`) for ETKDG so its convergence behavior, step
  caps, and final energies are **bit-for-bit unchanged**. `mmff_min.rs` becomes
  a one-line call into the shared core. This satisfies the "one source of
  L-BFGS" invariant.
- **`molrs-python`**: extend `PyPotentials` (`molrs-python/src/forcefield.rs`)
  with `minimize` / `minimize_batch`; add `PyOptReport` `#[pyclass]`; register
  in the module and add stubs to `molrs-python/python/molrs/molrs.pyi`.

## Constraints & Invariants

- **Force sign**: `Potential::eval` returns `forces = -gradient`; the optimizer
  steps along `-H·g = +H·F`. Must not be re-flipped during extraction.
- **`F = f64`** throughout the optimizer (matches the force-field path; never
  raw `f32`). Python accepts `float32`/`float64` input and up-casts to `f64`
  for the Rust call, returning `float64`.
- **Immutability at the Python boundary**: input ndarrays are never mutated;
  optimized coordinates are returned as fresh arrays (Rust `&mut [F]` operates
  on an internal copy of the caller's data).
- **Homogeneity (batch)**: `minimize_batch` assumes one shared topology; it does
  not inspect per-structure connectivity. The `(B, N, 3)` shape + `N`-vs-atom
  -count check is the only guard; heterogeneous input is a documented misuse.
- **ETKDG unchanged**: the conformer pipeline must produce identical conformers
  and stage reports before/after the extraction (regression-tested).
- **FFI safety**: no panics cross PyO3; shape/size errors become `ValueError`,
  internal failures become `RuntimeError` with context.

## Test Criteria

### Unit Tests (`molrs-ff`)

1. **Minimum of a known quadratic**: a single harmonic bond
   `E = ½k(r−r₀)²` from two atoms relaxes to `|r−r₀| < 1e-6 Å`, `final_energy
   ≈ 0`, `converged = true`.
2. **fmax convergence semantics**: at the returned point `final_fmax <= fmax`
   when `converged`; when `max_steps` is hit first, `converged = false` and
   `n_steps == max_steps`.
3. **Idempotence at the minimum**: re-running `minimize` on an already-relaxed
   structure returns `n_steps` small (≤ 1 outer iteration) and unchanged coords.
4. **Trust region**: with a tiny `max_step`, no single step displaces any
   coordinate component by more than `max_step` (instrumented).
5. **Edge cases**: `n_atoms = 1` (no internal coords; converges immediately),
   empty coords, `coords.len() % 3 != 0` → `Err`.
6. **Batch size/shape errors**: `coords.len() != n_structs * n_atoms * 3` →
   `Err`; `n_structs = 0` → empty report vec.
7. **Batch equals serial**: `minimize_batch` over B identical copies yields B
   reports each equal (within 1e-10) to the single `minimize` result — proves
   no cross-structure state leakage and rayon determinism.

### Integration Tests

8. **ETKDG regression**: for {ethane, ethylene, benzene, butane, caffeine},
   `Conformer.generate()` after the extraction produces conformers and stage
   reports identical (coords within 1e-9) to a captured pre-refactor baseline.
9. **Python round-trip**: `MMFFTypifier().build(mol).minimize(coords)` lowers
   the MMFF94 energy monotonically to a stationary point; returned array shape
   is `(N, 3)`, input array is unmodified.
10. **Python batch**: `minimize_batch` on a `(B, N, 3)` stack returns
    `(B, N, 3)` and `B` `OptReport`s; mismatched `N` raises `ValueError`.

### Numerical Validation

11. **Gradient check** (reuses the `.claude/notes/testing.md` standard): finite-difference
    gradient (`h = 1e-5 Å`) of the energy the optimizer sees matches the
    analytical `-forces` to max-component error `< 1e-5 kcal/mol/Å` (guards the
    sign convention end-to-end).
12. **molpy parity**: for a fixed molecule + MMFF/harmonic potential and
    identical start coords, molrs `minimize` and molpy's numpy `LBFGS.run`
    (`molpy/optimize/lbfgs.py`) reach the same minimum energy within
    1e-3 kcal/mol and final coords RMSD < 1e-3 Å (after rigid alignment).
13. **RDKit parity (MMFF)**: relaxing an RDKit-embedded conformer with molrs
    `minimize` (MMFF94 `Potentials`) reaches a final energy within
    1e-2 kcal/mol of RDKit `MMFFOptimizeMolecule` on the same start geometry.

## Performance Requirements

- Single `minimize` step overhead (excluding the `eval` itself) is O(`memory`·
  `n`); must not introduce per-step allocations beyond the L-BFGS history (no
  regression vs the current ETKDG cleanup wall-clock on benzene/caffeine, ±20%).
- `minimize_batch` with rayon scales near-linearly with thread count for
  `B ≫ n_threads`: a `B = 64`, ~30-atom batch completes in
  `≤ 1.5 × (serial_time / n_threads)` on the dev machine (records a baseline;
  >20% regression fails).

## Migration & Compatibility

- **Additive** at the Python layer: new `Potentials.minimize` /
  `minimize_batch` / `OptReport`; nothing removed.
- **Internal refactor** in Rust: `minimize_lbfgs` moves crate
  (`molrs-conformer` → `molrs-ff`); `molrs-conformer` keeps a thin wrapper so no
  downstream Rust caller breaks. No public Rust API of `molrs-conformer` changes.
- New `rayon` feature on `molrs-ff` is default-on; consumers building with
  `--no-default-features` get the serial batch path.

## Out of Scope

- **True vectorized / SIMD batched kernels** (one fused `(B,N,3)` energy/force
  evaluation). Batch here is rayon-parallel independent minimizations only;
  vectorized kernels are a possible follow-up spec.
- **Heterogeneous batch** (mixed topologies / per-structure `Potentials`).
- **Constraints** (fixed atoms, frozen bonds, cell relaxation), additional
  algorithms (FIRE, CG exposure), and PBC-aware minimization — future specs.
- **molpy backend switch**: wiring molpy's `LBFGS` to call this Rust path is a
  separate molpy-side change; this spec only provides and validates the molrs
  API (parity test #12 is the contract molpy will later consume).
