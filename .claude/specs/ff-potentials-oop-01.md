# Spec: molrs-ff OOP rewrite — Style.to_potential / ForceField.to_potentials / Potentials.calc_*(frame) / LBFGS class

## Summary
Rewrite the molrs-ff force-field/optimizer surface to mirror molpy's OOP
connection exactly: per-`Style` `to_potential()` builds a parameter-only
`Potential` (a class), `ForceField::to_potentials()` collects them into a
molecule-independent `Potentials`, and `Potentials::calc_energy(frame)` /
`calc_forces(frame)` take the frame **at call time**. The geometry optimizer
becomes an `LBFGS` **class** (`new(potential, cfg).run(frame, …)`). Delete the
rejected free-function surface: `ForceField::compile(frame)`,
`Potential::eval(coords)`, `molrs_ff::{minimize, minimize_batch}`, the free
`*_ctor` kernel constructors, and the `KernelRegistry` free-fn map.

## Motivation
The shipped surface (`compile(frame) -> Potentials`, `Potentials::eval(coords)`,
free `minimize`) was rejected: `compile` (pre-binding a force field to one
molecule's atom indices) is a meaningless step molpy never performs, and the
optimizer/kernels must be **classes**, not free functions
(`feedback-core-sinks-to-molrs`, `project-molrs-core-sink-roadmap` tracks 3+5).
molpy's model is the target (`molpy/core/forcefield.py::to_potentials`,
`molpy/potential/base.py::Potentials.calc_energy/calc_forces`,
`molpy/potential/utils.py::calc_energy_from_frame`,
`molpy/optimize/{base.py,lbfgs.py}` `Optimizer`/`LBFGS`):

- `Style.to_potential()` → a `Potential` holding only its type-indexed params.
- `ForceField.to_potentials()` → `Potentials` (no frame; molecule-independent).
- `Potentials.calc_energy(frame)` / `calc_forces(frame)` → frame supplies
  topology + per-element type labels + coords at call time.
- `LBFGS(potential, *, maxstep, memory, damping).run(structure, fmax, steps)`.

## Scope
- **Crates affected**: `molrs-ff` (Potential trait, every kernel, Potentials,
  ForceField, KernelRegistry, the optimizer module), `molrs-python` (bindings),
  `molrs-conformer` (ETKDG consumes the L-BFGS engine — must keep working).
- **Traits**: `Potential` signature changes from `eval(&[F]) -> (F, Vec<F>)` to
  `calc_energy(&Frame) -> F` + `calc_forces(&Frame) -> Vec<F>`; new
  `Style::to_potential(&self) -> Result<Box<dyn Potential>, String>`.
- **Data structures**: kernels store type→param tables (not pre-resolved atom
  indices); new `LBFGS` struct; `MinimizeOptions`→`LbfgsConfig`,
  `OptReport` kept.
- **Feature flags**: `rayon` (already present) for batch.

## Technical Design

### Potential trait (frame-driven)
```rust
pub trait Potential: Send + Sync {
    /// Total energy of this term for the molecule described by `frame`
    /// (topology + per-element `type` labels + atom coords). Resolves type
    /// labels against the kernel's own param table.
    fn calc_energy(&self, frame: &Frame) -> Result<F, String>;
    /// Forces (= -gradient), length `3 * n_atoms`, atom order = frame "atoms".
    fn calc_forces(&self, frame: &Frame) -> Result<Vec<F>, String>;
    /// One pass for both (default fans out; kernels may override for speed).
    fn calc_energy_forces(&self, frame: &Frame) -> Result<(F, Vec<F>), String> {
        Ok((self.calc_energy(frame)?, self.calc_forces(frame)?))
    }
    fn category(&self) -> &str; // "bond" | "angle" | "dihedral" | "pair" | ...
}
```
Each kernel (e.g. `DihedralOPLS`, `PairCoulCut`, `BondHarmonic`) holds a
`HashMap<String, Params>` (type label → params) and, at `calc_*`, reads its
block from the frame (`frame["dihedrals"]` etc.), pulls index + `type` columns,
resolves params, and accumulates — exactly molpy's `calc_energy_from_frame`
pattern, but the dispatch lives on the kernel (it knows its block/category).

### Style.to_potential (the FF↔Potential 衔接)
```rust
impl Style {
    /// Build the parameter-only Potential for this style from its Types.
    /// Returns None for styles with no kernel (e.g. AtomStyle).
    pub fn to_potential(&self) -> Result<Option<Box<dyn Potential>>, String>;
}
```
Dispatch on `(category, name)` — the role the `KernelRegistry` free-fn map plays
today — but as a method that constructs the kernel **class** from the style's
types. The registry becomes a `(category, name) -> fn(&Style)->Box<dyn Potential>`
or is folded into `Style::to_potential` via a match; no standalone free ctors.

### ForceField.to_potentials / Potentials
```rust
impl ForceField {
    /// Collect a Potential per style. No frame — molecule-independent.
    pub fn to_potentials(&self) -> Result<Potentials, String>;
}
impl Potentials {
    pub fn calc_energy(&self, frame: &Frame) -> Result<F, String>;        // Σ terms
    pub fn calc_forces(&self, frame: &Frame) -> Result<Vec<F>, String>;   // Σ terms
    pub fn calc_energy_forces(&self, frame: &Frame) -> Result<(F, Vec<F>), String>;
}
```

### LBFGS optimizer class
```rust
pub struct LbfgsConfig { pub fmax: F, pub max_steps: usize, pub maxstep: F, pub memory: usize }
// Defaults: fmax 0.05, max_steps 500, maxstep 0.2 Å, memory 8.

pub struct LBFGS<'p> { potential: &'p dyn Potential, cfg: LbfgsConfig, /* state */ }

impl<'p> LBFGS<'p> {
    pub fn new(potential: &'p dyn Potential, cfg: LbfgsConfig) -> Self;
    /// One step on a frame's coords (updated in place); returns (energy, fmax).
    pub fn step(&mut self, frame: &mut Frame) -> Result<(F, F), String>;
    /// Run to convergence; frame coords updated in place.
    pub fn run(&mut self, frame: &mut Frame) -> Result<OptReport, String>;
    /// Homogeneous batch: many frames sharing one Potential, rayon-parallel.
    pub fn run_batch(&self, frames: &mut [Frame]) -> Result<Vec<OptReport>, String>;
}
```
Mirrors molpy `Optimizer`/`LBFGS` (potential in ctor; `run(structure)`;
`step`). `OptReport { converged, n_steps, final_energy, final_fmax }` kept.

### The L-BFGS numeric engine is preserved
The two-loop recursion + backtracking line search (this session's validated
`optimize/lbfgs.rs` core, on a flat `&mut [F]` + an `(energy, forces)` closure)
stays as a **private engine**. `LBFGS::run` adapts: read coords from the frame
into a flat buffer, drive the engine with a closure that writes coords back into
a scratch frame and calls `potential.calc_forces`, write the result back.
**ETKDG keeps using the flat-coords engine directly** (its
`minimize_lbfgs_rms(coords, eval_closure)` path is unchanged) — so conformer
generation is unaffected. Only the *public* optimizer surface becomes the class.

### PyO3
- Add: `molrs.LBFGS(potentials, *, fmax=, max_steps=, maxstep=, memory=)` with
  `.run(frame) -> (frame_or_coords, OptReport)`, `.run_batch(frames)`,
  `.step(frame)`; `ForceField.to_potentials()`; `Potentials.calc_energy(frame)`
  / `calc_forces(frame)`.
- Remove: `ForceField.compile`, `Potentials.eval`, `Potentials.minimize`,
  `Potentials.minimize_batch`, `build_mmff_potentials` is reworked to return a
  `Potentials` via `to_potentials` (MMFFTypifier path).

## Constraints & Invariants
- `forces = -gradient`; per-term Newton's third law; `F = f64`.
- A `Potentials` is molecule-independent and reusable across frames with the
  same type vocabulary.
- Kernel math is **unchanged** — only how params/topology reach the kernel
  changes. All existing energy/gradient values must be preserved (the numerical
  tests below are the guard).
- ETKDG output bit-for-bit unchanged (it uses the private engine, not the class).

## Test Criteria
### Unit / Integration (molrs-ff)
1. **Kernel math preserved**: the existing finite-difference gradient tests for
   `DihedralOPLS`, `PairCoulCut`, harmonic bond/angle still pass, rewritten to
   the `calc_energy/calc_forces(frame)` API (build a small typed frame, assert
   energy + fd-force within 1e-5).
2. **Style.to_potential**: each style produces a kernel whose `calc_energy` on a
   known frame matches the pre-rewrite value.
3. **to_potentials reuse**: one `Potentials` evaluated on two different frames
   (same types, different coords/sizes) gives correct per-frame energies — proves
   molecule-independence (the property `compile` lacked).
4. **LBFGS class**: `LBFGS::new(pot, cfg).run(frame)` relaxes a harmonic system
   to `fmax < cfg.fmax`, `converged`; `run_batch` over a homogeneous set matches
   per-frame `run`; serial == rayon.
5. **Real MMFF**: ethane relaxes via `LBFGS` over MMFF `Potentials` (parity with
   this session's `lbfgs_minimize_relaxes_mmff_ethane`, new API).

### Regression
6. molrs-conformer suite unchanged (ETKDG still green; engine untouched).

## Migration & Compatibility
- **Breaking** within molrs-ff and its Python surface (intended): `compile`,
  `eval`, free `minimize`/`minimize_batch`, free `*_ctor` removed. No external
  consumers besides molpy (which is being realigned).
- Sequenced after `topology-paths-molgraph-01` is not required, but pairs with
  `ff-format-readers-01` (which produces the `ForceField` this consumes).

## Out of Scope
- FF-format readers (`ff-format-readers-01`).
- Typifier sink (B-line).
- Vectorized batched kernels (batch stays rayon-over-frames).
