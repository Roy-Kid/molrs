---
title: Combined Distribution Functions (CDF) — joint 2-D/3-D observable histograms
status: approved
created: 2026-06-26
---

# Combined Distribution Functions (CDF) — joint 2-D/3-D observable histograms

## Summary
Add **Combined Distribution Functions** — TRAVIS's single most-used analysis — to
`compute`. A CDF jointly histograms two (2-D) or three (3-D) `Observable`s from
link 01 *evaluated on the same selected tuples per frame*, producing a correlated
density such as RDF×ADF (distance vs. angle), distance×dihedral, or angle×angle.
This reveals structure that the marginal 1-D distributions cannot — e.g. that a
given coordination distance only occurs at a specific solvent orientation.

The CDF is a thin, generic composition over link 01: it pairs the per-sample
outputs of N observables into an N-tuple and bins them into an N-dimensional
`ndarray` histogram with per-axis ranges/bins, then normalizes. molrs has no joint
histogram today; `PMFT` (freud potential-of-mean-force-and-torque) is the closest
existing surface but is fixed to specific coordinate pairings and a −kT·ln(p)
output, not a general observable-pair density.

Library feature only — **no CLI**, no bindings, no plotting.

## Domain basis
- Combined Distribution Functions: Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**,
  51, 2007–2023; Brehm et al., *J. Chem. Phys.* **2020**, 152, 164105.
- A 2-D CDF is the joint probability density `p(x, y)` of two observables sampled
  on the same atom tuples; the marginals `∫p dy` and `∫p dx` recover the link-01
  1-D distributions (this is the consistency contract).
- Optional per-axis weighting mirrors link 01 (e.g. an ADF axis may carry the
  `sin θ` solid-angle correction); optional free-energy view `−kT·ln p` (kT in
  molrs units, `kB = 1.987204e-3 kcal/(mol·K)`) is offered as a derived field, not
  the primary output (keeps CDF a *density*, unlike PMFT).

## Implementation constraint — port from TRAVIS, do not reinvent

The reference implementation for this entire `travis-parity` chain is **TRAVIS**
(TRajectory Analyzer and VISualizer). Every analysis routine here MUST be a
**port of TRAVIS's actual source** — its data layout, binning, normalization,
weighting, and edge-case handling translated into Rust — **not** a from-scratch
reimplementation of the equations. Where TRAVIS already implements the analysis,
free improvisation is not permitted.

- **Source (download + extract before implementing):**
  `http://www.travis-analyzer.de/files/travis-src-220729.tar.gz`
- Identify the specific TRAVIS source file(s)/function(s) that implement this
  spec's analysis, and **cite them** — in the porting code's rustdoc/comments and
  in the commit message — so each ported routine is traceable to its origin.
- Keep numerical behavior **faithful to TRAVIS**. molrs idioms (ndarray, `SimBox`
  minimum-image, the `Compute`/`Observable` traits) may restructure the code, but
  must not change the result; document any deliberate deviation and why.
- Parity tests should check against TRAVIS output/conventions, not only against
  re-derived analytic values.

## Design
New submodule `compute/distribution/combined.rs` (same `compute` feature, same
module family as link 01 — CDF is the joint case of the 1-D DF).

New symbols:
- `CombinedDistribution` — implements `Compute`; holds 2 or 3 boxed `Observable`s
  plus per-axis `AxisSpec { bins, min, max, sin_weight }`. Per frame it zips the
  observables' per-tuple samples (which MUST be the same length — same selection
  arity count) into N-tuples and accumulates an N-D histogram.
- `CombinedDistributionResult` — N-D `ndarray` of counts + normalized joint
  density + per-axis edges; helper `marginal(axis)` returns the 1-D `DistributionResult`
  and `free_energy(temperature)` returns the `−kT·ln p` field (with an explicit
  floor for empty bins, documented).
- Reuses the link-01 `histogram1d` edge math per axis; the N-D accumulation is a
  flat-index `ndarray` (row-major), no new crate.

Constraints: the N observables must produce one sample per shared tuple in the
same order (validated — mismatched lengths are a typed `ComputeError`, never a
silent zip-truncation). 2-D and 3-D only (TRAVIS's practical range); higher
arities are out of scope.

Layer discipline: `compute` → `core` only; WASM-clean.

## Files to create or modify
- `molrs/src/compute/distribution/combined.rs` (new) — `CombinedDistribution` + result.
- `molrs/src/compute/distribution/mod.rs` (modify) — re-export.
- `molrs/src/compute/mod.rs` (modify) — re-export `CombinedDistribution`.
- `molrs/tests/compute/combined_distribution.rs` (new) — marginal-consistency + correlation tests.

## Tasks
- [ ] Write failing test: a 2-D distance×angle CDF whose marginals (∫ over each axis) equal the link-01 1-D distance DF and ADF within 1e-6.
- [ ] Write failing test: a constructed correlated dataset (angle determined by distance) shows density only on the diagonal band; an independent dataset shows a separable product `p(x,y)=p(x)p(y)`.
- [ ] Write failing test: mismatched observable sample counts → typed ComputeError (no silent truncation).
- [ ] Implement `CombinedDistribution` (2-D + 3-D) over an N-D ndarray histogram reusing link-01 axis math.
- [ ] Implement `marginal()` and `free_energy(T)` with documented empty-bin floor.
- [ ] Rustdoc with the joint-density definition + marginal-consistency contract + citations; run fmt/clippy/test.

## Testing strategy
- **Marginal consistency** (`scientific`): the headline contract — summing the joint
  density over one axis reproduces the corresponding link-01 1-D distribution.
- **Correlation detection**: synthetic correlated vs. independent samples produce
  diagonal-band vs. separable densities.
- **Normalization**: ∬ density = 1 (2-D) / ∭ = 1 (3-D) within 1e-6.
- **Validation**: arity/length mismatch → `ComputeError`; empty input → empty result.
- **Free energy**: `−kT·ln p` finite on populated bins, floored (not −∞/NaN) on empty.

## Third-party library analysis
- **N-D histogram**: evaluated `ndhistogram` (genuine N-D axes, MIT/Apache-2.0).
  **Recommend native `ndarray`** N-D arrays with a shared flat-index accumulator —
  consistent with link 01 and the existing RDF/`density`/`PMFT` histograms (PMFT
  already maintains multi-D histograms on ndarray), no new dependency, WASM-clean.
  `ndhistogram` is noted as a viable alternative if axis bookkeeping grows.
- No other third-party code required (pure composition over link 01).
- *(Versions to confirm at implementation time.)*

## Out of scope
- 4-D+ joint histograms; on-the-fly KDE smoothing.
- The `−kT·ln p` PMFT semantics as a *primary* output (that is freud `PMFT`'s job;
  CDF stays a density with free energy as a derived convenience).
- Reference-frame 3-D spatial density — that is link 03 (SDF).
- CLI, bindings, plotting.
