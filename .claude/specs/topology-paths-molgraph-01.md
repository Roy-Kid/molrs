# Spec: Graph-theoretic topology in molgraph; Atomistic.generate_topology

> **Implementation note (2026-06-10).** Per the directive "use petgraph like
> molpy, don't reinvent": instead of a new hand-rolled `MolGraph::paths_of_length`
> DFS, `Atomistic::generate_topology` **reuses the existing petgraph-backed
> `molrs-core/src/topology.rs` `Topology`** (`from_edges(...).angles()/.dihedrals()`).
> The graph-theory lives in `topology.rs` (petgraph); `Atomistic` is the domain
> leaf that builds it from bonds and names the result. The "molgraph
> paths_of_length primitive" sections below are superseded by this reuse.

## Summary
Move angle/dihedral generation out of the standalone `topology.rs` `Topology`
struct and into `molgraph` as a **generic graph-theory primitive** (enumerate
simple paths of length *k* over a relation kind's edges). Add a domain leaf
`Atomistic::generate_topology()` that materializes `angle` (length-2 path) and
`dihedral` (length-3 path) relations from bonds, expose it via PyO3, and retire
molpy's Python `core/topology.py` enumeration (molpy inherits the molrs method).

## Motivation
Angles and dihedrals are paths in the bond graph — an angle is a 2-edge path
(3 nodes `i-j-k`), a dihedral a 3-edge path (4 nodes `i-j-k-l`). That is pure
graph theory and belongs in `molgraph` (`molrs-core/src/molgraph.rs`), which
already *owns* the graph (interned-kind n-ary relations over a node SlotMap).
Today the logic lives in a parallel `molrs-core/src/topology.rs` `Topology`
struct that rebuilds its own adjacency, and molpy re-implements it again in
Python (`core/topology.py`, used by `Atomistic.get_topo`). Three copies of one
graph algorithm. Per *core sinks to molrs* (`feedback-core-sinks-to-molrs`),
there must be one implementation, in molgraph, inherited everywhere.

## Scope
- **Crates affected**: `molrs-core` (molgraph primitive + Atomistic method;
  absorb `topology.rs`), `molrs-python` (PyO3), downstream molpy (retire Python
  duplicate — tracked, not done here).
- **Traits/data**: new `MolGraph` method(s) for path enumeration; new
  `Atomistic::generate_topology`. `topology.rs` `Topology::{angles,dihedrals}`
  reduced to thin callers of the molgraph primitive (or removed).
- **Feature flags**: none.

## Technical Design

### API Surface

```rust
// molrs-core/src/molgraph.rs — generic, domain-agnostic (no "angle"/"dihedral").
impl MolGraph {
    /// All simple paths of exactly `k` edges over the 2-ary relation `kind`,
    /// returned as `k+1`-length node-id sequences. Undirected: each path is
    /// emitted once in a canonical orientation (first endpoint < last by node
    /// index) to avoid the reverse duplicate. Excludes paths that revisit a
    /// node (no i==k for k=2; no i==l for k=3).
    pub fn paths_of_length(&self, kind: KindId, k: usize) -> Vec<Vec<NodeId>>;
}
```

```rust
// molrs-core/src/atomistic.rs — domain leaf; names the paths.
impl Atomistic {
    /// Perceive angle (2-edge path) and dihedral (3-edge path) relations from
    /// the existing `bonds` relation and add them as `angle` / `dihedral`
    /// relation kinds. Idempotent: existing angle/dihedral rows are preserved
    /// unless `clear_existing`. Dedup: angles keyed `(i,j,k)` with `i < k`,
    /// dihedrals `(i,j,k,l)` with `j < k`.
    pub fn generate_topology(
        &mut self,
        gen_angle: bool,
        gen_dihedral: bool,
        clear_existing: bool,
    ) -> Result<(usize, usize), MolRsError>; // (n_angles_added, n_dihedrals_added)
}
```

PyO3: expose `Atomistic.generate_topology(gen_angle=True, gen_dihedral=True,
clear_existing=False)`. Because `molpy.Atomistic` subclasses
`molrs.Atomistic`, it inherits the method with no molpy code.

### Algorithm
- Build adjacency once from the `bonds` relation's 2-ary tuples.
- `paths_of_length(kind, 2)`: for each node `j`, every unordered neighbor pair
  `(i, k)`, `i < k` → `[i, j, k]`. (= angles)
- `paths_of_length(kind, 3)`: for each edge `(j, k)`, every `i ∈ adj(j)\{k}`
  and `l ∈ adj(k)\{j}`, `i != l`, canonicalized so `j < k` → `[i, j, k, l]`.
  (= dihedrals)
- Complexity O(Σ deg²) for angles, O(Σ deg(i)·deg(k)) over edges for dihedrals
  — the standard bounded enumeration.

### Integration Points
- `topology.rs` `Topology::{angles,dihedrals}` become thin wrappers over
  `MolGraph::paths_of_length` (or are deleted if no other caller needs the
  standalone struct). `impropers()` (center + 3 neighbors = a star, NOT a path)
  stays where it is or gets its own primitive — out of scope here.
- `Atomistic::generate_topology` uses `add_angle` / `add_dihedral` (exist).
- molpy: `Atomistic.get_topo(gen_angle, gen_dihe)` rewired to call the inherited
  molrs method; delete the Python angle/dihedral enumeration in
  `core/topology.py` (separate molpy PR; this spec only lands the molrs side +
  the binding).

## Constraints & Invariants
- molgraph stays **domain-agnostic**: no `angle`/`dihedral`/chemistry vocabulary
  in `paths_of_length`. Naming lives only in `Atomistic`.
- Deterministic ordering (sorted) so generated rows are reproducible.
- No duplicate relations on re-run (dedup against existing).
- `F`/`I`/`U` conventions unchanged.

## Test Criteria
### Unit Tests (molgraph)
1. `paths_of_length(bonds, 2)` on a known graph (ethane skeleton, branched,
   ring) returns the exact expected triplets, each once, `i < k`.
2. `paths_of_length(bonds, 3)` returns expected quartets, `j < k`, no `i==l`,
   no reverse duplicates.
3. Empty / single-edge / disconnected graphs → correct (often empty) results,
   no panic.

### Integration Tests (Atomistic)
4. Ethane (C2H6, 7 bonds): `generate_topology` adds **12 angles, 9 dihedrals**
   (matches the counts molpy's old enumeration produced — pinned).
5. Benzene / cyclohexane (rings): angle/dihedral counts match a reference;
   ring closure handled (no missing or doubled paths).
6. Idempotence: second call with `clear_existing=false` adds 0; with `true`
   regenerates the same set.

### Python
7. `molpy.Atomistic` (subclass) calls the inherited `generate_topology`;
   resulting angle/dihedral counts match `get_topo(gen_angle=True,
   gen_dihe=True)`'s historical output for ethane/butane.

## Migration & Compatibility
- Additive in molrs (new method); `topology.rs` internal refactor preserves any
  existing public behavior (or removes it if unused — check callers first).
- molpy `get_topo` rewiring + Python `topology.py` deletion is a follow-up molpy
  change gated on the wheel rebuild.

## Out of Scope
- Impropers (star enumeration, not a path) — separate primitive later.
- Deleting molpy's `core/topology.py` (molpy-side follow-up).
- Any FF / potential / optimizer work (specs `ff-potentials-oop-01`,
  `ff-format-readers-01`).
