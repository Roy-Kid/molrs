# ff-special-bonds-nblist — one generic potential path: neighbour list is the consumer's, ForceField owns special_bonds weights, one per-atom pair-kernel convention, MMFF demoted to a param loader

Status: **draft**
Scope: molrs-ff (potential, forcefield, typifier/mmff), molrs-python; reference consumer: molpack `LBFGSRelaxer`

## Summary

The force-field → potentials path carries **two pair conventions** and **one
misplaced responsibility**:

- Generic pair ctors (`lj/cut`, `coul/cut`) demand a caller-supplied `pairs`
  block of `atomi/atomj/type` (a **pre-combined per-pair type**), do **not**
  combine or scale, and silently **skip** when the block is absent
  (`potential/mod.rs:267-271`). So every non-MMFF force field (GAFF/LAMMPS, OPLS)
  needs its exclusions/1-4/combining pre-baked by the caller.
- MMFF's `mmff_vdw`/`mmff_ele` read `atomi/atomj/is_14` and combine **per-atom** +
  scale 1-4 internally, and MMFF builds its `pairs` block in its **own**
  `typifier/mmff/frame_builder.rs` — a parallel machine ("另起炉灶").

The unification: **one** pair convention, **one** neighbour-list source, **one**
generic path; MMFF becomes only parameter loading/definition.

**Key principle (user): the neighbour list is NOT a ForceField concern.**
`ForceField` = params + functional forms + special-bonds **weights**. The `pairs`
block (which pairs interact + the 1-4 flag) is built by the **consumer that owns
the nblist** — the optimizer/integrator. For geometry relaxation the LBFGS
optimizer builds it **once** (topology fixed); an MD integrator would rebuild it
periodically. `to_potentials(&frame)` stays as-is: it consumes a frame that
**already** has `pairs`; it never builds them.

## Domain basis

- `ForceField::to_potentials(&frame) -> Potentials` (`potential/mod.rs:252`): loops
  styles, maps `pair → "pairs"` block, **skips** a pair style if the block is absent
  (267-271). API and skip behaviour are unchanged by this spec.
- Exclusion logic (`typifier/mmff/frame_builder.rs:279-318`): `excluded_12` = bond
  ends, `excluded_13` = angle (i,k), `set_14` = dihedral (i,l); enumerate `i<j`
  minus 12/13, flag `is_14`. **Already lifted** to a generic free fn
  `intramolecular_pairs(frame) -> Block` (`atomi/atomj/is_14`) in `potential/mod.rs`
  (DONE this effort, compiles).
- Pair-ctor signature: `fn(style: &Params, types: &[(&str,&Params)], frame: &Frame) -> Result<Box<dyn Potential>>`.

## Design

1. **`ForceField` gains `special_bonds`** — per nonbonded kind (lj, coul) the
   `[1-2,1-3,1-4]` scale weights. Amber: lj `0/0/0.5`, coul `0/0/0.8333`. Fold in
   the `lj14scale`/`coulomb14scale` currently parked on the pair styles; the format
   readers (LAMMPS, OPLS) set them.
2. **The nblist is the consumer's.** `intramolecular_pairs(frame) -> Block` is the
   blessed free utility; consumers (the LBFGS relaxer; MD) call it and insert the
   `pairs` block **before** `to_potentials`. `ForceField`/`to_potentials` never
   build it. (DONE: the utility exists.)
3. **One pair-block convention = `atomi/atomj/is_14`** (MMFF's). Generic pair ctors
   (`lj/cut`, `coul/cut`, …) change to: read **per-atom** params (atoms-block `type`
   + the pair style's per-type params), combine via the mixing rule
   (Lorentz-Berthelot: `ε=√(εᵢεⱼ)`, `σ=(σᵢ+σⱼ)/2`), and apply the ff's
   special-bonds 1-4 weight when `is_14`. Drop the pre-combined per-pair `type`
   requirement. Update their unit tests + pairs-block fixtures.
4. **MMFF demoted to a param loader.** Delete `frame_builder.rs`'s private
   frame/pairs assembly; `MMFFTypifier` only typifies (atom_typing) → defines a
   `ForceField`; the topology Frame + nblist come from the general path (consumer
   builds `pairs`). `mmff_*` kernels stay registered and are consumed via generic
   `to_potentials` — no MMFF-specific build path ("严禁另起炉灶").
5. **`to_potentials` public API + skip-if-no-pairs behaviour unchanged.**

## Reference consumer (molpack)

`molpack.LBFGSRelaxer(ff)` — **force field only** (was `new(Arc<dyn Potential>)`). It
**lazy-compiles** the first time it is handed the molecule's frame during packing:
build the nblist via `intramolecular_pairs`, insert `pairs`, `ff.to_potentials` →
store the Potential, minimize. This requires the **`Relaxer::spawn` trait change** —
`spawn` must receive the molecule's Frame/topology (today `spawn(&ref_coords)` only).
The PyO3 binding receives a `molpy.ForceField`/`molrs.ForceField` over the interop
bridge (`forcefield_from_py`, sibling spec `interop-ffi-bridge`) — **no
`from_lammps_ff`**; reading the LAMMPS input file is `molrs::io::forcefield`'s job
(readers move out of `ff/forcefield/readers` into `io/forcefield`), proxied as
`molpy.io.forcefield`.

## Files (molrs)

- `ff/forcefield/mod.rs` — `special_bonds` field + accessors.
- `io/forcefield/{lammps,opls}.rs` (moved from `ff/forcefield/readers/`) — set
  `special_bonds`; reader relocation is part of this work (or a 3rd sibling spec).
- `ff/potential/pair/{lj_cut,coul_cut,…}.rs` — per-atom combine + `is_14` scaling;
  ctors read the atoms block + the ff's special_bonds. Tests updated.
- `ff/potential/mod.rs` — `intramolecular_pairs` (DONE); keep utility ↔ pair-ctor
  `is_14` block format in agreement.
- `typifier/mmff/` — delete `frame_builder.rs`'s frame/pairs assembly;
  `MMFFTypifier::build` = typify → `ForceField`; consumer builds the nblist.

## Testing

- A GAFF/LAMMPS frame + `intramolecular_pairs` + special_bonds → `to_potentials`
  builds LJ + Coulomb with correct 1-4 scaling; **energy parity** vs a reference
  (LAMMPS single-point, or molpy) within tolerance.
- An MMFF molecule via the generic path (typify → ff → consumer-built pairs →
  `to_potentials`) reproduces the **old `frame_builder` energy** (pin parity, then
  delete `frame_builder`).
- Generic `lj/cut` energy with per-atom combining matches a hand-combined
  reference; flagged 1-4 pairs are scaled by the ff's lj/coul 1-4 weight.
- molpack reference: relaxation-assisted pack **with non-bonded** runs end to end.

## Out of scope

- Cross-extension data-object interop (the `molrs-ffi` capsule bridge that lets a
  `molrs.ForceField` reach the molpack relaxer) — sibling spec `interop-ffi-bridge`.
  The two compose.
- uint index canonicalization — codified as the data contract in
  `interop-ffi-bridge`; the amber reader emits uint (DONE), the other molpy readers
  are a mechanical sweep.
