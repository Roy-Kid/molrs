# ff-perinstance-params — force fields assign **per-instance** parameters onto the structure; kernels evaluate from per-instance parameter columns (generic, multi-force-field); complete MMFF on this model and retire the standalone `MmffForceField` evaluator

Status: **draft** (2026-06-14)
Scope: molrs-ff (`potential`, `typifier/mmff`, `mmff/energy`→relocate, `mmff/{topo,params}`),
molrs-conformer (ETKDG decoupling), molrs-python.
Relationship: **consolidates and replaces the retired `ff-mmff-unify-generic-path`** (removed
2026-06-14 — its design used a "type-label → shared parameter table" kernel that cannot express MMFF's
per-instance parameters; see §Why). The done parts of that effort (typifier reuses the RDKit-validated
front-end for types + MMFF charges; the pair-list builder exists; `typify` is pairs-free and `build()`
owns the pair list) are recorded in §Current state and retained; its open criteria (energy parity,
ETKDG decouple, Python pair-list export, delete the standalone evaluator) are carried here. Also
**supersedes [`ff-special-bonds-nblist`](ff-special-bonds-nblist.md) `ac-005`** (MMFF demotion +
parity + delete — now owned here); that spec's other criteria (`special_bonds` + per-atom
`lj/cut`/`coul/cut` for the table force fields) stay independent and out of scope (§Out of scope).

## §Why the previous mechanism is wrong

Today MMFF runs through the same kernel mechanism as the table force fields:
each interaction carries a **string type-label** (e.g. a bond `"0_1_5"`), and the kernel constructor
looks that label up in the force field's **shared parameter table** (`type_map`, parsed from
`molrs/data/mmff94.xml`) to fetch the numbers. See `bond_harmonic_ctor` (`ff/potential/bond/harmonic.rs:101-113`),
`dihedral_opls_ctor` (`ff/potential/dihedral/opls.rs:95-98`), and the MMFF kernels
(`ff/potential/{bond,angle,dihedral,improper,pair}/mmff.rs`) — all identical "label → shared table".

This works for **table force fields** (GAFF / OPLS / LAMMPS): a finite typed parameter set, one row per
type combination. It **cannot** express MMFF94, whose parameters are genuinely **per-instance**:

1. **Equivalence-level fallback** — when an exact `(type_i, type_j, …)` row is absent, MMFF degrades the
   atom types through up to four/five equivalence levels (`mmff_def(t).eq_level`) until a row matches
   (`MmffForceField`'s `angle_lookup`/`oop_lookup`/`torsion_lookup`, `ff/mmff/energy/params.rs:245-320`).
   *Resolvable in the label model* by having the typifier emit the matched canonical label — already
   done for **oop** (`classify::resolve_oop_label`) and **angle** (`classify::resolve_angle_label`).
2. **Empirical rules** — when even equivalence fallback finds no row, MMFF **computes** the parameters
   from atomic constants (covalent radii, electronegativity, Herschbach–Laurie, ring size, …):
   `bond_empirical` (`params.rs:372`), `angle_empirical` (`:413`), `torsion_empirical` (~146 lines),
   default stretch-bend (`mmff_dfsb`). These values are **not in any fixed table**, so a kernel that
   looks up a shared table simply cannot find them. This is the hard blocker: benzene fails
   `mmff_stbn: unknown '1_37_37_37'`; ethylene's energy is off by 8.1 kcal/mol. The previous spec's
   `ac-004` parity is unreachable under "label → shared table".

Observed: under the generic path today, **e_ethane matches RDKit to 2.3e-5 kcal/mol** (gradient FD
7.5e-8) — proving the path is sound where every parameter is a table/equivalence hit; only the
empirical-rule molecules (aromatics / sp2) diverge.

## §Design — separate parameter **assignment** from energy **evaluation**

The general, force-field-agnostic model (chosen for extensibility to future force fields — GAFF,
CHARMM, AMBER, ML potentials — none of which should be forced into a single shared-table shape):

- **A force field is a *parameter assigner*.** Given a molecular structure it produces a **fully
  parameterized structure**: every bond/angle/dihedral/improper carries its **final numeric
  parameters**. *How* it resolves them is force-field-private — MMFF runs typing + equivalence fallback
  + empirical rules; a table force field looks each interaction up in its table; a future force field
  may infer them. For MMFF the assigner is the **`Typifier`**, operating on `Atomistic`.
- **A kernel is a pure evaluator of one functional form.** It reads **per-instance numeric parameters**
  (+ flat coords) and returns `(energy, forces)`. It makes **no** assumption that parameters come from
  a shared table and contains **no** force-field-specific resolution. Reusable across any force field
  that uses that functional form; a new functional form is a new kernel.
- **The interface between them is the parameterized structure** (per-instance parameter columns on the
  `Frame`). `Atomistic::to_frame()` projects the assigner's per-relation properties into `Frame`
  columns; `ForceField::to_potentials(frame)` wires each block to its kernel, which reads the columns.

This is already the shape of two existing pieces — confirming it fits the codebase, not fighting it:
- the **pair charge** is a per-atom `charge` column the Coulomb kernel reads (`pair/mmff.rs`,
  `pair/coul_cut.rs`), not a type lookup;
- **stretch-bend** `r0_ij`/`r0_kj`/`theta0` are per-angle columns baked by the typifier
  (`merge_stbn_r0`, `ff/typifier/mmff/mod.rs:128`) and read by `mmff_stbn_ctor`
  (`ff/potential/angle/mmff.rs:164-172`). Generalize this to every MMFF term.

Note: this does **not** force the table force fields to change. Their kernels (`harmonic`, `class2`,
`opls`, …) already build per-instance arrays internally from a label lookup — functionally a pure
evaluator with an in-ctor table read. They keep working unchanged. The contract is *"a kernel evaluates
from per-instance parameters"*; a table FF resolves them by lookup, MMFF by assignment. Only MMFF's
kernels move their resolution **out** (into the assigner) because MMFF's resolution is not a finite table.

## §Reuse, don't re-derive (numerical safety)

`MmffForceField` (`ff/mmff/energy/`) is the **only RDKit-validated** MMFF parameter resolution
(`tests/ff/mmff/energy.rs`: benzene / caffeine / … ≤1e-3 kcal/mol). It is *already* an
assigner+evaluator fused: `Topo::build(mol)` + `set_mmff_aromaticity` + `props` →
`enumerate_{bonds,angles,stretch_bends,oops,torsions,nonbonded}` (`energy/mod.rs:95-100`) call the
resolution in `energy/params.rs` (`bond_params`/`angle_params`/`torsion`/`oop_koop`/stbn + empirical)
to produce per-term structs, which then evaluate.

The work is to **split** this proven machine, not re-implement it: the **resolution** (`params.rs`
+ the `enumerate_*` driving it) becomes the assigner (reused by the typifier); the **term math**
(`bond.rs`/`angle.rs`/`oop.rs`/… `energy_grad`) is what the generic `mmff_*` kernels already do.
`Topo` (`ff/mmff/topo.rs`) and `set_mmff_aromaticity` (`ff/mmff/aromaticity.rs`) are already kept
modules; `params.rs` is relocated out of `energy/` into a kept location (e.g. `ff/mmff/params.rs`).
This keeps MMFF numbers bit-for-bit and lets `ac-003` parity hold by construction.

## §Current state (working tree, all green: 1446 lib/integration tests, clippy `-D warnings` clean)

DONE already (from the earlier MMFF-unify effort + this session):
- `typify` reuses `MmffMolProperties` (validated types + MMFF charges); `atom_typing.rs` deleted;
  Gasteiger dropped; `typify` is pairs-free; `build()` inserts the pair list then `to_potentials`.
- `nonbonded_pairs` logic exists as `pub fn intramolecular_pairs` (`ff/potential/mod.rs:46`) — **rename
  to `nonbonded_pairs` pending** (ac-006).
- OOP enumeration rewritten MMFF-correct: only degree-3 trigonal centres, three Wilson permutations
  sharing one `koop`, centre in the `atomj` column (`frame_builder.rs`); `classify::resolve_oop_label`
  + `resolve_angle_label` do the equivalence-fallback canonical key.
- `Style::to_potential` skips a bonded/pair style whose block is absent or has zero rows
  (`ff/potential/mod.rs`) — methane (no dihedrals/impropers/non-bonded pairs) now builds.
- Examples fixed (`typify_litfsi`/`typify_molecule`); the two `expect_err` OOP tests flipped to assert
  finite energy/forces; ac-003 parity test added — **e_ethane passes (2.3e-5)**.
- **bond term migrated to the new model** (proof-of-pattern): the typifier bakes per-bond `kb`/`r0`
  via the reused `bond_params` resolution (table → equivalence → empirical), and `mmff_bond_ctor`
  reads the `kb`/`r0` columns (no type-label lookup). `mmff::{topo, energy::params}` exposed
  `pub(crate)` for reuse (physical relocation of `params.rs` out of `energy/` deferred to the
  monolith-deletion step, ac-006). ethane parity unchanged (2.3e-5); ff suite green (67 tests).
  **Remaining terms (angle+stbn, torsion, oop) follow the same recipe.**

## §Tasks

1. **Relocate the MMFF parameter resolution to a kept location.** Move `ff/mmff/energy/params.rs`
   (resolution: `bond_params`, `angle_params`, torsion params, `oop_koop`, stbn, the empirical rules,
   `eq_level`, `bond_type`/`angle_type`) to `ff/mmff/params.rs` (or `ff/typifier/mmff/resolve.rs`),
   `pub(crate)`. Fix `use super::` paths (it depends only on kept `topo`/`tables`/`MmffVariant`).
   `MmffForceField` (still present until ac-007) keeps using it from the new location.
2. **Typifier assigns per-instance numeric parameters.** In the MMFF typifier, build `Topo` + the
   validated `types`/`charges`, and for each interaction call the relocated resolution to get final
   numbers; set them as `Atomistic` relation properties (so `to_frame` carries them as columns):
   - bonds: `kb`, `r0`  · angles: `ka`, `theta0` (+ existing `r0_ij`/`r0_kj`/`theta0` for stbn, moved
     into the typifier rather than the post-`to_frame` `merge_stbn_r0`)  · dihedrals: `v1`,`v2`,`v3`
     · impropers: `koop` (enumeration already MMFF-correct). The string `type` label may remain for
     inspection but is no longer the parameter source.
3. **MMFF kernels read per-instance columns.** Change `ff/potential/{bond,angle,dihedral,improper,pair}/mmff.rs`
   constructors from `type_map.get(label)` to reading the per-instance `get_float` columns (mirroring
   `mmff_stbn_ctor`'s `r0_ij` read). Drop the now-unneeded `resolve_*_label` string round-trip.
4. **RDKit parity, term by term.** Validate the generic path against `tests/ff/mmff/fixtures/*` after
   each term (bond → angle → stbn → torsion → oop), widening the `ac-003` test set from `e_ethane`
   to the full coverage (e_ethylene, e_benzene, e_butane, e_caffeine, s_aniline, s_acetamide, …).
5. **Decouple ETKDG from MMFF** (generation ≠ optimization). `generate_3d_impl`
   (`conformer/etkdg/mod.rs:50`) currently runs an in-generation MMFF cleanup
   (`if opts.mmff_cleanup_internal() { mmff_cleanup(...) }`, `:194`; `mmff_cleanup` at `:343` builds the
   standalone `MmffForceField` and minimises via the `mmff_min` module = `ff::optimize::minimize_lbfgs_rms`).
   Remove `mmff_cleanup`, the `mmff_min` module, the `mmff_cleanup_internal` option
   (`conformer/options.rs:135`), and the `MmffForceField`/`MmffMolProperties`/`MmffVariant` imports from
   `conformer/etkdg`. `generate_3d` returns the embed + ET-refined geometry (matching RDKit
   `EmbedMolecule`); MMFF cleanup becomes a caller-composed step (`MMFFTypifier::build` + generic
   `ff::optimize`).
6. **Rename `intramolecular_pairs` → `nonbonded_pairs`** and **expose it to Python** (mirror
   `extract_coords`; `molrs.pyi` entry).
7. **Delete the standalone evaluator (gated on ac-003 green).** Remove `ff/mmff/energy/` (`MmffForceField`
   + `impl Potential` + `MmffEnergyBreakdown` + the term-math structs) and `mmff/mod.rs` energy
   re-exports; `build_mmff_potentials_py` + its `lib.rs` registration + pyi; relocate
   `tests/ff/mmff/energy.rs`'s RDKit parity onto the generic path. **Keep** `mmff::{topo,aromaticity,
   atomtype,charges}` + `MmffMolProperties` + the relocated `params` resolution.

## §Files (molrs)

- `ff/mmff/params.rs` (NEW, relocated from `ff/mmff/energy/params.rs`) — the assigner's resolution.
- `ff/typifier/mmff/{frame_builder,mod,classify}.rs` — assign per-instance numeric params on `Atomistic`;
  fold `merge_stbn_r0` into the typifier; `resolve_*_label` becomes value resolution.
- `ff/potential/{bond,angle,dihedral,improper,pair}/mmff.rs` — read per-instance columns.
- `ff/potential/mod.rs` (+ `molrs.pyi`) — `intramolecular_pairs` → `nonbonded_pairs`; expose to Python.
- `ff/mmff/energy/`, `ff/mmff/mod.rs` — **delete** (after ac-003) the evaluator; keep front-end + params.
- `conformer/etkdg/{mod,mmff_min}.rs`, `conformer/options.rs` — decouple (ac-005).
- `molrs-python/src/{forcefield,lib}.rs`, `molrs-python/python/molrs/{__init__.py,molrs.pyi}` — delete
  `build_mmff_potentials`; add `nonbonded_pairs`.
- `tests/ff/mmff/energy.rs` — relocate parity onto the generic path; `tests/embed/etkdg.rs` — decoupled.

## §Out of scope

- MMFF94s variant (oop/torsion `_S` tables) on the generic path — MMFF94 only; the front-end carries
  the variant for a follow-up.
- `special_bonds` on `ForceField` + generic `lj/cut`/`coul/cut` per-atom unification
  (`ff-special-bonds-nblist` ac-001/003/004) — MMFF's 1-4 scaling is internal, independent.
- molpack `LBFGSRelaxer` + the interop FFI capsule bridge (`interop-ffi-bridge`).
- Migrating the **table** force fields (GAFF/OPLS/LAMMPS) off in-ctor label lookup — unnecessary; their
  kernels already evaluate from per-instance arrays. Only the contract ("kernels evaluate from
  per-instance parameters") is made explicit.
