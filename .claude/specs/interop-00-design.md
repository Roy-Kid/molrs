# interop-00-design — top-level design: how molpy / molrs / a downstream Rust app interlock, and what the user-facing API looks like

Status: **design** (review before implementing the 3 sibling specs)
Umbrella over: `interop-ffi-bridge`, `ff-special-bonds-nblist`, `interop-wrapup`

Purpose: settle the **three-party interaction model** and the **concrete API surface**
(Python + Rust) *before* implementation, so the three implementation specs build to
one agreed shape. Review the "Open API decisions" at the bottom.

## The three parties

| Party | Is | Owns |
|---|---|---|
| **molpy** | pure-Python molecule building + file IO. Its `Frame` **is** a `molrs.Frame` (embeds molrs). | builders, format loaders; emits molrs frames (uint indices per the data contract) |
| **molrs** | Rust core `molcrafts-molrs` + Python ext `molrs` + stable FFI `molrs-ffi` + C-ABI `molrs-capi`. | the data types (Frame/Block/Box/ForceField), the **capsule transport**, `io::forcefield` readers, `ff::to_potentials`, `intramolecular_pairs`, the kernels |
| **downstream Rust/PyO3 app** (molpack, others) | links `molcrafts-molrs` (core — the kernels/ForceField it computes with) **+ `molrs-ffi`** (the bridge). | its domain logic only; **no own data type, no marshalling** |

**Linking (Cargo):** a consumer deps `molcrafts-molrs` + `molrs-ffi` (the latter
**pyo3-free** — used only for the handle *types* `FrameRef`/`ForceFieldRef`). At the
Python boundary it accepts `molrs.Frame` / `molrs.ForceField` (and molpy's, which
carry the same capsule), resolves the capsule **to `molrs_ffi::FrameRef` in a few
local helper functions** (the cxxapi `frame_clone_from_addr` pattern), operates on the
**real core `&molrs::Frame`** lent by `FrameRef::with_frame`, and returns molrs
objects. There is exactly one data layer: molrs's. The consumer owns its ~30-line
resolver locally (like molrs-cxxapi does); **molrs-ffi gains no pyo3 code**.

## Data ownership (the load-bearing rule)

- **molrs** owns data types + capsule transport + io readers + `to_potentials` +
  `intramolecular_pairs` + kernels.
- **`ForceField`** = params + functional forms + **special_bonds weights**. NOT the
  neighbour list.
- **consumer** (the LBFGS relaxer; an MD integrator) **owns the neighbour list**: it
  calls `intramolecular_pairs`, inserts the `pairs` block, calls `to_potentials`,
  minimises. Owns domain logic.
- **nobody** marshals or duplicates the data layer.

## User-facing API — Python (molpy + molpack only; molrs is NEVER imported)

**molpy is the Python facade**; it proxies molrs's readers and types. The user never
types `molrs`. Frames are `molpy.Frame` (= `mp.Frame`) — **there is no `mpk.Frame`**.

```python
import molpy, molpack

# 1. molpy builds the molecule -> mp.Frame
chain = molpy.builder.polymer("{[<]CCO[>]}|23|")

# 2. molpy PROXIES the per-format force-field reader (which lives in molrs::io::forcefield)
ff = molpy.io.forcefield.read_lammps("melt.ff")

# 3. molpack: the relaxer takes ONLY the force field; it LAZY-compiles (builds its
#    neighbour list + potential) the first time it is handed the molecule's frame
#    during packing — NOT at construction.
relaxer = molpack.LBFGSRelaxer(ff)
target  = molpack.Target(chain, count=1).with_relaxer(relaxer)
result  = molpack.Molpack().with_periodic_box(...).pack([target, *others])

result.frame    # a mp.Frame  (Python returns molpy frames)
```

What the user never sees: `molrs`, `to_dict`/`from_dict`, a `from_lammps_ff`, a
`mpk.Frame`, or any deep copy.

## User-facing API — Rust (a downstream consumer)

Bridge is **free helper functions, owned by the consumer** (developer-friendly per
review), `verb_noun`, `py↔ref` symmetric. Each is the ~4-line cxxapi
`frame_clone_from_addr` resolver. Rust returns a `molrs.Frame`.

```rust
use molpack::interop::{frame_from_py, forcefield_from_py};  // LOCAL to the consumer

#[pyfunction]
fn my_op<'py>(py: Python<'py>, frame: &Bound<'py, PyAny>, ff: &Bound<'py, PyAny>)
    -> PyResult<Bound<'py, PyAny>>
{
    let fref = frame_from_py(frame)?;           // capsule -> molrs_ffi::FrameRef (zero-copy)
    let ff   = forcefield_from_py(ff)?;         // capsule -> molrs_ffi::ForceFieldRef
    fref.with_frame_mut(|f| {                   // f: &mut molrs::Frame — the REAL core type
        f.set_block("pairs", intramolecular_pairs(f));
        let pots = ff.with_forcefield(|ff| ff.to_potentials(f))?;
        // … minimise; write coords back into f (shared store, visible in Python) …
        Ok(())
    })?;
    Ok(frame.clone())   // in-place result: hand back the same molrs.Frame, no construction
}
```

`FrameRef::with_frame`/`with_frame_mut` (`molrs-ffi/src/store.rs:117/126`) lend the
**core `&molrs::Frame`**, so the consumer calls `to_potentials`/`intramolecular_pairs`
directly — no handle-API re-export needed. `frame_to_py` is only for genuinely-fresh
result frames (placed copies); in-place mutation needs no constructor.

## How the three implementation specs realise this

- **interop-ffi-bridge** → `molrs-ffi` gains `ForceFieldRef` (mirror `FrameRef`,
  pyo3-free); molrs-python gains `ForceField._ffi_forcefield_capsule()` (mirror Frame's);
  molpack gains a local `interop.rs` (capsule→ref resolvers) + the uint data contract.
  Replaces molpack's `frame_marshal`. **No new crate, no `molrs_ffi::bridge`.**
- **ff-special-bonds-nblist** → `molrs.io.forcefield.read_lammps`, `ForceField`
  special_bonds, `intramolecular_pairs` (done), per-atom + `is_14` pair kernels, MMFF
  demotion, and the relaxer that takes a `ForceField` and builds its own nblist.
- **interop-wrapup** → polish the prelude/naming + `docs/interop.md` recipe & example.

## Resolved API decisions (locked)

1. **Python facade = molpy.** The Python side never imports `molrs`; molpy proxies the
   readers + types (`molpy.io.forcefield.read_lammps`, `molpy.Frame`). molpack is fed
   `mp.Frame`; **there is no `mpk.Frame` — ever.**
2. **Relaxer = `LBFGSRelaxer(ff)`** — force field only. It **lazy-compiles** (builds its
   neighbour list via `intramolecular_pairs` + `to_potentials`) the **first time it
   receives the molecule's frame during packing**, not at construction. → the `Relaxer`
   trait's `spawn` must be handed the molecule's Frame/topology (today it only gets
   `ref_coords`); `LBFGSRelaxer` holds the `ForceField` until then.
3. **Bridge = free functions, owned by the consumer (revised — was `molrs_ffi::bridge`).**
   `frame_from_py`/`frame_to_py`, `forcefield_from_py`/`forcefield_to_py` live **in
   molpack** (`molpack::interop`), not in molrs-ffi — each is the ~4-line cxxapi
   `frame_clone_from_addr` resolver. molrs-ffi stays **pyo3-free**; the consumer deps it
   only for the `FrameRef`/`ForceFieldRef` *types*. Rationale: one consumer today → no
   shared abstraction yet (cxxapi sets the precedent: it keeps its own resolver). If a
   2nd Rust consumer appears, lift molpack's `interop.rs` into molrs-ffi behind an
   **optional, off-by-default `pyo3` feature** (wasm/cxxapi unaffected). No new crate.
4. **Readers = per-format**: `read_lammps`, `read_opls`, … (no single auto-detect entry).
5. **Return types**: Python returns `mp.Frame`; Rust returns `molrs.Frame`.
6. **Naming (locked now):**
   - bridge fns: `{frame,forcefield}_{from_py,to_py}` in **`molpack::interop`** (consumer-local).
   - new molrs-ffi handle: `ForceFieldRef` + `with_forcefield` (mirror `FrameRef`/`with_frame`).
   - molrs-python producer: `ForceField._ffi_forcefield_capsule()` (mirror Frame's).
   - capsule names: keep existing `"molrs.FrameRef"`; add `"molrs.ForceFieldRef"`.
     (Version tags `.v1` DEFERRED — not load-bearing for one consumer.)
   - readers: `molrs::io::forcefield::{read_lammps, read_opls, ForceFieldReader}`,
     proxied as `molpy.io.forcefield.*`.
   - relaxer: `molpack.LBFGSRelaxer(ff)`.

Implement in order: `interop-ffi-bridge` → `ff-special-bonds-nblist` → `interop-wrapup`.
Note the **`Relaxer::spawn` signature change** (frame/topology at spawn) is part of
`ff-special-bonds-nblist`'s molpack reference work, driven by decision 2.
