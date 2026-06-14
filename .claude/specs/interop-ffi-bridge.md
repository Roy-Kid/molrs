# interop-ffi-bridge — one blessed zero-copy path for downstream Rust/PyO3 apps to consume molrs (and molpy) data objects

Status: **implemented** (2026-06-13) — ForceFieldRef + FF/Frame capsules + molpack
`interop.rs`; frame_marshal deleted; verified (molrs-ffi 14, molrs-python 349,
molpack 148 tests green). Dict frame input dropped (locked decision). Version-tag
guard (ac-005) deferred.
Scope: molrs-ffi, molrs-python, molrs-capi, (reference consumer: molpack-python)

## Summary

Downstream Rust/PyO3 applications that depend on molrs (today `molpack`; tomorrow
others) must interoperate with molpy/molrs **Python data objects** — `Frame`,
`ForceField`, `Block`, `Box`. Today every consumer **re-invents a data layer and
marshals by hand**: `molpack/python/src/frame_marshal.rs` does a `to_dict` →
parse → `from_dict` round-trip on **every** boundary crossing. That is the
friction this spec removes:

- a de-facto parallel `Frame`/`ForceField` representation per consumer (the
  forbidden "mpk.Frame");
- an O(N) deep copy of all columns on every call in and out;
- silent dtype/signedness bugs (molpy emits `int32` atom indices, molrs ff reads
  `UInt` — the marshaller collapsed both to `Int`, breaking `to_potentials`);
- duplicated maintenance and a different bug surface per consumer.

molrs and each consumer are **separate PyO3 cdylibs**, so a `molrs.Frame`
pyclass cannot be `.extract()`'d as a Rust object inside `molpack` (PyO3
pyclasses are not shared across extensions). The fix is **not** more marshalling —
it is a single, blessed, **zero-copy** consumer path through molrs's already-existing
stable-FFI layer. molpack is the reference consumer; the recipe must generalize
to any Rust app depending on molrs/molpy.

## Domain basis — the infrastructure already exists

- Crates: `molcrafts-molrs` (Rust core) · `molrs-python` (PyO3) · `molrs-ffi`
  (stable handles: `FrameRef`, `FrameId`, `BlockHandle`) · `molrs-capi`
  (C ABI: `MolrsFrameHandle`, `MolrsForceFieldHandle`; `frame.rs`/`forcefield.rs`/
  `block.rs`/`simbox.rs`) · `molrs-cxxapi` (C++ bridge over molrs-ffi).
- `molrs.Frame._ffi_frameref_capsule()` (`molrs-python/src/frame.rs:388-410`)
  returns a `PyCapsule` named `"molrs.FrameRef"` whose `void*` is a
  `*mut *mut FrameRef` — a **clone of the frame's handle sharing the same
  underlying data, no deep copy**. The capsule destructor reclaims the boxed
  `FrameRef`.
- `molrs-cxxapi::frame_clone_from_addr(addr)` (`molrs-cxxapi/src/lib.rs:447-473`)
  is the documented consumer resolver: `PyCapsule_GetPointer` → `usize` →
  `Box<FrameRef>`. This is the exact pattern molpack must use instead of marshalling.

## Gaps to close

1. **Only `Frame` exposes a capsule.** `molrs.ForceField` has no `_ffi_*_capsule()`
   (verified: `dir(molrs.ForceField)` has no `ffi`/`capsule` member). `Block`/`Box`
   likewise as needed.
2. **Consumer-facing resolver helper.** Each consumer hand-rolls the ~4-line
   capsule→`FrameRef` resolver (cxxapi `frame_clone_from_addr`) plus the return path.
   This is **intentionally consumer-local** (revised), not a molrs-ffi module — keeps
   molrs-ffi pyo3-free. Cheap to duplicate; extract behind an optional `pyo3` feature
   only when a 2nd consumer exists.
3. **Handle API coverage — RESOLVED, not a gap.** `FrameRef::with_frame(|f: &Frame| …)`
   lends the **core `&molrs::Frame`** directly, so consumers call the full molrs core API
   (`to_potentials`, `intramolecular_pairs`, column reads) inside the closure — no need to
   re-export a read/write surface through handles. (`ForceFieldRef::with_forcefield` mirrors this.)
4. **Data contract not codified.** Atom-connectivity index columns
   (`atomi/atomj/atomk/atoml`, ids) must be **`UInt`** (molrs `U = u32`) end to end;
   molpy must emit uint (partly done: `molpy/io/forcefield/amber.py`). Required
   blocks/columns + dtype conventions documented once.
5. **No version guard.** The capsule handle layout is tied to a molrs-ffi ABI
   version; consumer and molrs-python must agree, with a clear error on mismatch.

## Design — the blessed path

Key realisation (verified): `molrs_ffi::FrameRef::with_frame(|f: &Frame| …)`
(`molrs-ffi/src/store.rs:117`) lends the **real core `&molrs::Frame`**, and
`ForceField::to_potentials(&self, frame: &Frame)` (`potential/mod.rs:322`) consumes
exactly that. So a consumer that resolves the capsule to a `FrameRef` calls the full
core API directly — **no handle-API re-export, no marshalling, no consumer data type.**

- **Consumers link `molrs-ffi`** (the stable handle layer, **pyo3-free**) only for the
  handle *types* (`FrameRef`, `ForceFieldRef`). A `molrs.Frame`/`molrs.ForceField`
  (incl. molpy's) crosses as its FFI capsule; the consumer resolves it and operates on
  the core `&molrs::Frame`/`&ForceField` lent by `with_frame`/`with_forcefield` — zero copy.
- **The resolver helpers are CONSUMER-LOCAL FREE functions** (revised — *not* a
  `molrs_ffi::bridge` module; molrs-ffi gets no pyo3). For molpack: `molpack::interop`
  (`python/src/interop.rs`), each ~4 lines = the cxxapi `frame_clone_from_addr` pattern:
  - `frame_from_py(obj: &Bound<PyAny>) -> PyResult<molrs_ffi::FrameRef>`
  - `forcefield_from_py(obj: &Bound<PyAny>) -> PyResult<molrs_ffi::ForceFieldRef>`
  - `frame_to_py<'py>(py, …) -> PyResult<Bound<'py, PyAny>>` — only for genuinely-fresh
    result frames; in-place mutation via `with_frame_mut` returns the same Python object.
  Accepts any object exposing the capsule (`molrs.Frame`, `molpy.Frame`, …). The Python
  facade is **molpy** — users call `molpy.io.forcefield.read_lammps`, never `molrs`.
  Precedent: molrs-cxxapi keeps its own `frame_clone_from_addr`; it does **not** live in
  molrs-ffi. If a 2nd Rust consumer appears, lift `interop.rs` into molrs-ffi behind an
  **optional, off-by-default `pyo3` feature** (wasm/cxxapi unaffected).
- **molrs-ffi** gains `ForceFieldRef` (newtype over a shared-`Rc` core `ForceField`) +
  `with_forcefield(|ff: &ForceField| …)` — mirroring `FrameRef`/`with_frame`. **No pyo3.**
  (Required: a raw `Arc<molrs::ForceField>` cast across cdylibs is unsound — molpack links
  molrs core with `default-features=false`, molrs-python with `io+ff+…`; differing
  features can change layout. The stable-layer handle's uniform feature surface is the fix.)
- **molrs-python** adds `ForceField._ffi_forcefield_capsule()` (mirror the existing
  `Frame._ffi_frameref_capsule()`), and — only if a fresh-frame return path is needed —
  a `Frame`-from-FFI-handle constructor.
- **Data contract** (documented + enforced at the seam): uint atom indices, required
  blocks, column names — kills the marshalling-era signedness class of bug.
- **Version guard**: DEFERRED. Keep the existing `"molrs.FrameRef"` name; add
  `"molrs.ForceFieldRef"`. ABI `.vN` tags are a later hardening, not load-bearing now.

## Files

- `molrs-ffi/src/store.rs` (or sibling) — **new `ForceFieldRef`** handle + `with_forcefield`,
  mirroring `FrameRef`/`with_frame`. **No pyo3.**
- `molrs-python/src/forcefield.rs` — `_ffi_forcefield_capsule()` (mirror `frame.rs`'s);
  optional `Frame`-from-FFI-handle ctor only if a fresh-frame return path is needed.
- `molpack/python/Cargo.toml` — add `molcrafts-molrs-ffi` path dep (pin 0.1.1, manual
  per the molrs path/version rule).
- `molpack/python/src/interop.rs` (new, ~30 lines) — `frame_from_py`/`forcefield_from_py`/
  `frame_to_py`; the cxxapi `frame_clone_from_addr` resolver, consumer-local.
- `molpack/python/src/frame_marshal.rs` — **delete**; bindings switch to `interop`.
- `docs/interop.md` — DEFERRED (write once there's a 2nd consumer to bless a shared path).
- `molrs-capi/src/forcefield.rs` — out of scope here (C-ABI consumers; molpack uses the
  Rust closure handles, not capi).

## Testing

- Round-trip: a `molrs.Frame` → `frame_from_capsule` → mutate via handle →
  observable in Python **without** a deep copy (assert shared backing, e.g. an
  in-place column write is visible on the original Python object).
- `molpy.Frame` resolves through the same path unchanged.
- ForceField capsule round-trips; consumer reads styles/params via the handle.
- Version-mismatch capsule → clean error, not UB.
- molpack reference: build + pack via `bridge` with zero `to_dict`/`from_dict`.

## Out of scope

- The ff potential unification (special_bonds in ForceField, per-atom + `is_14`
  pair kernels, MMFF demotion to a pure param loader, `io::forcefield` reader
  move). Tracked separately — this spec is **only** the cross-extension data-object
  interop mechanism. The two compose: once `bridge` lands, the relaxer takes a
  `molrs.ForceField` over the bridge and builds its own neighbour list.
