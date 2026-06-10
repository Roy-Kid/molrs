# molrs — Evolving Decisions

Short-lived working notes captured by `/mol:note`. Stable entries are
promoted into `CLAUDE.md` and removed from here.

Format per entry:

```
## YYYY-MM-DD — <topic>
**Decision:** <one-liner>
**Why:** <motivation — constraint, incident, experiment result>
**Status:** provisional | hardening | promoted (→ CLAUDE.md §section)
```

Run `/mol:note sweep` monthly to surface stale entries (> 90 days without
status change) and conflicts with `CLAUDE.md`.

---

## 2026-05-28 — BLAS/LAPACK backend selection is the binary's job, not molrs's

**Decision:** `molrs-core/Cargo.toml` keeps `ndarray-linalg = "0.18"` with
no backend feature pre-selected (`openblas-system` / `netlib-static` /
`intel-mkl-*` / etc.). Picking a backend is the responsibility of the
top-level binary that consumes molrs (test runner, downstream app), not
of molrs-core itself.

**Why:**
- ndarray-linalg README, verbatim: "If you are creating a library
  depending on this crate, we encourage you not to link any backend."
  Cargo features are additive — if molrs-core picks `openblas-system`,
  every downstream is forced onto OpenBLAS forever.
- ndarray-linalg 0.18 backends: `openblas-{system,static}`,
  `netlib-{system,static}`, `intel-mkl-*` variants. **No `accelerate`
  feature**; Apple Silicon + Accelerate.framework is not officially
  supported by ndarray-linalg in this version.
- Consequence: `cargo test --all-features` on a fresh checkout will
  fail to link with `Undefined symbols: _cblas_sgemv, _dgetrf_, ...`
  unless the developer provides a backend externally.

**How to actually run `--all-features` tests locally:**

Either (a) the canonical `blas-src` / `lapack-src` dev-dependency
pattern in the test crate, e.g. in `molrs-core/Cargo.toml`:

```toml
[dev-dependencies]
openblas-src = { version = "0.10", features = ["system"] }
```

plus `#[cfg(test)] extern crate openblas_src;` at the top of
`molrs-core/src/lib.rs`, plus `brew install openblas` (it provides
CBLAS, unlike `brew install lapack` which is Fortran-ABI only).

Or (b) opt-in via CLI on the developer's machine without touching
Cargo.toml — but cargo doesn't have a clean per-invocation override
for downstream features; (a) is the canonical path.

**Action item (not blocking):** the `cargo test --all-features` line in
`CLAUDE.md` is misleading because it won't run on a clean checkout. We
should either drop `--all-features` from CLAUDE.md's quick-start, or
adopt option (a) and document `brew install openblas` as prerequisite.

**Status:** provisional — captured during `frame-block-subclass` impl
when the user's local `cargo test --all-features` couldn't link;
unrelated to that spec; not blocking.

## 2026-05-13 — Frame is pure `HashMap<String, Block>`, no Grid special case

**Decision:** Remove `Grid` (`grid.rs`) and `UniformGridField` (`field.rs`)
from `molrs-core`. Frame stores only `HashMap<String, Block>` + `meta` +
`SimBox`. No grid-specific methods on Frame or Block. Grid semantics belong at
the I/O boundary (CHGCAR/Cube reader → Block columns + spatial metadata in
Frame meta), not as privileged types in the core data model.

**Why:**
- `Grid` is just `{named arrays} + {dim, origin, cell, pbc}` — named arrays
  are Block columns, spatial metadata is Frame meta. No new type needed.
- `UniformGridField` duplicates Grid's spatial definition and `FieldEncoding`
  had only one variant — premature abstraction with no callers.
- Frame having `HashMap<String, Grid>` alongside `HashMap<String, Block>` is
  a special case that complicates the API (7 extra methods, separate Zarr
  code path, separate wasm index path, separate Python class).

**Status:** provisional
