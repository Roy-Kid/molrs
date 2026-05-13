# molrs — Evolving Decisions

Short-lived working notes captured by `/molrs-note`. Stable entries are
promoted into `CLAUDE.md` and removed from here.

Format per entry:

```
## YYYY-MM-DD — <topic>
**Decision:** <one-liner>
**Why:** <motivation — constraint, incident, experiment result>
**Status:** provisional | hardening | promoted (→ CLAUDE.md §section)
```

Run `/molrs-note sweep` monthly to surface stale entries (> 90 days without
status change) and conflicts with `CLAUDE.md`.

---

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
