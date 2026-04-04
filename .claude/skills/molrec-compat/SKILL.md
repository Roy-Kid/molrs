---
name: molrec-compat
description: Evaluate molrec spec compatibility for a file format. Spawns a product-manager agent to audit naming, metadata, grid mapping, and propose spec improvements.
argument-hint: "<format-name>"
user-invocable: true
---

Evaluate MolRec compatibility for format: **$ARGUMENTS**

## Trigger

`/molrec-compat <format-name>`

Examples:

- `/molrec-compat cube` — evaluate Gaussian Cube integration
- `/molrec-compat chgcar` — evaluate VASP CHGCAR integration
- `/molrec-compat lammps-dump` — evaluate LAMMPS dump integration

## Workflow

### Step 1: Gather context (parallel)

Launch up to 3 Explore agents in parallel:

1. **Format reader/writer** — Read `molrs-core/src/io/<format>.rs` to understand what fields are parsed, how they map to Frame/Grid/Block, and what metadata is stored.

2. **MolRec spec** — Read the relevant spec docs:
   - `/home/jicli594/work/molcrafts/molrec/docs/spec/frame.md`
   - `/home/jicli594/work/molcrafts/molrec/docs/spec/types.md`
   - `/home/jicli594/work/molcrafts/molrec/docs/spec/observables.md`

3. **Existing tests & cross-format comparison** — Read:
   - molrec test for this format: `/home/jicli594/work/molcrafts/molrec/src/molrec/tests/test_<format>.py`
   - Other format readers for naming comparison (e.g., compare cube vs chgcar field names)

### Step 2: Product Manager evaluation

Spawn the **product-manager** agent (see `.claude/agents/product-manager.md`) with full context from Step 1. The agent evaluates 6 dimensions:

1. **Field naming alignment** — Do column names match molrec spec conventions?
2. **Grid data model fit** — Are grid/array keys descriptive and consistent?
3. **Metadata completeness** — Is format-provided information captured structurally?
4. **Observable readiness** — Can grid data promote to observables without manual metadata?
5. **Roundtrip fidelity** — Is information preserved through MolRec → Zarr → reload?
6. **API ergonomics** — Is the Python API discoverable and consistent?

### Step 3: Cross-format consistency check

Compare naming conventions with at least one other format that stores similar data:

| Aspect | Format A | Format B | Consistent? |
|--------|----------|----------|-------------|
| Grid key | `"chgcar"` | `"cube"` | Both format-named |
| Density array | `"total"` | `"density"` | ⚠️ Divergent |
| Comment | `meta["title"]` | `meta["comment1"]` | ⚠️ Divergent |
| Atom symbol | `"symbol"` | `"symbol"` | ✅ |

### Step 4: Generate report

Output the compatibility report following the product-manager's output format:

```
# MolRec Compatibility Report: <Format>
## Summary
## Mapping Audit
## Friction Points
## Recommendations
## Compatibility Score
```

### Step 5: Propose action items

For each friction point with severity HIGH or above, propose a concrete action:

- **Spec change** → draft the spec diff
- **Reader change** → describe the code change with file paths
- **Naming convention** → propose the canonical name and migration path

Present action items to the user for prioritization.
