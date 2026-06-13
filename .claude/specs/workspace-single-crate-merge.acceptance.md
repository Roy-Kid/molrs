---
slug: workspace-single-crate-merge
criteria:
  - id: ac-001
    summary: only molcrafts-molrs remains as a publishable lib workspace member
    type: code
    evaluator_hint: "grep [workspace] members in Cargo.toml"
    pass_when: |
      Root Cargo.toml [workspace] members no longer lists molrs-core, molrs-io,
      molrs-signal, molrs-compute, molrs-ff, or molrs-conformer; the only
      publishable lib member among the former 7 is `molrs` (package
      molcrafts-molrs). molrs-cxxapi (staticlib) may remain; molrs-ffi stays
      excluded. The six sub-crate src/ trees now live under molrs/src/{core,io,
      signal,compute,ff,conformer}/.
    status: verified
    last_checked: 2026-06-13
  - id: ac-002
    summary: cargo build and cargo check --all-features are green
    type: code
    evaluator_hint: ""
    pass_when: |
      `cargo build` and `cargo check --all-features` succeed; `cargo fmt --all
      --check` and `cargo clippy -- -D warnings` succeed against the merged crate.
    status: verified
    last_checked: 2026-06-13
  - id: ac-003
    summary: no-default and each single-feature subset compiles
    type: code
    evaluator_hint: "cargo check --no-default-features --features <one>"
    pass_when: |
      `cargo check -p molcrafts-molrs --no-default-features` compiles, and each of
      `--features io`, `--features ff`, `--features signal`, `--features compute`,
      `--features conformer`, `--features smiles` (over --no-default-features)
      compiles independently.
    status: verified
    last_checked: 2026-06-13
  - id: ac-004
    summary: subsystem-OFF build does not compile that subsystem's unique deps
    type: code
    evaluator_hint: "cargo tree -p molcrafts-molrs --no-default-features --features ff"
    pass_when: |
      `cargo tree -p molcrafts-molrs --no-default-features --features ff` shows no
      flate2, petgraph, or io-only once_cell node; `--features io` shows no
      roxmltree node; `--no-default-features` shows no rustfft/rand/petgraph/
      flate2/roxmltree node. Subsystem-unique deps are `optional = true` and
      feature-gated in molrs/Cargo.toml.
    status: verified
    last_checked: 2026-06-13
  - id: ac-005
    summary: full --all-features test suite passes at the ~1248-test floor
    type: runtime
    evaluator_hint: "cargo test --all-features; count >= 1248"
    pass_when: |
      `cargo test --all-features` passes with zero failures and the total executed
      test count is >= 1248 (the pre-merge workspace floor). IO data-driven tests
      still iterate every file in tests-data/<format>/ via the local common helper.
    status: verified
    last_checked: 2026-06-13
  - id: ac-006
    summary: IO data-driven tests still iterate tests-data/ from the new location
    type: runtime
    evaluator_hint: "marker: io; common::format_files"
    pass_when: |
      molrs/tests/io retains the local `common` module resolving `../tests-data`
      (or $MOLRS_TESTS_DATA), and the format tests still call
      common::format_files("<format>") over every real file (no synthetic
      happy-path string). The IO test binary runs green under --all-features.
    status: verified
    last_checked: 2026-06-13
  - id: ac-007
    summary: downstream public paths resolve unchanged
    type: code
    evaluator_hint: "doctest / probe build against merged crate"
    pass_when: |
      A consumer with molcrafts-molrs (features=["full","zarr","filesystem",
      "smiles"]) resolves molrs::Frame, molrs::io::read_xyz, molrs::smiles::*,
      molrs::ff::*, molrs::conformer::Conformer, molrs::compute::*, and
      molrs::signal::* exactly as before the merge — no rename, no path change.
    status: verified
    last_checked: 2026-06-13
  - id: ac-008
    summary: all four binder crates build against the merged crate
    type: code
    evaluator_hint: "ffi/cxxapi: cargo check; wasm: wasm32 target; python: cargo check"
    pass_when: |
      molrs-ffi (cargo check), molrs-cxxapi (cargo check; molrs_io:: refs rewritten
      to molrs::io::), molrs-wasm (cargo check --target wasm32-unknown-unknown), and
      molrs-python (cargo check) each build with deps repointed to molcrafts-molrs +
      matching features and no remaining reference to the removed sub-crate names.
    status: verified
    last_checked: 2026-06-13
  - id: ac-009
    summary: publish.yml has exactly one Rust cargo publish
    type: code
    evaluator_hint: "grep cargo publish / publish-if-new in publish.yml"
    pass_when: |
      .github/workflows/publish.yml contains exactly one Rust crates.io publish
      step (for molcrafts-molrs); the publish-core/io/signal/compute/ff/conformer
      jobs are gone. publish-wasm and build-python/publish-python jobs are
      unchanged.
    status: verified
    last_checked: 2026-06-13
  - id: ac-010
    summary: sub-crate names are not yanked; merged crate version bumped
    type: docs
    evaluator_hint: ""
    pass_when: |
      The spec/changelog records that the six sub-crate crates.io names are NOT
      yanked (molpack's 0.1.0 pin keeps resolving) and that molcrafts-molrs takes a
      coordinated minor/major version bump as a breaking change; molpack migration
      is logged as an out-of-tree follow-up.
    status: pending
---

# Acceptance criteria — workspace-single-crate-merge

Binding contract for `workspace-single-crate-merge.md`. The merge lands as a
single atomic change (forward incremental phasing is impossible — every sub-crate
depends on `core` and the facade re-exports every sub-crate, so absorbing `core`
while a sub-crate stays separate forms a `cyclic package dependency`). These 10
criteria are the **end-state** bars, verified once the change lands.

- **ac-001 / ac-009** are the structural headline outcomes: one publishable lib
  member, one Rust publish job.
- **ac-002 / ac-003 / ac-004** form the build matrix: full build green, every
  feature subset compiles, and the dependency-isolation invariant (the core
  motivation — subsystem-OFF must not compile that subsystem's unique deps) holds
  via `cargo tree`.
- **ac-005 / ac-006** are the two `runtime` bars: the ~1248-test floor passes and
  IO data-driven tests still iterate `tests-data/` from the relocated test tree.
- **ac-007 / ac-008** guard the zero-API-change contract: downstream paths and
  all four binder crates resolve/build against the merged crate.
- **ac-010** (`docs`) records the versioning discipline: no yank, coordinated
  bump, molpack follow-up out of tree.

No `ui_runtime` criteria — this refactor touches no frontend.
