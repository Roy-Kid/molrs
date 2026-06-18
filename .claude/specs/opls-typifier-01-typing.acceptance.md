---
slug: opls-typifier-01-typing
criteria:
  - id: ac-001
    summary: OplsTypingMeta parses class/def/overrides/priority/layer from OPLS XML
    type: code
    evaluator_hint: "test: ff::typifier::opls meta parse"
    pass_when: |
      read_opls_typing_xml_str over a real oplsaa.xml excerpt yields an
      OplsTypingMeta keyed by opls_NNN, where a known modern row (e.g. opls_135)
      carries class="CT", a non-empty def SMARTS, and a row with overrides parses
      the comma list into Vec<String>; a row missing def yields def=None.
    status: verified
  - id: ac-002
    summary: priority resolution matches molpy _OplsAtomTypifier
    type: code
    evaluator_hint: "test: ff::typifier::opls priority"
    pass_when: |
      For constructed type sets, the computed priority equals molpy's rule:
      explicit `priority` wins; else (+len(overrides)) − (times overridden);
      plus layer*stride. The highest-priority type is selected for an atom
      matched by multiple defs.
    status: verified
  - id: ac-003
    summary: read_opls_xml potential parsing unchanged (no regression)
    type: code
    evaluator_hint: "test: existing opls reader tests still pass"
    pass_when: |
      All existing OplsXmlReader tests (ff/forcefield/readers/opls.rs) pass
      unchanged; read_opls_typing_xml_str is additive and does not alter the
      ForceField produced by read_opls_xml.
    status: verified
  - id: ac-004
    summary: SMARTS atom typing on real molecules incl. recursive def
    type: scientific
    evaluator_hint: "test: ff::typifier::opls typing on tests-data molecules"
    pass_when: |
      annotate_opls assigns opls_NNN atom types over molecules read from
      tests-data/, including at least one def using recursive $() SMARTS; typed
      atoms carry type, class, and charge props. No synthetic happy-path inputs.
    status: verified
  - id: ac-005
    summary: malformed SMARTS def fails fast
    type: code
    evaluator_hint: "test: ff::typifier::opls error path"
    pass_when: |
      A type whose def is an unparseable SMARTS string causes OplsTypifier
      construction (or typify) to return Err, never silently dropping the type.
    status: verified
  - id: ac-006
    summary: lint, type check, and test suite clean
    type: runtime
    pass_when: |
      `cargo fmt --all --check`, `cargo clippy --all-targets --all-features
      -- -D warnings`, and `cargo test --all-features` all exit 0.
    status: verified
    note: |
      RESOLVED 2026-06-18: full `--all-features` gate now exits 0 with the brew
      openblas env (LIBRARY_PATH + DYLD_FALLBACK_LIBRARY_PATH +
      RUSTFLAGS="-L/opt/homebrew/opt/openblas/lib -lopenblas"). fmt --all --check
      clean; clippy --all-targets --all-features -D warnings clean; test
      --all-features all binaries green, 0 failed. Prior blockers (compute/fit
      WIP, io/xtc teammate WIP, blas link) resolved via upstream merge + openblas
      env. See [[project_molrs_allfeatures_blas]].

      Partially verified. fmt clean; clippy + tests GREEN for every in-scope
      feature set (io,signal,smiles,ff,conformer): 1191 passed / 0 failed,
      ff-scoped clippy clean. The literal `--all-features` invocation does NOT
      yet exit 0 because the `compute` feature is broken by PRE-EXISTING,
      UNRELATED working-tree WIP (uncommitted edits under molrs/src/compute/ +
      an untracked molrs/src/compute/fit/ whose mod.rs declares
      `pub mod raw_computes;`/`pub mod spectral;` but raw_computes.rs is missing,
      so `compute::spectra` fails to import `compute::fit::forward_fft_onesided`).
      This is independent of the OPLS typifier (ff-only) and was left untouched
      per scope. Re-run `--all-features` once the compute WIP lands.
---

# Acceptance criteria

- **ac-001 / ac-002**: typing 元数据 plumb + 优先级算法逐位复刻 molpy `_OplsAtomTypifier`（class/def/overrides/priority/layer）。
- **ac-003**: 势能 reader 零回归——typing-meta 是独立 sink，不碰 `read_opls_xml`。
- **ac-004**: 真实分子上的 SMARTS 分型（含递归 `$()`），按 MANDATORY IO 规则迭代 `tests-data/`，禁合成 happy-path 数据；per-atom 全量 parity 是链 3/3 的门。
- **ac-005**: 坏 SMARTS def fail-fast（对齐 molpy "broken FF def → raise"）。
- **ac-006**: cargo 质量闸（fmt/clippy/check/test）。
