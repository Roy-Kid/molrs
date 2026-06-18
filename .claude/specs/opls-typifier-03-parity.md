---
title: OPLS-AA typifier parity 验证台（vs molpy OplsTypifier）
status: approved
created: 2026-06-18
---

# OPLS-AA typifier parity 验证台

> OPLS typifier 链 3/3。依赖 [opls-typifier-01-typing](opls-typifier-01-typing.md) + [opls-typifier-02-assign](opls-typifier-02-assign.md)（+ 可选 [ff-parameter-estimator](ff-parameter-estimator.md)）。镜像 [gaff-typifier-04-parity](gaff-typifier-04-parity.md) 的对照范式，但 ground truth 是 molpy 现有 `OplsTypifier`（而非 antechamber）。

## Summary

建立一个对照验证台，证明 molrs Rust `OplsTypifier`（原子分型 + 成键参数）与 molpy 现有 Python `OplsTypifier` 在一组真实分子上**逐原子类型一致**、**成键参数在物理容差内一致**。这是反转 B线、删除 molpy Python 分型代码（[opls-typifier-downsink](opls-typifier-downsink.md)，molpy 侧）之前的硬门。

## Domain basis

下沉的正确性判据 = 与既有、已验证的 molpy `OplsTypifier` 行为等价。对照维度：(1) 每原子 `opls_NNN` type 100% 一致（分型是离散的，必须精确）；(2) 每 bond/angle/dihedral 的参数在容差内（bond r₀ 0.02 Å、angle θ₀ 3°、力常数 rtol 0.10）——容差吸收 RB→OPLS、nm→Å/kJ→kcal 换算的浮点差。fixtures 用真实分子（含 PEO 醚、烷烃、芳香、含杂原子体系），覆盖通配端二面角与 overrides/layer 分型分支。

## Design

- `gen_opls_fixtures.py`（molpy 侧脚本，或 fixtures 仓库）跑 molpy `OplsTypifier` 产出 per-atom type + per-term params 的 JSON ground truth，存入 `tests-data/`（binding-neutral 根目录，gitignored，`fetch-test-data.sh` 拉取），镜像 GAFF/MMFF 的 RDKit/antechamber fixture 范式。
- `molrs/tests/ff/typifier/opls_parity.rs`：迭代 `tests-data/opls/` 每个分子，`OplsTypifier::from_xml_str(oplsaa).typify(mol)` → 比对 JSON。`type:scientific` 门：per-atom type 100% 一致；params 容差内。
- fixtures 缺席时**干净跳过**（对齐 GAFF parity 范式）。
- estimator 启用/关闭两种模式都对照：关闭时与 molpy strict 行为对齐；启用时对照 molpy + estimator（若 molpy 侧也启用）或单独验证估计项的溯源/penalty。

## Files to create or modify

- `tests-data/opls/` (new, 数据仓库) — 真实分子 + molpy ground-truth JSON。
- `scripts/gen_opls_fixtures.py`（molpy 仓库或 fixtures 仓库）(new) — 跑 molpy OplsTypifier 生成 ground truth。
- `molrs/tests/ff/typifier/opls_parity.rs` (new) — 对照测试，迭代 tests-data/opls/。

## Tasks

- [ ] Write gen_opls_fixtures.py to dump molpy OplsTypifier per-atom types + per-term params as JSON for a real-molecule set (PEO ether, alkane, aromatic, heteroatom)
- [ ] Add the molecule set + ground-truth JSON to the tests-data repo under opls/
- [ ] Write opls_parity.rs iterating tests-data/opls/, asserting per-atom type 100% match and per-term params within tolerance; skip cleanly when fixtures absent
- [ ] Verify parity holds with estimator off; record estimator-on provenance behavior
- [ ] Run cargo fmt --all --check && cargo clippy -- -D warnings && cargo test --all-features --features slow-tests

## Testing strategy

- **Scientific (gated)** — per-atom `opls_NNN` type 100% identical to molpy; per-term params within bond r₀ 0.02 Å / angle θ₀ 3° / force const rtol 0.10. Iterate every file in `tests-data/opls/`; skip cleanly if absent.
- **Coverage** — fixture set must include wildcard-end dihedrals and overrides/layer-resolved atom types.
- **Build smoke** — fmt/clippy/check/test green.

## Out of scope

- **molpy Python 分型代码的删除/rewire** → molpy 侧 [opls-typifier-downsink](opls-typifier-downsink.md)。
- **CL&P/CL&Pol parity** — v2（layer 分支保留但不在本 fixture 集）。
- **性能基准** — parity 只验正确性；性能另议。
