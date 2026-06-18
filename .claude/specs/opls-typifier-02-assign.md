---
title: OPLS-AA 成键参数匹配下沉 molrs（specificity + 通配 + layer 排序）
status: approved
created: 2026-06-18
---

# OPLS-AA 成键参数匹配下沉 molrs

> OPLS typifier 链 2/3。依赖 [opls-typifier-01-typing](opls-typifier-01-typing.md)（提供 typed `Atomistic`：原子带 `type`/`class`）。**完成 [opls-ef-01-kernels-seam](opls-ef-01-kernels-seam.md) "typifier 不下沉（B线）" 的反转**：OPLS 成键参数赋值不再在 molpy SMARTS-typify 后由 Python 完成，而在 Rust 内闭环。

## Summary

把 OPLS-AA 的**成键参数匹配**（给定原子 type→class，从 ForceField 的 bond/angle/dihedral 表择最匹配项）从 molpy（`typifier/atomistic.ForceField{Bond,Angle,Dihedral}Typifier`）下沉到 molrs。势能参数表已在 Rust ForceField 里（`read_opls_xml` 产出 bond/angle harmonic、dihedral opls，按 class 键控，单位已换算、RB→OPLS 已做）。但 ForceField 的 `get_bondtype(c1,c2)` 是**精确键 + 对称**、**无通配/无特异性排序**（`ff/forcefield/mod.rs:254`）；OPLS 大量用通配端（`X-CT-CT-X`）与 class/type 两级匹配，故需在 typifier 内实现 molpy `_sequence_score`/`_end_score` 的特异性排序匹配器。匹配到 → 写参数；匹配不到 → 交 [ff-parameter-estimator](ff-parameter-estimator.md)（链共享）。

## Domain basis

molpy 的成键匹配（`typifier/atomistic.py:25-95,141-321`，本 spec 复刻）：

- **特异性 `_end_score`**：单端 pattern 对单原子——精确 type 匹配=3，class 匹配=1，通配 `X`/`*`=0，不匹配=None。
- **`_sequence_score`**：对成键项的 pattern 元组双向（forward + reversed，成键项端对端对称）求 `_end_score` 之和；任一端 None 则整体 None。
- **排序键 `(score, layer)`**：所有候选里取 score 最高；并列时取 layer 最高（CL&P/CL&Pol overlay 压 OPLS-AA）。`layer` 由 atomtype 的 `layer` 属性经 `_build_type_class_layer` 解析（链 1/3 的 typing-meta 已携带 layer；本 spec 复刻 class→layer 映射）。

OPLS 二面角用 OPLS 4-cosine（`dihedral:opls`，f1..f4，已由 reader 从 RB 转换）；bond/angle 用 harmonic（k0/r0、k0/theta0）。匹配产出的参数须按目标 Style 的 `params` 形状写回。

## Design

**1. 候选表与映射（复用，不从 XML 重建）**

`OplsTypifier` 在构造期从 `self.ff` 抽出按 class 键控的成键候选表：遍历 `get_style("bond","harmonic").defs`（`StyleDefs::Bond(Vec<BondType>)`）、angle harmonic、dihedral opls，得 `[(class 元组, &Type)]`（镜像 molpy `_bond_table`/`_angle_table`/`_dihedral_table`）。type→class 与 class→layer 映射由链 1/3 的 `OplsTypingMeta` 提供（复刻 `_build_type_class_layer`）。

**2. 特异性排序匹配器（typifier 内，非 ForceField）**

新增 `ff/typifier/opls/assign.rs`：`_end_score`/`_sequence_score` 的 Rust 实现（纯函数，单测覆盖 3/1/0/None + 双向对称）。`assign_bonded(mol_typed, &ff, &meta)`：对 typed `Atomistic` 枚举的每个 bond/angle/dihedral，解析端点 class，扫候选表取 `(score, layer)` 最大者，写回该 Type 的 `params` 到该项（labeled `Atomistic` 的成键 prop，对齐 molpy `term.data.update(**type.params.kwargs)`）。**不**改 ForceField 的 `get_bondtype`（保持精确语义；特异性匹配是 typifier 的职责）。

**3. no-match 钩子（estimator 接缝）**

当某项无任何特异性匹配（score 全 None）时：若 estimator 启用 → 调 [ff-parameter-estimator](ff-parameter-estimator.md) 的 `estimate_{bond,angle,dihedral}`；否则按 strict 行为（strict=true → `Err`；strict=false → 原样返回无参项）。这是与 GAFF 链 [gaff-typifier-03-assign](gaff-typifier-03-assign.md) 共享的同一接缝语义。

**4. 闭环到 potentials**

`OplsTypifier::build(mol)`（镜像 `MMFFTypifier::build`，`ff/typifier/mmff/mod.rs:85`）：`typify`（链 1/3 原子分型）→ `assign_bonded`（本 spec）→ labeled `Atomistic` → `to_frame()` → `ff.to_potentials(frame)`。1-2/1-3 排除 + 1-4 缩放由 ForceField 的 `special_bonds`（reader 已设 `[0,0,0.5]`）+ consumer 的 `intramolecular_pairs` 处理（[ff-special-bonds-nblist](ff-special-bonds-nblist.md)）。

## Files to create or modify

- `molrs/src/ff/typifier/opls/assign.rs` (new) — `_end_score`/`_sequence_score` + `assign_bonded` + estimator 接缝。
- `molrs/src/ff/typifier/opls/mod.rs` — `typify` 后串 `assign_bonded`；加 `build()`；候选表构造。
- `molrs/tests/ff/typifier/opls.rs` — 成键匹配集成测试（真实分子）。

## Tasks

- [ ] Write failing unit tests for _end_score / _sequence_score (3/1/0/None, forward+reversed symmetry) in molrs/src/ff/typifier/opls/assign.rs
- [ ] Implement _end_score / _sequence_score in molrs/src/ff/typifier/opls/assign.rs
- [ ] Write failing tests for assign_bonded: exact-vs-wildcard specificity winner + layer tiebreak + params written in target Style shape
- [ ] Implement candidate-table build (bond/angle/dihedral by class) and assign_bonded with (score, layer) ranking in molrs/src/ff/typifier/opls
- [ ] Write failing tests for no-match seam (strict=true → Err; strict=false + no estimator → unparam'd term)
- [ ] Wire the no-match seam to ff-parameter-estimator (optional, opt-in) and add OplsTypifier::build (typify→assign→to_frame→to_potentials)
- [ ] Add rustdoc; note the B-line reversal of opls-ef-01-kernels-seam
- [ ] Run cargo fmt --all --check && cargo clippy -- -D warnings && cargo test --all-features

## Testing strategy

- **Unit** — `_end_score`/`_sequence_score`: exact 3 / class 1 / wildcard 0 / None; reversed-orientation match.
- **Unit** — ranking: a fully-resolved pattern beats a wildcard one; equal score → higher layer wins.
- **Integration (real data)** — `assign_bonded` over typed molecules from `tests-data/`; every bond/angle/dihedral gets params matching molpy `OplsTypifier` within tolerance (full parity = chain 3/3).
- **Edge** — no-match seam: strict=true Err; strict=false returns unparam'd term when estimator absent.
- **Build smoke** — fmt/clippy/check/test green.

## Out of scope

- **缺参估计算法本体** → [ff-parameter-estimator](ff-parameter-estimator.md)（本 spec 只接缝）。
- **原子分型** → 链 1/3。
- **improper 匹配** — OPLS improper 少见；v1 仅 bond/angle/dihedral，improper 随后。
- **pair/charge 赋值** — charge 在链 1/3 由 atom style 解析；pair 组合规则是 kernel 的事（[ff-special-bonds-nblist](ff-special-bonds-nblist.md)）。
- **改 ForceField `get_bondtype` 语义** — 保持精确，特异性匹配在 typifier。
