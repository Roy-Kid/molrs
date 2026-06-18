---
title: OPLS-AA 分层依赖原子分型引擎（%opls_NNN）下沉 molrs
status: approved
created: 2026-06-18
---

# OPLS-AA 分层依赖原子分型引擎（%opls_NNN）下沉 molrs

> OPLS typifier 链的分型扩展。**执行顺序**：在 [opls-typifier-02-assign](opls-typifier-02-assign.md) 之后、[opls-typifier-03-parity](opls-typifier-03-parity.md) 之前（编号 04 仅表撰写次序）。补齐 [opls-typifier-01-typing](opls-typifier-01-typing.md) 跳过的 **133 个 `%opls_NNN` 依赖型 def**（157 个 def 中约 100 个）——chain-3 全量 per-atom parity 的硬前置。

## Summary

molpy 的 OPLS 分型不是单遍 SMARTS：133 个原子类型的 `def` 含 `%opls_NNN` 记号（如 `opls_146` 的 `[H][C;%opls_145]`），意为"该邻居原子在**前序迭代**中已被赋型 `opls_NNN`"。molpy 用 `DependencyAnalyzer`（Kahn 拓扑分层 + Tarjan SCC 识别环）+ `LayeredTypingEngine`（逐层匹配 + 环组定点迭代）+ 匹配期消费"当前每原子赋型表"的谓词来解析。chain-1 只覆盖 56 个独立 SMARTS、跳过 `%`-def。本 spec 把这套分层依赖引擎下沉 Rust：(1) molrs SMARTS 加一个**通用的"外部原子标签"上下文谓词**解析 `%LABEL` + 匹配期查上下文标签表；(2) `ff/typifier/opls` 加依赖分析（Kahn+Tarjan）+ 分层定点驱动；(3) chain-1 的 `annotate_opls` 退化为本引擎的 level-0 情形。结果：覆盖全部 def，达成与 molpy 的 per-atom parity。

## Domain basis

molpy 实现（本 spec 复刻，已勘查定位）：

1. **`%opls_NNN` 语义** — `src/molpy/typifier/graph.py:428-441`（`_atom_primitive_matches` 的 `has_label` 分支）：`[X;%opls_NNN]` 命中当且仅当该原子的当前赋型 `atomtype == "opls_NNN"`。SMARTS 解析把 `%opls_NNN` 转为 `has_label` 原语（`parser.smarts`）。
2. **依赖提取** — `graph.py:223-254`（`extract_dependencies`）：扫 def 的 `has_label` 原语，剥 `%` 收集被引类型名集合。
3. **依赖分层** — `dependency_analyzer.py:42-95`（`_compute_levels`，Kahn 算法）：无依赖→level 0，仅依赖 level-0→level 1，……；`_detect_circular_groups`（Tarjan SCC，`:97-141`）把环组并到 `max_level+1`。
4. **分层驱动** — `layered_engine.py:55-180`：按 level 顺序匹配；普通层一遍（`_resolve_level`，把当前赋型写到图顶点供谓词读，`:120-126`），环层定点迭代（`_resolve_circular`，至多 `max_iterations=10`，`assignments == prev` 收敛，`:145-180`）。
5. **冲突解析** — 每原子多命中按 priority 排序取胜（layer ≫ overrides ≫ specificity，复刻自 chain-1 的 `_OplsAtomTypifier` 算法，`atomistic.py:452-528`）。跨迭代已赋型原子不变，只新增赋型 / 环层重匹配。

参考分子（必须正确分型）：苯 `opls_145`(芳C)→`opls_146`(芳H 邻 `%opls_145`)；甲苯 `opls_145`→`opls_148`(CH₃)；醇 `opls_154`(O)→`opls_155`(H 邻 `%opls_154`)；三醇 `opls_173/174`→`opls_171`。

## Design

**1. molrs SMARTS：通用上下文标签谓词（core 扩展，additive）**

在 `core/chem/smarts` 加一个**领域无关**的原子查询变体（如 `AtomQuery::HasContextLabel(String)`）+ 解析 `%LABEL` 记号 + 匹配期通过 `MolContext` 携带一个 `&HashMap<AtomId, String>`（"当前外部标签表"），谓词为 `ctx.label(atom) == Some(label)`。镜像现有 `RecursiveEvaluator` 经 context 求值递归子模式的方式（`matcher.rs:51-68`）。**通用、非 OPLS 专属**——OPLS 层只是把"当前赋型表"作为该标签表传入。`find_matches` 增一个带上下文的入口（或 `SmartsPattern::find_matches_with_labels(mol, &labels)`），旧入口不变（空标签表）。

**2. `ff/typifier/opls/deps.rs` — 依赖分析**

`OplsDependencyAnalyzer`：对每个 `OplsTypeRow` 的 def 提取 `%opls_NNN` 依赖集（解析期已得 `HasContextLabel` 列表）；Kahn 拓扑分层（`Vec<Vec<type_name>>` by level）；Tarjan SCC 识别环组并入 `max_level+1`。镜像 `dependency_analyzer.py`。

**3. `ff/typifier/opls/layered.rs` — 分层定点驱动**

`LayeredTypingEngine`：按 level 顺序，对每层用 chain-1 的匹配 + priority 解析（消费当前赋型表作为标签上下文）；普通层一遍，环层定点迭代（至多 N=10，赋型表相等即收敛）。每轮把新赋型并入 `HashMap<AtomId,String>`，下一层/下一轮匹配时作为标签上下文。

**4. 整合 chain-1**

chain-1 的 `annotate_opls`（独立 SMARTS 分型）退化为本引擎的 level-0 路径；`OplsTypifier::typify` 改为调 `LayeredTypingEngine` 驱动全流程。priority/overrides/layer 解析复用不变。chain-2 的 `assign_bonded` 不受影响（仍消费 typed atoms 的 `class`，现在覆盖全部原子）。

## Files to create or modify

- `molrs/src/core/chem/smarts/{parser.rs,ast.rs,matcher.rs,mod.rs}` — 加 `%LABEL` 解析 → `AtomQuery::HasContextLabel`；matcher 经 context 查标签表；`find_matches_with_labels` 入口（旧入口不变）。
- `molrs/src/ff/typifier/opls/deps.rs` (new) — `OplsDependencyAnalyzer`（提取依赖 + Kahn 分层 + Tarjan SCC）。
- `molrs/src/ff/typifier/opls/layered.rs` (new) — `LayeredTypingEngine`（分层 + 定点迭代）。
- `molrs/src/ff/typifier/opls/typing.rs` — `annotate_opls` 整合为引擎驱动（level-0 + 多层）。
- `molrs/src/ff/typifier/opls/mod.rs` — `OplsTypifier::typify` 走 LayeredTypingEngine。
- `molrs/tests/ff/typifier/opls.rs` + smarts 单测 — 依赖/分层/定点/真实分子全覆盖分型。

## Tasks

- [ ] Write failing tests for SMARTS %LABEL parse → AtomQuery::HasContextLabel and context-label matcher predicate (find_matches_with_labels) in core/chem/smarts
- [ ] Implement %LABEL parsing + HasContextLabel variant + context-label predicate (additive; old find_matches entry unchanged) in molrs/src/core/chem/smarts
- [ ] Write failing tests for OplsDependencyAnalyzer (dep extraction, Kahn levels, Tarjan SCC circular groups) matching molpy fixtures
- [ ] Implement OplsDependencyAnalyzer in molrs/src/ff/typifier/opls/deps.rs
- [ ] Write failing tests for LayeredTypingEngine (level-by-level + fixed-point convergence for circular groups)
- [ ] Implement LayeredTypingEngine in molrs/src/ff/typifier/opls/layered.rs
- [ ] Write failing tests for full layered typing on real molecules (benzene opls_145→146, toluene→148, alcohol 154→155, triol 171) covering the previously-skipped %-defs
- [ ] Integrate annotate_opls/OplsTypifier::typify to drive via LayeredTypingEngine; ensure chain-1 standalone-SMARTS path is the level-0 case
- [ ] Add rustdoc; document the general (non-OPLS) context-label SMARTS extension
- [ ] Run cargo fmt --all --check && cargo clippy -- -D warnings && cargo test (ff feature set)

## Testing strategy

- **Unit** — `%LABEL` parse → `HasContextLabel`; matcher predicate true iff context label matches; old `find_matches` unaffected (empty context).
- **Unit** — dependency extraction (`%opls_NNN` → dep set); Kahn levels (0/1/2 chain); Tarjan SCC on a constructed A→B→A cycle → one circular group at max_level+1.
- **Unit** — `LayeredTypingEngine`: level ordering; circular fixed-point converges (assignment equality) within max_iterations.
- **Integration (real data)** — full typing over `tests-data/` molecules requiring `%`-deps (benzene/toluene/alcohol/triol): per-atom types match molpy `OplsTypifier`; coverage now includes the 133 `%`-defs.
- **Build smoke** — fmt/clippy/test green (ff feature set; literal `--all-features` may stay blocked by unrelated compute WIP — note as in chain-1/2).

## Out of scope

- **成键参数匹配 / 估计 / parity 验证台** → 02-assign / [ff-parameter-estimator](ff-parameter-estimator.md) / 03-parity。
- **把 `%LABEL` 做成 OPLS 专属** — 故意做成通用 context-label SMARTS 扩展（GAFF/未来力场可复用）。
- **molpy 侧删除/rewire** → molpy [opls-typifier-downsink](../../../molpy/.claude/specs/opls-typifier-downsink.md)（消费 spec）。
- **环依赖的语义改良** — 复刻 molpy 的 max_iterations 定点行为，不引入新的环消解策略。
