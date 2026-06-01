---
title: SMARTS 环大小区间 + 环连接度原语 (SMARTS Ring Queries)
status: approved
created: 2026-06-01
chain: core-perception
chain_index: 2
depends_on: [etkdg-smarts-01-engine]
---

# SMARTS 环大小区间 + 环连接度原语 (SMARTS Ring Queries)

## Summary

给 `molrs_core::smarts` 引擎补两个 RDKit SMARTS 原语：**环大小区间** `r{lo-hi}` /
`r{lo-}` / `r{-hi}`（ETKDGv3 macrocycle 扭转模式逐条都用，如 `r{9-}`）与**环键连接度**
`x<n>`（处于 n 条环键中的原子）。这消除 `etkdg-smarts-02-torsions` 在
`molrs-embed/src/distgeom/torsion_prefs.rs` 里临时"strip 这些 token + 匹配后再 post-check"
的 shim，让全部约束回到 SMARTS 引擎本体表达。

## Domain basis

Daylight/RDKit SMARTS 环原语：
- `r<n>`（已支持）：在大小为 n 的环中。
- `r{lo-hi}`：在大小落在 [lo,hi] 的某个环中（RDKit `RANGE` 查询）。
- `x<n>`：ring-bond connectivity = 该原子参与的环键数 == n。

参考实现（BSD-3，port 语义并署名）：

- `/Users/roykid/work/rdkit/Code/GraphMol/SmilesParse/SmartsParse.cpp`（`r{}` 区间与 `x` 文法）
- `/Users/roykid/work/rdkit/Code/GraphMol/QueryOps.cpp`（`makeAtomInRingOfSizeQuery` 区间、`makeAtomRingBondCountQuery`）

## Design

- `parser.rs`：词法/文法支持 `r{lo-hi}` / `r{lo-}` / `r{-hi}` 和 `x<n>`，产出对应原语 AST 节点。
- `ast.rs`：求值——`r{}` 用 `RingInfo`（原子是否在大小处于区间的环中，对齐 RDKit 语义：
  存在某环大小 ∈ 区间）；`x<n>` = 该原子的环键计数（邻接键中 `is_bond_in_ring` 为真的数目）。
- 复用 `molrs_core::rings::RingInfo`（`ring_sizes` / `is_bond_in_ring` / 每原子环大小集合）。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/smarts/parser.rs` — (modify) 解析 `r{}` 区间 + `x<n>`
- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/smarts/ast.rs` — (modify) 两个原语的求值
- `/Users/roykid/work/molcrafts/molrs/molrs-core/tests/core/smarts.rs` — (modify) 对照 RDKit 的区间/连接度用例

## Tasks

- [ ] Parse `r{lo-hi}` / `r{lo-}` / `r{-hi}` ring-size-range primitive in `parser.rs`
- [ ] Parse `x<n>` ring-bond-connectivity primitive in `parser.rs`
- [ ] Evaluate ring-size-range against RingInfo (atom in some ring with size in range) in `ast.rs`
- [ ] Evaluate `x<n>` (count of incident ring bonds) in `ast.rs`
- [ ] Add RDKit-compared tests for both primitives (incl. real macrocycle patterns) in `tests/core/smarts.rs`
- [ ] Run full check (`cargo test --all-features`, `cargo clippy -- -D warnings`)

## Testing strategy

### 对照 RDKit（科学验证）
- 模式（取自 ETKDG macrocycle 表，含 `r{9-}`、`x2`、`x3`）× 分子（环己烷、12 元大环、
  萘[稠环]、立方烷状/螺环若干、链状）：molrs `find_matches` 与 RDKit
  `GetSubstructMatches(uniquify=False)` 集合一致。
- `r{9-}` 在 6 元环上不匹配、在 12 元环上匹配；`x2` 命中单环原子、`x3` 命中稠环桥原子。

### 回归（去 shim 前置）
- 现有 189/189 SMARTS 用例保持全过（新增原语不破坏既有匹配）。

## Out of scope

- `torsion_prefs.rs` 的 shim 删除本身（molrs-embed，本 spec 落地后作为行为保持清理提交，
  由现有 `tests/embed/torsions.rs` 守护）。
- 其它未用 SMARTS 特性（手性、同位素等）。
