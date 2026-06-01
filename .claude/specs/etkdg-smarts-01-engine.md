---
title: SMARTS 子结构匹配引擎 (SMARTS Engine)
status: code-complete
created: 2026-06-01
chain: etkdg-smarts
chain_index: 1
depends_on: []
---

# SMARTS 子结构匹配引擎 (SMARTS Engine)

## Summary

在 `molrs-core` 中新建 `molrs-core/src/smarts/`：一个 SMARTS 子结构查询引擎
（解析器 + 子图同构匹配器），覆盖 ETKDGv3 实验扭转偏好表所需的全部 SMARTS 特性，
**包括递归 SMARTS `[$(...)]`**（RDKit 的 v2 / macrocycle 扭转表里各有 ~120 处）。
产出对照 RDKit `GetSubstructMatches` 逐分子等价的匹配结果，并保留 SMARTS 原子映射
（`:1`…`:4`），供 `etkdg-smarts-02-torsions` 把每条扭转模式落到具体的四原子。

这是 molrs 缺失的通用能力（`mmff94-etkdg-03/04` 因无 SMARTS 引擎只能用代表性子集，
导致柔性分子构象偏离 RDKit）。本 spec 只做引擎，不接扭转表。

## Domain basis

SMARTS 语言：Daylight SMARTS 规范
(https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html)。

参考实现（BSD-3，port 语义并署名）：

- `/Users/roykid/work/rdkit/Code/GraphMol/SmilesParse/SmartsParse.cpp`（SMARTS 文法）
- `/Users/roykid/work/rdkit/Code/GraphMol/Substruct/SubstructMatch.cpp`（VF2 风格匹配 + 递归 SMARTS 求值）
- `/Users/roykid/work/rdkit/Code/GraphMol/QueryAtom.h` / `QueryBond.h`（原子/键查询基元）

匹配语义以 RDKit `GetSubstructMatches(uniquify=False)` 为准（ETKDG 用非唯一化匹配）。

### 必须支持的 SMARTS 特性子集（由 ETKDG 三张表实测决定）

- **原子基元**：脂肪/芳香元素（大写 `C N O S ...` / 小写 `c n o ...`）、`*`、`a`、`A`、
  `#<n>`（原子序数）、`H<n>`（总氢数）、`X<n>`（连接数）、`D<n>`（显式度）、
  `R` / `R<n>`（环成员/环数）、`r<n>`（最小环大小）、`+` / `-` / `+<n>` / `-<n>`（形式电荷）、
  原子映射 `:<n>`。
- **原子逻辑**：隐式高优先级 `&`、低优先级 `;`、或 `,`、非 `!`。
- **递归 SMARTS** `[$(...)]`（可嵌套；以候选原子为根做子匹配）。
- **键基元**：`-` `=` `#` `:` `~` `@` `!@`，及其逻辑组合（如 `!@;-`、`-,:`）；
  相邻原子默认键为"单键或芳香键"。
- **结构**：分支 `( )`、环闭合数字（含双位 `%nn`）。

显式排除（ETKDG 表未用，超出本 spec）：手性 `@/@@`、原子量同位素查询、组件级
`( ).( )`、reaction SMARTS。

## Design

### 模块布局

```
molrs-core/src/smarts/
├── mod.rs       // 公共 API：SmartsPattern、match_*
├── ast.rs       // 解析后的查询 AST（QueryAtom / QueryBond / 逻辑节点）
├── parser.rs    // SMARTS 字符串 → 查询图（含递归 $()）
└── matcher.rs   // 子图同构（递归回溯 + 递归 SMARTS 求值）
```

### 公共 API

```rust
// molrs-core::smarts

/// 编译后的 SMARTS 查询。
pub struct SmartsPattern { /* query graph + 原子映射标签 */ }

impl SmartsPattern {
    /// 解析 SMARTS。语法错误返回 Err。
    pub fn parse(smarts: &str) -> Result<SmartsPattern, MolRsError>;

    /// 所有匹配（非唯一化），每个匹配是 query-atom-index → mol-AtomId 的映射，
    /// 顺序与 query 原子顺序一致（含 `:n` 标签可经 map_label() 取回）。
    pub fn find_matches(&self, mol: &MolGraph) -> Vec<Vec<AtomId>>;

    /// 是否存在至少一个匹配。
    pub fn has_match(&self, mol: &MolGraph) -> bool;

    /// query 原子序 → SMARTS 原子映射号（`:1` 等），无标签则 None。
    pub fn map_label(&self, query_atom: usize) -> Option<u32>;
}
```

### 与现有代码的关系

- 纯查询，只读 `MolGraph`；复用 `molrs_core::rings::find_rings`（`R`/`r` 基元）、
  `molrs_core::element::Element`、芳香性信息（键级 1.5 / 现有感知）。
- 不依赖 molrs-ff / molrs-embed。新增 `pub mod smarts;` 到 `molrs-core/src/lib.rs`。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/smarts/mod.rs` — (new) `SmartsPattern` 公共 API
- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/smarts/ast.rs` — (new) 查询 AST + 基元求值
- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/smarts/parser.rs` — (new) SMARTS 解析器（含递归）
- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/smarts/matcher.rs` — (new) 子图同构匹配器
- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/lib.rs` — (modify) `pub mod smarts;` + re-export `SmartsPattern`
- `/Users/roykid/work/molcrafts/molrs/molrs-core/tests/core/smarts.rs` — (new) 对照 RDKit 的匹配测试

## Tasks

- [ ] Implement query AST + atom/bond primitive evaluation against MolGraph in `smarts/ast.rs`
- [ ] Implement SMARTS parser (atoms, logical ops, bonds, branches, ring closures) in `smarts/parser.rs`
- [ ] Add recursive-SMARTS `$(...)` parsing + evaluation
- [ ] Implement subgraph-isomorphism matcher (backtracking, non-uniquified) preserving atom-map labels in `smarts/matcher.rs`
- [ ] Wire `SmartsPattern{parse,find_matches,has_match,map_label}` in `smarts/mod.rs`; update `lib.rs`
- [ ] Generate RDKit reference matches + write comparison tests in `tests/core/smarts.rs`
- [ ] Run full check (`cargo test --all-features`, `cargo clippy -- -D warnings`)

## Testing strategy

### 对照 RDKit（科学验证）
- 取 ETKDGv3 三张表里的一组代表性 SMARTS（含递归 `$()`、`!@;-`、`H<n>`、`X<n>`、
  芳香/脂肪、电荷）× 一组分子（丁烷、苯、吡啶、联苯、丙氨酸、咖啡因、一个酯、一个酰胺、
  一个大环），molrs `find_matches` 的匹配原子集合（按 `:n` 标签归一后）与 RDKit
  `mol.GetSubstructMatches(Chem.MolFromSmarts(p), uniquify=False)` **完全一致**（集合相等）。
- 解析鲁棒性：非法 SMARTS → `Err`，不 panic。

### 边界
- 递归 SMARTS 嵌套（`[$([CX3]=[OX1])]` 等）正确求值。
- 无匹配分子返回空；`has_match` 与 `find_matches` 一致。

## Out of scope

- 手性 / 同位素 / reaction SMARTS / 组件级查询。
- SMILES 写出、规范化、子结构唯一化去重（ETKDG 用非唯一化）。
- 性能优化（先正确；大规模索引留后续）。
