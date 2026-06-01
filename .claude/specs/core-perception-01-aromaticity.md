---
title: Atomistic 芳香性感知 (Core Aromaticity Perception)
status: approved
created: 2026-06-01
chain: core-perception
chain_index: 1
depends_on: []
---

# Atomistic 芳香性感知 (Core Aromaticity Perception)

## Summary

在 `molrs-core` 中实现一个**对齐 RDKit 默认芳香性模型**的通用芳香性感知，作为
`Atomistic` 的方法 `perceive_aromaticity(&mut self)`：环感知 + π 电子计数 + RDKit
芳香规则，给芳香原子置 `is_aromatic` 属性、给芳香键置键级 1.5。这是 molrs-core 当前
缺失的权威能力——现状把芳香性散落在 `molrs-ff/src/mmff/aromaticity.rs`（MMFF 专有模型）
与 `molrs-embed/src/distgeom/perceive.rs`（嵌入用的临时感知），且 `molrs_core::SmartsPattern`
的 `a`/`c`/`:` 原语此前只能靠测试里"移植 RDKit flag"才匹配得上。

入口放在 `Atomistic`（非裸 `MolGraph`）：感知需要 element 完整这一不变式，而 `Atomistic`
正是保证它的 newtype（`Deref<Target=MolGraph>`）。SMARTS 引擎读 `is_aromatic`/键级 1.5，
经 Deref 直接吃 `&Atomistic`，所以 `atomistic.perceive_aromaticity(); pattern.find_matches(&atomistic)`
即用。MMFF 的**专有**芳香性模型保持独立（定义不同，不并入本通用模型）。

## Domain basis

目标 = RDKit 默认芳香性模型 `AROMATICITY_RDKIT`：对每个 SSSR 环（及稠环并集）按
Hückel 4n+2 计 π 电子，原子可贡献电子数由其杂化/邻接/电荷/外部双键决定。

参考实现（BSD-3，port 语义并署名）：

- `/Users/roykid/work/rdkit/Code/GraphMol/Aromaticity.cpp`
  （`setAromaticity`、`applyHuckel`、`isAtomCandidate`、`getMinMaxAtomElecs`、稠环合并）
- 与 SMARTS `a`/`A`/`:` 的语义一致性以 RDKit `Atom::GetIsAromatic()` / `Bond::GetIsAromatic()` 为准。

## Design

### 模块布局

```
molrs-core/src/aromaticity.rs   // 通用芳香性感知（自由函数 + Atomistic 方法）
```

### 公共 API

```rust
// molrs-core::aromaticity
/// 感知并就地标注芳香性：芳香原子 set is_aromatic=true，芳香键 order=1.5。
/// 返回被标为芳香的原子数。RDKit AROMATICITY_RDKIT 模型。
pub fn perceive_aromaticity(mol: &mut MolGraph) -> usize;

// molrs-core::atomistic — 新增方法（薄封装，权威入口）
impl Atomistic {
    pub fn perceive_aromaticity(&mut self) -> usize;
}
```

### 与现有代码的关系

- 自由函数操作 `&mut MolGraph`；`Atomistic::perceive_aromaticity` 转调它（DerefMut）。
- 复用 `molrs_core::rings::find_rings`（SSSR）与 `Element`。
- 不改 `molrs-ff/mmff/aromaticity.rs`（MMFF 专有模型独立保留）。
- `molrs-embed/distgeom/perceive.rs` 之后可改为委托本函数（不在本 spec，属 chain 后续/清理）。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/aromaticity.rs` — (new) 感知算法
- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/atomistic.rs` — (modify) 加 `perceive_aromaticity` 方法
- `/Users/roykid/work/molcrafts/molrs/molrs-core/src/lib.rs` — (modify) `pub mod aromaticity;` + re-export
- `/Users/roykid/work/molcrafts/molrs/molrs-core/tests/core/aromaticity.rs` — (new) 对照 RDKit 的逐原子/逐键测试

## Tasks

- [ ] Implement ring π-electron counting + atom candidacy (port `getMinMaxAtomElecs`, `isAtomCandidate`) in `aromaticity.rs`
- [ ] Implement Hückel 4n+2 over SSSR rings + fused-ring union handling (port `applyHuckel`/`setAromaticity`)
- [ ] Write back `is_aromatic` atom prop + bond order 1.5 for aromatic atoms/bonds
- [ ] Add `Atomistic::perceive_aromaticity`; wire `pub mod aromaticity` + re-export in lib.rs
- [ ] Generate RDKit per-atom/per-bond aromatic-flag fixtures + write comparison tests
- [ ] Run full check (`cargo test --all-features`, `cargo clippy -- -D warnings`)

## Testing strategy

### 对照 RDKit（科学验证）
- 一组分子（苯、吡啶、吡咯、咪唑、呋喃、萘、吲哚、嘌呤/咖啡因、联苯、环己烷[非芳香]、
  环戊二烯[非芳香]、吡啶酮、苯酚、吡喃鎓）：molrs `perceive_aromaticity` 后每个原子的
  `is_aromatic` 与 RDKit `mol.GetAtomWithIdx(i).GetIsAromatic()` **逐原子一致**，每条键的
  芳香性与 RDKit `bond.GetIsAromatic()` 一致。
- 与 SMARTS 端到端：感知后用 `molrs_core::SmartsPattern` 跑 `a`/`c`/`[cX3]` 等芳香原语，
  匹配集合与 RDKit `GetSubstructMatches` 一致（确保感知模型与 SMARTS 语义吻合）。

### 边界
- 非芳香环（环己烷、1,3-环戊二烯）不被误标。
- 幂等：连续两次 `perceive_aromaticity` 结果一致。

## Out of scope

- MMFF 专有芳香性（`molrs-ff/mmff/aromaticity.rs` 不动）。
- 把 `distgeom/perceive.rs` 改为委托（属 chain 后续清理）。
- 其它芳香性模型（MDL / Simple）；只做 RDKit 默认模型。
