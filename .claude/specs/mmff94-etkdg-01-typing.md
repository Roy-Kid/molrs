---
title: MMFF94 原子分型 + 芳香性 + 参数表 + 电荷 (MMFF94 Typing)
status: code-complete
created: 2026-06-01
chain: mmff94-etkdg
chain_index: 1
depends_on: []
---

# MMFF94 原子分型 + 芳香性 + 参数表 + 电荷 (MMFF94 Typing)

## Summary

在 `molrs-ff` 中**全新重建（greenfield）** 一个忠实于 RDKit 的 MMFF94 前端模块
`molrs-ff/src/mmff/`，覆盖三件事：(1) 完整的 ~99 个 MMFF94 数值原子类型分配；
(2) MMFF94 **自有的芳香性感知模型**（molrs-core 当前没有任何芳香性感知，只在键级
1.5 上做约定）；(3) MMFF94 **自有的部分电荷模型**——基于 formal-charge 分摊 +
bond-charge-increment（BCI），取代当前 `molrs-ff` 里用 Gasteiger 电荷近似 MMFF 静电
的做法。本 spec 不实现能量项（见 `mmff94-etkdg-02-energy`），只产出"分子 → MMFF 原子
类型 + 部分电荷 + 完整参数表查询"这一层，等价于 RDKit 的 `MMFFMolProperties`。

现有 `molrs-ff/src/typifier/mmff/` 的近似实现整体退役（标记 `#[deprecated]` 并从
`Potential` 装配路径移除），由新的 `molrs-ff/src/mmff/` 取代。所有 MMFF94 参数表
（`MMFFSYMB / MMFFDEF / MMFFAROM / MMFFBOND / MMFFBNDK / MMFFANG / MMFFSTBN /
MMFFDFSB / MMFFOOP / MMFFTOR / MMFFVDW / MMFFCHG / MMFFPBCI / MMFFHDEF`）作为静态
内嵌数据 port 自 RDKit 的同名表。

## Domain basis

MMFF94 力场及其原子分型、芳香性模型、电荷模型定义于：

- Halgren, T. A. "Merck molecular force field. I. Basis, form, scope,
  parameterization, and performance of MMFF94." *J. Comput. Chem.* **17**,
  490–519 (1996). （以及 Part II–V，490–641）
- Halgren, T. A. "MMFF VI. MMFF94s option for energy minimization studies."
  *J. Comput. Chem.* **20**, 720–729 (1999).

参考实现（BSD-3，直接 port 并在模块 docstring 署名，沿用 `gasteiger.rs` 先例）：

- `$RDBASE/Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp`
  （`setMMFFAromaticity`、`setMMFFHeavyAtomType`、`setMMFFHydrogenType`、
  `computeMMFFCharges`）
- `$RDBASE/Code/ForceField/MMFF/Params.cpp` / `Params.h`（内嵌参数表）

### 芳香性模型（MMFF 专有，区别于 RDKit 默认芳香性）

MMFF94 在 SSSR 的 5/6 元环上独立判定芳香性：检测 π 电子数满足 4n+2、环内全部为可
参与共轭的原子类型，并据此把环原子重指派到芳香类型（如 C→CB/C5A/C5B、N→NPYL/N5A
等）。这一过程在 RDKit 中是 `setMMFFAromaticity`，与通用芳香性分开。

### 电荷模型（BCI）

MMFF 部分电荷：`q_i = q0_i + Σ_j ω_{ij}`，其中 `q0_i` 是 formal-charge 经
`MMFFPBCI` 分摊后的初始电荷，`ω_{ij}` 是键 `i–j` 的 bond-charge-increment（来自
`MMFFCHG`，缺失时回退到 `MMFFPBCI` 的 partial-bond-charge-increment 公式
`ω_{ij} = pbci_j − pbci_i`）。

## Design

### 模块布局

```
molrs-ff/src/mmff/
├── mod.rs            // pub use; MmffMolProperties 入口
├── tables.rs         // 全部静态参数表（port 自 RDKit Params.cpp）
├── aromaticity.rs    // setMMFFAromaticity 等价物
├── atomtype.rs       // 重原子 + 氢原子类型分配
└── charges.rs        // BCI 部分电荷
```

### 公共 API

```rust
// molrs-ff::mmff

/// MMFF94 / MMFF94s 变体开关。
pub enum MmffVariant { Mmff94, Mmff94s }

/// 分子的 MMFF 性质：每原子数值类型 + 部分电荷 + 参数表句柄。
/// 等价于 RDKit 的 MMFFMolProperties。
pub struct MmffMolProperties {
    atom_types: Vec<u8>,        // 每原子 MMFF 数值类型 (1..=99)
    partial_charges: Vec<f64>,  // 每原子 MMFF 部分电荷
    variant: MmffVariant,
}

impl MmffMolProperties {
    /// 对分子做：环感知 → 芳香性 → 原子分型 → 电荷。
    /// 失败（含不支持原子）返回 Err。
    pub fn compute(mol: &MolGraph, variant: MmffVariant)
        -> Result<MmffMolProperties, MolRsError>;

    pub fn atom_type(&self, idx: usize) -> u8;
    pub fn partial_charge(&self, idx: usize) -> f64;
    pub fn is_setup_complete(&self) -> bool;
}

/// 只读参数表查询（供 -02-energy 消费）。
pub fn mmff_tables() -> &'static MmffTables;
```

### 与现有代码的关系

- `molrs-ff/src/typifier/mmff/`：标记 `#[deprecated(note = "replaced by molrs_ff::mmff")]`，
  从默认 `Potential` 装配移除；保留编译以免破坏下游，下一个 minor 删除。
- 复用 `molrs_core::rings::find_rings`（SSSR）作为芳香性/分型的输入。
- 复用 `molrs_core::element::Element` 的 `atomic_number` / `by_symbol`。
- **不复用** `molrs_core::gasteiger`——MMFF 用自有 BCI 电荷。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/mod.rs` — (new) 模块入口、`MmffMolProperties`、`MmffVariant`
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/tables.rs` — (new) 全部 MMFF94 静态参数表
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/aromaticity.rs` — (new) MMFF 芳香性感知
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/atomtype.rs` — (new) 重原子 + 氢原子分型
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/charges.rs` — (new) BCI 部分电荷
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/lib.rs` — (modify) `pub mod mmff;` + re-export；`typifier::mmff` 加 `#[deprecated]`
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/tests/ff/mmff/typing.rs` — (new) 对照 RDKit 的分型/电荷集成测试
- `/Users/roykid/work/molcrafts/molrs/molrs-python/src/lib.rs` — (modify) 暴露 `compute_mmff_properties(mol)` 给 Python，用于 RDKit 对照

## Tasks

- [ ] Add MMFF94 aromaticity perception in `mmff/aromaticity.rs` (port `setMMFFAromaticity`)
- [ ] Embed full MMFF94 parameter tables in `mmff/tables.rs` (port from RDKit `Params.cpp`)
- [ ] Implement heavy-atom + hydrogen MMFF type assignment (all ~99 types) in `mmff/atomtype.rs`
- [ ] Implement BCI partial-charge model in `mmff/charges.rs` (formal-charge sharing + bond-charge increments)
- [ ] Wire `MmffMolProperties::compute` in `mmff/mod.rs`; deprecate `typifier::mmff`; update `lib.rs`
- [ ] Expose `compute_mmff_properties` via molrs-python for RDKit cross-validation
- [ ] Write RED tests in `tests/ff/mmff/typing.rs` comparing atom types + charges against RDKit
- [ ] Run full check (`cargo test --all-features`, `cargo clippy -- -D warnings`)

## Testing strategy

### 快乐路径 / 对照 RDKit
- 一组覆盖性分子（甲烷、乙烯、苯、吡啶、咪唑、苯胺、乙酰胺、硝基苯、苯磺酸、
  咖啡因）逐原子 MMFF 数值类型等于 RDKit `AllChem.MMFFGetMoleculeProperties(mol).GetMMFFAtomType(i)`。
- 同组分子逐原子 MMFF 部分电荷与 RDKit `GetMMFFPartialCharge(i)` 差 < 1e-4。
- 芳香性：苯/吡啶/咪唑的环原子被指派到 MMFF 芳香类型（CB / C5A / C5B / NPYL / N5A …）。

### 边界情况
- 含过渡金属/不支持原子 → `MmffMolProperties::compute` 返回明确 `Err`（不静默给错值），
  `is_setup_complete()` 为 false。
- 带 formal_charge 的离子（铵、羧酸根）→ 电荷总和等于体系净电荷（< 1e-6）。

### MMFF94 vs MMFF94s
- 至少一个 sp2 氮体系（如苯胺）验证 94 与 94s 变体的类型/电荷差异符合 RDKit。

## Out of scope

- 能量项与梯度（属 `mmff94-etkdg-02-energy`）。
- ETKDG / 构象生成（属 03、04）。
- UFF 力场。
- 删除 `typifier/mmff`（本 spec 只 deprecate，不删）。
