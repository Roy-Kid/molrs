---
title: DistGeom bounds 矩阵 + ETKDGv3 知识 + smoothing (DistGeom Bounds)
status: code-complete
created: 2026-06-01
chain: mmff94-etkdg
chain_index: 3
depends_on: []
---

# DistGeom bounds 矩阵 + ETKDGv3 知识 + smoothing (DistGeom Bounds)

## Summary

在 `molrs-embed` 中**全新重建** RDKit 距离几何（DistGeom）的"约束生成"层，对齐
ETKDGv3：从分子拓扑构建上下距离界矩阵（bounds matrix），做三角 + 四点
（triangle + tetrangle）不等式 smoothing，并 port ETKDGv3 的**实验扭转偏好**
（experimental torsion，CrystalFF）、**basic-knowledge** 项（共轭体系平面性、
1-3 角项）与**小环/大环（14+ 元）专门扭转项**，外加手性/improper 约束的装配。

本 spec 只产出"分子 → bounds 矩阵 + 约束集合（扭转/手性/improper）"，不做嵌入与
最小化（属 `mmff94-etkdg-04-embed`）。它取代当前 `molrs-embed/src/distance_geometry.rs`
里自研的近似界（只有三角 smoothing、无实验扭转、无四点 smoothing）。

## Domain basis

- 距离几何 / bounds smoothing：Blaney & Dixon, "Distance Geometry in Molecular
  Modeling," *Rev. Comput. Chem.* **5**, 299 (1994)；Crippen & Havel,
  *Distance Geometry and Molecular Conformation* (1988)。
- ETKDG：Riniker & Landrum, "Better Informed Distance Geometry: Using What We
  Know To Improve Conformation Generation," *J. Chem. Inf. Model.* **55**,
  2562–2574 (2015). https://doi.org/10.1021/acs.jcim.5b00654
- ETKDGv3（小环/大环）：Wang, Witek, Landrum, Riniker, "Improving Conformer
  Generation for Small Rings and Macrocycles Based on Distance Geometry and
  Experimental Torsional-Angle Preferences," *J. Chem. Inf. Model.* **60**,
  2044–2058 (2020). https://doi.org/10.1021/acs.jcim.0c00025

参考实现（BSD-3，port 并署名）：

- `$RDBASE/Code/DistGeom/BoundsMatrix.h`、`TriangleSmooth.cpp`、`DistGeomUtils.cpp`
- `$RDBASE/Code/GraphMol/DistGeomHelpers/BoundsMatrixBuilder.cpp`
  （`setTopolBounds`、`set12Bounds`、`set13Bounds`、`set14Bounds`、`initBoundsMat`）
- `$RDBASE/Code/GraphMol/DistGeomHelpers/Embedder.cpp`（`getExperimentalTorsions`、
  ETKDGv3 参数 `ETKDGv3()`）
- `$RDBASE/Code/ForceField/CrystalFF/TorsionAngleM6.cpp`（实验扭转 M6 项）
- ETKDGv3 实验扭转 SMARTS/参数数据表（port 自 RDKit Data 与 CrystalFF）

## Design

### 模块布局

```
molrs-embed/src/distgeom/
├── mod.rs            // pub use；约束集合 DgConstraints
├── bounds.rs         // BoundsMatrix + 1-2/1-3/1-4 拓扑界
├── smooth.rs         // 三角 + 四点 smoothing
├── torsion_prefs.rs  // ETKDGv3 实验扭转表（SMARTS → V/multiplicity）
├── knowledge.rs      // basic-knowledge：共轭平面性 + 1-3 角项
└── chirality.rs      // 手性 + improper 约束
```

### 公共 API

```rust
// molrs-embed::distgeom

/// ETKDG 档位（本 spec 默认 v3）。
pub enum EtkdgVersion { Etdg, Etkdgv2, Etkdgv3 }

/// 对称的上下距离界矩阵（n×n）。
pub struct BoundsMatrix { /* lower/upper triangular storage */ }
impl BoundsMatrix {
    pub fn lower(&self, i: usize, j: usize) -> f64;
    pub fn upper(&self, i: usize, j: usize) -> f64;
}

/// 嵌入阶段消费的约束集合。
pub struct DgConstraints {
    pub bounds: BoundsMatrix,
    pub experimental_torsions: Vec<TorsionConstraint>,
    pub chiral: Vec<ChiralConstraint>,
    pub improper: Vec<ImproperConstraint>,
}

/// 主入口：分子 → 完整 ETKDGv3 约束。
pub fn build_constraints(mol: &MolGraph, version: EtkdgVersion)
    -> Result<DgConstraints, MolRsError>;

/// 三角 + 四点 smoothing；返回是否一致（无 lower>upper）。
pub fn smooth_bounds(bounds: &mut BoundsMatrix) -> Result<(), MolRsError>;
```

### 与现有代码的关系

- 新建 `distgeom/` 子模块；现有 `distance_geometry.rs` 的近似界生成在
  `mmff94-etkdg-04-embed` 中随旧 pipeline 一并退役。
- 复用 `molrs_core::rings::find_rings`（环 → 1-3/1-4 与环闭合界、大环判定）。
- 复用 `molrs_core::element::Element` 半径用于 vdW 下界。
- 实验扭转需要子结构匹配（SMARTS）；若 molrs 暂无 SMARTS 引擎，本 spec 内
  实现"够用子集"匹配器（仅覆盖 ETKDGv3 扭转模式所需的原子/键查询），并在
  docstring 标注其能力边界。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/distgeom/mod.rs` — (new) `DgConstraints`、`build_constraints`、`EtkdgVersion`
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/distgeom/bounds.rs` — (new) `BoundsMatrix` + 1-2/1-3/1-4 拓扑界
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/distgeom/smooth.rs` — (new) 三角 + 四点 smoothing
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/distgeom/torsion_prefs.rs` — (new) ETKDGv3 实验扭转参数表 + 匹配
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/distgeom/knowledge.rs` — (new) 共轭平面性 + 1-3 角项
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/distgeom/chirality.rs` — (new) 手性 + improper 约束
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/lib.rs` — (modify) `pub mod distgeom;` + re-export
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/tests/embed/distgeom.rs` — (new) 对照 RDKit bounds 矩阵 + smoothing 测试
- `/Users/roykid/work/molcrafts/molrs/molrs-python/src/lib.rs` — (modify) 暴露 `molecule_bounds_matrix(mol)` 给 Python

## Tasks

- [ ] Implement `BoundsMatrix` + 1-2/1-3/1-4 topological bounds in `distgeom/bounds.rs` (port `setTopolBounds`)
- [ ] Implement triangle + tetrangle smoothing in `distgeom/smooth.rs` (port `TriangleSmooth`)
- [ ] Embed ETKDGv3 experimental-torsion parameter tables + matcher in `distgeom/torsion_prefs.rs`
- [ ] Implement basic-knowledge terms (conjugated planarity, 1-3 angle terms) in `distgeom/knowledge.rs`
- [ ] Implement chiral + improper constraint assembly in `distgeom/chirality.rs`
- [ ] Wire `build_constraints` (ETKDGv3) in `distgeom/mod.rs`; update `lib.rs`
- [ ] Expose `molecule_bounds_matrix` via molrs-python; write RED tests comparing to RDKit `GetMoleculeBoundsMatrix`
- [ ] Run full check (`cargo test --all-features`, `cargo clippy -- -D warnings`)

## Testing strategy

### 对照 RDKit（科学验证）
- 一组分子（丁烷、环己烷、苯、联苯、甘氨酸、一个 12 元大环、一个含手性中心分子）
  的 bounds 矩阵上/下界与 RDKit `rdDistGeom.GetMoleculeBoundsMatrix(mol)` 逐元素
  差 < 1e-3（注意 RDKit 默认在 builder 内已做 smoothing；本 spec 比对 smoothing 后矩阵）。
- 实验扭转：联苯 / 正丁烷的扭转约束（中心键周围）的偏好角与 ETKDGv3 表一致。

### smoothing 正确性
- smoothing 后任意三元组满足三角不等式 `upper[i][j] ≤ upper[i][k]+upper[k][j]` 且
  `lower[i][j] ≥ |lower[i][k]−upper[k][j]|`，无 `lower > upper`。
- 人为注入不一致界 → `smooth_bounds` 报错或收敛到一致并标记。

### 手性
- 含一个 R 手性中心的分子，生成的手性约束体积符号与 CIP 指派一致。

## Out of scope

- 4D 嵌入、metrization 采样、坐标生成（属 04）。
- 一阶 ET 误差函数最小化与 MMFF 清理（属 04）。
- 通用 SMARTS 引擎（只实现 ETKDGv3 扭转所需子集）。
- `distance_geometry.rs` 的物理删除（在 04 随 pipeline 退役）。
