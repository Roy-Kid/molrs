---
title: MMFF94/MMFF94s 能量项 + 解析梯度 (MMFF94 Energy)
status: code-complete
created: 2026-06-01
chain: mmff94-etkdg
chain_index: 2
depends_on: [mmff94-etkdg-01-typing]
---

# MMFF94/MMFF94s 能量项 + 解析梯度 (MMFF94 Energy)

## Summary

在 `molrs-ff/src/mmff/` 中实现**真正的 MMFF94 函数形式**及其解析梯度，消费
`mmff94-etkdg-01-typing` 产出的 `MmffMolProperties`（原子类型 + 部分电荷 + 参数表），
装配成一个 `Potential`（`eval(&coords) -> (energy, gradient)`）。这是对现有
`molrs-ff` 用通用谐振键 / OPLS 二面角 / LJ 对势 + Gasteiger 电荷"冒充"MMFF 的彻底替换
——那套近似不是 MMFF94 的能量形式，本 spec 用 RDKit 一致的七项能量取代它。

七个能量贡献（MMFF94 标准形式）：

1. **Bond stretch**（四次/cubic-stretch）
2. **Angle bend**（三次，含近线性特例）
3. **Stretch-bend coupling**
4. **Out-of-plane bend**（Wilson 角）
5. **Torsion**（3 项 Fourier）
6. **van der Waals**（buffered 14-7）
7. **Electrostatic**（buffered Coulomb，用 -01 的 MMFF 电荷）

并提供 MMFF94s（static）变体开关（影响 OOP/共轭氮平面性等）。能量单位 kcal/mol，
坐标单位 Å。

## Domain basis

MMFF94 能量函数定义（Halgren, *J. Comput. Chem.* **17**, 490–641, 1996，Part I–V）：

- Bond stretch: `EB = 143.9325 · (kb/2) · Δr² · (1 + cs·Δr + 7/12·cs²·Δr²)`，
  `cs = −2 Å⁻¹`，`Δr = r − r0`。
- Angle bend: `EA = 0.043844 · (ka/2) · Δθ² · (1 + cb·Δθ)`，`cb = −0.007 deg⁻¹`；
  线性角（θ0=180°）用 `EA = 143.9325 · ka · (1 + cosθ)`。
- Stretch-bend: `ESB = 2.51210 · (kba_ijk·Δr_ij + kba_kji·Δr_kj) · Δθ`。
- Out-of-plane: `EOOP = 0.043844 · (koop/2) · χ²`（χ 为 Wilson 离面角）。
- Torsion: `ET = 0.5 · (V1(1+cosφ) + V2(1−cos2φ) + V3(1+cos3φ))`。
- vdW: buffered 14-7，`EvdW = εij · (1.07 R*ij/(Rij+0.07 R*ij))⁷ · (1.12 R*ij⁷/(Rij⁷+0.12 R*ij⁷) − 2)`。
- Electrostatic: `EQ = 332.0716 · qi qj / (D·(Rij + δ))`，`δ = 0.05 Å`（buffering），
  1-4 作用按 0.75 缩放。

参考实现（BSD-3，port 并署名）：

- `$RDBASE/Code/ForceField/MMFF/BondStretch.cpp`
- `$RDBASE/Code/ForceField/MMFF/AngleBend.cpp`
- `$RDBASE/Code/ForceField/MMFF/StretchBend.cpp`
- `$RDBASE/Code/ForceField/MMFF/OopBend.cpp`
- `$RDBASE/Code/ForceField/MMFF/TorsionAngle.cpp`
- `$RDBASE/Code/ForceField/MMFF/Nonbonded.cpp`
- `$RDBASE/Code/GraphMol/ForceFieldHelpers/MMFF/Builder.cpp`（项的枚举与 1-4 掩码）

## Design

### 模块布局

```
molrs-ff/src/mmff/
├── energy/mod.rs        // MmffForceField: 装配 + Potential 实现
├── energy/bond.rs       // bond stretch + grad
├── energy/angle.rs      // angle bend + grad（含线性特例）
├── energy/stretchbend.rs
├── energy/oop.rs        // Wilson 离面 + grad
├── energy/torsion.rs    // 3 项 Fourier + grad
└── energy/nonbonded.rs  // buffered-14-7 vdW + buffered Coulomb + 1-4 掩码
```

### 公共 API

```rust
// molrs-ff::mmff

/// 装配好的 MMFF94 力场，绑定一个分子的拓扑 + MmffMolProperties。
pub struct MmffForceField { /* 项列表、1-4 掩码、电荷 */ }

impl MmffForceField {
    /// 从分子 + MMFF 性质构建全部能量项。
    pub fn build(mol: &MolGraph, props: &MmffMolProperties)
        -> Result<MmffForceField, MolRsError>;

    /// 逐项能量分解（用于对照 RDKit 调试）。
    pub fn energy_terms(&self, coords: &[f64]) -> MmffEnergyBreakdown;
}

/// 实现既有 Potential trait：eval -> (energy_kcal_per_mol, gradient)。
impl Potential for MmffForceField {
    fn eval(&self, coords: &[f64]) -> (f64, Vec<f64>);
}

pub struct MmffEnergyBreakdown {
    pub bond: f64, pub angle: f64, pub stretch_bend: f64,
    pub oop: f64, pub torsion: f64, pub vdw: f64, pub electrostatic: f64,
    pub total: f64,
}
```

### 与现有代码的关系

- 实现 `molrs_ff::potential::Potential`（`eval(&[f64]) -> (f64, Vec<f64>)`），与现有
  优化器/接口兼容。坐标布局为扁平 `[x0,y0,z0,x1,...]`。
- `mmff94-etkdg-04-embed` 的二阶清理直接 `build` 本力场并交给最小化器。
- 旧的 `potential/bond|angle|dihedral|improper|pair` 中专为"mmff94 别名"走的分支退役。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/energy/mod.rs` — (new) `MmffForceField`、`Potential` 实现、breakdown
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/energy/bond.rs` — (new) 四次键伸缩 + 梯度
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/energy/angle.rs` — (new) 三次角弯曲 + 线性特例 + 梯度
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/energy/stretchbend.rs` — (new) stretch-bend 耦合 + 梯度
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/energy/oop.rs` — (new) Wilson 离面 + 梯度
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/energy/torsion.rs` — (new) 3 项 Fourier 扭转 + 梯度
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/energy/nonbonded.rs` — (new) buffered-14-7 vdW + buffered Coulomb + 1-4 掩码
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/src/mmff/mod.rs` — (modify) re-export energy API
- `/Users/roykid/work/molcrafts/molrs/molrs-ff/tests/ff/mmff/energy.rs` — (new) 对照 RDKit 能量 + 有限差分梯度测试
- `/Users/roykid/work/molcrafts/molrs/molrs-python/src/lib.rs` — (modify) 暴露 `mmff_energy(mol, coords)` 给 Python

## Tasks

- [x] Implement bond stretch (quartic) energy + analytical gradient in `energy/bond.rs`
- [x] Implement angle bend (cubic + linear special case) energy + gradient in `energy/angle.rs`
- [x] Implement stretch-bend coupling energy + gradient in `energy/stretchbend.rs`
- [x] Implement out-of-plane (Wilson) bend energy + gradient in `energy/oop.rs`
- [x] Implement 3-term Fourier torsion energy + gradient in `energy/torsion.rs`
- [x] Implement buffered-14-7 vdW + buffered Coulomb + 1-4 scaling in `energy/nonbonded.rs`
- [x] Assemble `MmffForceField` (build + Potential + breakdown) in `energy/mod.rs`; MMFF94/94s switch carried via `MmffMolProperties::variant()` (note: the Oop/Tor s-variant tables are not yet ported, so 94 and 94s share the Oop/Tor set for now — same caveat as the typing layer)
- [x] Write RED→GREEN tests in `tests/ff/mmff/energy.rs` (RDKit total + per-term breakdown + central-difference FD gradient + invariance + timing). NOTE: `mmff_energy` Python binding deferred — out of scope for this job (constraint: do not touch molrs-python).
- [x] Run full check: `cargo test -p molcrafts-molrs-ff` (143 passed) + `cargo clippy` (energy/ clean; one pre-existing `tables.rs` test warning left untouched per constraint)

## Testing strategy

### 对照 RDKit（科学验证）
- 一组分子（乙烷、乙烯、苯、丁烷的多个构象、咖啡因），用 RDKit 嵌入得到的坐标喂给
  molrs，**总能量**与 RDKit `MMFFGetMoleculeForceField(mol).CalcEnergy()` 差 < 1e-3 kcal/mol。
- **逐项分解**：bond/angle/stretch-bend/oop/torsion/vdW/electrostatic 各项与 RDKit
  分项（通过其 contrib 接口或独立小体系）一致，单项差 < 1e-3。

### 梯度正确性
- 对随机扰动坐标，解析梯度与中心差分（步长 1e-5 Å）一致，max abs error < 1e-5。
- `−gradient` 即力；平移整体不改变能量（不变性检查）。

### 性能（防退化）
- N≈50 原子分子单次 `eval` 在记录基线内（非键项不得退化为未截断的纯 O(N²) 主导
  导致超阈值）；记录基线数值，回归 > 20% 报警。

## Out of scope

- 原子分型 / 芳香性 / 电荷（属 -01）。
- 构象嵌入 / ETKDG / 最小化器（属 04）。
- 周期性边界、静电的 Ewald/PME（MMFF94 用直接 buffered Coulomb）。
- 删除旧 `potential/*` 通用核（仅退役 mmff94 别名分支）。
