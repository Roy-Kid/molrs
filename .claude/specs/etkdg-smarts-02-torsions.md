---
title: ETKDGv3 完整实验扭转表 + 接入 (ETKDG Torsions)
status: code-complete
created: 2026-06-01
chain: etkdg-smarts
chain_index: 2
depends_on: [etkdg-smarts-01-engine]
---

# ETKDGv3 完整实验扭转表 + 接入 (ETKDG Torsions)

## Summary

把 RDKit 的三张 ETKDGv3 实验扭转偏好表（`torsionPreferences_v2.in` 378 行 +
`torsionPreferences_smallrings.in` 116 行 + `torsionPreferences_macrocycles.in`
379 行）完整 port 为内嵌数据，并用 `etkdg-smarts-01-engine` 的 SMARTS 引擎驱动匹配，
取代 `mmff94-etkdg-03-bounds` 里 `distgeom/torsion_prefs.rs` 的代表性子集。结果是
ETKDGv3 的实验扭转项逐键对齐 RDKit `getExperimentalTorsions`，从而让 `generate_3d`
对柔性分子的构象逼近 RDKit（闭合 `mmff94-etkdg-04-embed` 的 ac-001/ac-002）。

## Domain basis

- ETKDGv3：Wang, Witek, Landrum, Riniker, *J. Chem. Inf. Model.* **60**, 2044 (2020)。
- 扭转势形式（六项）：`V = Σ_{k=1..6} V_k·(1 + s_k·cos(k·φ))`。

参考实现（BSD-3，port 并署名）：

- `/Users/roykid/work/rdkit/Code/GraphMol/ForceFieldHelpers/CrystalFF/torsionPreferences_{v2,smallrings,macrocycles}.in`
  （数据；格式 `[SMARTS, s1,V1, s2,V2, …, s6,V6]`）
- `/Users/roykid/work/rdkit/Code/GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.cpp`
  （`getExperimentalTorsions`：版本选择=2、smallring/macrocycle 叠加、模式优先级 = 表中先匹配先用、
  环大小门槛 macrocycle≥9）
- `/Users/roykid/work/rdkit/Code/GraphMol/ForceFieldHelpers/CrystalFF/TorsionAngleM6.cpp`（M6 势）

## Design

### 数据与匹配

- `torsion_tables.rs`：三张表内嵌为 `&[(&str /*smarts*/, [Sign;6], [f64;6])]`（脚本从 `.in` 提取）。
- 每条模式在 build 时 `SmartsPattern::parse` 一次（可缓存）；对分子 `find_matches`，
  用 `:1..:4` 标签定位四原子，按 RDKit 规则（首个匹配该可旋转键的模式胜出）给每条
  可旋转键分配一组 `(V[6], s[6])`。
- macrocycle 模式仅用于环大小 ≥ 9 的环内键；smallring 模式用于小环；其余用 v2。

### 接入

- 重写 `molrs-embed/src/distgeom/torsion_prefs.rs`：删除代表性子集，改为消费上表 +
  SMARTS 引擎，输出与现有 `DgConstraints.experimental_torsions` 相同的结构（保持
  `build_constraints` 签名不变，`mmff94-etkdg-04-embed` 的 `etmin` 无需改）。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/distgeom/torsion_tables.rs` — (new) 三张内嵌表
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/distgeom/torsion_prefs.rs` — (modify) 用 SMARTS 引擎驱动，取代子集
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/Cargo.toml` — (modify, if needed) 已依赖 molrs(core)，SMARTS 引擎随之可用
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/tests/embed/torsions.rs` — (new) 对照 RDKit getExperimentalTorsions
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/tests/embed/etkdg.rs` — (modify) 收紧 ac-001 RMSD 断言（含丙氨酸 < 0.5 Å）

## Tasks

- [ ] Write an extractor script to port the 3 `.in` tables into `torsion_tables.rs` (SMARTS + 6×(sign,V))
- [ ] Rewrite `torsion_prefs.rs` to drive matching via `SmartsPattern` + per-bond first-match-wins selection
- [ ] Apply RDKit version/smallring/macrocycle layering rules (macrocycle ring-size >= 9)
- [ ] Write RED tests comparing per-bond torsion params to RDKit `getExperimentalTorsions` in `tests/embed/torsions.rs`
- [ ] Re-run ETKDG RMSD; tighten `etkdg.rs` ac-001 assertions (alanine < 0.5 Å)
- [ ] Run full check (`cargo test --all-features`, `cargo clippy -- -D warnings`)

## Testing strategy

### 对照 RDKit（科学验证）
- 对一组分子（丁烷、联苯、丙氨酸、甲基乙酸酯、酰胺、一个 12 元大环、一个小环体系），
  每条可旋转键分配到的扭转模式 SMARTS + `(V[6], s[6])` 与 RDKit
  `rdMolTransforms`/`getExperimentalTorsions`（经其调试接口或 `EmbedMolecule` 内部）一致；
  无 SMARTS 引擎残留的"子集 fallback"路径。
- 模式优先级：当多个模式匹配同一键，选择与 RDKit 相同（表内先到先得）。

### 端到端（闭合 spec 04）
- 重跑 `generate_3d` vs RDKit ETKDGv3 best-fit RMSD：乙醇/丁烷/苯保持 < 0.5 Å，
  **丙氨酸从 0.755 降到 < 0.5 Å**；构象 MMFF 能量与 RDKit 相对差 < 10%。

## Out of scope

- 改 `etmin` / 嵌入算法本身（仅替换扭转来源）。
- basic-knowledge 平面化项（已在 03 的 `knowledge.rs`）。
- ETKDG v1 表（只做 ETKDGv3 用的 v2 + smallring + macrocycle）。
