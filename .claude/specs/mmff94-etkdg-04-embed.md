---
title: 4D 嵌入 + ET 最小化 + MMFF 清理 + 重试 pipeline (ETKDG Embed)
status: code-complete
created: 2026-06-01
chain: mmff94-etkdg
chain_index: 4
depends_on: [mmff94-etkdg-02-energy, mmff94-etkdg-03-bounds]
---

# 4D 嵌入 + ET 最小化 + MMFF 清理 + 重试 pipeline (ETKDG Embed)

## Summary

把 `mmff94-etkdg-03-bounds` 的约束与 `mmff94-etkdg-02-energy` 的 MMFF94 力场组装成
完整的 ETKDGv3 构象生成 pipeline，并以此**重写 `generate_3d`**。流程对齐 RDKit
`EmbedMolecule`：

1. **4D 距离几何嵌入**：metrization 采样从 bounds 取一个距离矩阵 → 度量矩阵
   特征分解 → 在 4 维生成初始坐标（第 4 维抑制手性陷阱）。
2. **一阶 ET 最小化**：用距离违约（`DistViolationContrib`）+ 手性违约
   （`ChiralViolationContrib`）+ 实验扭转项（CrystalFF M6）构成误差函数，最小化
   并投影回 3D。
3. **二阶 MMFF94 清理**：构建 `MmffForceField`（消费 -02）做最终能量最小化。
4. **手性强制 + stereo 检查**：校验四面体/双键立体未反转。
5. **失败重试循环**：对应 RDKit `maxIterations`——单次嵌入不一致/手性错误/收敛失败
   时换随机种子重试，必要时 `useRandomCoords` 兜底；全失败才返回 Err。

旧实现（`builder.rs` 的 FragmentRules、`optimizer.rs` 的玩具能量模型、
`distance_geometry.rs` 的近似界、`rotor_search.rs`、`fragment_data.rs`）整体退役删除。
`generate_3d` 函数签名与 `EmbedReport` 结构保留；`EmbedOptions` / `EmbedAlgorithm`
改造为 ETKDG 参数。

## Domain basis

- ETKDG / ETKDGv3：见 `mmff94-etkdg-03-bounds` 的参考文献（Riniker & Landrum 2015；
  Wang et al. 2020）。
- 4D 嵌入与误差函数最小化：Crippen & Havel (1988)；Blaney & Dixon (1994)。

参考实现（BSD-3，port 并署名）：

- `$RDBASE/Code/GraphMol/DistGeomHelpers/Embedder.cpp`
  （`EmbedMolecule`、`embedPoints`、`_minimizeWithExpTorsions`、`_generateInitialCoords`、
  重试与 `useRandomCoords` 逻辑、`maxIterations` 默认）
- `$RDBASE/Code/DistGeom/DistGeomUtils.cpp`（`computeInitialCoords`、`embedConfsFromDistMat`）
- `$RDBASE/Code/DistGeom/DistViolationContrib.cpp`、`ChiralViolationContrib.cpp`
- `$RDBASE/Code/ForceField/CrystalFF/TorsionAngleM6.cpp`

## Design

### 模块布局

```
molrs-embed/src/etkdg/
├── mod.rs        // 顶层 pipeline：embed → ET-min → MMFF-clean → check → retry
├── embed4d.rs    // metrization 采样 + 4D 初始坐标
├── etmin.rs      // 一阶误差函数（dist + chiral + exp-torsion）+ 最小化
└── retry.rs      // maxIterations 重试 + useRandomCoords 兜底
```

### 公共 API（保留签名，改造 options）

```rust
// molrs-embed

pub fn generate_3d(mol: &Atomistic, opts: &EmbedOptions)
    -> Result<(Atomistic, EmbedReport), MolRsError>;  // 签名不变

/// 改造后的 ETKDG 选项（取代旧 embed_algorithm / speed 字段）。
pub struct EmbedOptions {
    pub etkdg_version: EtkdgVersion,   // 默认 Etkdgv3
    pub add_hydrogens: bool,
    pub rng_seed: Option<u64>,         // None = 随机；Some = 完全可复现
    pub max_iterations: usize,         // 0 = RDKit 默认启发式 (≈10×尝试)
    pub use_random_coords_fallback: bool,
    pub mmff_cleanup: bool,            // 默认 true：二阶 MMFF94 最小化
    pub mmff_variant: MmffVariant,
}

// EmbedReport / StageReport / StageKind 保留；StageKind 调整为
// {BuildConstraints, Embed4D, EtMinimize, MmffCleanup, StereoCheck}
```

### 与现有代码的关系

- 消费 `molrs_embed::distgeom::build_constraints`（-03）与
  `molrs_ff::mmff::MmffForceField`（-02）。
- **删除**：`builder.rs`、`optimizer.rs`、`distance_geometry.rs`、`rotor_search.rs`、
  `fragment_data.rs`、`geom.rs`（如仅服务旧 pipeline）、`stereo_guard.rs` 中仅旧用法的部分
  （stereo 检查迁入 `etkdg/mod.rs`）。
- `EmbedAlgorithm` 枚举删除（ETKDG 是唯一算法）；`ForceFieldKind`/`EmbedSpeed` 删除
  或并入新 options。这是对 embed 公共 API 的**破坏性变更**，需在 CHANGELOG 标注，
  并同步更新 molrs-python 的 `PyEmbedOptions`。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/etkdg/mod.rs` — (new) 顶层 pipeline + stereo 检查
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/etkdg/embed4d.rs` — (new) metrization + 4D 初始坐标
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/etkdg/etmin.rs` — (new) 一阶 ET 误差函数 + 最小化
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/etkdg/retry.rs` — (new) maxIterations 重试 + 兜底
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/lib.rs` — (modify) 重写 `generate_3d`，re-export etkdg
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/options.rs` — (modify) 改造 `EmbedOptions`，删除 `EmbedAlgorithm`/`EmbedSpeed`
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/report.rs` — (modify) `StageKind` 改为 ETKDG 阶段
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/src/{builder,optimizer,distance_geometry,rotor_search,fragment_data}.rs` — (delete) 旧 pipeline 退役
- `/Users/roykid/work/molcrafts/molrs/molrs-embed/tests/embed/pipeline.rs` — (modify) 重写为 ETKDG + RDKit 对照测试
- `/Users/roykid/work/molcrafts/molrs/molrs-python/src/lib.rs` — (modify) 更新 `PyEmbedOptions` 字段

## Tasks

- [ ] Implement metrization sampling + 4D initial coordinate generation in `etkdg/embed4d.rs`
- [ ] Implement first-stage ET error function (dist + chiral + exp-torsion) + minimizer in `etkdg/etmin.rs`
- [ ] Wire second-stage MMFF94 cleanup consuming `MmffForceField` in `etkdg/mod.rs`
- [ ] Implement stereo enforcement + check (no chirality inversion) in `etkdg/mod.rs`
- [ ] Implement maxIterations retry loop + useRandomCoords fallback in `etkdg/retry.rs`
- [ ] Rewrite `generate_3d` + reshape `EmbedOptions`/`StageKind`; update molrs-python `PyEmbedOptions`
- [ ] Delete retired modules (builder, optimizer, distance_geometry, rotor_search, fragment_data) and fix `lib.rs`
- [ ] Rewrite `tests/embed/pipeline.rs` as ETKDG + RDKit cross-validation
- [ ] Run full check (`cargo test --all-features`, `cargo clippy -- -D warnings`)

## Testing strategy

### 对照 RDKit（科学验证）
- 一组分子（乙醇、丁烷、苯、咖啡因、含手性中心分子、12 元大环），固定 `rng_seed`，
  molrs ETKDGv3 生成的最优构象与 RDKit `EmbedMolecule(..., ETKDGv3())` 的构象做
  best-fit RMSD < 0.5 Å（允许镜像/对称等价匹配）。
- 生成构象的 MMFF94 能量与 RDKit 同分子 ETKDG+MMFF 优化后能量同量级（相对差 < 10%）。

### 鲁棒性（runtime）
- `tests-data/smi/rdkit_problems.smi` 的 71 个分子：molrs 成功率 ≥ RDKit 在同集合的
  成功率，且严格 ≥ 旧 FragmentRules 实现（记录三者数字）。
- 固定 `rng_seed` → 两次 `generate_3d` 坐标逐元素一致（完全可复现）。
- 空分子 / 单原子 / 断开多组分 → 行为与 RDKit 一致（成功或明确 Err）。

### 立体化学
- 含手性中心分子生成后 `stereo` 检查零反转告警；E/Z 双键构型保持。

### 退役验证
- `builder.rs` / `optimizer.rs` / `distance_geometry.rs` / `rotor_search.rs` /
  `fragment_data.rs` 已删除，`cargo build` 无对其的悬挂引用；`EmbedAlgorithm` 不再导出。

## Out of scope

- 多构象集合生成（`EmbedMultipleConfs`）与 RMSD 去重——本 spec 只做单构象；多构象可后续 spec。
- 构象能量排序 / Boltzmann 加权。
- 周期性体系 / 晶体结构生成。
- MMFF 之外的清理力场（UFF cleanup 不在范围）。
