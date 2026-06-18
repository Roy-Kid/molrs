---
title: 力场角度/相位单位约定归一（kernels 吃弧度，readers 边界换算）
status: approved
created: 2026-06-18
---

# 力场角度/相位单位约定归一

> molrs-ff 核心约定迁移。**起因**：OPLS typifier 下沉后 `OplsTypifier::build()` 的角能量严重错误——OPLS reader 存弧度 θ₀，但 angle kernel ctor 做 `.to_radians()`（假设度），双重换算。在 OPLS 水角平衡位（1.911 rad）产出 **~264 kcal/mol 本应为 0 的伪角能量**。必须原子化迁移，否则修 OPLS 会回归 LAMMPS/MMFF。

## Summary

molrs 的角/二面角/improper kernel 一律在构造期对 `theta0`/`phase`/`chi0` 调 `.to_radians()`（假设存的是度）；但三个来源**约定分裂**：OPLS reader 存**弧度**（GROMACS 原生）、LAMMPS reader 存**度**、MMFF 存**度**。于是 OPLS 路径双重换算、角能量错数百 kcal/mol；LAMMPS/MMFF 因"度+kernel 换算"恰好对。CLAUDE.md 的核心约定是**内部弧度**。本 spec 把约定统一为 **"kernel 消费弧度；每个 reader 在边界把 angle/phase/chi0 归一化到弧度"**（对齐 OPLS reader 已做的 nm→Å、kJ→kcal 边界归一化模式），原子化落地以免拿 OPLS bug 换 LAMMPS/MMFF 回归。

## Domain basis

物理上角谐势 `½k(θ−θ₀)²` 要求 θ₀ 与 θ 同单位（弧度）。当前 kernel 内 `θ₀_used = stored_θ₀.to_radians()`：
- LAMMPS 存 109.66°（度）→ `to_radians()` → 1.914 rad ✓
- OPLS 存 1.911（弧度）→ `to_radians()` → 0.0334 rad ✗（dθ≈1.878 → 伪能量 ~264 kcal/mol）

正确约定（与 molrs "内部弧度" + "reader 边界归一化" 一致）：kernel 直接消费弧度；LAMMPS/MMFF reader 在读入时 deg→rad；OPLS reader 已是弧度、不变。

## Design

**1. kernel 去掉 `.to_radians()`（改为消费弧度）**
- `ff/potential/angle/harmonic.rs:111-114`（删 `.to_radians()`，结构体 doc line 12 "stored in radians" 本就如此，修正 ctor 与之一致）
- `ff/potential/angle/class2.rs:121`、`ff/potential/angle/mmff.rs:81`(MMFFAngleBend)、`:198`(MMFFStretchBend)
- 二面角/improper 相位同款：`dihedral/periodic.rs:83,97,176`、`dihedral/charmm.rs:97,127`、`dihedral/class2.rs:97`、`improper/periodic.rs:96,126`、`improper/harmonic.rs:100,128`
- 各 kernel doc-block 改为"consumes radians"。

**2. reader 边界 deg→rad 归一化（把换算从 kernel 移到 reader）**
- **LAMMPS** `readers/lammps.rs`：angle `theta0` `:214`、improper `chi0` `:274`、dihedral phase `d` `:246` 读入时 `.to_radians()`；重写头注 unit doc-block（line 26-29 "stored in degrees…kernels call to_radians" → "normalized to radians at read"）。
- **MMFF** `typifier/mmff/mod.rs:201-214`（存 `theta0` 处）deg→rad；**审计独立的 `mmff/energy/*` 路径**（端到端用度 + `RAD2DEG`）——确认它要么自洽不经通用 kernel、要么一并迁移；不可半迁移。
- **OPLS** `readers/opls.rs`：已弧度，**不变**。

**3. 测试反转 + 新增缺失的回归**
- 反转 `tests/ff/potential/angle.rs:72 compile_path_converts_degrees_to_radians` 与 `tests/ff/readers/lammps.rs:36 angle_phase_stays_in_degrees` → 断言"存储即弧度"。
- **新增 OPLS 平衡能量测试**：θ=θ₀ 时角能量 ≈ 0（当前未被任何测试捕获——OPLS 测试只验 `is_finite()` + FD 力自洽，bug 漏网）。

**原子性**：1+2+3 必须同一改动落地。只改 kernel 会修 OPLS 却坏 LAMMPS/MMFF（`angle_phase_stays_in_degrees` + GAFF relaxation 角能量）。

## Files to create or modify

- `ff/potential/angle/{harmonic,class2,mmff}.rs` — 删 `.to_radians()`，doc 改 "radians"。
- `ff/potential/dihedral/{periodic,charmm,class2}.rs`、`ff/potential/improper/{periodic,harmonic}.rs` — 相位同款。
- `ff/forcefield/readers/lammps.rs` — angle/dihedral-phase/improper deg→rad + doc。
- `ff/typifier/mmff/mod.rs`（+ 审计 `ff/mmff/energy/*`）— θ₀ deg→rad，确保 MMFF 能量不变。
- `tests/ff/potential/angle.rs`、`tests/ff/readers/lammps.rs` — 反转约定断言。
- `tests/ff/typifier/opls_parity.rs` 或 `tests/ff/potential/opls.rs` — 新增 θ=θ₀ 角能量≈0 测试。
- `CLAUDE.md` — 在核心约定里明确 "angle/phase/chi0 内部弧度；reader 边界归一化"。

## Tasks

- [ ] Write a failing OPLS equilibrium-angle-energy test (θ=θ0 ⇒ angle energy ≈ 0) — currently RED due to the double-conversion
- [ ] Write/invert reader-convention tests (lammps angle theta0 stored as radians; compile_path expects radians) — RED
- [ ] Add deg→rad normalization in LAMMPS reader (angle theta0, dihedral phase, improper chi0) + rewrite its unit doc-block
- [ ] Add deg→rad normalization in the MMFF typifier theta0 storage; audit mmff/energy/* path stays correct (RDKit parity unchanged)
- [ ] Drop .to_radians() from angle (harmonic/class2/mmff) + dihedral (periodic/charmm/class2) + improper (periodic/harmonic) kernels; update doc-blocks to "radians"
- [ ] Update CLAUDE.md core convention (angle/phase/chi0 radians internal, normalized at reader boundary)
- [ ] Verify atomically: OPLS build() angle energy ≈0 at equilibrium; LAMMPS angle energies unchanged; MMFF RDKit parity (e_ethane ~2e-5) unchanged; full angle/dihedral/improper energy suites green
- [ ] Run cargo fmt --all --check && cargo clippy -- -D warnings && cargo test (ff feature set)

## Testing strategy

- **Scientific (new, RED-first)** — OPLS angle energy = 0 at θ=θ₀ (catches the ~264 kcal/mol bug).
- **Regression (must stay green)** — LAMMPS angle-harmonic energy; MMFF angle bend + stretch-bend RDKit parity (e_ethane ~2.3e-5); dihedral/improper phase energies.
- **Unit (inverted)** — reader stores radians; kernel consumes radians (no conversion).
- **Build smoke** — fmt/clippy/test green (ff feature set; `--all-features` may stay blocked by unrelated blas/teammate WIP).

## Out of scope

- **改 molpy** — molpy 的角约定与本 spec 无关。
- **非角度单位**（长度/能量）— 已在 reader 边界归一化，不动。
- **新增 kernel/力场** — 仅迁移现有 kernel 的角/相位单位约定。
- **GAFF estimator 弧度 fixture**（`tests/ff/typifier/estimate.rs`）— 只做相似度比较、不喂角 kernel，自洽不动。
