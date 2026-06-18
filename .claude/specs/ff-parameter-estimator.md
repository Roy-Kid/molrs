---
title: 基于相似性的力场缺失成键参数估计器（Rust，FF 无关）
status: approved
created: 2026-06-18
---

# 基于相似性的力场缺失成键参数估计器（Rust）

> 力场无关的缺参兜底器，挂在 OPLS [opls-typifier-02-assign](opls-typifier-02-assign.md) 与 GAFF [gaff-typifier-03-assign](gaff-typifier-03-assign.md) 的成键匹配 no-match 分支。与 OPLS 链一并启动。**取代**已作废的 molpy-Python 版（molpy `.claude/specs/ff-parameter-estimator.md`，superseded）——算法/Domain basis 整体转录至此并以 Rust 落地。

## Summary

新增 `ff/typifier/estimate/`：`ParameterEstimator`，当 bond/angle/dihedral 在精确 + class + 通配匹配下均缺失时，用 parmchk2 式类比级联（精确 → 等价 → 对应 → 通配）+ CGenFF 式加性惩罚（内原子加权）+ GAFF 经验兜底公式（Badger 键 k、角 θ₀ 均值、角 K_θ 经验式、二面角回退通用项/近零势垒）补齐，**完全不做从头量化计算**。估计器以可选注入挂到 OPLS/GAFF 两个 assign 步骤的 no-match 分支（精确匹配始终优先；strict=true 不介入、仍 `Err`；默认关闭 = 行为零变化）。每个估计项写溯源 `{analog, penalty, method}` 与置信度分级（penalty <10 可靠 / 10–50 谨慎 / >50 差）。

## Domain basis

类比/相似性补参数是力场领域成熟标准做法（parmchk2、CGenFF、MATCH、OpenFF、MMFF94 均用）。条件：必报溯源 + penalty；相似性尊重元素/杂化/周期；最近类比**直接复制**（不平均）；二面角回退通用项或近零势垒，**绝不伪造势垒**。

1. **parmchk2 类比级联** — Wang et al., *J. Mol. Graph. Model.* 2006, 25:247–260, DOI:10.1016/j.jmgm.2005.12.005。每缺失项层级搜索：(1) 精确 type → (2) 等价原子类型（共振/几何孪生, penalty 0）→ (3) 对应原子类型（按属性差异累加 penalty）→ (4) 通配符。最低总 penalty 胜，直接复制其参数。
2. **加性惩罚 + 内原子加权** — CGenFF: Vanommeslaeghe & MacKerell, *J. Chem. Inf. Model.* 2012, 52:3144–3154 (DOI:10.1021/ci300363c) 及 52:3155–3168 (DOI:10.1021/ci3003649)。`penalty = Σ_i w_pos(i)·subst_penalty(t_i,p_i)`；内原子（角中心、二面角两内原子、improper 中心）×10。阈值带：penalty <10 可靠 / 10–50 谨慎 / >50 差需优化。
3. **GAFF 经验兜底公式** — Wang et al., *J. Comput. Chem.* 2004, 25:1157–1174, DOI:10.1002/jcc.20035（基于元素，力场无关）：
   - **键力常数**：Badger 型反幂律（按元素对标定常数）。exact 形式与元素对常数表从 GAFF 原文/parmchk2 源码逐字转录，经验公式单测钉死。
   - **角 θ₀**：共享中心 B 的已有角 θ₀(A-B-A)、θ₀(C-B-C) 的均值。
   - **角 K_θ**：GAFF 经验式（常数 143.9、按元素 Z/C 因子、依赖 θ₀ 与键长）。exact 形式/Z/C 表逐字转录，单测钉死。
   - **二面角**：优先回退最通用已有扭转项（通配端，按两中心原子键控）；否则近零势垒 + 高 penalty，绝不伪造；多重周期项整组复制。
4. **单位对齐（correctness 约束）**：molrs reader 已把 OPLS（GROMACS nm/kJ）与 GAFF（AMBER Å/kcal）统一换算到 **molrs 内部单位（Å, kcal/mol, rad, e）**（见 `readers/opls.rs` 头注 + gaff parser）。**最近类比复制天然单位安全**（复制同表项的 molrs-单位值）。GAFF 经验公式天生产出 AMBER 单位（≈molrs 单位），但写入前仍须确认目标 Style 的单位约定并对齐（OPLS bond/angle 与 GAFF 同为 molrs 内部单位 → 无需再换；记录此前提为回归测试约束）。
5. **关键陷阱**：① 不伪造本应近零的二面角；② 区分"FF 故意省略二面角靠 1-4"与"真缺失需估"——v1 保守：仅当该中心键无任何扭转项时才估计；③ 多重周期整组复制；④ 不同力场等价类型表不可互串；⑤ 经验角 K_θ 单位/约定回归测试钉死。

## Design

**1. 模块归属**

新建 `molrs/src/ff/typifier/estimate/`：
- `mod.rs` — `ParameterEstimator`（持等价/对应替换表 + 元素对/Z·C 常数表，构造期从内嵌数据加载并缓存，镜像 `MMFFParams` 内嵌范式）。公开 `estimate_bond/estimate_angle/estimate_dihedral`。内部 helper 前导下划线（`_analogy_score`、`_substitution_penalty`、`_empirical_bond_k`、`_empirical_angle_theta0`、`_empirical_angle_k`、`_generic_dihedral`）。
- `tables.rs` — GAFF 等价/对应替换表 + 经验常数（编译期内嵌，镜像 `molrs::data::MMFF94_XML`；数据资产放 `molrs/data/`，从 GAFF 原文/parmchk2 逐字转录）。

**2. 调用时机（接缝，与 assign 共享）**

估计器以可选注入（`Option<&ParameterEstimator>` 或 typifier 字段，默认 `None`）挂到 OPLS [opls-typifier-02-assign] 与 GAFF [gaff-typifier-03-assign] 的成键 no-match 分支。仅当特异性匹配全失败且 estimator 启用时调用，对单个缺失项即时估计后返回。精确/通配匹配始终优先；strict=true 完全不介入（仍 `Err`）；未注入（默认）时行为零变化。**非** eager 全量 pass。

**3. 类比级联（复用 assign 的打分基元）**

构建在 OPLS [opls-typifier-02-assign] 的 `_end_score`/`_sequence_score` 之上：把精确/class/通配三档放宽为"元素/杂化感知的相似度 + 加性惩罚"，扫同一候选表找最近类比（`(penalty 升序)` 取最小）。最近类比 → **复制其 `params`**（单位安全）。内原子 ×10 加权（角中心、二面角两内原子）。

**4. 经验兜底**

无任何类比时：键 k 用 Badger 反幂律（`tables.rs` 元素对常数）；角 θ₀ 取共享中心已有角均值（从候选表查 A-B-A/C-B-C）；角 K_θ 用 GAFF 经验式（143.9 + Z/C 表）；二面角优先通用通配项，否则近零势垒 + 高 penalty。产出 molrs 单位 `Params`。

**5. 溯源（新约定）**

每个估计项在其 prop/`params` 写：`estimated=true`、`estimate_method`∈{analogy, empirical, generic-wildcard}、`estimate_penalty`、`estimate_analog`（源 Type name 或空）。命名对齐 molpy atom 级 `source` 先例。错误处理 fail-fast，与 typifier 既有 `Err` 一致。

## Files to create or modify

- `molrs/src/ff/typifier/estimate/mod.rs` (new) — `ParameterEstimator` + estimate_* + 私有 helper。
- `molrs/src/ff/typifier/estimate/tables.rs` (new) — 等价/对应表 + Badger/Z·C 经验常数（内嵌）。
- `molrs/data/gaff_equiv.json` (+ `bond_empirical.json` / `angle_empirical.json`) (new) — 从 GAFF 原文/parmchk2 逐字转录的数据资产。
- `molrs/src/ff/typifier/mod.rs` — `pub mod estimate;` + re-export `ParameterEstimator`。
- `molrs/tests/ff/typifier/estimate.rs` (new) — 经验公式钉死、留一法、溯源/penalty、二面角不伪造。
- `molrs/tests/ff/typifier/estimate_parity.rs` (new) — parmchk2 金标准对照（gated；fixtures 缺席干净跳过）。

## Tasks

- [x] Write failing unit tests for empirical formulas (Badger bond k, angle θ₀ mean, GAFF Eq.5 K_θ) against transcribed reference values in molrs/tests/ff/typifier/estimate.rs
- [x] Transcribe GAFF/parmchk2 constants into molrs/data/*.json and implement _empirical_bond_k / _empirical_angle_theta0 / _empirical_angle_k in molrs/src/ff/typifier/estimate
- [x] Write failing tests for analogy cascade + additive penalty (nearest-analog copy, inner-atom ×10, leave-one-out recovery, source fields, penalty tiers)
- [x] Implement ParameterEstimator analogy cascade (on _end_score/_sequence_score) + estimate_bond/angle/dihedral with provenance emission
- [x] Write failing tests for dihedral generic fallback + near-zero barrier + multi-periodicity group copy
- [x] Implement _generic_dihedral fallback, near-zero-barrier-with-high-penalty, multi-periodicity group copy
- [x] Wire optional estimator injection into OPLS opls-typifier-02-assign and GAFF gaff-typifier-03-assign no-match seams (opt-in; strict=true unaffected) — OPLS wired (`OplsTypifier::with_estimator`/`with_default_estimator`); GAFF typifier not implemented yet, but the `Estimator` trait is FF-agnostic and `ParameterEstimator::new` takes any `ForceField` so GAFF reuses it as-is.
- [x] Write parmchk2 gold-standard cross-validation test (gated, skips when fixtures absent) in molrs/tests/ff/typifier/estimate_parity.rs
- [x] Add rustdoc with units; document the estimate_* provenance convention
- [~] Run cargo fmt --all --check && cargo clippy -- -D warnings && cargo test --all-features — fmt/clippy clean; `cargo test --features "io,signal,smiles,ff,conformer"` green (1232 passed). Literal `--all-features` blocked by an unrelated `blas` link failure (no system BLAS on this arm64 host) — same posture as chain-1/2.

## Testing strategy

- **Unit** — empirical formulas reproduce transcribed GAFF reference values (rtol 1e-3; formula pin, not data-fit).
- **Unit** — analogy: nearest-analog copy (not averaged); inner-atom ×10; penalty tiers <10/10–50/>50 at boundaries.
- **Scientific** — leave-one-out: delete a known OPLS/GAFF term, estimator recovers within bond r0 0.02 Å / angle θ0 3° / force const rtol 0.10.
- **Edge** — dihedral never fabricates a rigid barrier (no analog + no generic → near-zero k + high penalty); multi-periodicity copied as one group.
- **Edge** — seam: strict=true unaffected (Err); estimator absent → behavior unchanged.
- **Scientific (gated)** — parmchk2 frcmod cross-validation; skips cleanly without AmberTools fixtures.
- **Build smoke** — fmt/clippy/check/test green.

## Out of scope

- **从头量化计算 / 任何 QM 拟合** — 明确排除。
- **OpenFF 式 AM1-Wiberg 键级插值** — 无键级，v1 不做。
- **OPLS/CL&P 专属等价类型表 curate** — v1 仅 GAFF 等价表；OPLS/CL&P 走元素/杂化通用相似 + GAFF 经验兜底。
- **improper 估计** — v1 优先级最低、可选。
- **eager 全量补全** — 仅 assign no-match 按需触发。
