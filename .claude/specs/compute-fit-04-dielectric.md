---
title: 计算与拟合分离 — 介电 ε(ω) raw+fit 收尾 (Phase 04)
status: approved
created: 2026-06-18
---

# 计算与拟合分离 — 介电 ε(ω) raw+fit 收尾 (Phase 04)

## Summary

补齐 compute↔fit 分离链留下的最后一处捆绑路径：**介电谱 ε(ω)**。phase 03 因「无 ε(ω) raw+fit 替代核」而 DEFER 了 `einstein_helfand_spectrum`/`green_kubo_spectrum` 的删除；本阶段建出该核（raw 偶极/电流 ACF compute + ε(ω) transform Fit，镜像 phase 01 的 VDOS/IR/Raman 模式），把 molpy `DielectricSusceptibility` 重指到 raw+fit，然后删除遗留 free fns + 内联 helper + 其 PyO3 绑定。完成后 molrs 中**每一个 compute 都只返回原始数据**，所有加窗/FFT/拟合都在 `compute::fit`，链条全闭。属同一 BREAKING 释出（molcrafts-molrs 已在 phase 03 跳到 0.2.0）。

## Domain basis

不引入新物理——ε(ω) 公式与 phase 01/03 一致：`(ε*(ω) − ε∞)/(ε0 − ε∞) = 1 − iω ∫₀^∞ Φ(t) e^{−iωt} dt`，其中 Φ(t)=⟨δM(0)·δM(t)⟩/⟨δM²⟩ 是涨落（去均值）偶极自相关；幅度 (ε0−ε∞) 来自零延迟方差 ⟨M²⟩（Neumann/Kirkwood），静态极限走既有 `static_dielectric_constant`（Neumann 1983）。变换层对 raw ACF 的导数做加窗 one-sided FT（即既有 `windowed_acf_derivative_spectrum` 逻辑），保证损耗谱有限。参考：Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986)；Neumann, *Mol. Phys.* **50**, 841 (1983)；*J. Chem. Phys.* **159**, 134505 (2023)。raw ACF 必须非归一化（或附 ⟨M²⟩），且携带 V/T/Ewald-BC + 电流归一化元数据（链上三不变量）。

## Design

- **Raw compute（只返回原始曲线/元数据）**：复用 phase 01 的 `DebyeRelaxation`（已返回 raw 偶极 ACF + ⟨M(0)²⟩ + V/T/Ewald-BC）作为 Einstein-Helfand/介电弛豫路线的 raw 入口；电流路线复用 `GreenKuboConductivity` 的 raw 电流 ACF。若需要可加一个轻量 `DielectricACF` 别名 compute，但优先复用既有 raw computes，避免重复。
- **Fit/transform（`compute::fit`）**：新增 ε(ω) 变换 Fit，消费 raw ACF（+ 元数据）→ `DielectricSpectrumResult { frequencies, eps_real, eps_imag }`。命名沿用 method-first：`EinsteinHelfandSpectrum` 与 `GreenKuboSpectrum` 两个 Fit 结构（或单 `DielectricSpectrum{route}`，实现者择清晰者），各自**逐位复现**对应旧 free fn 的输出（回归锁）。变换体抽取/迁移自 dielectric.rs 的 `windowed_acf_spectrum`/`acf_to_spectrum`/`windowed_acf_derivative_spectrum` 进 `compute::fit`，窗一律复用 `molrs::signal`。
- **删除遗留**：`einstein_helfand_spectrum`、`green_kubo_spectrum` free fns + 上述内联 helper（迁移后死亡的部分）+ 其 deprecated PyO3 绑定（`dielectric_einstein_helfand_spectrum`/`dielectric_green_kubo_spectrum`）。保留纯 raw 的 `static_dielectric_constant*`/`compute_dipole_moment`/`compute_current_density`/`decompose_current`（定义量，非拟合）。
- **molpy 重指**：`DielectricSusceptibility`（dielectric.py）从旧 free fn 改为 raw compute（DebyeRelaxation/GreenKuboConductivity）+ 新 ε(ω) Fit；保留公开输出（回归锁）；不留 Python 端谱数学。
- **版本收尾**：molrs-python `pyproject.toml` 与 molrs-ffi 自身版本与 molcrafts-molrs 0.2.0 对齐（breaking 释出一致）；不发布。

## Files to create or modify

- `molrs/src/compute/fit/spectral.rs`（或新增 `compute/fit/dielectric_spectrum.rs`）— ε(ω) 变换 Fit + `DielectricSpectrumResult`，迁移 dielectric 谱变换体。
- `molrs/src/compute/dielectric.rs` — 删 `einstein_helfand_spectrum`/`green_kubo_spectrum` + `windowed_acf_spectrum`/`acf_to_spectrum`/`windowed_acf_derivative_spectrum`（迁移后）；保留 raw + 静态介电。
- `molrs/src/compute/fit/raw_computes.rs` — 若需 `DielectricACF` 别名 compute（否则复用 DebyeRelaxation/GreenKuboConductivity）。
- `molrs/src/compute/mod.rs` — re-export 更新。
- `molrs-python/src/dielectric.rs` + `lib.rs` — 删两个 deprecated 谱绑定；新增 ε(ω) Fit 绑定（若暴露）；收敛注册。
- `molrs-python/python/molrs/dielectric.py` — 删谱 shim，保留 raw+fit 入口。
- `molpy/src/molpy/compute/dielectric.py` — `DielectricSusceptibility` 重指 raw compute + ε(ω) Fit。
- `molrs-python/pyproject.toml`、`molrs-ffi/Cargo.toml` — 版本对齐 0.2.0。

## Tasks

- [ ] Write failing tests: ε(ω) Fit reproduces legacy einstein_helfand_spectrum / green_kubo_spectrum output bit-for-bit on the same raw ACF (molrs/src/compute/fit/spectral.rs)
- [ ] Implement the ε(ω) transform Fit(s) in compute::fit, migrating the dielectric windowed-derivative-FT body; route through molrs::signal windows
- [ ] Remove einstein_helfand_spectrum/green_kubo_spectrum + windowed_acf_spectrum/acf_to_spectrum/windowed_acf_derivative_spectrum from dielectric.rs; update compute/mod.rs re-exports
- [ ] Remove the two deprecated dielectric-spectrum PyO3 bindings + python shims; converge registration
- [ ] Repoint molpy DielectricSusceptibility to raw compute + ε(ω) Fit (preserve public output)
- [ ] Align molrs-python pyproject.toml + molrs-ffi version to 0.2.0 (no publish)
- [ ] Rebuild wheel; run gates (cargo test --features compute green; clippy/fmt clean on touched; grep-clean of removed dielectric-spectrum symbols; molpy compute tests green)

## Testing strategy

- Regression lock: `EinsteinHelfandSpectrum`/`GreenKuboSpectrum` Fit applied to the raw dipole/current ACF reproduces legacy `einstein_helfand_spectrum`/`green_kubo_spectrum` `frequencies`/`eps_real`/`eps_imag` bit-for-bit (or documented float tol).
- Physics: ε(ω=0) recovers the Neumann static dielectric constant; loss spectrum ε″ finite at Nyquist (derivative-FT path); molpy `DielectricSusceptibility` output pinned to pre-migration within tolerance.
- Grep-clean: zero live refs to `einstein_helfand_spectrum`/`green_kubo_spectrum`/`windowed_acf_spectrum`/`windowed_acf_derivative_spectrum`/dielectric `acf_to_spectrum` across molrs/molrs-python/molpy.
- Edge: empty/too-short ACF errors; missing ⟨M²⟩/V/T metadata is a compile-time/typed error (amplitude unrecoverable without it).

## Out of scope

- New physics or new spectral routes beyond the two existing dielectric ones.
- Touching static_dielectric_constant*/compute_dipole_moment/decompose_current (raw/defined quantities).
- The unrelated untracked io/trajectory xtc/trr work and ff/typifier work.
- Publishing/tagging the 0.2.0 release.
