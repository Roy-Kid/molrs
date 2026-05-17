---
title: 振动光谱 (Vibrational Spectra)
status: code-complete
created: 2026-05-17
---

# 振动光谱 (Vibrational Spectra)

## Summary

在 molrs 中实现三种经典振动光谱计算：功率谱/振动态密度（VDOS）、红外（IR）光谱和拉曼（Raman）光谱。算法改编自 SchNetPack 的 `md.data.spectra` 模块，但输入接口重新设计为直接接收 ndarray 时间序列（而非 HDF5），ACF 使用 molrs-signal 已有的非归一化 Wiener-Khinchin 实现（保留正确振幅），并增加可选的谐振量子校正因子。计算放置在 `molrs-compute::spectra` 子模块中，复用 `molrs-signal` 的 ACF、窗函数和频率网格基础设施。

## Domain basis

### Power spectrum (VDOS)

Velocity autocorrelation function (VACF):

`C_vv(t) = ⟨v(0)·v(t)⟩`

VDOS:

`G(ω) = ∫ C_vv(t) · w(t) · e^{−iωt} dt`

where `w(t)` is the cosine-squared window. Units: arbitrary (intensities not normalized).

Reference: Dickey & Paskin, Phys. Rev. 188, 1407 (1969); SchNetPack `md.data.spectra.power_spectrum`.

### IR spectrum

Dipole flux via central finite difference:

`Ṁ(t) ≈ (M(t+Δt) − M(t−Δt)) / (2·Δt)`

Dipole flux ACF:

`C_J(t) = ⟨Ṁ(0)·Ṁ(t)⟩`

IR absorption coefficient:

`I(ω) ∝ ω² · ∫ C_J(t) · w(t) · e^{−iωt} dt`

The `ω²` factor converts dipole flux spectrum to dipole-derivative spectrum (standard IR lineshape). Units: arbitrary.

Reference: Caillol, Levesque & Weis, J. Chem. Phys. 85, 6645 (1986); SchNetPack `md.data.spectra.ir_spectrum`.

### Raman spectrum

Polarizability derivative isotropy/anisotropy decomposition — define:

`ᾱ = Tr(α) / 3` (isotropic part)
`β² = ½ · [(α₁₁−α₂₂)² + (α₂₂−α₃₃)² + (α₃₃−α₁₁)² + 6·(α₁₂²+α₁₃²+α₂₃²)]` (anisotropic part)

ACF of isotropic and anisotropic components → FT → Raman I(ω) with optional Bose factor and cross-section correction.

`I_iso(ω) ∝ ∫ ⟨ᾱ̇(0)·ᾱ̇(t)⟩ · w(t) · e^{−iωt} dt`
`I_aniso(ω) ∝ ∫ ⟨β̇²(0)·β̇²(t)⟩ · w(t) · e^{−iωt} dt`

Bose factor (optional, controlled by `temperature_k`):

`n(ω,T) + 1 = 1 / (1 − e^{−ℏω/k_BT})`

Parallel/perpendicular decomposition (when `averaged=true`):

`I_∥(ω) = I_iso(ω) + (4/3)·I_aniso(ω)`
`I_⊥(ω) = (1/3)·I_aniso(ω)`

Reference: Berne & Pecora, Dynamic Light Scattering, Wiley (1976); SchNetPack `md.data.spectra.raman_spectrum`.

### Frequency conversion

From angular frequency `ω` (rad/fs) to wavenumber `ν̃` (cm⁻¹):

`ν̃ = ω / (2π · c · 10¹³)⁻¹ = ω · 33356.4 cm⁻¹·fs/rad`

where `c = 299792458 m/s`.

### Optional quantum correction (all three spectrum types)

Harmonic quantum correction factor (HQC):

`Q(ω) = (βℏω/2) / sinh(βℏω/2)`  with `β = 1/(k_B·T)`

Not applied by default; caller may multiply `intensities *= Q(ω)`.

Reference: Ramirez, Lopez-Ciudad, *J. Chem. Phys.* 115, 5723 (2001); **Bader & Berne, *J. Chem. Phys.* 100, 8359 (1994)**.

## Design

### 数据结构

- `Spectrum` — 通用振动光谱结果，包含 `frequencies_cm1` 和 `intensities`。用于 VDOS 和 IR 光谱。
- `RamanSpectrum` — 拉曼光谱结果，包含 `isotropic`、`anisotropic`分量；当启用 `averaged=true` 时额外输出 `parallel` 和 `perpendicular` 方向分辨谱。
- `WindowType::CosineSq` — 在 `molrs-signal` 中新增的 squared-cosine 窗函数变体。

### 公共 API

```rust
// molrs-compute::spectra

pub struct Spectrum {
    pub frequencies_cm1: Array1<f64>,
    pub intensities: Array1<f64>,
    pub resolution: usize,
    pub n_frames: usize,
}

pub struct RamanSpectrum {
    pub frequencies_cm1: Array1<f64>,
    pub isotropic: Array1<f64>,
    pub anisotropic: Array1<f64>,
    pub parallel: Option<Array1<f64>>,
    pub perpendicular: Option<Array1<f64>>,
    pub resolution: usize,
    pub n_frames: usize,
}

pub fn power_spectrum(
    velocities: &Array2<f64>,   // (n_frames, n_dof)
    dt_fs: f64,
    resolution: usize,
) -> Result<Spectrum, ComputeError>;

pub fn ir_spectrum(
    dipole_moments: &Array2<f64>,   // (n_frames, 3)
    dt_fs: f64,
    resolution: usize,
) -> Result<Spectrum, ComputeError>;

pub fn raman_spectrum(
    polarizabilities: &Array2<f64>,   // (n_frames, 6) Voigt notation
    dt_fs: f64,
    resolution: usize,
    incident_frequency_cm1: f64,      // for cross-section correction
    temperature_k: f64,               // for Bose factor; 0.0 = no Bose correction
    averaged: bool,                    // compute parallel/perpendicular?
) -> Result<RamanSpectrum, ComputeError>;
```

### 计算管线（三者共享）

1. 按需计算派生量（如 IR 的偶极子通量 `Ṁ` 使用中心差分，Raman 的极化率各向异性 `β²`）
2. 每分量 ACF：使用 `acf_fft_with_planner`（复用 planner）
3. 按光谱类型求和/组合 ACF（trace 求和 vs 各向异性组合）
4. 截断至 `resolution` 个延迟步
5. 应用 squared-cosine 窗（新增 `WindowType::CosineSq`）
6. 零填充至 2 的幂，单边 FFT（复用 `dielectric.rs` 中的 `acf_to_spectrum` 模式）
7. 取实部（偶极通量/极化率 ACF 的 FT 实部即谱）
8. 频率轴从 rad/fs 转换为 cm⁻¹
9. 应用物理前因子（IR 的 `ω²`，Raman 的 Bose 因子等）
10. 返回 `Spectrum` / `RamanSpectrum`

### 与现有基础设施的关系

- 复用 `molrs-signal` 的 `acf_fft_with_planner`（Wiener-Khinchin 非归一化 ACF）
- 复用 `molrs-signal` 的 `frequency_grid`（添加 rad/fs → cm⁻¹ 转换层）
- 新增 `WindowType::CosineSq` 变体（SchNetPack 默认窗）
- `dielectric.rs` 中的 `acf_to_spectrum` 模式直接复用作为 `spectra/mod.rs` 的共享工具函数
- 不带 `Compute` trait 实现——振动光谱是纯函数式 API（如 `dielectric.rs` ），不适宜转换为 stateless `Compute` 结构

### 与 SchNetPack 的关键差异

| 方面 | SchNetPack | molrs |
|------|-----------|-------|
| 数据来源 | HDF5Loader | 调用方提供 ndarray |
| ACF 归一化 | z-score（破坏振幅） | 非归一化（物理正确） |
| 窗函数 | 仅 cosine_sq | CosineSq + 已有的 Hann/Blackman |
| 单位 | 通过 ASE 单位系统 | 物理常数直接转换 cm⁻¹ |
| Raman 常数 | 4.160440e-18（未文档化） | 省略，标记为任意单位 |
| 量子校正 | 无 | 可选 HQC（caller 选择应用） |

### 生命周期与所有权

所有函数纯函数式：输入 ndarray 只读借用，输出新分配的 `Spectrum` 或 `RamanSpectrum`。无内部可变状态，无全局注册表。遵循 `molrs-compute` 的 stateless 约定。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs-signal/src/window.rs` — (modify) 添加 `WindowType::CosineSq` 变体和对应实现
- `/Users/roykid/work/molcrafts/molrs/molrs-compute/src/spectra/mod.rs` — (new) 模块定义，共享工具函数（频谱预处理管线、频率转换、输入验证）
- `/Users/roykid/work/molcrafts/molrs/molrs-compute/src/spectra/power_spectrum.rs` — (new) `power_spectrum()` 函数 + 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs-compute/src/spectra/ir_spectrum.rs` — (new) `ir_spectrum()` 函数 + 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs-compute/src/spectra/raman_spectrum.rs` — (new) `raman_spectrum()` 函数 + 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs-compute/src/lib.rs` — (modify) 添加 `pub mod spectra;` 声明和 re-exports

## Tasks

- [ ] Add `CosineSq` variant to `WindowType` in `molrs-signal/src/window.rs` with `w[n] = cos²(π·n / (2·(N−1)))` formula and unit tests
- [ ] Write failing tests for `power_spectrum()` in `molrs-compute/src/spectra/power_spectrum.rs`
- [ ] Implement `power_spectrum()` in `molrs-compute/src/spectra/power_spectrum.rs` — VACF → FT → VDOS pipeline
- [ ] Write failing tests for `ir_spectrum()` in `molrs-compute/src/spectra/ir_spectrum.rs`
- [ ] Implement `ir_spectrum()` in `molrs-compute/src/spectra/ir_spectrum.rs` — central-difference dipole flux → ACF → FT → ω² scaling
- [ ] Write failing tests for `raman_spectrum()` in `molrs-compute/src/spectra/raman_spectrum.rs`
- [ ] Implement `raman_spectrum()` in `molrs-compute/src/spectra/raman_spectrum.rs` — isotropic/anisotropic decomposition, Bose factor, parallel/perpendicular
- [ ] Wire shared helpers (ACF-to-spectrum, frequency conversion, input validation) in `molrs-compute/src/spectra/mod.rs` and add `pub mod spectra` to `molrs-compute/src/lib.rs`
- [ ] Run full check + test suite (`cargo test --all-features`, `cargo clippy -- -D warnings`)

## Testing strategy

### 快乐路径

- `power_spectrum`：对单频正弦波速度序列（`v_x(t) = sin(ω₀·t)`，其他分量=0），验证输出频谱在 `ω₀` 处（转换为 cm⁻¹ 后）存在峰值
- `ir_spectrum`：对已知振荡偶极矩 `M_z(t) = sin(ω₀·t)`，验证 IR 谱在 `ω₀` 处有吸收峰，且常数偶极矩序列产生零谱
- `raman_spectrum`：对 Voigt 记法极化率序列，验证 `isotropic` 和 `anisotropic` 输出非空、形状正确、均方根非零；启用 `averaged=true` 时验证 `parallel` / `perpendicular` 为 `Some`
- 频率转换验证：对已知信号验证峰值 cm⁻¹ 位置与预期相符（33356.4 scale factor）

### 边界情况

- 输入帧数 < 2 → 返回 `ComputeError::EmptyInput`
- `resolution` > 可用帧数 → 自动截断至 `n_frames - 1`
- `dt_fs ≤ 0` → 返回 `ComputeError::OutOfRange`
- Raman 的 `temperature_k = 0.0` → 不加 Bose 校正（只需文档化，不报错）
- Raman 的 `incident_frequency_cm1 = 0.0` → 视为任意单位，cross-section 校正因子取 1

### 科学验证

- 正弦波速度验证 VDOS 峰值位置，容差 < 1 cm⁻¹（取决于频率网格分辨率）
- 非归一化 ACF 的 VDOS 总积分应与 `⟨v²⟩` 成正比（Parseval 定理验证）
- 恒定零信号的频谱应为全零（数值噪声 < 1e-12）

## Out of scope

- **纯振动光谱计算流程**本身是本规范的完整范围。不包括以下内容：
  - HDF5 或其他文件格式的输入适配器（调用方负责从轨迹读取数据并提供 ndarray）
  - 多分子/多体系平均（调用方可多次调用并自行平均）
  - 振动模式分析（VMD/NMA 等简正模式分析不在本规范范围内）
  - 线型展宽（不均匀展宽、均匀展宽等后处理）
  - Python 绑定（可通过现有 PyO3 桥接调用，但不在本规范中）
  - META 声子分析等更高级的振动特性
  - `dielectric.rs` 的重构或移动（两个模块独立共存）
