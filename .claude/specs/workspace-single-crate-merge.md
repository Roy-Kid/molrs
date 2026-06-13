---
title: Merge molrs 7-crate workspace into one crate molcrafts-molrs (feature-gated modules)
status: code-complete
created: 2026-06-13
revised: 2026-06-13
---

# Merge molrs 7-crate workspace into one crate molcrafts-molrs (feature-gated modules)

## Summary

把 molrs workspace 的 7 个成员 crate（`molrs-core` / `molrs-io` / `molrs-signal` / `molrs-compute` / `molrs-ff` / `molrs-conformer` 以及门面 `molrs`）合并成**单个**可发布 crate `molcrafts-molrs`（lib 名 `molrs`）。各子系统从"门面 re-export 外部 crate"变为"门面内部的 feature-gated **模块**"——源码被物理吸收进 `molrs/src/<subsystem>/`：`core` 常开，`io` / `signal` / `compute` / `ff` / `conformer` 各自由 feature 门控。4 个 binder crate（`molrs-ffi` / `molrs-cxxapi` / `molrs-wasm` / `molrs-python`）与外部消费者 `molpack` **保持独立**，仅把依赖从 6 个子 crate **重指**到合并后的 `molcrafts-molrs` + 对应 feature。下游公开 API 表面**零变化**：`molrs::Frame`、`molrs::io::read_xyz`、`molrs::smiles::*`、`molrs::ff::*`、`molrs::conformer::*`、`molrs::compute::*`、`molrs::signal::*` 全部照旧可用。

**迁移必须原子落地（避免依赖环）**：因每个子 crate 都依赖 `core`、而门面 re-export 全部子 crate，任何"先吸收 core、暂留 io/ff 为独立 crate"的增量切法都会构成 normal-dependency 环（`molrs-io` → 门面 → `molrs-io`），cargo 直接报 `cyclic package dependency`。因此本 spec 不拆成前向链，而是作为**一次性原子变更**实施：先完成全部"源码搬迁 + 引用重写 + 依赖折叠 + binder 重指"，再做第一次 `cargo check`。下方 Tasks 是该单次变更内的执行清单（非各自可独立编译的提交）。

## Design

**端状态拓扑**：`[workspace] members` 中此前 7 个成员里只剩 `molrs`（包名 `molcrafts-molrs`，lib `molrs`）一个可发布 lib crate；`molrs-core` / `molrs-io` / `molrs-signal` / `molrs-compute` / `molrs-ff` / `molrs-conformer` 六个目录从 members 移除，其 `src/` 被吸收为 `molrs/src/{core,io,signal,compute,ff,conformer}/`。`molrs-ffi` 仍以自己的 `[workspace]` 独立（保留在 root 的 `exclude`）；`molrs-cxxapi` 仍是 members（它是 staticlib，非发布 lib），但依赖重指。

**门面 lib.rs 从 re-export 改模块**：每个 `#[cfg(feature="X")] pub use molrs_X as X;` 改为 `#[cfg(feature="X")] pub mod X;`，由吸收进来的源码支撑；`pub use molrs_core::*;` 改为 `pub mod core;` + `pub use crate::core::*;`（顶层 `molrs::Frame` 等核心类型对下游仍可达）。`smiles` 不再是独立 crate，而是 `crate::io::smiles`（与现状一致，io 内部模块）。docs.rs 配置（`[package.metadata.docs.rs] features=["full"]`）保持不变。

**源码内交叉引用重写**（在被吸收的 6 个成员 `src/` 内，约 807 处、166 个 `.rs` 文件，含 doctests/doc-comment 路径如 conformer 的 `molrs::system::atomistic::Atomistic`）：
- `molrs::` （core 的 workspace 别名）→ 619 处 → `crate::core::`
- `molrs_core::` → 63 处 → `crate::core::`
- `molrs_io::` → 70 处 → `crate::io::`
- `molrs_ff::` → 27 处 → `crate::ff::`
- `molrs_compute::` → 24 处 → `crate::compute::`
- `molrs_conformer::` → 3 处 → `crate::conformer::`
- `molrs_signal::` → 1 处 → `crate::signal::`

**端状态 feature 表**（合并后写进 `molrs/Cargo.toml`）：
- `io = ["dep:flate2", "dep:once_cell"]`
- `smiles = ["io", "dep:petgraph"]`
- `signal = ["dep:rustfft"]`
- `compute = ["signal", "dep:rand"]`  （rustfft 经由 signal）
- `ff = ["dep:rustfft", "dep:roxmltree"]`
- `conformer = ["ff", "dep:once_cell", "dep:rand"]`
- `zarr = ["dep:zarrs"]`，`filesystem = ["zarr", "zarrs/filesystem"]`，`blas = ["dep:ndarray-linalg"]`，`rayon = ["dep:rayon"]`（default），`full = ["io","compute","smiles","ff","conformer","signal"]`，`default = []`

**关键成本（必须落到 manifest）**：此前每个子 crate 里"白送"的子系统独占依赖（`flate2`/`once_cell`/`roxmltree`/`rustfft`/`rand`/`petgraph`）合并后必须 `optional = true` 并 feature 接线，使得关掉某子系统的 build **不编译**其独占依赖。共享依赖由其任一 owner 启用：`rustfft`（signal+ff+compute）、`rand`（compute+conformer）、`once_cell`（io+conformer）。`ndarray`/`slotmap`/`smallvec`/`serde`/`serde_json`/`libm` 是 core 常开依赖，保持非 optional。

**Binder 重指**（4 crate 保持独立，仅改依赖；必须与源码搬迁在同一变更内完成，否则其 path 依赖断裂）：
- `molrs-ffi`（lib `molrs_ffi`，独立 workspace）：`molcrafts-molrs-core{default-features=false}` → `molcrafts-molrs{default-features=false}`（只拿 core）；deprecated `f64/i64/u64` forward 重指合并 crate。
- `molrs-cxxapi`（lib `molrs_cxxapi`）：core + `molrs_io` + molrs-ffi → `molcrafts-molrs{features=["io"]}` + molrs-ffi；其 `zarr=["molrs_io/filesystem"]` → `["molcrafts-molrs/filesystem"]`；源码 2 处 `molrs_io::` → `molrs::io::`。
- `molrs-wasm`（package `molrs`，lib `molwasm`）：core(zarr) + molrs-ffi + io/compute/conformer(optional) → `molcrafts-molrs{default-features=false, features=["zarr"]}` + molrs-ffi；自身 `io=["dep:molrs-io"]` → `io=["molcrafts-molrs/io"]`，compute/conformer/smiles 同理。
- `molrs-python`（lib `molrs_python`）：core(zarr)+io(filesystem,smiles)+ff+compute+conformer+signal+molrs-ffi → `molcrafts-molrs{features=["full","zarr","filesystem","smiles"]}` + molrs-ffi。

**测试迁移（遵守 CLAUDE.md IO Testing Rules）**：每个成员的 `tests/` 树搬入合并 crate 的 `molrs/tests/`，镜像模块布局（如 `molrs/tests/io/data/<format>.rs`、`molrs/tests/io/common.rs`）。io 测试目标本地 `common` helper（`common::{tests_data_dir,data_path,format_files}` 经 `CARGO_MANIFEST_DIR/../tests-data` 或 `$MOLRS_TESTS_DATA` 解析）必须保留；因合并 crate `molrs/` 仍在 workspace root 下一层（与旧 `molrs-io/` 同深），`../tests-data` 解析不变。`src/` 内 `#[cfg(test)]` 单元测试随模块迁移、保持纯函数（不读真实文件）。

**版本与 crates.io 注意**：6 个子 crate 名已（或将）在 crates.io 以 0.1.0 存在且被 molpack pin。合并后它们从 members 移除即停止发布——这对任何 crates.io 上子 crate 名的消费者（今天只有 molpack）是 **breaking change**。需对 `molcrafts-molrs` 做一次 minor/major 版本跳并协调 molpack bump。**不要 yank** 已发布的子 crate 版本——molpack 的 pin 在其迁移前仍解析到已发布的 0.1.0。

## Files to create or modify

- `Cargo.toml` — `[workspace] members` 移除 `molrs-core`/`molrs-io`/`molrs-signal`/`molrs-compute`/`molrs-ff`/`molrs-conformer`（保留 `molrs`、`molrs-cxxapi`、`exclude=["molrs-ffi"]`）；移除已无成员的 `[workspace.dependencies]` 子 crate 别名（`molrs`/`molrs_io`/`molrs_compute`/`molrs_ff`/`molrs_conformer`/`molrs_signal`）。
- `molrs/Cargo.toml` — 折叠 6 子 crate 的全部依赖；落实端状态 feature 表（独占依赖 `optional=true` + feature 接线）；移除 `dep:molrs_*` 依赖项。
- `molrs/src/lib.rs` — `pub use molrs_core::*;` → `pub mod core; pub use crate::core::*;`；每个 `pub use molrs_X as X;` → `#[cfg(feature="X")] pub mod X;`；`smiles` 指向 `crate::io::smiles`。
- `molrs/src/core/` (new) — 吸收 `molrs-core/src/` 全部（含 `data/mmff94.xml` 资源路径）。
- `molrs/src/io/` (new) — 吸收 `molrs-io/src/`（含 `smiles/`）。
- `molrs/src/signal/` (new) — 吸收 `molrs-signal/src/`。
- `molrs/src/compute/` (new) — 吸收 `molrs-compute/src/`。
- `molrs/src/ff/` (new) — 吸收 `molrs-ff/src/`。
- `molrs/src/conformer/` (new) — 吸收 `molrs-conformer/src/`。
- `molrs/tests/` (new tree) — 迁移 6 成员的 `tests/`（`core/`、`io/`(+`io/common.rs`)、`signal/`、`compute/`、`ff/`、`embed/`），镜像模块布局；保留 io `common` + `../tests-data` 解析。
- `molrs/benches/` (new tree) — 迁移 `molrs-core/benches/` 与 `molrs-compute/benches/`，更新 `[[bench]]` 入口至 `molrs/Cargo.toml`。
- `molrs-ffi/Cargo.toml` — `molcrafts-molrs-core` → `molcrafts-molrs{default-features=false}`；`f64/i64/u64` forward 重指。
- `molrs-cxxapi/Cargo.toml` — 依赖与 `zarr` feature 重指到 `molcrafts-molrs`。
- `molrs-cxxapi/src/lib.rs` — 2 处 `molrs_io::` → `molrs::io::`。
- `molrs-wasm/Cargo.toml` — 依赖与 `io/compute/conformer/smiles` feature 重指到 `molcrafts-molrs`。
- `molrs-python/Cargo.toml` — 6 子 crate 依赖合为 `molcrafts-molrs{features=["full","zarr","filesystem","smiles"]}` + molrs-ffi。
- `.github/workflows/publish.yml` — 7 个 cargo-publish job（`publish-core/io/signal/compute/ff/conformer/facade`）折叠为 1 个 `publish-molrs`（发 `molcrafts-molrs`）；`publish-wasm`/`build-python`/`publish-python` 不变；更新拓扑注释。
- `.claude/specs/INDEX.md` — 新增本 spec 行（newest on top）。

## Tasks

> 以下为**单次原子变更**内的执行顺序；在全部完成前不要期待中间态可 `cargo check`（见 Summary 的依赖环说明）。

- [x] Move all six member src/ trees into molrs/src/{core,io,signal,compute,ff,conformer} in one pass (molrs/src/)
- [x] Rewrite all cross-crate refs to crate:: paths: molrs::/molrs_core:: → crate::core::, molrs_io:: → crate::io::, molrs_ff:: → crate::ff::, molrs_compute:: → crate::compute::, molrs_conformer:: → crate::conformer::, molrs_signal:: → crate::signal:: (~807 refs, 166 files incl. doctests)
- [x] Rewrite facade re-exports to feature-gated modules in molrs/src/lib.rs (pub mod core + pub use crate::core::*; #[cfg(feature="X")] pub mod X; smiles → crate::io::smiles)
- [x] Update molrs/Cargo.toml: fold all sub-crate deps, mark subsystem-unique deps optional (flate2/once_cell/roxmltree/rustfft/rand/petgraph), wire the end-state feature table (io/smiles/signal/compute/ff/conformer/zarr/filesystem/blas/rayon/full)
- [x] Remove the 6 sub-crate dirs from [workspace] members and drop their [workspace.dependencies] aliases in Cargo.toml
- [x] Move the 6 members' tests/ and benches/ into molrs/tests and molrs/benches, preserving the io common helper + ../tests-data resolution and [[bench]] entries (molrs/tests/, molrs/benches/)
- [x] Repoint molrs-ffi and molrs-cxxapi Cargo.toml to molcrafts-molrs and rewrite cxxapi's molrs_io:: refs (molrs-ffi/Cargo.toml, molrs-cxxapi/Cargo.toml, molrs-cxxapi/src/lib.rs)
- [x] Repoint molrs-wasm and molrs-python Cargo.toml to molcrafts-molrs with matching features (molrs-wasm/Cargo.toml, molrs-python/Cargo.toml)
- [x] Collapse publish.yml's 7 Rust cargo-publish jobs into one publish-molrs job and update the topology comment (.github/workflows/publish.yml)
- [x] Add the INDEX entry and the breaking-version-bump + molpack-follow-up note (.claude/specs/INDEX.md)
- [x] Run full check + test suite (cargo fmt --all --check; cargo clippy -- -D warnings; cargo check) plus cargo test --all-features and per-feature-subset compile checks, then delete the six emptied sub-crate dirs

## Testing strategy

- **Happy path**: `cargo test --all-features` 整套 workspace 测试（当前 ~1248 个为下限）对合并后 `molcrafts-molrs` 全绿；`cargo build` 与 `cargo check --all-features` 无 warning（clippy `-D warnings`）。
- **Feature-subset 编译**: `cargo check --no-default-features` 以及每个单 feature（`--features io`、`--features ff`、`--features signal`、`--features compute`、`--features conformer`、`--features smiles`）各自编译通过；`--features full` 等价于旧门面 full。
- **依赖隔离（关键成本验证）**: 关掉某子系统的 build 不编译其独占依赖——`cargo tree -p molcrafts-molrs --no-default-features --features ff` 中**无** `flate2`/`petgraph`/`once_cell`（io 独占）节点；`--features io` 不拉 `roxmltree`/`rustfft`（ff/signal 独占）；`--no-default-features` 不拉任何 optional 子系统依赖。
- **IO 数据驱动不退化**: io 测试仍经本地 `common::format_files("<format>")` 遍历 `tests-data/<format>/` 每个真实文件（非硬编码子集），`../tests-data` / `$MOLRS_TESTS_DATA` 解析在新 `molrs/tests/` 位置仍成立。
- **下游公共路径不变（API 表面零变化）**: 编译一个引用 `molrs::Frame`、`molrs::io::read_xyz`、`molrs::smiles::*`、`molrs::ff::*`、`molrs::conformer::Conformer`、`molrs::compute::*`、`molrs::signal::*` 的探针（doctest 或 binder build）全部解析通过。
- **Binder build**: `molrs-ffi`、`molrs-cxxapi`、`molrs-wasm`（`wasm-pack build` 或 `cargo check --target wasm32-unknown-unknown`）、`molrs-python`（`maturin build` 或 `cargo check`）各自对合并 crate 构建通过。
- **Edge cases**: workspace 中仅 `molcrafts-molrs` 一个可发布 lib member（former 6 已移除）；docs.rs 配置 `features=["full"]` 未改动；`rayon` 仍为 default 转发；6 子 crate 名**未被 yank**（molpack 0.1.0 pin 仍可解析）。
- **CI**: `publish.yml` Rust 侧恰好 1 个 `cargo publish`（`molcrafts-molrs`）；wasm-npm 与 python-wheel matrix 不变。

## Out of scope

- molpack 的本仓库迁移（`molcrafts-molrs-core`+`molcrafts-molrs-io` → `molcrafts-molrs{default-features=false}`+`io` feature、10 处 `molrs_io::`→`molrs::io::` 重写）——molpack 在独立 repo，作为 companion change 在 molrs 发布后处理；本 spec 必须 **不破坏** molpack 当前 pin 的 0.1.0（故不 yank 子 crate 名）。
- 任何公开 API 重设计 / 符号增删（行为与 API 表面逐项不变）。
- 任何性能改动 / 算法改动 / 单位或科学行为改动。
- `molrs-capi`（独立 cdylib，非 workspace member，非 Rust 消费者）——不在合并范围。
- 把任一子系统拆得更细或调整模块边界——仅做"crate→module"物理吸收，模块内部结构原样保留。

## Acceptance Criteria

See `workspace-single-crate-merge.acceptance.md` for the binding contract (10 criteria, types: code / runtime / docs).
