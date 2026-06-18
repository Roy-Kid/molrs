---
title: OPLS-AA SMARTS 原子分型下沉 molrs（typing-meta + SMARTS 分型）
status: approved
created: 2026-06-18
---

# OPLS-AA SMARTS 原子分型下沉 molrs

> OPLS typifier 链 1/3（+ 共享 [ff-parameter-estimator](ff-parameter-estimator.md)）。**反转** [opls-ef-01-kernels-seam](opls-ef-01-kernels-seam.md) 的"typifier 不下沉（B线）"决策——Rust SMARTS 引擎（`core/chem/smarts`）现已具备，foyer 式 OPLS 分型可在 Rust 落地。镜像 `ff/typifier/mmff` 的"分型元数据与势能参数分开读"范式。

## Summary

把 OPLS-AA 的**原子分型**从 molpy（`typifier/atomistic._OplsAtomTypifier` + `graph.SMARTSGraph`）下沉到 molrs Rust。现有 `read_opls_xml`（`ff/forcefield/readers/opls.rs`）只读 `name`+`mass`，**故意丢弃**了分型必需的 SMARTS 元数据（reader 注释："Reconciling class↔type per atom is the typifier's job"）。本 spec：(1) 新增 `OplsTypingMeta`（每个 `opls_NNN` → {class, SMARTS `def`, overrides, priority, layer}），经一个读取器从同一 OPLS XML 解析（镜像 `read_mmff_params_xml_str`/`MMFFParams`）；(2) 新增 `ff/typifier/opls/`，用 molrs 原生 SMARTS（`SmartsPattern::parse` → `find_matches`）+ overrides/priority/layer 排序做原子分型，产出每原子带 `type`/`class`/`charge` 的 labeled `Atomistic`。成键参数赋值在链 2/3（[opls-typifier-02-assign](opls-typifier-02-assign.md)）。

## Domain basis

foyer/OpenMM 式 OPLS-AA 分型：每个原子类型携带一个 SMARTS `def` 模式与 `overrides` 列表；一个原子可匹配多个类型，按**特异性优先级**择一。molpy `_OplsAtomTypifier._extract_patterns` 的优先级算法（本 spec 须逐位复刻）：

- 显式 `priority` 属性优先采用（整数）。
- 否则按 overrides 关系计算：`priority -= 1`（每被一个其它类型 override）；`priority += len(overrides)`（本类型 override 了几个）。
- overlay 层：`priority += int(layer) * _LAYER_PRIORITY_STRIDE`（CL&P/CL&Pol 读在 OPLS-AA 之上，高层无条件压低层）。

仅携带 `def` 的类型参与自动分型；legacy 行（oplsaa.xml 1–134 行，无 `def`）不可被 SMARTS 匹配，仅供手工赋型/读 LAMMPS（写入 Out of scope）。SMARTS 语义须与 molpy `parser.smarts` + `graph.SMARTSGraph` 一致（含递归 `$()`，molrs matcher 已支持，`matcher.rs:51-68`）。

## Design

**1. 模块归属**

新建 `molrs/src/ff/typifier/opls/`，镜像 `ff/typifier/mmff/`：
- `mod.rs` — `OplsTypifier { meta: OplsTypingMeta, ff: ForceField }`，实现 `Typifier` trait（`ff/typifier/mod.rs:15`，`fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String>`）。构造 `from_xml_str(xml)`（一次解析出 meta + ff，镜像 `MMFFTypifier::from_xml_str`）。
- `meta.rs` — `OplsTypingMeta`（`HashMap<String, OplsTypeRow>`，key = `opls_NNN`）与 `OplsTypeRow { class: String, def: Option<String>, overrides: Vec<String>, priority: Option<i64>, layer: u32 }`。镜像 `MMFFParams`/`MMFFAtomProp`（`ff/typifier/mmff/params.rs`）。
- `typing.rs` — SMARTS 分型 + 优先级排序（`annotate_opls`，镜像 `frame_builder::annotate_mmff`）。

**2. typing 元数据读取（plumb 现已丢弃的属性）**

在 `ff/forcefield/xml.rs` 增 `read_opls_typing_xml_str(xml) -> Result<OplsTypingMeta, String>`（镜像 `read_mmff_params_xml_str`，line 94）：扫 `<AtomTypes>` 的每个 `<Type>`，读 `name`/`class`/`def`/`overrides`/`priority`/`layer`。`read_opls_xml`（势能参数）**保持不变**——typing-meta 是独立的"later sink"，与势能 ForceField 解耦，二者从同一 XML 读出但各管各的。

**3. 原子分型（`annotate_opls`）**

- 对每个携带 `def` 的 `OplsTypeRow`，`SmartsPattern::parse(def)`（`core/chem/smarts/mod.rs:74`）编译一次（缓存于 typifier 构造期），失败 fail-fast `Err`（对齐 molpy "broken FF def → raise"）。
- 对分子跑 `pattern.find_matches(mol)`（`matcher.rs:252`，`Vec<Vec<AtomId>>`），target 原子取每个 match 的 root（query-atom 0）。
- 每个原子收集所有命中类型，按上述 priority（含 overrides/layer）取最高者；并列时报错或按确定性 tiebreak（与 molpy 行为对齐——见 parity 链 3/3）。
- 写回 labeled `Atomistic`：每原子 `type`=`opls_NNN`、`class`=其 class、`charge`=（由势能 ForceField 的 atom style 按 type 解析，`get_style("atom","full").get_atomtype(type).params.get("charge")`）。
- 返回 labeled `Atomistic`（成键项的 type 标注留给链 2/3）。`to_frame()`（`atomistic.rs:484`）由 consumer 调。

**4. 复用既有基础设施（不重造）**

SMARTS 解析/匹配全用 `core/chem/smarts`（`SmartsPattern::parse`/`find_matches`/`has_match`，递归 `$()` 已支持）。元素/键/键级/芳香性走 `Atomistic` 现有访问器（`get_atom`→`element` prop、`bonds()`/`bond_endpoints()`、bond `order`/`is_aromatic` prop）。**不**引入新 SMARTS 引擎，**不**改 `read_opls_xml` 的势能解析。

## Files to create or modify

- `molrs/src/ff/typifier/opls/mod.rs` (new) — `OplsTypifier` + `Typifier` impl + `from_xml_str`。
- `molrs/src/ff/typifier/opls/meta.rs` (new) — `OplsTypingMeta` + `OplsTypeRow`。
- `molrs/src/ff/typifier/opls/typing.rs` (new) — `annotate_opls`（SMARTS 分型 + 优先级排序）。
- `molrs/src/ff/typifier/mod.rs` — `pub mod opls;` + re-export `OplsTypifier`。
- `molrs/src/ff/forcefield/xml.rs` — 新增 `read_opls_typing_xml_str`。
- `molrs/tests/ff/typifier/opls.rs` (new) — 分型集成测试（迭代 `tests-data/` 真实分子，禁合成 happy-path 数据）。

## Tasks

- [ ] Write failing tests for OplsTypingMeta parse (class/def/overrides/priority/layer round-trip from real oplsaa.xml rows) in molrs/tests/ff/typifier/opls.rs
- [ ] Implement read_opls_typing_xml_str in molrs/src/ff/forcefield/xml.rs and OplsTypingMeta/OplsTypeRow in molrs/src/ff/typifier/opls/meta.rs
- [ ] Write failing tests for priority resolution (explicit priority, overrides +/-, layer stride) matching molpy _OplsAtomTypifier
- [ ] Implement priority/overrides/layer ranking in molrs/src/ff/typifier/opls/typing.rs
- [ ] Write failing tests for SMARTS atom typing on real molecules (tests-data) including recursive SMARTS def_ patterns
- [ ] Implement annotate_opls (SmartsPattern::parse cache + find_matches + per-atom best-type pick + type/class/charge write-back) and OplsTypifier::typify
- [ ] Add rustdoc with units/conventions for OplsTypifier public API; note legacy-no-def rows are out of scope
- [ ] Run cargo fmt --all --check && cargo clippy -- -D warnings && cargo test --all-features

## Testing strategy

- **Unit** — `OplsTypingMeta` parse: real `<Type>` rows (def/overrides/priority/layer) round-trip; malformed `def` → `Err`.
- **Unit** — priority algorithm parity with molpy: constructed override/layer cases produce the same winning type.
- **Integration (real data)** — `annotate_opls` over molecules in `tests-data/` (per MANDATORY IO rule: iterate real files, no synthetic happy-path); assert per-atom `opls_NNN` types match molpy `OplsTypifier` reference (full per-atom parity is the chain-3 gate).
- **Edge** — recursive `$()` SMARTS def matches; an atom with no matching def → consistent behavior (Err or unassigned per molpy).
- **Build smoke** — fmt/clippy/check/test green.

## Out of scope

- **成键参数赋值**（bond/angle/dihedral matching）→ 链 2/3 [opls-typifier-02-assign](opls-typifier-02-assign.md)。
- **缺参估计** → [ff-parameter-estimator](ff-parameter-estimator.md)。
- **legacy 无-`def` 行（oplsaa.xml 1–134）的自动分型** — 不可 SMARTS 匹配，排除；仅现有手工/LAMMPS 路径。
- **CL&P / CL&Pol 专属类型表的 curate** — layer 机制保留接口，具体表 v2。
- **改动 `read_opls_xml` 的势能解析** — 不动，typing-meta 独立读。
- **PyO3 暴露 `PyOplsTypifier`** — 链 3/3 parity 通过后随 molpy 消费 spec 一起；本 spec 只到 Rust 内部 API。
