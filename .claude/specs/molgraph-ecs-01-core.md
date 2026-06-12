---
title: MolGraph 重构为 ECS（molrs-core）——纯数据 world + 对齐列存 + 系统即自由函数
status: draft
created: 2026-06-08
chain: molgraph-ecs
chain_index: 1
owner: molrs-core
supersedes: molgraph-ecs（无后缀，旧稿）
---

# molgraph-ecs-01-core

> Chain `molgraph-ecs`：**01-core**（本 spec，molrs-core）→ 02-pybind（molrs-python 暴露
> entity 句柄 + 零拷贝列 + 自由函数）→ 03-molpy（molpy 句柄视图）。**一条 arc，底层彻底重构，
> 不兼容旧 API/旧 to_frame/旧调用点**——这些随重构一起改写。

## 1. Overview

把 `MolGraph`/`Atomistic`/`CoarseGrain` 重构为 **ECS world**：

- **数据即 world**：图实例**就是** world——持 entity 句柄 + 对齐列存 component + kind-tagged 关系。
  **每结构一个 world**，无全局单例、无 scheduler、无 tick。用户写 `Atomistic()` 即用，**永不手创
  World**。
- **system 即自由函数**：一切算法/变换/投影是吃 `&/&mut world` 的**自由函数**
  （`perceive_aromaticity(mol)`、`translate(mol, d)`、`to_frame(mol)`）。数据结构上**不挂领域算法
  方法**，只留存储原语。
- **零硬编码字段**：不散落字段字面量——**引用内置 key/field 约定的常量**（§3.4），**直接按约定名
  访问、缺了/类型错就报错**（无 `unwrap_or` 兜底、无 `"X"` 占位）。

现状 `SlotMap<NodeId, Atom{HashMap<String,PropValue>}>`（AoS）导致每属性访问一次 HashMap+FFI、
无法零拷贝批处理、几何写死 `x/y/z`、数据与行为耦在对象上。molrs **已约 90% 是自由函数**
（`perceive_aromaticity`/`find_rings`/`gasteiger`/`add_hydrogens`/`find_chiral_centers`/
`detect_rotatable_bonds` 均自由函数），只差「列存 + 剥几何/frame 方法包装 + 统一字段约定」。

## 2. Domain basis

- ECS：**Entity** = 纯 id（稳定 generational `SlotMap` 键 `NodeId`/`RelationId`）；**Component** =
  按实体的列数据；**System** = 吃 world 的函数。**朴素 ECS**——不要 archetype 自动迁移、scheduler、
  query planner（分子拓扑相对静态，重型 ECS 是过度工程）。
- **列存底座新建，不复用 `Block`**：`Block` 强制各列等行数（`block/mod.rs:218` ragged 即报错），
  与「按实体可缺」相反。正解 = **共享行序的对齐稠密列 + 每列 null 掩码**（Arrow/dataframe 模型）。
- 字段约定：molpy `core/fields.py` 已有 canonical 名（element/charge/mol_id/x/y/z/mass/type/…）；
  molrs-core 引入对应**常量约定模块**，两侧对齐、引用而非字面量。

## 3. Design

### 3.1 World = 图实例

```rust
pub struct MolGraph {                  // = world（Graph/Atomistic/CoarseGrain 同此核）
    nodes: EntityTable,                // 句柄↔行 + 节点 component 列（§3.3）
    relations: Vec<EntityTable>,       // 每 kind 一张：句柄↔行 + endpoints + 该 kind 的 component 列
    kind_name: Vec<String>, kind_arity: Vec<usize>, name_to_kind: HashMap<String, KindId>,
    // 无全局 WORLD、无单例、无 system 注册表、无 coord_keys
}
```
节点与每个 relation kind 都是同一个 `EntityTable`（句柄↔行 + 对齐列存）——node/relation component
走**同一套**机制，消解 dual-store 同步问题。`Atomistic`/`CoarseGrain` = 预注册领域 kind 的 world
（bonds/angles/dihedrals/impropers；CG: bonds）+ 领域 builder（§3.5）；`Atomistic()` = 建空 world +
注册 kind，列**用才长**。

### 3.2 数据结构 API = 只剩存储原语（方法）

`MolGraph`/叶子保留为方法的仅 world 存储接口：
- entity：`spawn() -> NodeId`、`despawn(NodeId)`、`entities()`、`contains(NodeId)`。
- component（**类型化访问器**，非泛型 `get<T>`，避免 downcast）：`get_f64/get_i32/get_str(e, key)
  -> Result<…>`（缺/类型错 = `Err`）、`set_f64/…(e, key, v)`、`has(e, key)`、`remove(e, key)`、
  `column_f64(key) -> &[f64]` + `validity(key) -> &Bitmap`（零拷贝对齐切片，长度 = n_rows）。
- relation：`register_kind(name, arity)`、`add_relation(kind, &[NodeId])`、`relations(kind)`、
  `get_relation`/`remove_relation`、`n_relations(kind)`；relation prop 走同一列存（按 (kind, 行)）。
- world 间：`adopt(other)`（单图**移动**，零拷贝；§02 用）、`merge(other)`（合两图，**必重铸**
  world-局部句柄 + 行追加，O(N)，非零拷贝）。

**移除（改自由函数，§3.4）**：`translate`/`rotate`/`to_frame`/`from_frame`/`read_frame` 及任何
计算/感知/变换/投影方法。grep 确认 `impl MolGraph`/`impl Atomistic` 仅存储原语。

### 3.3 Component = 共享行序对齐稠密列 + 每列 null 掩码

```rust
struct EntityTable {
    keys: SlotMap<NodeId, u32>,        // 句柄 → 行号(packed)，O(1) 随机
    rows: Vec<NodeId>,                 // 行号 → 句柄（迭代序 = 对齐序）
    cols: HashMap<String, Column>,
}
enum Column { F64(Vec<f64>, Bitmap), I32(Vec<i32>, Bitmap), Str(Vec<String>, Bitmap), Bool(Vec<bool>, Bitmap) }
// 每列长度 == rows.len()；第 i 个值对应 rows[i]；Bitmap 标该行是否有此 component（稀疏 = null）。
```
- **共享行序对齐**：一个 `keys` 被所有列共用 → 列 `i` 与列 `i` 同一实体。故 (a) 对 numpy **零拷贝
  且对齐**，(b) `to_frame` 零拷贝直出（列已对齐成表），(c) 按句柄 O(1) 随机。
- **稀疏靠 null 掩码**：实体可有 `charge` 而无 `port`（Bitmap=0）。分子数据多稠密、少稀疏，代价极小。
- **惰性建列、首写定类型**：首次 `set_f64(e,"charge",1.0)` 建 `F64` 列；**类型冲突 = `Err`**
  （后续 `set_str` 同名列报错，不静默改型/coerce——我们控全部调用方）。无 schema 注册、无预声明、
  无 world 初始化。
- **删除 = swap-remove 行 + 更新 `keys`**：外部**句柄稳定**（slotmap 键不变），仅内部行紧凑。

### 3.4 零硬编码字段 = 引用内置约定 + 直接访问 + 错即报错

不发明 coord_keys 运行时通道、不散落 `"x"`/`"element"` 字面量。引入 molrs **字段约定模块**
（与 molpy `core/fields.py` 对齐的 canonical 常量；e.g. `keys::X`/`keys::Y`/`keys::Z`/`keys::ELEMENT`/
`keys::CHARGE`/`keys::BEAD_TYPE`/…）：
- 所有需要具体字段的代码（几何、感知、to_frame、领域 builder）**引用约定常量**，源码无字段字面量
  （grep 通用 `molgraph.rs` 无 `"x"/"y"/"z"/"element"/"bond"` 等字面量）。
- **直接访问、错即报错**：`translate` 读 `keys::X/Y/Z`，节点缺坐标 → `Err`（**不** `unwrap_or(0.0)`）；
  领域 builder 写约定字段；**消除 `symbol = … or "X"` 占位**。
- 约定是单一真相源——引用它不算硬编码；改名只改约定一处。

### 3.5 System = 自由函数；叶子 = 领域 world + 领域 builder

- **System**（吃 `&/&mut world`）：感知/分析 `perceive_aromaticity(&mut mol)`/`find_rings(&mol)`/
  `find_chiral_centers(&mol)`/`compute_gasteiger_charges(&mut mol)`/`add_hydrogens(&mut mol)`；
  几何 `translate(&mut mol, d)`/`rotate(&mut mol, …)`（读约定坐标键）；投影 `to_frame(&mol)`/
  `from_frame`；typify 及 ff/conformer/io 入口同。**无调度器/无 tick——调它才跑**（molrs 现状即此，
  本 spec 只剥残留方法包装 e.g. `Atomistic::perceive_aromaticity`）。
- **叶子**：`Atomistic`/`CoarseGrain` 预注册领域 kind + 领域 builder 自由函数（领域词只在叶子模块）：
  `atomistic::add_atom(&mut mol, element)`（spawn + 写 `keys::ELEMENT`）、
  `atomistic::add_bond(&mut mol, i, j)`（add_relation + 写 `keys::ORDER`=1.0）、
  `coarsegrain::add_bead(&mut mol, ty)`（写 `keys::BEAD_TYPE`）。通用 `Graph` world 不含领域词。

### 3.6 不变量

- 句柄稳定（删其它实体不动本句柄；删后失效可检测）。
- `to_frame(mol)` 按 kind registry 输出（节点块 + 每非空 kind 一块），列由对齐列存零拷贝直出；
  坐标输出为约定标量列（`keys::X/Y/Z`）。round-trip 各 kind 计数一致。
- 零硬编码字段：通用层无字段字面量；缺字段/类型错一律报错。

## 4. Files

- `molrs-core/src/keys.rs`(new) — 字段约定常量（与 molpy fields 对齐）。
- `molrs-core/src/molgraph.rs` — node 存储 → `EntityTable`（对齐列 + null 掩码 + 句柄↔行）；
  `MolGraph` 只留存储原语 + 类型化访问器；剥几何/frame 方法。
- `molrs-core/src/{geometry.rs(new), frame_io}` — `translate`/`rotate`/`to_frame`/`from_frame` 自由
  函数，读约定坐标键、错即报错。
- `molrs-core/src/{atomistic,coarsegrain}.rs` — 叶子 = 领域 world + 领域 builder 自由函数；删方法包装。
- `molrs-core/src/{aromaticity,rings,stereo,hydrogens,gasteiger,rotatable,mapping,smarts}.rs` —
  统一为自由函数 + 类型化列访问 + 引用约定常量（`node.get("x")` → `mol.get_f64(e, keys::X)?`）。
- `molrs-ff`/`molrs-conformer`/`molrs-io` — 所有消费方改列存 API + 约定常量（~300+ 点，inversion 量级）。
- molrs umbrella `pub use` 自由函数 + `keys`。

## 5. Tasks

- [ ] Write failing tests: aligned-column store (zero-copy `&[f64]` aligned to rows + null bitmap, O(1) by-handle, swap-remove keeps handles stable); type-conflict → `Err`; missing field → `Err` (no `unwrap_or`); no field literals in generic `molgraph.rs` (grep)
- [ ] Build `EntityTable` (handle↔row + aligned columns + null masks) + typed accessors; replace AoS node store; relations reuse the same table per kind
- [ ] Add `keys` field-convention module (aligned with molpy fields); route every field access through it; geometry/builders read convention keys and error on missing
- [ ] Move `translate`/`rotate`/`to_frame`/`from_frame` to free functions; strip the `Atomistic::perceive_aromaticity` wrapper; leaves = domain builder free functions
- [ ] Migrate molrs-core/ff/conformer/io consumers to typed column access + convention keys; umbrella re-exports
- [ ] `cargo test --all-features` green; `cargo clippy -- -D warnings`; criterion: build O(N), column access O(1) zero-copy

## 6. Testing strategy

- World/discipline: grep shows no algorithm methods on the data struct; systems are module free functions taking the world; `Atomistic()` + builders build a graph with zero World/registry setup.
- Aligned columns: entity with `charge` not `port` (null bit); `column_f64(keys::CHARGE)` is a contiguous `&[f64]` of length n_rows aligned to `rows`; O(1) by-handle get; build N atoms O(N); swap-remove a middle entity keeps other handles valid.
- Errors not fallbacks: `get_f64(e, keys::X)` on a node without coords → `Err`; `set_str` on an `F64` column → `Err`; no `unwrap_or(0.0)` / no `"X"` placeholder in source.
- No hardcoded fields: generic `molgraph.rs` has no field literals; all canonical names come from `keys`.
- Parity: `perceive_aromaticity`/`find_rings`/`add_hydrogens`/`gasteiger`/`to_frame` results identical to pre-ECS on the existing molrs corpus; `cargo test --all-features` green.

## 7. Out of scope

- Python 暴露（句柄 + 零拷贝列 + 自由函数）→ molgraph-ecs-02-pybind（同 arc，rebase 本 spec）。
- molpy 视图层 → molgraph-ecs-03-molpy（同 arc）。
- 重型 ECS（archetype/scheduler/query planner）——明确不做。
- groups/containment 行为——独立后续 spec（沿用预留占位）。
