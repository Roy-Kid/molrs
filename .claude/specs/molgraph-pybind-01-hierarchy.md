---
title: 暴露 MolGraph 层级到 Python（Graph ← Atomistic / CoarseGrain）+ 绑定扩面 + impropers
status: code-complete
created: 2026-06-01
chain: molgraph-pybind
chain_index: 1
depends_on: []
---

# 暴露 MolGraph 层级到 Python + 绑定扩面 + impropers

## Summary

`molrs-core` 已经有「通用图作基、AA/CG 作 newtype 特化」的层级：

- `molgraph.rs::MolGraph` — 通用图（节点是 atoms **或** beads；bonds/angles/dihedrals；
  每实体一个 `props` 属性袋）。
- `atomistic.rs::Atomistic(MolGraph)` — `Deref`/`DerefMut` 特化，不变量「每节点有 `element`」。
- `coarsegrain.rs::CoarseGrain(MolGraph)` — 特化，不变量「每节点有 `bead_type`」，有 `add_bead`。

但 `molrs-python` 只导出了一个**扁平的** `Atomistic`（`#[pyclass(name="Atomistic")]`），
方法面只有 7 个（`add_atom`/`set_atom_prop`/`add_bond`/`set_bond_order`/`n_atoms`/`n_bonds`/
`to_frame`），既看不到通用基 `MolGraph`，也看不到 `CoarseGrain`，更没有角/二面角/迭代/
属性读取等。后果：molpy 只能在 Python 里平行重写一套图，再用会丢信息的
`to_molrs`/`from_molrs` 胶水来回拷贝（见 molpy spec
[`atomistic-cg-on-molrs-molgraph`](../../../molpy/.claude/specs/atomistic-cg-on-molrs-molgraph.md)）。

本 spec（molgraph-pybind 链 1/1）做 molpy 重构的**先决项 P0**：

1. 在 `molrs-core` 补 `Improper`（与 `Angle`/`Dihedral` 对称），使四级拓扑完整。
2. 在 `molrs-python` **如实暴露这条层级**：通用基 `Graph`（`#[pyclass(subclass)]`）←
   `Atomistic` / `CoarseGrain`（`#[pyclass(extends=Graph)]`），并把绑定面补齐到「足以
   背书一个全功能图容器」。
3. 让 molpy 的 `Atomistic`/`CoarseGrain` 能直接子类化 `molrs.Graph`（spike 已验证可行，
   见 §Domain basis）。

不含 molpy 侧改写（那是 molpy spec 的 P1–P5）。

## Domain basis

- molpy 父设计 spec：`molpy/.claude/specs/atomistic-cg-on-molrs-molgraph.md`（层级模型、
  绑定最小集 §5、决策 D2 impropers / D6 层级落点）。
- **PyO3 子类化 spike（2026-06-01，已验证）**：给 `PyAtomistic` 加 `#[pyclass(..., subclass)]`
  重建后，Python 端 `class Struct(molrs.Atomistic, SpatialMixin, MembershipMixin)` +
  `class Atomistic(Struct)` 成立：协作 MRO（`Atomistic → Struct → molrs.Atomistic →
  SpatialMixin → MembershipMixin → object`）、实例化、`isinstance`、Rust 方法/属性、
  mixin 经 `self` 调 Rust、Rust 背书实例上挂 Python 属性、领域子类方法——全部生效；
  既有 `to_molrs`/embed 不回归（CAT 仍 n=32）。**唯一约束**：molrs `#[new]` 不收参数，
  子类需 `__new__` 垫片——本 spec 通过让 `#[new]` 接受并忽略 `*args/**kwargs` 消除该约束。
- PyO3 继承机制：`#[pyclass(subclass)]` 基类 + `#[pyclass(extends=Base)]` 子类。
- 既有绑定风格参考：`molrs-python/src/molgraph.rs`（`set_atom_prop` 通用 setter 范式、
  `index_to_atom_id`/`atom_id_to_index` 的 index↔AtomId 映射、`molrs_error_to_pyerr`）。

## Design

### 1. molrs-core：补 `Improper`（对称 Angle/Dihedral）

`molgraph.rs` 现有 `Angle{atoms:[AtomId;3], props}` + `AngleId`、`Dihedral{atoms:[AtomId;4],
props}` + `DihedralId`，及 `add/remove/get/iter/n_*`、`remove_atom` 级联删除。照搬一套：

- `pub struct ImproperId;`（slotmap key）、`pub struct Improper { pub atoms: [AtomId;4], pub props: HashMap<String,PropValue> }`。
- `MolGraph` 加 `impropers: SlotMap<ImproperId, Improper>`；`add_improper(i,j,k,l)` /
  `remove_improper` / `get_improper` / `get_improper_mut` / `impropers()` 迭代 / `n_impropers`。
- `remove_atom` 级联删除引用该原子的 impropers（与 angles/dihedrals 同处补一段）。
- `to_frame`（`molrs-core` 内）输出 impropers 块（与 angles/dihedrals 块对称），
  使 Python `to_frame()` 拿到完整四级拓扑。

> 选择：在 core 加一等 `Improper`（D2 方案 A），而非「带 `kind` 属性的 dihedral」。
> 理由：保持四级拓扑对称 + `to_frame` 无特判；改动量与 Angle/Dihedral 一条线对称。

### 2. molrs-python：暴露层级 + 绑定扩面

#### 2.1 三个 pyclass（继承链）

```rust
#[pyclass(name = "Graph", subclass, unsendable)]
pub struct PyGraph { pub(crate) inner: MolGraph }   // 通用基，承载下表所有方法

#[pyclass(name = "Atomistic", extends = PyGraph, unsendable)]
pub struct PyAtomistic;   // 仅 add_atom 便捷 + element 不变量（构造时校验/写入）

#[pyclass(name = "CoarseGrain", extends = PyGraph, unsendable)]
pub struct PyCoarseGrain; // 仅 add_bead 便捷 + bead_type 不变量
```

- `PyGraph::#[new]` 签名 `(*args, **kwargs)` 并**忽略**多余参数 —— 这样 molpy 的
  `Struct(molrs.Graph, *Mixins)` 子类无需写 `__new__` 垫片（消除 spike 发现的唯一约束）。
- `Atomistic`/`CoarseGrain` 经 `extends=PyGraph` 继承全部通用方法；各自只加领域便捷构造
  （`add_atom(symbol,...)` 写 `element`；`add_bead(bead_type,...)` 写 `bead_type`）。
- **向后兼容**：现有 Python 代码 `molrs.Atomistic()` + 7 个老方法签名不变（老方法变成继承自
  `Graph`，签名一致）。`to_molrs` 等调用方零改动。

#### 2.2 通用方法面（挂在 `Graph` 上，转发 `molrs-core`，无新算法）

| 类别 | 新增 Python 方法（转发目标） |
|---|---|
| 原子属性 | `get_atom_prop(i,key)`、`set_atom_prop(i,key,value)`（已实现，通用 setter 范式）、`del_atom_prop(i,key)`、`atom_keys(i)`、`atom_column(key)→list`（批量列） |
| 原子 | `remove_atom(i)`、`atom_ids()`/`iter_atoms()`、`neighbors(i)→list[int]` |
| 键 | `get_bond_atoms(b)`、`get_bond_prop/set_bond_prop/del_bond_prop/bond_keys`、`remove_bond`、`bond_ids` |
| 角 | `add_angle(i,j,k)`、`get_angle_atoms`、`get/set/del_angle_prop`、`angle_keys`、`remove_angle`、`angle_ids`、`n_angles`(getter) |
| 二面角 | `add_dihedral(i,j,k,l)` + 同上一套、`n_dihedrals` |
| improper | `add_improper(i,j,k,l)` + 同上一套、`n_impropers` |
| 几何 | `translate(vec3)`、`rotate(axis,angle,about=None)`（core 已有）、`coords()→Nx3`、`set_coords(arr)` |
| 合并 | `extend(other: Graph)`（按 id 偏移并入另一张图的原子/键/角/二面角/improper + props） |

- 所有 `*_prop` 的 value 走通用 `int|float|str → PropValue::{Int(i32)|F64|Str}` 派发
  （沿用已实现的 `set_atom_prop`），**不为单属性开命名参数**（formal_charge 教训）。
- index↔Id：对外一律暴露**稳定 0 基 index**（沿用现有 `index_to_atom_id`/
  `atom_id_to_index`），角/键/二面角/improper 同理给各自的 index 视图。

#### 2.3 注册 + stub

- `molrs-python/src/lib.rs`：`add_class::<PyGraph>()` + `PyAtomistic` + `PyCoarseGrain`。
- `molrs-python/python/molrs/*.pyi`：补 `Graph`/`CoarseGrain` + 新方法的类型存根。

## Files

- `molrs-core/src/molgraph.rs` — 加 `Improper`/`ImproperId` + add/remove/get/iter/n + remove_atom 级联 + to_frame 块。
- `molrs-python/src/molgraph.rs` — 重构为 `PyGraph`(基) + `PyAtomistic`/`PyCoarseGrain`(extends)，补 §2.2 全部方法。
- `molrs-python/src/lib.rs` — 注册三个类。
- `molrs-python/python/molrs/__init__.py` + `*.pyi` — 导出 + 存根。
- `molrs-core/tests/` + `molrs-python` 内联/pytest — 见 Testing。

## Tasks

- [ ] T1 core：`Improper` 数据结构 + CRUD + 级联 + to_frame；`cargo test -p molcrafts-molrs-core`。
- [ ] T2 pybind：`PyGraph` 基类（`#[new]` 吞 `*args/**kwargs`）承载现有 7 法 + §2.2 新法。
- [ ] T3 pybind：`PyAtomistic`/`PyCoarseGrain` 改 `extends=PyGraph`，保留领域便捷构造。
- [ ] T4：`lib.rs` 注册 + `.pyi` 存根。
- [ ] T5：`maturin develop --release` 重建；跑 §Testing。
- [ ] T6：molpy 回归——`to_molrs`/embed/`tests/test_embed`、`test_io`、`test_parser` 全绿（向后兼容）。

## Testing / 验收

见 `molgraph-pybind-01-hierarchy.acceptance.md`。要点：

- molrs-core：impropers 的 CRUD + remove_atom 级联 + to_frame 块，`cargo test` 通过。
- 层级：`molrs.Graph`/`Atomistic`/`CoarseGrain` 存在；`issubclass(molrs.Atomistic, molrs.Graph)`；
  `class S(molrs.Graph, Mixin): pass; class A(S): pass; A()` 可实例化且 Rust/mixin 方法都生效
  （spike 的正式化，且**无需** `__new__` 垫片）。
- 绑定面：每个新方法一条 round-trip（建图→读回→等值）。
- 向后兼容：molpy `to_molrs`/embed 不回归（CAT n=32、带电单体氢数正确）。

## Out of scope

- molpy 侧 `Atomistic`/`CoarseGrain`/`Entity` 改写、删 `_molrs.py`（molpy spec 的 P1–P5）。
- `perceive_topology()` 把 get_topo 下沉 Rust（molpy spec D3，后续）。
- 零拷贝列 buffer（molpy spec D4，先用 `atom_column`/`to_frame` 兜底）。
- 任何 CG 投影/派生量算子（cg-redesign spec NG2）。
