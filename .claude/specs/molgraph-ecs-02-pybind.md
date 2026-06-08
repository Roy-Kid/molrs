---
title: 把 ECS world 暴露到 Python（molrs-python）——entity 句柄 + 零拷贝列 + 自由函数 system
status: draft
created: 2026-06-08
chain: molgraph-ecs
chain_index: 2
depends_on: [molgraph-ecs-01-core]
owner: molrs-python
supersedes: molgraph-views-01-stable-handles
---

# molgraph-ecs-02-pybind

> Chain `molgraph-ecs`：01-core（molrs-core ECS）→ **02-pybind**（本 spec）→ 03-molpy。
> 把 01 的 ECS world 如实暴露到 Python：**entity 句柄 = 不透明 int、component 列 = 零拷贝 numpy
> view、system = 模块自由函数**。本 spec 兑现 arc 的 zero-copy 收益。

## 1. Overview

01 让 molrs-core 成 ECS world（句柄 entity + 对齐列存 + 自由函数 system）。本 spec 在 Python 把它
暴露成同形：

- **Entity = 稳定不透明句柄（int）**：`spawn()` 返回稳定句柄（generational `SlotMap` 键的 ffi `u64`）；
  删其它实体不使本句柄失效、无位置平移；失效句柄读写报错。取代旧绑定「迭代序 0-based 位置」。
- **Component 列 = 零拷贝 numpy view**：`column(key)` 直接把对齐稠密列（+validity）映射成 numpy 数组
  （写穿到 Rust），批处理 system 一次取整列、无逐实体 FFI。
- **System = 模块自由函数**：`molrs.perceive_aromaticity(mol)`、`molrs.translate(mol, d)`、
  `molrs.to_frame(mol)`——不在类上挂领域算法方法。
- **叶子可子类化**：`PyAtomistic`/`PyCoarseGrain` 标 `subclass`（03 的 molpy 以叶子为基类的前置）。
- **零拷贝 `adopt`**：`adopt(dst, src)` 移动单图存储，供 03 零拷贝接管 molrs 产出的图。

## 2. Domain basis

- 依赖 01：句柄 = entity id、对齐列存（零拷贝切片 + validity）、字段约定 `keys`、自由函数 system。
- `slotmap` 键 ↔ `u64`（`as_ffi`/`from_ffi`，generational，失效可检测）。
- 零拷贝：01 的列是连续 `&[f64]`（按行对齐）+ bitmap → PyO3 `numpy`/buffer 协议映射成 view；
  对象生命周期与借用安全由持有 `Py<PyGraph>` 强引用保证（列 view 借 world）。

## 3. Design

### 3.1 PyGraph（通用 world 门面）

- entity：`spawn() -> int(handle)`、`despawn(h)`、`entities() -> list[int]`、`has_entity(h)`。
- component（类型化 + 约定键，缺/类型错抛错）：`get(h, key)`/`set(h, key, v)`/`has(h, key)`/
  `del(h, key)`；**`column(key) -> np.ndarray`（零拷贝 view）** + `validity(key) -> np.ndarray[bool]`。
- relation：`register_kind(name, arity)`、`kinds()`、`add_relation(kind, list[int]) -> int(handle)`、
  `relation_nodes(kind, rh)`、relation 的 component 同 `*_relation_*`、`remove_relation`、`n_relations`。
- `adopt(other)`（移动）。**不**暴露 `translate`/`to_frame` 等方法（见 §3.2）。

### 3.2 System = 模块级自由函数

PyO3 模块函数（非方法）：`molrs.perceive_aromaticity(mol)`、`molrs.find_rings(mol)`、
`molrs.add_hydrogens(mol)`、`molrs.compute_gasteiger_charges(mol)`、`molrs.translate(mol, delta)`、
`molrs.rotate(mol, axis, angle, about=None)`、`molrs.to_frame(mol) -> Frame`、`molrs.from_frame(frame)`。
转发到 01 的 core 自由函数。**类上无算法方法**（与 01 数据/行为分离一致）。

### 3.3 叶子 + adopt

- `PyAtomistic`/`PyCoarseGrain` `#[pyclass(extends=PyGraph, subclass)]`：`#[new]` 经 core 叶子注册
  领域 kind；领域 builder 暴露为模块函数或叶子薄方法（`molrs.add_atom(mol, "C")` 风格，与 system 一致）。
- `adopt(dst, src)`：core `adopt` 的 Python 入口，零拷贝接管 `Conformer`/SMILES 产出图（取代旧逐节点
  深拷贝）。`subclass` 修复当前「`class S(molrs.Atomistic)` → TypeError: not an acceptable base type」。

### 3.4 字段约定到 Python

molrs 暴露 `keys` 约定（如 `molrs.keys.X`/`molrs.keys.ELEMENT` 或字符串常量），Python 侧引用约定名,
不散落字面量;`column`/`get` 用约定键,缺/类型错抛 Python 异常。

## 4. Files

- `molrs-python/src/molgraph.rs` — PyGraph/叶子句柄化 + 类型化 component + `column` 零拷贝（numpy/
  buffer 协议）+ `subclass` + `adopt`；删类上的算法方法。
- `molrs-python/src/lib.rs` — 注册模块级 system 自由函数 + `keys` 约定 + 三类。
- `molrs-python/src/{io,conformer,forcefield}.rs` — 跟随句柄化 + 叶子构造（用 `adopt`/`wrap`）。
- `molrs-python/python/molrs/molrs.pyi` — 句柄/列/自由函数/约定存根。
- `molrs-python/tests/test_ecs_pybind.py` — 句柄稳定/零拷贝列/自由函数 system/subclass/adopt 契约。

## 5. Tasks

- [ ] Failing pytest: stable handles (remove middle entity → others resolve, no reindex); `column(keys.CHARGE)` is a zero-copy numpy view (mutating it writes through); systems are module functions (`molrs.perceive_aromaticity(mol)`, no method); `class S(molrs.Atomistic)` instantiable; `adopt` zero-copy (source emptied)
- [ ] Handle-ize PyGraph (entity + relation) over 01's `SlotMap` keys; typed component access via convention keys; `column`/`validity` zero-copy numpy
- [ ] Expose systems as module-level free functions; remove algorithm methods from the classes
- [ ] Add `subclass` to leaves; expose `adopt`; update io/conformer/forcefield construction
- [ ] Refresh `.pyi` + `keys`; rebuild wheel; run contract + existing molrs-python suite green

## 6. Testing strategy

- Stable handles: `spawn`/`add_relation` return stable ints; `despawn(mid)` leaves others valid; stale handle raises.
- Zero-copy: `col = mol.column(keys.X)` is a numpy view; `col[0] = 9.0` reflects in `mol.get(h0, keys.X)` (write-through); no per-element PyList build.
- Systems-as-functions: `molrs.perceive_aromaticity(mol)` works; `hasattr(molrs.Atomistic, "perceive_aromaticity")` is False (it's a module fn).
- Subclass + adopt: `class S(molrs.Atomistic): pass; S()` ok; `dst.adopt(src)` → `dst` populated, `src` emptied, handles valid.
- Errors: `mol.get(h, keys.X)` on missing → raises; `mol.set(h, keys.CHARGE, "x")` type-conflict → raises.
- Regression: existing `molrs-python/tests/` green after the reshape.

## 7. Out of scope

- molrs-core ECS 本身（01，前置依赖）。
- molpy 视图层（03）。
- 句柄连续化 / 暴露 SlotMap 细节——句柄保持不透明。
