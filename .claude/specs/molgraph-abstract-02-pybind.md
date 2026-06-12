---
title: 在 Python 暴露反转后的 MolGraph 层级（PyGraph 通用基 + 领域方法下沉叶子）
status: approved
created: 2026-06-08
chain: molgraph-abstract
chain_index: 2
depends_on: [molgraph-abstract-01-core]
---

# Python 暴露反转后的 MolGraph 层级（molrs-python）

## Summary

在 spec 1（molrs-core 反转为 interned kind 标签关系存储）落地后，把反转后的层级**如实暴露到 Python**：
`molrs.Graph`（`PyGraph`）作为抽象基，只承载领域无关的通用关系 API（注册 kind+arity、按 kind 增删查迭
关系、每关系属性袋）与**无字段 `add_node()`**；领域命名方法（`add_atom`/`add_bond`/`add_angle`/
`add_dihedral`/`add_improper`、`add_bead`/CG 同类）从 `PyGraph` **移到** `PyAtomistic`/`PyCoarseGrain`
叶子。向后兼容是硬约束：`molrs.Atomistic().add_atom("C")`/`add_bond`/`add_angle`/… 必须继续可用，
`issubclass(molrs.Atomistic, molrs.Graph)` 仍为 True，molpy 的 `to_molrs`/embed 不回归。Python 侧
kind 仍以**字符串**寻址（Python 无 Rust enum）——但每个字符串在 Rust 边界解析为缓存的 `KindId`，
不在 Python 每次调用时重复 hash 进数据路径。本 spec 仅改 molrs-python；molpy 接入是其自己的后续 spec。

## Domain basis

- 直接依赖 spec [`molgraph-abstract-01-core`](molgraph-abstract-01-core.md)（必须先落地：core 的
  interned 通用关系 API + 无字段 `add_node` + 叶子 kind 注册 + typed facade）。
- 本 spec 反转 [`molgraph-pybind-01-hierarchy`](molgraph-pybind-01-hierarchy.md) 在 Python 侧的形态：
  该 spec（ac-002/ac-004/ac-005）让 `PyGraph` 承载全部领域方法、`PyAtomistic`/`PyCoarseGrain` 为 marker
  子类；本 spec 把领域方法移到叶子，`PyGraph` 只留通用关系 API。重新评估理由同 spec 1（基应领域无关）。
- PyO3 继承机制与既有约束沿用 `molgraph-pybind-01-hierarchy`：`#[pyclass(subclass)]` 基 +
  `#[pyclass(extends=PyGraph)]` 叶子；`PyGraph::#[new]` 吞 `*args/**kwargs`（molpy 子类无需 `__new__`
  垫片，该不变量保留）。
- 既有绑定风格与索引模型参考现有 `molrs-python/src/molgraph.rs`（`py_to_prop`/`prop_to_py` 派发、
  0 基 index↔Id 解析、`molrs_error_to_pyerr`）——复用，不重写。无物理方程。

## Design

### 1. PyGraph：仅通用关系 API + add_node（领域无关）

- 保留：节点属性/几何/迭代通用面——`add_node()`（**新增，无字段**，对应 core `add_node`）、`remove_node`/
  `get_node_prop`/`set_node_prop`/`del_node_prop`/`node_keys`/`node_column`、`neighbors`、`coords`/
  `set_coords`/`translate`/`rotate`、`extend`、`to_frame`、`n_nodes`(getter)。
- **新增通用关系面**（字符串 kind 在 Rust 边界解析为缓存 KindId，转发 core）：`register_kind(kind: str,
  arity: int)`、`kinds()`、`add_relation(kind: str, nodes: list[int]) -> int`、`get_relation_nodes(kind,
  r) -> list[int]`、`get/set/del_relation_prop(kind, r, key, value)`、`relation_keys(kind, r)`、
  `remove_relation(kind, r)`、`n_relations(kind)`。
- **移除自 PyGraph**：`add_atom`/`add_bead`/`add_bond`/`set_bond_order`/`add_angle`/`add_dihedral`/
  `add_improper` 及各自的 `get_*_atoms`/`*_keys`/`*_prop`/`remove_*`/`n_*`（领域命名面，下沉叶子）。

### 2. PyAtomistic / PyCoarseGrain：领域命名方法（extends=PyGraph）

- `PyAtomistic`（`extends=PyGraph`）：`#[new]` 经 core `Atomistic` 注册 atom/bond/angle/dihedral/improper
  kind（KindId 缓存在 Rust 侧）；提供 `add_atom(symbol, x=None,y=None,z=None) -> int`（写 `element`，内部
  `add_node` + `set_node_prop`）、`add_bond(i,j)`/`set_bond_order(i,j,order)`、`add_angle(i,j,k)`、
  `add_dihedral(i,j,k,l)`、`add_improper(i,j,k,l)`，及领域命名的 `get_bond_atoms`/`n_bonds`/`n_angles`/
  `n_dihedrals`/`n_impropers`——内部转发到 core 叶子的 typed 方法 / 通用关系 API（按缓存 KindId）。
- `PyCoarseGrain`（`extends=PyGraph`）：注册 CG kind；`add_bead(bead_type, ...)`（写 `bead_type`）+ CG 方法。
- 继承链 + `issubclass(molrs.Atomistic, molrs.Graph)` / `issubclass(molrs.CoarseGrain, molrs.Graph)`
  保持 True；`molrs.Graph` 仍可作 molpy 的最后一个基类，且 `class S(molrs.Graph, Mixin)` 无需 `__new__`
  垫片（沿用 pybind-01 ac-003）。

### 3. 注册 + stub

- `molrs-python/src/lib.rs`：仍 `add_class::<PyGraph>()` + `PyAtomistic` + `PyCoarseGrain`（类不变，
  方法面迁移）。
- `molrs-python/python/molrs/molrs.pyi`：把领域方法存根从 `Graph` 移到 `Atomistic`/`CoarseGrain`，
  并为 `Graph` 补通用关系方法 + `add_node` 的存根。

## Files to create or modify

- `molrs-python/src/molgraph.rs` — `PyGraph` 改为仅通用关系 API + `add_node` + 通用 `*_relation` 面
  （字符串 kind→缓存 KindId）；领域命名方法移入 `PyAtomistic`/`PyCoarseGrain`（`extends=PyGraph`），
  转发 core 叶子 typed 方法；调整内部 index↔Id helper 走通用关系 API。
- `molrs-python/src/lib.rs` — 确认三类注册不变（仅符号变更时更新 import）。
- `molrs-python/python/molrs/molrs.pyi` — 领域方法存根移到叶子；`Graph` 补 `add_node` + `*_relation` 存根。

## Tasks

- [ ] Write failing pytest for the inverted Python surface: `molrs.Graph` exposes `add_node`/`register_kind`/`add_relation`/`n_relations` but `hasattr(molrs.Graph, "add_atom")` is False; `molrs.Atomistic().add_atom("C")` still works and `issubclass(molrs.Atomistic, molrs.Graph)` is True
- [ ] Implement `PyGraph` generic relation API + field-less `add_node` in `molrs-python/src/molgraph.rs` (string kind resolved to cached KindId at the Rust boundary), removing the domain-named methods from `PyGraph`
- [ ] Implement domain-named convenience methods on `PyAtomistic`/`PyCoarseGrain` (`add_atom`/`add_bond`/`set_bond_order`/`add_angle`/`add_dihedral`/`add_improper`, `add_bead`, plus `n_*`/`get_*_atoms`) forwarding to the core leaf typed methods (`molrs-python/src/molgraph.rs`)
- [ ] Update `lib.rs` registration if symbols changed and refresh `molrs.pyi` stubs (move domain method stubs to leaves, add `Graph.add_node`/`*_relation` stubs)
- [ ] Rebuild the wheel (`maturin develop --release`) and run the inverted-surface + backward-compat pytest
- [ ] Run molpy backward-compat regression: `python -m pytest molpy/tests/test_embed tests/test_io tests/test_parser` green (no regression)

## Testing strategy

- Happy path: `g = molrs.Graph(); g.register_kind("bond", 2); a=g.add_node(); b=g.add_node();
  r=g.add_relation("bond", [a,b])`; assert `g.n_relations("bond")==1`, `g.get_relation_nodes("bond", r)
  == [a,b]`, relation prop round-trips int/f64/str.
- Field-less node: `g.add_node()` creates a node with no `element` (`get_node_prop(i,"element")` is None);
  `g.set_node_prop(i, "element", "C")` then attaches it (this is the molpy `_bind_entity` enabler).
- Base no longer carries domain methods: `hasattr(molrs.Graph, "add_atom")` and `hasattr(molrs.Graph,
  "add_bond")` are False (domain face lives on leaves only).
- Leaves keep the domain surface: `a = molrs.Atomistic(); a.add_atom("C"); a.add_atom("O"); a.add_bond(0,1);
  a.add_angle(0,1,? )` etc.; `n_atoms`/`n_bonds`/`n_angles`/`n_dihedrals`/`n_impropers` correct;
  `a.set_bond_order(0,1,2.0)` works; `molrs.CoarseGrain().add_bead("W")` writes `bead_type`.
- Inheritance intact: `issubclass(molrs.Atomistic, molrs.Graph)` and `issubclass(molrs.CoarseGrain,
  molrs.Graph)` True; `isinstance(molrs.Atomistic(), molrs.Graph)` True; `class S(molrs.Graph, Mixin):
  pass; class A(S): pass; A()` instantiable with no `__new__` shim (regression for pybind-01 ac-003).
- to_frame: `molrs.Atomistic` with all kinds populated → `to_frame()` yields atoms/bonds/angles/dihedrals/
  impropers blocks (registry-driven from core).
- Backward-compat (load-bearing, mirrors pybind-01 ac-005): molpy `to_molrs`/`from_molrs`/embed unchanged
  and green — CAT monomer via `Generate3D` still n=32, charged `[N+]`/`[N-]` H-counts correct;
  `python -m pytest molpy/tests/test_embed tests/test_io tests/test_parser` all pass.

## Out of scope

- molpy 把 `_bind_entity` 的 `symbol = str(ent.data.get("element") or "X"); add_atom` 硬编码迁到新
  `add_node` + 通用 `set_node_prop`，并收缩 `Entity` 为纯 view——molpy 自己的 spec（refinement of
  `molpy/.claude/specs/atomistic-cg-on-molrs-molgraph.md`）。
- 层级容器 / 分组的行为实现（residue/chain/CG-mapping）——独立后续 spec；spec 1 仅预留 core 占位。
- molrs-core 关系存储本身（spec 1，前置依赖；本 spec 假定其已落地）。
- 零拷贝列 buffer（沿用 `node_column`/`to_frame` 兜底，pybind-01 既有 Out of scope）。
