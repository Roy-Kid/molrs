---
title: 反转 MolGraph 分层——领域无关的 kind 标签 n 元关系存储（molrs-core）
status: approved
created: 2026-06-08
chain: molgraph-abstract
chain_index: 1
depends_on: [molgraph-pybind-01-hierarchy]
---

# 反转 MolGraph 分层——kind 标签 n 元关系存储（molrs-core）

## Summary

把 `MolGraph` 从「五个硬编码领域 arena（atoms/bonds/angles/dihedrals/impropers）」反转为
**领域无关的基**：基只提供 kind 标签的 n 元关系存储——节点（1 元）、边（2 元）、高阶超边
（3 元 angle、4 元 dihedral/improper……），每条关系实例带一个 schemaless 属性袋。领域词汇全部
下沉到 `Atomistic`/`CoarseGrain` 叶子：叶子注册各自的 kind 与其元数（bond=2、angle=3、dihedral=4、
improper=4），并暴露领域便捷方法（`add_atom`/`add_bond`/`add_angle`/`add_dihedral`/`add_improper`、
`add_bead`/CG 同类）包装基的通用关系 API。基新增**无字段 `add_node()`** 构造器。dihedral 与 improper
不再是两个一等 struct，而是同为 4 元关系、靠 **kind 标签**区分。

**性能与类型安全是一等约束（资深 Rust 架构师评审要求，2026-06-08）**，故 kind 标签在**实现层不得是
按访问 hash 的 `String`**，关系端点不得是无条件堆分配的 `Vec`：

- kind 用 **interned `KindId`（数组索引）+ `enum Kind` 已知种类**；`String` 只活在注册边界与
  `to_frame` 列命名。叶子在 `new()` 时把各 kind 的 `KindId` 缓存为字段，便捷方法零查表。
- `Relation::nodes` 用 `SmallVec<[NodeId; 4]>`（现有 5 种 kind 全 ≤4 元 → 内联、无堆分配，
  保持今天 `[AtomId; N]` 的 cache 行为）；`props` 用 `Option<HashMap<..>>`（空属性不占 map）。
- 叶子保留 **arity 编译期检查的 typed facade**（`add_bond(i,j)`/`add_dihedral(i,j,k,l)` 仍是定参
  签名）；通用 `add_relation(kind, &[..])` 只服务真正动态的新 kind 路径。
- 保留类型可辨的 `BondId`（`= RelationId` 别名 + 叶子侧 typed `bonds()` 迭代器），使
  `aromaticity.rs`/`stereo_guard.rs` 里把 bond id 当图算法 map key 的 `HashSet<BondId>` 不丢失类型语义。

本 spec 仅改 molrs-core；Python 重新暴露与 molpy 接入是后续 spec（见 Out of scope）。

## Domain basis

本 spec **revisits** 既有 spec [`molgraph-pybind-01-hierarchy`](molgraph-pybind-01-hierarchy.md)
（status: code-complete）的两项设计决策，并故意反转之：

- 该 spec 的 **D2 方案 A** 选择「在 core 加一等 `Improper` struct，而非带 `kind` 属性的 dihedral」，
  理由是「保持四级拓扑对称 + `to_frame` 无特判」。本 spec 改采当时被拒的 **D2 方案 B**：dihedral/
  improper 收敛为 kind 标签的 4 元关系。重新评估理由：方案 A 的「对称」是靠把同一段
  CRUD/级联/iter/to_frame/from_frame 逻辑**复制五份**换来的（`molgraph.rs:303-349` 的 remove_atom
  四臂级联、`:843-926` 的五段 to_frame 列构建、`:720-759` 的 per-arena merge）。**这份复制已经产生一个
  潜伏 bug：`merge`（`:720-759`）静默漏搬 impropers**——正因为手抄 arena 容易漏一个。registry 驱动的
  单存储用一份逻辑取代五份特例，天然修掉该 bug，且天然支持「同元数、不同语义」（dihedral vs improper
  均 4 元）以及未来任意新定长 kind（无需改 core）。
- 该 spec 让 `Graph`/`MolGraph` 直接承载全部领域方法（`add_atom`/`add_bond`/…），叶子是近空 newtype。
  本 spec 反转：领域方法下沉到叶子，基只留通用关系 API。

注意：被引 spec 已实现（code-complete），故本 spec 是**后续重构**（follow-on refactor），不是
greenfield；不改写旧 spec 文件，仅在此引用并说明决策为何被重审。本 spec 的**最强当下理由是去重正确性**
（修掉 merge 漏 impropers），扩展性是次要收益。无物理方程（纯数据结构 / 拓扑重构），故不含数值验证目标。

## Design

### 1. 关系存储模型（基，领域无关，interned kind）

抽象原语 = **「一组定长（fixed-arity）关系 + 每实例可选属性袋」，按 interned `KindId` 分桶**：

```rust
new_key_type! { pub struct NodeId; pub struct RelationId; pub struct GroupId; }
pub type AtomId = NodeId;        // 减少叶子/下游 churn
pub type BondId = RelationId;    // 类型别名：保留 aromaticity/stereo 的 map-key 语义

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind { Bond, Angle, Dihedral, Improper, Custom(u16) }  // 已知种类 + 开放

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct KindId(u16);          // 注册时分配的稠密索引

pub struct Relation {
    pub nodes: SmallVec<[NodeId; 4]>,                 // ≤4 元内联，无堆分配
    pub props: Option<HashMap<String, PropValue>>,    // 空属性不占 map
}

pub struct MolGraph {
    nodes:       SlotMap<NodeId, Atom>,
    kinds:       Vec<SlotMap<RelationId, Relation>>,  // 按 KindId.0 索引（数组索引，零 hash）
    kind_arity:  Vec<usize>,                          // 按 KindId.0
    kind_name:   Vec<String>,                         // 仅注册边界 / to_frame 命名用
    name_to_kind: HashMap<String, KindId>,            // 仅注册时解析一次
    adjacency:   HashMap<NodeId, Vec<RelationId>>,    // 由 arity==2 关系维护，供 neighbors
    groups:      SlotMap<GroupId, Group>,             // 预留：见 §5，本 spec 不实现行为
}
```

要点：**按访问路径（`relations`/`n_relations`/`add_relation`）一律走 `KindId`→`Vec` 数组索引，绝不
hash `String`**。`String` 只在 `register_kind` 解析一次与 `to_frame` 命名时出现。

### 2. 基的通用关系 API（无领域词）

- 节点：`add_node() -> NodeId`（**无字段**，prop 袋为 `None`）；`add_node_with(Atom) -> NodeId`；
  `remove_node`/`get_node`/`get_node_mut`/`nodes()`/`n_nodes`。`remove_node` 级联删除所有 `nodes`
  含该节点的关系——**遍历 `kinds` 全部桶（registry 驱动，一份逻辑）**，替代当前四臂手抄级联。
- kind 注册：`register_kind(&mut self, kind: Kind, arity: usize) -> KindId`（幂等；同 arity 重复注册
  返回既有 `KindId`，冲突 arity 返回 `MolRsError::validation`）；`kind_id(Kind) -> Option<KindId>`；
  `arity(KindId) -> usize`；`kind_ids() -> impl Iterator<Item = KindId>`（稳定顺序，供 to_frame）。
- 关系 CRUD（按 `KindId`）：`add_relation(KindId, &[NodeId]) -> Result<RelationId>`（校验 len==arity、
  每个 node 存在；arity==2 时维护 adjacency）；`get_relation`/`get_relation_mut`/`remove_relation`；
  `relations(KindId) -> impl Iterator<Item = (RelationId, &Relation)>`；`n_relations(KindId) -> usize`。
- 几何/遍历领域无关：`neighbors`（经 arity==2 关系）、`translate`、`rotate`、`merge`/`extend`
  （**registry 驱动地搬运所有 kind 桶 + props**，一份逻辑——天然覆盖 impropers，修掉现 merge bug）。

### 3. 叶子注册领域 kind + typed facade

- `Atomistic::new()` 注册 `Bond=2`/`Angle=3`/`Dihedral=4`/`Improper=4`，**把返回的 `KindId` 缓存为
  字段**（`bond_kind`/`angle_kind`/…）；typed 便捷方法 `add_atom_bare(symbol)`/`add_atom_xyz(..)`
  （写 `element`，复用 `add_node_with`）、`add_bond(i,j)`、`add_angle(i,j,k)`、`add_dihedral(i,j,k,l)`、
  `add_improper(i,j,k,l)`——**定参签名保留编译期 arity 检查**，内部调
  `add_relation(self.bond_kind, &[..])`。`n_bonds`/… 转发 `n_relations(self.bond_kind)`。
- 叶子提供 typed 迭代器 `bonds() -> impl Iterator<Item = (BondId, &Relation)>`（`BondId = RelationId`
  别名）；`aromaticity.rs`/`stereo_guard.rs` 经此保留 `HashSet<BondId>`/`HashMap<BondId,_>` 语义。
- `CoarseGrain::new()` 注册 CG kind（bead 隐含 1 元 node、CG-bond=2）；`add_bead`/`add_bead_bare` 写
  `bead_type`。
- 旧的基方法 `MolGraph::add_atom`/`add_bond`/`add_angle`/`add_dihedral`/`add_improper` 及一等
  `Bond`/`Angle`/`Dihedral`/`Improper` struct + 其专属 `*Id`（`BondId` 改为 `RelationId` 别名保留）
  **从基移除**。

### 4. registry 驱动的 to_frame / from_frame（无 per-kind 特判）

- `to_frame`：节点块同现状。拓扑块改为**遍历 `kind_ids()`**，对每个非空 kind 产出一个块，名取
  `kind_name`，列为 `atomi/atomj/atomk/atoml`（保持现有命名以不破坏既有读者）。bond `order` 属性沿用
  现有 F 列写法。删除当前五段复制。
- `from_frame`：遍历 frame 中每个已注册 kind 对应的块，按 arity 读 `atomi..atoml` 列重建关系。
  **遵守 molrs IO Testing Rules**：round-trip 测试用程序内建图（非合成 IO 文件），不触碰 `tests-data/`。

### 5. 预留 containment 轴（本 spec 仅占位，不实现行为）

层级容器（residue ⊃ atoms、chain ⊃ residues、CG atom→bead 映射）是**变长、嵌套、有向所有权**，
**不是**定长对等关系，**不得**建模为 relation kind（否则 group 节点会污染 `nodes` arena，破坏所有
迭代 `nodes()` 的消费方）。为避免未来二次迁移（再过一遍 cascade + PyO3 index↔id 层），本 spec **现在
就预留形状**：`new_key_type! { pub struct GroupId; }` + `MolGraph.groups: SlotMap<GroupId, Group>`
（空、未接线）+ `struct Group { members: Vec<NodeId>, parent: Option<GroupId>, props: Option<HashMap<..>> }`
占位定义 + 一句模块级 doc：「`kinds`/relations 仅用于定长对等拓扑；containment 将作为独立 `groups`
registry 落地（后续 spec），不复用关系存储」。`groups` 的增删查/级联/序列化**不在本 spec 范围**。

## Files to create or modify

- `molrs-core/src/molgraph.rs` — 核心重写：`Kind`/`KindId`/`Relation`(SmallVec+Option props)/`NodeId`/
  `RelationId`/`GroupId`/`Group` + interned kind registry 存储；通用关系 API（register_kind/add_relation/
  get/remove/relations/n_relations）；无字段 `add_node`；registry 驱动 remove_node 级联、merge/extend、
  to_frame/from_frame；删除一等 `Bond`/`Angle`/`Dihedral`/`Improper` + 基的 `add_atom`/`add_bond`/
  `add_angle`/`add_dihedral`/`add_improper`；保留 `BondId = RelationId` 别名 + 预留 `groups` 字段。
- `molrs-core/src/atomistic.rs` — `Atomistic::new` 注册 atom/bond/angle/dihedral/improper kind 并缓存
  `KindId` 字段；typed 便捷方法 + typed `bonds()` 迭代器；`n_*` 转发 `n_relations`。
- `molrs-core/src/coarsegrain.rs` — `CoarseGrain::new` 注册 bead/CG-bond kind + 缓存 KindId；`add_bead`。
- `molrs-core/Cargo.toml` — 加 `smallvec` 依赖（若工作区未引入）。
- `molrs-core/src/lib.rs` — 更新被删/改符号的 `pub use` re-export。
- `molrs-core/src/aromaticity.rs` + `molrs-conformer/src/stereo_guard.rs` — 把对一等 `Bond`/`get_bond`/
  `bonds()` 的引用改用叶子 typed `bonds()` + `BondId` 别名，保留 `HashSet<BondId>` map-key 语义。
- molrs-core/molrs-ff/molrs-conformer 中其余引用旧 `add_bond`/`Bond`/`Dihedral` 的消费方——经 `cargo
  build` 错误定位后逐个改用叶子便捷方法 / 通用关系 API（Task T8）。

## Tasks

- [ ] Write failing tests for the interned kind-registry relation store in `molrs-core/src/molgraph.rs` `#[cfg(test)]`: register_kind returns dense KindId, add_relation by KindId, get/iter/remove_relation, optional per-relation prop bag, field-less add_node (empty props), dihedral & improper as 4-ary relations distinguished by Kind, remove_node cascade across ALL kinds
- [ ] Implement `Kind`/`KindId`/`Relation` (`SmallVec<[NodeId;4]>` + `Option` props) + interned kind-bucketed storage (`kinds: Vec<SlotMap<..>>` array-indexed, no per-access String hash) in `MolGraph` (`molrs-core/src/molgraph.rs`); add `smallvec` to `molrs-core/Cargo.toml`
- [ ] Implement the generic relation API + field-less `add_node`/`add_node_with` + registry-driven `remove_node` cascade in `MolGraph`, removing first-class `Bond`/`Angle`/`Dihedral`/`Improper` and the base `add_atom`/`add_bond`/`add_angle`/`add_dihedral`/`add_improper`; keep `BondId = RelationId` alias
- [ ] Rewrite `merge`/`extend` + registry-driven `to_frame`/`from_frame` with no per-kind special-casing (`molrs-core/src/molgraph.rs`); add a regression test asserting `merge` transfers ALL kinds incl. impropers (the current `:720-759` bug), and a program-built to_frame/from_frame round-trip test (no tests-data IO)
- [ ] Reserve the containment axis: define `GroupId`/`Group` + empty `MolGraph.groups` field + module doc stating containment is NOT a relation kind (`molrs-core/src/molgraph.rs`); no group behavior implemented
- [ ] Write failing tests for the leaves' kind registration + typed convenience methods + typed `bonds()` iterator (`molrs-core/src/atomistic.rs`, `molrs-core/src/coarsegrain.rs`)
- [ ] Implement `Atomistic::new`/`CoarseGrain::new` kind registration with cached `KindId` fields + typed arity-checked convenience methods (`add_atom*`/`add_bond`/`add_angle`/`add_dihedral`/`add_improper`, `add_bead*`) wrapping `add_relation`; forward `n_*` to `n_relations`; expose typed `bonds()`
- [ ] Port `molrs-core/src/aromaticity.rs` + `molrs-conformer/src/stereo_guard.rs` to the leaf typed `bonds()` / `BondId` alias, preserving `HashSet<BondId>`/`HashMap<BondId,_>` keys; fix remaining in-crate consumers of the removed base API (locate via `cargo build`)
- [ ] Run full check + test suite (`cargo fmt --all`, `cargo clippy -- -D warnings`, `cargo test -p molcrafts-molrs-core`) green

## Testing strategy

- Happy path: `register_kind(Kind::Bond, 2)` returns a dense `KindId`; `add_node()` (field-less) ×4;
  `add_relation` for each kind; assert `n_relations(kind)` per kind, `relations(kind)` iterates the right
  node tuples, optional prop bag round-trips int/f64/str, and a relation with no props keeps `props == None`.
- kind discrimination: a `Dihedral` and an `Improper` relation over the same 4 nodes coexist, are counted
  separately, iterate under their own `KindId` — confirming arity alone does not merge them.
- Interned keys: registering the same `Kind` twice returns the same `KindId`; conflicting arity errors.
- Field-less node: `add_node()` creates a node with `props == None`; `translate` skips it without panic;
  `set_node_prop` later attaches `element`.
- Edge cases: `add_relation` wrong arity → Err; relation referencing a removed node → Err; `remove_node`
  cascades across **all** registered kinds (regression for the old per-arena cascade).
- **merge regression**: build a graph with bonds+angles+dihedrals+**impropers**, `merge` into another,
  assert the destination's `n_relations` for **every** kind equals the sum — explicitly covering the
  current `merge` impropers-drop bug.
- Leaves: `Atomistic` typed methods produce relations under the right `KindId`; `n_bonds`/`n_angles`/
  `n_dihedrals`/`n_impropers` equal `n_relations(kind)`; typed `bonds()` yields `BondId`-keyed pairs usable
  in a `HashSet<BondId>`; `CoarseGrain::add_bead` writes `bead_type`.
- Frame round-trip: program-built graph → `to_frame` emits one block per non-empty kind (atoms + bonds +
  angles + dihedrals + impropers) → `from_frame` reconstructs identical per-kind counts. No synthetic IO
  files (graph built in-test, per molrs IO Testing Rules).
- Containment reservation: `GroupId`/`Group`/`groups` compile and the module doc is present; no behavior.
- `cargo test -p molcrafts-molrs-core` green (existing ~396-test suite survives the API migration), and
  `aromaticity.rs`/`stereo_guard.rs` keep typed `BondId` keys (no clippy warnings).

## Out of scope

- molrs-python 重新暴露反转后的层级——本链 spec 2 `molgraph-abstract-02-pybind`。
- molpy 侧把 `_bind_entity` 的 `symbol = str(ent.data.get("element") or "X"); molrs.Graph.add_atom`
  硬编码迁到新 `add_node` + 通用 `set_node_prop`，并把 `Entity` 收缩为纯 view——molpy 自己的 spec
  （refinement of `molpy/.claude/specs/atomistic-cg-on-molrs-molgraph.md`），在本链之后。
- **containment / 分组的行为实现**（residue/chain/CG-mapping 的增删查/级联/序列化）——本 spec 仅预留
  `GroupId`/`Group`/`groups` 占位与 doc 约束；其落地是独立后续 spec。
- 既有 spec `molgraph-pybind-01-hierarchy.md` 文件本身不改写（仅引用）。
