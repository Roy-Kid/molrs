---
slug: molgraph-abstract-01-core
criteria:
  - id: ac-001
    summary: MolGraph 提供 interned kind 标签 n 元关系 API + 无字段 add_node（KindId 数组索引，无 String hash）
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core"
    pass_when: |
      （实现修正：register_kind 按字符串名注册 register_kind(name, arity) -> KindId，名字由叶子给，
      base 无 Kind 化学枚举。）molrs-core::MolGraph 暴露 register_kind(name, arity)（稠密 KindId；同名同
      arity 幂等返回，冲突 arity panic）、kind_id(name)/arity/kind_ids，以及 add_relation(KindId, &[NodeId])
      -> Result/get_relation/get_relation_mut/remove_relation/relations(KindId)/n_relations(KindId)，和无字段
      add_node() -> NodeId（新节点空 props）。关系按 KindId 走 Vec 数组索引存取（kinds: Vec<SlotMap<..>>，
      访问路径不 hash String；String 仅在注册与 to_frame 命名出现）。Relation::nodes 为 SmallVec<[NodeId;4]>，
      props 为 HashMap。add_relation 在 len != arity 或引用不存在 node 时返回 Err。≥5 条单测覆盖。
      `cargo test -p molcrafts-molrs-core` 全绿。
    status: verified
  - id: ac-002
    summary: dihedral 与 improper 同为 4 元关系、靠 Kind 区分（删除一等 struct）
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core"
    pass_when: |
      molrs-core 不再定义一等 Bond/Angle/Dihedral/Improper struct（统一为 Relation；Bond/.. 与 BondId/..
      作类型别名移到 crate::atomistic）。两条同为 4 元、kind 名不同（"dihedrals" vs "impropers"）的关系可在
      同一 MolGraph 上覆盖相同 4 个 node 共存，n_relations 各为 1，各自经 relations(KindId) 迭代到正确 node
      元组；新增单测显式断言「同元数不同 kind 不合并」。
    status: verified
  - id: ac-003
    summary: MolGraph 与化学词汇彻底分离——全部领域方法/类型在叶子，基只有通用图 API
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core"
    pass_when: |
      （设计修正 2026-06-08：原"保留 typed facade on base"被推翻；用户要求 MolGraph 与具体化学词汇彻底
      分离。）MolGraph 基**不含任何化学词汇**：无 Kind::Bond/Angle/.. 枚举（register_kind 按字符串名 +
      arity），无 add_bond/add_angle/add_dihedral/add_improper、无 bonds()/angles()/get_bond/n_bonds/
      neighbor_bonds，也无 add_atom/atoms()/get_atom（节点 API 用中性 add_node/add_node_with/get_node/
      nodes()/n_nodes/remove_node + 通用 neighbors）。grep 确认 molgraph.rs 不再出现 bond/angle/dihedral/
      improper 标识符。Atomistic::new 注册 bond=2/angle=3/dihedral=4/improper=4 并缓存 KindId 字段；独占
      add_atom*/add_bond(i,j)/add_angle(i,j,k)/add_dihedral(i,j,k,l)/add_improper(..)/bonds()/get_bond/
      n_bonds/.. 及 BondId/Bond 等领域类型别名；n_bonds==n_relations(kind)；产出的 BondId 可用作
      HashSet<BondId> 键。CoarseGrain::new 注册 CG kind，独占 add_bead/beads/n_beads。全部 all-atom 化学
      消费方（aromaticity/rings/stereo/hydrogens/gasteiger/rotatable/smarts）签名改吃 &Atomistic。
      `cargo test -p molcrafts-molrs-core` 全绿。
    status: verified
  - id: ac-004
    summary: registry 驱动 remove_node 级联 + merge（修 impropers 漏搬 bug）+ to_frame/from_frame，无 per-kind 特判
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core"
    pass_when: |
      remove_node(n) 删除所有「nodes 含 n」的关系，跨全部已注册 kind（新增单测：建 bond+angle+dihedral+
      improper 后删中心 node，四类计数全部归零）。merge/extend 经 registry 搬运全部 kind 桶——新增回归单测：
      源图含 impropers，merge 后目标 n_relations 每个 kind 等于两图之和（显式覆盖现 merge :720-759 漏搬
      impropers 的潜伏 bug）。to_frame 遍历 kind_ids 对每个非空 kind 产出一块（程序内建图断言 atoms/bonds/
      angles/dihedrals/impropers 块均在），列名沿用 atomi/atomj/atomk/atoml；from_frame 反向按 arity 重建，
      round-trip 各 kind 计数一致。round-trip 测试用程序内建 MolGraph（不读 tests-data，遵守 IO Testing
      Rules）。`cargo test -p molcrafts-molrs-core` 全绿。
    status: verified
  - id: ac-005
    summary: 预留 containment 轴（GroupId/Group/groups 占位 + doc 约束），不实现行为
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core && grep doc"
    pass_when: |
      molrs-core 定义 GroupId（new_key_type）+ struct Group{members, parent, props} + MolGraph.groups:
      SlotMap<GroupId, Group> 字段（空、未接线），整体编译通过；molgraph.rs 模块级 doc 含「kinds/relations
      仅用于定长对等拓扑；containment 将作为独立 groups registry 落地，不复用关系存储」之意的声明。本 spec
      不对 groups 实现任何增删查/级联/序列化（断言其不出现在 to_frame/merge/remove_node 路径）。
    status: verified
---

# Acceptance — molgraph-abstract-01-core

molrs-core 关系存储反转的验收（chain molgraph-abstract 1/2）。全部 `type: code`，经
`cargo test -p molcrafts-molrs-core` 判定。已折入资深 Rust 架构师评审要求的缓解项。

- ac-001：interned `KindId`（数组索引，无 String hash）的通用关系 API + `SmallVec` nodes + `Option`
  props + 无字段 `add_node`。
- ac-002：dihedral/improper 收敛为 `Kind` 标签 4 元关系，一等 struct 已删除（D2 方案 B 取代方案 A）。
- ac-003：领域 typed 便捷方法（缓存 KindId、编译期 arity）下沉叶子，typed `bonds()` 保留 `BondId` map-key
  语义，基不再承载领域词。
- ac-004：registry 驱动级联删除 + merge（**修掉 impropers 漏搬 bug**）+ to_frame/from_frame，无 per-kind
  特判，round-trip 守 IO 规则。
- ac-005：预留 containment 轴占位 + doc 约束，不实现行为（防二次迁移）。
