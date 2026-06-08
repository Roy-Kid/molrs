---
slug: molgraph-abstract-02-pybind
criteria:
  - id: ac-001
    summary: molrs.Graph 暴露通用 kind 标签关系 API + 无字段 add_node，且不再带领域方法
    type: code
    evaluator_hint: "maturin develop --release && python -m pytest（反转面）"
    pass_when: |
      重建 wheel 后：molrs.Graph 暴露 add_node()（无参建节点）、register_kind(kind: str, arity: int)、
      kinds()、add_relation(kind, list[int])->int、get_relation_nodes/get_relation_prop/set_relation_prop/
      del_relation_prop/relation_keys/remove_relation/n_relations(kind)；且 hasattr(molrs.Graph,"add_atom")
      与 hasattr(molrs.Graph,"add_bond") 均为 False（领域面已移出基）。字符串 kind 在 Rust 边界解析为缓存
      KindId（不在数据路径每次 hash）。一条建图 round-trip（register_kind→add_node×N→add_relation→读回
      nodes+prop）等值通过。
    status: pending
  - id: ac-002
    summary: 领域方法落在 Atomistic/CoarseGrain 叶子，继承链 + issubclass 完好
    type: code
    evaluator_hint: "python -m pytest（叶子面 + 继承）"
    pass_when: |
      a = molrs.Atomistic(); a.add_atom("C"); a.add_atom("O"); a.add_bond(0,1); a.add_angle(0,1,? 合法第三点);
      a.add_dihedral(...); a.add_improper(...) 均成功，n_atoms/n_bonds/n_angles/n_dihedrals/n_impropers
      计数正确，a.set_bond_order(0,1,2.0) 生效；molrs.CoarseGrain().add_bead("W") 写 bead_type。
      issubclass(molrs.Atomistic, molrs.Graph) 与 issubclass(molrs.CoarseGrain, molrs.Graph) 均 True；
      isinstance(a, molrs.Graph) True；class S(molrs.Graph, Mixin): pass; class A(S): pass; A() 可实例化
      （无 __new__ 垫片，沿用 pybind-01 ac-003）。
    status: pending
  - id: ac-003
    summary: 无字段 add_node 可承载 element 经通用 set_node_prop（molpy _bind_entity enabler）
    type: code
    evaluator_hint: "python -m pytest（add_node + set_node_prop）"
    pass_when: |
      g = molrs.Graph(); i = g.add_node(); 该节点初始无 "element"（get_node_prop(i,"element") 为 None）；
      g.set_node_prop(i, "element", "C") 后 get_node_prop(i,"element")=="C"。证明不再强制 add_atom(symbol)
      必填字段——消除 molpy `symbol = ... or "X"` 硬编码所需的底层能力到位。
    status: pending
  - id: ac-004
    summary: to_frame 仍按 registry 输出完整拓扑块；向后兼容 molpy 不回归
    type: code
    evaluator_hint: "python -m pytest molpy/tests/test_embed tests/test_io tests/test_parser"
    pass_when: |
      molrs.Atomistic 建满五类拓扑后 to_frame() 含 atoms/bonds/angles/dihedrals/impropers 块；
      molpy to_molrs/from_molrs 与 embed 路径零改动可用：CAT 单体经 Generate3D 仍 n=32、带电单体
      [N+]/[N-] 氢数正确；`python -m pytest molpy/tests/test_embed tests/test_io tests/test_parser` 全绿。
    status: pending
---

# Acceptance — molgraph-abstract-02-pybind

molrs-python 暴露反转后层级的验收（chain molgraph-abstract 2/2，依赖 spec 1）。全部 `type: code`。

- ac-001：`molrs.Graph` 仅通用关系 API + `add_node`，领域方法已移出基（字符串 kind → 缓存 KindId）。
- ac-002：领域方法落叶子，`issubclass`/继承/无 `__new__` 垫片实例化完好。
- ac-003：无字段 `add_node` + `set_node_prop` 承载 element（molpy `_bind_entity` enabler）。
- ac-004：registry 驱动 `to_frame` 五块；molpy `to_molrs`/embed 向后兼容不回归（镜像 pybind-01 ac-005）。
