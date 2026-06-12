---
slug: molgraph-pybind-01-hierarchy
criteria:
  - id: ac-001
    summary: molrs-core 一等 Improper（对称 Angle/Dihedral）+ remove_atom 级联 + to_frame 块
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core"
    pass_when: |
      MolGraph 提供 add_improper(i,j,k,l)/remove_improper/get_improper/get_improper_mut/
      impropers()/n_impropers，与 Angle/Dihedral 一一对称；remove_atom(a) 同时删除所有引用 a
      的 improper（新增单测覆盖）；MolGraph::to_frame 输出 impropers 块（含端点 index 与 props）。
      `cargo test -p molcrafts-molrs-core` 全绿，且新增 impropers 测试 >=3 条（add/get/iter、
      remove_improper、remove_atom 级联）。
    status: verified
    last_checked: "2026-06-01"
    note: "molrs-core 加 Improper(struct/field/CRUD/get_*_mut/级联/iter/n/to_frame/from_frame)；新增 4 条 improper 测试（crud、invalid_atom、remove_atom 级联、frame roundtrip）；cargo test -p molcrafts-molrs-core = 396 passed, 4 ignored。"
  - id: ac-002
    summary: Python 暴露 Graph(基) ← Atomistic / CoarseGrain 继承链
    type: code
    evaluator_hint: "import molrs; issubclass/isinstance 检查"
    pass_when: |
      molrs.Graph、molrs.Atomistic、molrs.CoarseGrain 三个类均可 import；
      issubclass(molrs.Atomistic, molrs.Graph) 与 issubclass(molrs.CoarseGrain, molrs.Graph)
      均为 True；molrs.Atomistic() / molrs.CoarseGrain() / molrs.Graph() 均可实例化；
      a=molrs.Atomistic(); a.add_atom("C"); isinstance(a, molrs.Graph) 为 True。
    status: verified
    last_checked: "2026-06-01"
  - id: ac-003
    summary: molpy 可子类化 molrs.Graph + 纯 Python mixin，无需 __new__ 垫片
    type: code
    evaluator_hint: "正式化 spike：Struct(molrs.Graph, *Mixins) 多继承"
    pass_when: |
      下述 Python 成立且无异常：
        class SpatialMixin: 
            def n2(self): return self.n_atoms * 2
        class Struct(molrs.Graph, SpatialMixin):
            def __init__(self, name=None):
                super().__init__(); self.name = name      # 注意：无需自定义 __new__
        class Atomistic(Struct):
            def kind(self): return "AA"
        a = Atomistic(name="w"); a.add_atom("C"); a.add_atom("H"); a.add_bond(0,1)
      验证：MRO 含 [Atomistic, Struct, molrs.Graph, SpatialMixin, object]；a.n_atoms==2；
      a.n2()==4；a.name=="w"；a.kind()=="AA"；isinstance(a, molrs.Graph)。
      关键：molrs.Graph 的 #[new] 接受并忽略 *args/**kwargs，故 Struct 不需要 __new__ 垫片。
    status: verified
    last_checked: "2026-06-01"
  - id: ac-004
    summary: 通用方法面 round-trip（原子/键/角/二面角/improper CRUD + props + 迭代）
    type: code
    evaluator_hint: "每个新方法一条建图→读回→等值"
    pass_when: |
      在 molrs.Graph 上：add_atom×3 + add_bond + add_angle + add_dihedral + add_improper 后，
      n_atoms/n_bonds/n_angles/n_dihedrals/n_impropers 计数正确；
      get_*_atoms 返回正确端点 index；set_atom_prop/get_atom_prop/del_atom_prop/atom_keys、
      及 bond/angle/dihedral/improper 的同类 *_prop 往返一致；
      neighbors(i)、atom_column("x")、coords()/set_coords、translate(vec)、extend(other)
      行为正确（extend 后计数 = 两图之和，端点 index 正确偏移）；to_frame() 含五类块。
    status: verified
    last_checked: "2026-06-01"
  - id: ac-005
    summary: 向后兼容——既有 molpy to_molrs / embed 不回归
    type: code
    evaluator_hint: "molpy 既有用例 + 测试"
    pass_when: |
      molrs.Atomistic() + 现有 7 个方法签名不变；molpy to_molrs/from_molrs 与 embed 路径零改动可用：
      CAT 单体经 molpy.compute.embed.Generate3D 仍 n=32、带电单体 [N+]/[N-] 氢数正确；
      `rtk proxy python -m pytest molpy/tests/test_embed tests/test_io tests/test_parser` 全绿。
    status: verified
    last_checked: "2026-06-01"
---

# Acceptance — molgraph-pybind-01-hierarchy

P0（molrs 侧）验收 —— **5/5 verified（2026-06-01）**，已独立复跑确认：
- ac-001：molrs-core Improper（CRUD/级联/to_frame）；`cargo test -p molcrafts-molrs-core` 396 passed。
- ac-002：`molrs.Graph ← Atomistic / CoarseGrain` 继承链成立。
- ac-003：`class Struct(molrs.Graph, Mix)` 无 `__new__` 垫片可实例化（`Graph::#[new]` 吞 `*args/**kwargs`），MRO 正确。
- ac-004：五级拓扑 CRUD + props 往返 + neighbors/atom_column/coords/translate/extend + to_frame 五块。
- ac-005：纯增量不回归——CAT 带电单体 n=32；molpy `test_embed/test_io/test_parser/test_core` 1222 passed, 1 skipped, 1 xfailed。

下游消费方（embed.rs / forcefield.rs / io.rs）随 `PyAtomistic` 改为 `extends=PyGraph`（数据移入基类）已一并修好。
注：`Atom.props` 在 molrs-core 是私有，原子属性读经类型化访问器（`get_str/get_int/get_f64` 顺序探测），未改 core。
