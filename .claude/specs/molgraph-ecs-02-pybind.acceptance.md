---
slug: molgraph-ecs-02-pybind
criteria:
  - id: ac-001
    summary: Entity = 稳定不透明句柄；删其它实体不失效、无平移
    type: code
    evaluator_hint: "cd molrs-python && maturin develop --release && python -m pytest tests/test_ecs_pybind.py"
    pass_when: |
      spawn()/add_relation() 返回稳定句柄（int）。建 a,b,c + 关系 r(a,c)，despawn(b) 后 a、c 句柄仍
      get 成功，relation_nodes(r)==[a,c]（不前移），has_entity(b) False，对已删句柄 get 抛错。
  - id: ac-002
    summary: Component 列 = 零拷贝 numpy view（写穿）
    type: code
    evaluator_hint: "python -m pytest tests/test_ecs_pybind.py"
    pass_when: |
      col = mol.column(keys.X) 为 numpy view；col[0]=9.0 后 mol.get(h0, keys.X)==9.0（写穿到 Rust，
      非逐元素 PyList 拷贝）；validity(key) 返回 bool 掩码。get/set 用约定键，缺/类型错抛 Python 异常。
  - id: ac-003
    summary: System = 模块自由函数；类上无算法方法
    type: code
    evaluator_hint: "python -m pytest tests/test_ecs_pybind.py"
    pass_when: |
      molrs.perceive_aromaticity(mol)/find_rings(mol)/add_hydrogens(mol)/translate(mol,d)/
      to_frame(mol) 为模块函数且生效；hasattr(molrs.Atomistic,"perceive_aromaticity") 与
      hasattr(molrs.Graph,"translate") 均 False（算法不在类上）。
  - id: ac-004
    summary: 叶子可子类化 + adopt 零拷贝接管
    type: code
    evaluator_hint: "python -m pytest tests/test_ecs_pybind.py && python -m pytest tests/"
    pass_when: |
      class S(molrs.Atomistic): pass; S() 可实例化（叶子带 subclass，不再 TypeError）；
      issubclass(S, molrs.Graph)。src 为含原子+键的图，dst.adopt(src) 后 dst 计数=src 原值、
      src 空、dst 句柄有效。SMILES to_atomistic / Conformer 产出图可零拷贝接管。
      `cd molrs-python && python -m pytest tests/` 全绿（既有套件 + ECS 契约）。
---

# Acceptance — molgraph-ecs-02-pybind

molrs-python 暴露 ECS world 的验收（chain molgraph-ecs 2/3，依赖 01）。全部 `type: code`。

- ac-001：稳定不透明句柄，删其它实体不失效。
- ac-002：component 列零拷贝 numpy view（写穿）。
- ac-003：system 为模块自由函数，类上无算法方法。
- ac-004：叶子可子类化 + adopt 零拷贝 + 既有套件不回归。
