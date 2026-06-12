---
slug: molgraph-ecs-01-core
criteria:
  - id: ac-001
    summary: World = 图实例（无全局/无 scheduler/无手创 World）；数据结构只剩存储原语
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core && grep"
    pass_when: |
      MolGraph/Atomistic/CoarseGrain 即 world：持 entity 句柄↔行 + 列存 component + kind-tagged
      关系；无全局 WORLD、无单例、无 system 注册表/scheduler。`impl MolGraph`/`impl Atomistic` 仅含
      存储原语（spawn/despawn/entities/contains、类型化 get/set/has/remove、column/validity、
      register_kind/add_relation/relations/remove_relation、adopt/merge）；grep 确认其上无
      translate/rotate/to_frame/perceive_aromaticity 等算法方法。Atomistic::new() + 领域 builder
      建图全程零 World/registry 初始化。`cargo test -p molcrafts-molrs-core` 全绿。
  - id: ac-002
    summary: 所有算法/变换/投影 = 吃 world 的自由函数
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core"
    pass_when: |
      perceive_aromaticity(&mut mol)/find_rings(&mol)/find_chiral_centers(&mol)/
      compute_gasteiger_charges(&mut mol)/add_hydrogens(&mut mol)/translate(&mut mol, d)/
      rotate(&mut mol, …)/to_frame(&mol)/from_frame 全部为模块自由函数、world 首参；
      残留方法包装（Atomistic::perceive_aromaticity）已删。无调度器/无 tick——调用即跑。
  - id: ac-003
    summary: Component = 共享行序对齐稠密列 + 每列 null 掩码；零拷贝、对齐、O(1)、句柄稳定
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core"
    pass_when: |
      node 存储为 EntityTable（keys: 句柄→行, rows: 行→句柄, cols: 名→(Vec<T>,Bitmap)）。所有列
      共享同一行序：列 i ↔ rows[i] 同一实体。column_f64(key) 返回**连续 &[f64]（长度 = n_rows、按行
      对齐）** + validity bitmap，可零拷贝映射；按句柄 get 为 O(1)；建 N 实体 O(N)。删中间实体
      swap-remove 行后**外部句柄仍稳定**、其它实体 get 正确。一实体有 charge 无 port（null 位）。
      ≥7 单测（建列/null/对齐/O(1)/swap-remove 句柄稳定/相邻列同实体）。
  - id: ac-004
    summary: 零硬编码字段——引用约定常量、直接访问、缺/类型错即报错（无兜底/无占位）
    type: code
    evaluator_hint: "cargo test -p molcrafts-molrs-core && grep"
    pass_when: |
      新增 keys 字段约定模块（canonical 常量，与 molpy fields 对齐）；几何/感知/to_frame/领域 builder
      全部引用 keys::* 而非字面量——grep 通用 molgraph.rs 无 "x"/"y"/"z"/"element"/"bond" 等字段字面量。
      get_f64(e, keys::X) 在无坐标节点 → Err（源码无 unwrap_or(0.0)）；set_str 覆盖 F64 列 → Err；
      领域 builder 无 `or "X"` 占位（无 element 的节点最终无 element 列值/为 null）。
  - id: ac-005
    summary: 行为对齐既有语料 + 全特性绿 + clippy + 性能
    type: code
    evaluator_hint: "cargo test --all-features && cargo clippy -- -D warnings"
    pass_when: |
      perceive_aromaticity/find_rings/add_hydrogens/compute_gasteiger_charges/to_frame 在既有 molrs
      测试语料上与 ECS 化前结果一致；`cargo test --all-features` 全绿、`cargo clippy -- -D warnings`
      干净；criterion 验证 build O(N)、column 访问 O(1) 零拷贝。
---

# Acceptance — molgraph-ecs-01-core

molrs-core ECS 重构验收（chain molgraph-ecs 1/3）。全部 `type: code`。

- ac-001：World = 图实例（无全局/scheduler/手创）；数据结构只剩存储原语。
- ac-002：算法/变换/投影皆吃 world 的自由函数。
- ac-003：共享行序对齐稠密列 + null 掩码（零拷贝/对齐/O(1)/句柄稳定）。
- ac-004：零硬编码字段——引用约定、直接访问、错即报错。
- ac-005：行为对齐 + 全绿 + clippy + 性能。
