# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

| 2026-06-12 | [core-drop-petgraph](core-drop-petgraph.md) | approved | molrs-core | 从 molrs-core 移除 petgraph：`Topology`/`topo_distances`/`find_rings` 改在 MolGraph 原生邻接快照上跑（angles/dihedrals/impropers 枚举、BFS 距离、flood-fill 连通分量、Horton SSSR 全是手写，仅 3 处 `petgraph::algo::connected_components` 计数需换原生），删 `petgraph = "0.8"`；行为逐字节等价，topo_distances 不退化（PoC 1.4–1.5×）；取代 topology-paths-molgraph-01；SMARTS VF2（molrs-io）留作独立 vendor spec |
| 2026-06-12 | [gaff-typifier-01-parser](gaff-typifier-01-parser.md) | approved | molrs-ff | GAFF M1 chain 1/4：原生 gaff.dat(GAFF 1.x) reader → molrs ForceField/Params + 编译期内嵌（镜像 MMFF94_XML，留在 molrs-ff）；gaff.dat 从 openmmforcefields(MIT) vendored+版本钉死，参数对照其转换 ffxml 作 AmberTools-free ground truth；覆盖 MASS/BOND/ANGLE/DIHEDRAL/IMPROPER/NONBON；负周期多项二面角续行累加 + improper 无 IDIVF 为载重不变量 |
| 2026-06-12 | [gaff-typifier-02-typing](gaff-typifier-02-typing.md) | approved | molrs-ff | GAFF M1 chain 2/4：非共轭原子分型引擎（H/C/N/O/S/P/卤素局部判定），镜像 typifier/mmff，复用 rings/aromaticity/hybrid；ATOMTYPE_GFF.DEF 为基准；ca 入 nb 出的刻意非对称；共轭/杂芳输入必须报错不静默误型 |
| 2026-06-12 | [gaff-typifier-03-assign](gaff-typifier-03-assign.md) | approved | molrs-ff | GAFF M1 chain 3/4：键/角/二面角/improper 参数分配，X 通配回退（精确优先、最具体通配次之）独立于通用 store；多项二面角组装；improper PK 直用；产出已参数化 Frame→to_potentials |
| 2026-06-12 | [gaff-typifier-04-parity](gaff-typifier-04-parity.md) | approved | molrs-ff | GAFF M1 chain 4/4：antechamber 对照验证台（gen_gaff_fixtures.py 跑 antechamber -at gaff → SDF+JSON；逐原子 100% 一致 type:scientific 门），fixtures 缺席时干净跳过；镜像 MMFF/RDKit fixture 范式 |
| 2026-06-12 | [gaff-typifier-05-charges](gaff-typifier-05-charges.md) | approved | molpy-wrapper, molrs-ff | GAFF 电荷：与 openmm 一致 = AM1-BCC 委托（antechamber -c bcc / openff am1bcc），不在 molrs 重写半经验 QM；电荷在委托层算好写进 frame charge 列，molrs 只消费；逐原子对照 omff/antechamber（gated），净电荷守恒、禁 Gasteiger 替代；未来 QM-free 走 NAGL GNN |
| 2026-06-10 | [topology-paths-molgraph-01](topology-paths-molgraph-01.md) | superseded | molrs-core, molrs-python | 下沉 #1：角/二面角=键图里的 k-边路径 + `Atomistic::generate_topology`。**superseded by [core-drop-petgraph](core-drop-petgraph.md)** —— 当时按"用 petgraph 不要重造轮子"改成复用 petgraph Topology；core-drop-petgraph 据 PoC 反转该决策、原生化并删依赖 |
| 2026-06-10 | [ff-potentials-oop-01](ff-potentials-oop-01.md) | code-complete | molrs-ff, molrs-python, molrs-conformer | 下沉 #2：镜像 molpy OOP — `Style::to_potential`/`ForceField::to_potentials`(无 frame)/`Potentials::calc_energy\|calc_forces(frame)` + `LBFGS` 优化器类(`new(pot,cfg).run(frame)`)；删 compile/eval/自由 minimize/自由 ctor；ETKDG 用私有引擎不变 |
| 2026-06-10 | [ff-format-readers-01](ff-format-readers-01.md) | code-complete | molrs-ff, molrs-python | 下沉 #3：molrs 直接读力场格式 → molrs.ForceField（单位归一化在 reader）；首个 OplsXmlReader（GROMACS nm/kJ→Å/kcal，RB→OPLS f1..f4）；对照 molpy numpy OPLS |
| 2026-06-10 | [opls-ef-01-kernels-seam](opls-ef-01-kernels-seam.md) | code-complete | molrs-ff | OPLS-AA E/F：补 OPLS 4-cosine 二面角 + 通用 coul/cut 库仑 kernel（bond/angle 复用 harmonic），定义 molpy SMARTS-typify→typed Frame→molrs compile→minimize 接缝；排除/1-4 scaling/几何组合规则在 molpy 侧烘焙进 pairs 块，kernel 拓扑无关；对照 molpy numpy OPLS。typifier 不下沉（B线） |
| 2026-06-10 | [geometry-optimizer-01-generic-batch](geometry-optimizer-01-generic-batch.md) | code-complete | molrs-ff, molrs-conformer, molrs-python | 通用几何优化器下沉：把 ETKDG 私有 L-BFGS 抽进 molrs-ff 成 force-field-agnostic minimize（fmax 收敛，复用 two-loop+line search）+ 同构体系批量 minimize_batch（rayon par_chunks，(B,N,3)）+ PyO3 暴露 Potentials.minimize/minimize_batch；ETKDG 行为不变；对照 molpy numpy-LBFGS & RDKit MMFFOptimize |
| 2026-06-08 | [molgraph-ecs-01-core](molgraph-ecs-01-core.md) | draft | molrs-core | MolGraph→ECS：图实例即 world(纯数据,无全局/scheduler/手创)；system 即吃 world 的自由函数(perceive_aromaticity(mol)…)；component=共享行序对齐稠密列+null 掩码(零拷贝/对齐/O(1)/句柄稳定)；零硬编码字段=引用内置 key 约定+直接访问+错即报错。chain molgraph-ecs 1/3(inversion 量级,breaking molrs) |
| 2026-06-08 | [molgraph-ecs-02-pybind](molgraph-ecs-02-pybind.md) | draft | molrs-python | 暴露 ECS world 到 Python：entity 稳定句柄(int)+component 列零拷贝 numpy view(写穿)+system 模块自由函数(molrs.perceive_aromaticity(mol))+叶子可子类化+零拷贝 adopt(chain molgraph-ecs 2/3,依赖 01) |
| 2026-06-08 | [molgraph-abstract-01-core](molgraph-abstract-01-core.md) | code-complete | molrs-core | 反转 MolGraph 为领域无关 interned kind 标签 n 元关系存储（SmallVec nodes/Option props/KindId 数组索引）；领域词下沉 Atomistic/CoarseGrain 叶子；dihedral/improper→4 元+kind；修 merge 漏 impropers bug；预留 containment 轴（chain molgraph-abstract 1/2，refines molgraph-pybind-01 D2） |
| 2026-06-08 | [molgraph-abstract-02-pybind](molgraph-abstract-02-pybind.md) | approved | molrs-python | Python 暴露反转层级：PyGraph 仅通用关系 API + 无字段 add_node，领域方法移到 PyAtomistic/PyCoarseGrain 叶子；向后兼容 molpy to_molrs/embed 不回归（chain molgraph-abstract 2/2，依赖 01） |
| 2026-06-01 | [molgraph-pybind-01-hierarchy](molgraph-pybind-01-hierarchy.md) | code-complete | molrs-core, molrs-python | 暴露 MolGraph 层级到 Python（Graph ← Atomistic/CoarseGrain）+ 角/二面角/impropers/属性/extend 绑定扩面；molpy 子类化后端的先决项 P0（5/5 acceptance verified） |
| 2026-06-01 | [core-perception-01-aromaticity](core-perception-01-aromaticity.md) | approved | molrs-core | Atomistic::perceive_aromaticity()，对齐 RDKit 默认芳香性模型（chain core-perception 1/2） |
| 2026-06-01 | [core-perception-02-smarts-rings](core-perception-02-smarts-rings.md) | approved | molrs-core | SMARTS r{lo-hi} 区间 + x<n> 环连接度，去掉 torsion_prefs shim（chain core-perception 2/2） |
| 2026-06-01 | [etkdg-smarts-01-engine](etkdg-smarts-01-engine.md) | code-complete | molrs-core | SMARTS 子结构匹配引擎（含递归 $()），对照 RDKit GetSubstructMatches（chain etkdg-smarts 1/2） |
| 2026-06-01 | [etkdg-smarts-02-torsions](etkdg-smarts-02-torsions.md) | code-complete | molrs-embed | ETKDGv3 完整实验扭转表 + SMARTS 接入，闭合 mmff94-etkdg-04 RMSD（chain etkdg-smarts 2/2） |
| 2026-06-01 | [mmff94-etkdg-01-typing](mmff94-etkdg-01-typing.md) | code-complete | molrs-ff | MMFF94 原子分型 + 芳香性 + 参数表 + BCI 电荷（对照 RDKit，chain mmff94-etkdg 1/4） |
| 2026-06-01 | [mmff94-etkdg-02-energy](mmff94-etkdg-02-energy.md) | code-complete | molrs-ff | MMFF94/MMFF94s 七项能量 + 解析梯度（对照 RDKit，chain mmff94-etkdg 2/4） |
| 2026-06-01 | [mmff94-etkdg-03-bounds](mmff94-etkdg-03-bounds.md) | code-complete | molrs-embed | DistGeom bounds 矩阵（对齐 RDKit 0.0）+ 三角 smoothing + 手性；实验扭转部分（待 etkdg-smarts）（chain mmff94-etkdg 3/4） |
| 2026-06-01 | [mmff94-etkdg-04-embed](mmff94-etkdg-04-embed.md) | code-complete | molrs-embed | 4D 嵌入 + ET 最小化 + MMFF 清理 + 重试,重写 generate_3d；删除/reshape 推迟（chain mmff94-etkdg 4/4） |
| 2026-05-17 | [vibrational-spectra](vibrational-spectra.md) | approved | molrs-compute, molrs-signal | 功率谱 (VDOS)、红外、拉曼光谱 — ndarray 输入，复用 ACF/窗函数/频率网格 |
| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|
| 2026-06-10 | [unit-conversion-system](unit-conversion-system.md) | shipped | molrs-core | pint-style UnitRegistry + Dimension + Quantity; MD units (kcal/mol, eV, hartree, Å, bohr, °C) |

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
