# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

| 2026-06-13 | [interop-wrapup](interop-wrapup.md) | draft | molrs-ffi, molrs-python, molrs-ff, molrs-capi, docs (consumer: molpack) | 两份实现 spec 落地**之后**的收尾:整体 review + 把 FFI/bridge 层做成"产品级",让别的项目低摩擦地继续开发。(1) 端到端 review —— 真零拷贝(共享后端可验证)、**uint 索引契约覆盖所有 molpy reader**、pair kernel 能量 parity、MMFF parity 后删 `frame_builder`、**无残留** `frame_marshal`/`from_lammps_ff`/MMFF 私有机器;一套 pairs 约定/一个 nblist 工具/一种 capsule 模式。(2) FFI 易用性 + **命名统一** —— `molrs-ffi::bridge` 一套 `verb_noun`、`py↔ref` 对称命名,capsule 名带版本;**一行采纳路径**(`use molrs_ffi::prelude::*` → `obj.as_frame_ref()?`/`frame_ref.into_py(py)?`,隐藏 addr/capsule 细节);单一 error 类型;扁平公开面(`molrs::io::forcefield::{read_lammps,read_opls,ForceFieldReader}`,无深路径泄漏)。(3) `docs/interop.md` 钦定配方 + ~10 行可跑示例 + 数据契约 + 命名约定,molpack 作参考。依赖 [interop-ffi-bridge](interop-ffi-bridge.md) + [ff-special-bonds-nblist](ff-special-bonds-nblist.md);命名决定要在外部项目 pin 公开面**之前**落 |
| 2026-06-12 | [gaff-typifier-01-parser](gaff-typifier-01-parser.md) | approved | molrs-ff | GAFF M1 chain 1/4：原生 gaff.dat(GAFF 1.x) reader → molrs ForceField/Params + 编译期内嵌（镜像 MMFF94_XML，留在 molrs-ff）；gaff.dat 从 openmmforcefields(MIT) vendored+版本钉死，参数对照其转换 ffxml 作 AmberTools-free ground truth；覆盖 MASS/BOND/ANGLE/DIHEDRAL/IMPROPER/NONBON；负周期多项二面角续行累加 + improper 无 IDIVF 为载重不变量 |
| 2026-06-12 | [gaff-typifier-02-typing](gaff-typifier-02-typing.md) | approved | molrs-ff | GAFF M1 chain 2/4：非共轭原子分型引擎（H/C/N/O/S/P/卤素局部判定），镜像 typifier/mmff，复用 rings/aromaticity/hybrid；ATOMTYPE_GFF.DEF 为基准；ca 入 nb 出的刻意非对称；共轭/杂芳输入必须报错不静默误型 |
| 2026-06-12 | [gaff-typifier-03-assign](gaff-typifier-03-assign.md) | approved | molrs-ff | GAFF M1 chain 3/4：键/角/二面角/improper 参数分配，X 通配回退（精确优先、最具体通配次之）独立于通用 store；多项二面角组装；improper PK 直用；产出已参数化 Frame→to_potentials |
| 2026-06-12 | [gaff-typifier-04-parity](gaff-typifier-04-parity.md) | approved | molrs-ff | GAFF M1 chain 4/4：antechamber 对照验证台（gen_gaff_fixtures.py 跑 antechamber -at gaff → SDF+JSON；逐原子 100% 一致 type:scientific 门），fixtures 缺席时干净跳过；镜像 MMFF/RDKit fixture 范式 |
| 2026-06-12 | [gaff-typifier-05-charges](gaff-typifier-05-charges.md) | approved | molpy-wrapper, molrs-ff | GAFF 电荷：与 openmm 一致 = AM1-BCC 委托（antechamber -c bcc / openff am1bcc），不在 molrs 重写半经验 QM；电荷在委托层算好写进 frame charge 列，molrs 只消费；逐原子对照 omff/antechamber（gated），净电荷守恒、禁 Gasteiger 替代；未来 QM-free 走 NAGL GNN |
| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
