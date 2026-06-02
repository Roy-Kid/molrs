# molrs — Spec Index

One row per spec produced by `/molrs-spec`. Newest on top.

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

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /molrs-impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
