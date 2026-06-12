---
title: Drop petgraph from molrs-core; native MolGraph adjacency for topology algorithms
status: code-complete
created: 2026-06-12
supersedes: topology-paths-molgraph-01
---

# Drop petgraph from molrs-core; native MolGraph adjacency for topology algorithms

## Summary
从 `molrs-core` 移除 `petgraph` 依赖。`Topology`、`Atomistic::{generate_topology, topo_distances}` 与 `chem::rings::find_rings` 当前都通过把键关系临时物化成一份一次性的 `petgraph::UnGraph` 来跑图算法；这些方法的主体本就是手写在 petgraph 邻接之上的（角=2 边路径、二面角=3 边路径、improper=星枚举、BFS 距离、连通分量泛洪、Horton 最小环基），唯一真正的 petgraph 算法调用是三处 `petgraph::algo::connected_components` 计数。本 spec 把这些算法改为直接在 `Topology` 自有的原生 CSR 邻接快照上运行（该快照一次性从 `Atomistic::neighbor_bonds` / `MolGraph::neighbor_relations` 过滤出键 `KindId` 构建），把计数换成原生泛洪（`connected_components().iter().max()+1`），删除 `petgraph = "0.8"`。对每个公共入口（`generate_topology` 计数、`topo_distances` 集合、连通分量标号、环感知）行为逐字节等价，且 `topo_distances` 不退化（PoC 实测在 10k/100k 链上 1.4–1.5× 更快）。本 spec **取代** `topology-paths-molgraph-01`：那条 spec 因当时"像 molpy 一样用 petgraph、不要重造轮子"的指令而选择复用 petgraph `Topology`，本 spec 反转该决策——真正复杂的图算法（VF2 子图同构）只在 molrs-io SMARTS 里，不在 core，故"不要重造轮子"对 molrs-core 不适用。

## Domain basis
全部为标准、教科书级图论，不涉及任何物理方程，无需 scientist 介入：
- **键图（bond graph）**：无向图 `G=(V,E)`，顶点=原子、边=键。
- **角 = 2 边路径**：中心节点 `j` 的每对邻居 `(i,k)`，规范去重 `i<k`，产出 `[i,j,k]`。`Σ_j C(deg(j),2)`。
- **proper 二面角 = 3 边路径**：每条中心边 `(j,k)`（规范 `j<k`），`i∈adj(j)\{k}`、`l∈adj(k)\{j}` 且 `i≠l`，产出 `[i,j,k,l]`。`Σ_edges deg(j)·deg(k)` 级别。
- **improper = 度≥3 的星枚举**：中心节点 `c` 的邻居的所有有序 3 组合 `[c,i,j,k]`（`i<j<k`）。
- **单源最短路（无权 BFS）**：`source` 距离 0，邻居逐跳 +1，不可达为 `-1`（`Topology::distances`）/ 省略（`topo_distances`）。复杂度 `O(V+E)`。
- **连通分量**：泛洪标号（0 基、连续）；分量数 `C = max(label)+1`。
- **最小环基 / SSSR（Horton）**：候选环 = 对每条边临时删除后做 BFS 最短路回连；按长度升序后在边-关联向量上做 GF(2) 高斯消元贪心选取线性无关者；环秩 `cycle_rank = E - V + C`。复杂度 `O(E·(V+E))` 候选生成、`O(R·E²)` 独立性检验；分子图小而稀疏故快。
环不变量 `E - V + C` 是本次唯一用到 petgraph 计数 `C` 的地方，原生泛洪同样给出 `C`。

## Design
**采用方案 (a)：保留 `Topology` 作为算法宿主，但其内部以原生邻接快照替换 `petgraph::UnGraph`。** 公共签名（`Topology` 全部方法、`Atomistic::{generate_topology, topo_distances, bond_topology}`、`chem::rings::find_rings(&Atomistic)->RingInfo`）一律不变。

理由（对比方案 (b)"把方法搬到 MolGraph/Atomistic 并删除 Topology"）：
- 这些算法已经全部手写在 petgraph 的邻接之上，petgraph 只当容器 + 三处计数用——替换容器即可，无需触碰算法逻辑或任何调用方/测试。
- **molgraph 必须保持领域无关**（取自被取代 spec 的好不变量）：把 BFS/SSSR 搬到 `MolGraph` 会让通用图沾上拓扑算法，搬到 `Atomistic` 又会在领域叶子里重复 SSSR 的 GF(2) 机器；方案 (a) 让 `Topology`（一个纯整数索引、领域无关的连通性结构）继续承载这些图论，干净分层。
- 方案 (b) 会扰动 26 个 `system::topology` 测试与 6 个 `chem::rings` 测试的调用面，零收益。

具体内部改造：
- `Topology` 由 `struct Topology { graph: UnGraph<(),()> }` 改为持有原生 CSR/邻接快照，例如 `struct Topology { adj: Vec<Vec<usize>>, edges: Vec<[usize;2]> }`（顶点为连续 `0..n` 索引；`edges` 保留插入顺序以维持 `bonds()`/边索引语义与 `delete_bond(idx)` 的现有约定）。`from_edges`/`with_atoms`/`add_*`/`delete_*` 在此结构上重建；`add_bond` 去重、`bonds()` 顺序、`degree`/`neighbors`/`are_bonded` 行为保持。
- `angles()/dihedrals()/impropers()/distances()/connected_components()` 主体逐字段迁移到 `adj`（已是手写邻接遍历，仅把 `self.graph.neighbors(n)` 换成 `&self.adj[n]`，把 `edge_indices/edge_endpoints` 换成 `self.edges`）。保持现有规范去重与确定性/排序顺序。
- `n_components()`：`petgraph::algo::connected_components` → `self.connected_components().iter().max().map_or(0, |m| m+1)`（空图返回 0）。
- `find_rings()`（topology.rs）与 `bfs_skip_edge`：把 `EdgeIndex` 跳边语义改为按端点对 `(min,max)` 跳过；`cycle_rank` 用原生 `n_components()`。GF(2)/`edge_lookup`/`TopologyRingInfo` 不变。
- `Atomistic::generate_topology`：删去构建 `Topology::from_edges` 之前那段把 `relation_nodes` 物化成 `edges` 的 petgraph 准备代码无须改（它喂的是原生 `Topology`），但需让 `topo_distances` 走原生路径。**`topo_distances`**：将 PoC `topo_distances_native` 主体折入 `topo_distances`（`SecondaryMap`-keyed BFS 直接在 `neighbor_relations` 过滤键上跑，免去 AtomId↔连续索引重映射与 `bond_topology()` 物化），删除临时 `topo_distances_native` 方法。`bond_topology()` 仍可保留（构建原生 `Topology`），供 `find_rings`/角二面角枚举使用。
- `chem::rings::find_rings`：去掉 `UnGraph` 构建与 `petgraph::algo::connected_components`，改在从 `Atomistic` 直接构建的原生邻接（`atom_to_idx` + 邻接表）上跑同一 Horton+GF(2)；`bfs_shortest_path` 改原生跳边。`RingInfo` 结构与所有公共方法不变。
- 暴露/复用一个领域无关的原生路径/邻居枚举作为角二面角的共享基础（实现被取代 spec 原意的 `paths_of_length` 等价物，但落在 `Topology` 上而非 `MolGraph`，保持 molgraph 纯净）。命名只在 `Atomistic` 出现。
- `molrs-core/Cargo.toml`：删除 `petgraph = "0.8"`。删除 `topology.rs`/`rings.rs` 顶部 petgraph `use`。
- 删除 `benches/core/topology.rs` 里的 `bench_topo_distances_native` A/B 变体及 `criterion_group!` 中的引用；保留 `bench_topo_distances`/`bench_generate_topology` 作为不退化基准。

## Files to create or modify
- `molrs-core/src/system/topology.rs` — `Topology` 改原生邻接快照；迁移 `angles/dihedrals/impropers/distances/connected_components`；`n_components` 原生计数；`find_rings`/`bfs_skip_edge` 去 petgraph；新增原生-vs-旧 parity 单元测试。
- `molrs-core/src/system/atomistic.rs` — `topo_distances` 主体替换为原生 BFS；删除 `topo_distances_native`；把其 parity 测试转为永久回归测试。
- `molrs-core/src/chem/rings.rs` — `find_rings`/`bfs_shortest_path` 改原生邻接；去 petgraph `use` 与 `connected_components` 计数。
- `molrs-core/Cargo.toml` — 删除 `petgraph = "0.8"`。
- `molrs-core/benches/core/topology.rs` — 删除 `bench_topo_distances_native` 及其 `criterion_group!` 引用。

## Tasks

- [x] Write failing tests for native Topology parity (molrs-core/src/system/topology.rs) — angles/dihedrals/impropers/distances/connected_components on ethane, benzene, naphthalene, cyclohexane, linear chain, empty/single-edge/disconnected/unknown-source; pin ethane 12 angles / 9 dihedrals
- [x] Implement native adjacency snapshot in Topology and migrate angles/dihedrals/impropers/distances/connected_components off petgraph (molrs-core/src/system/topology.rs)
- [x] Implement native n_components and de-petgraph find_rings + bfs_skip_edge in Topology (molrs-core/src/system/topology.rs)
- [x] Write failing regression test for topo_distances native parity (molrs-core/src/system/atomistic.rs) — convert topo_distances_native_matches_petgraph into a permanent topo_distances test over ethane all-sources, 12-chain, unknown-source
- [x] Implement topo_distances native BFS body and delete topo_distances_native (molrs-core/src/system/atomistic.rs)
- [x] Write failing tests for native find_rings parity (molrs-core/src/chem/rings.rs) — 6-ring, naphthalene, cyclohexane, linear-no-ring, empty, disconnected
- [x] Implement de-petgraph find_rings + bfs_shortest_path on native adjacency (molrs-core/src/chem/rings.rs)
- [x] Remove petgraph dependency from molrs-core/Cargo.toml and delete bench_topo_distances_native from molrs-core/benches/core/topology.rs
- [x] Run full check + test suite (cargo fmt --check, clippy --workspace --all-targets --locked -D warnings, cargo test --workspace)

## Testing strategy
- **Happy path**: ethane `generate_topology(true,true,false)` 返回 `(12, 9)` 且 `n_angles()==12`、`n_dihedrals()==9`；线性链 `topo_distances(endpoint)` 给出 `0..=n-1` 距离集合;6 元环 `find_rings` 给 1 个 6 环、所有原子/键 in-ring。
- **Native-vs-old parity**: 对 ethane(branched)、benzene/naphthalene/cyclohexane(ring)、linear chain，逐一断言原生 `angles/dihedrals/impropers/distances/connected_components/find_rings` 与改造前 petgraph 输出（排序后集合 + 计数）一致;`topo_distances` 对 ethane 全源 + 12-chain + unknown-source 与历史输出一致。
- **Edge cases**: 空图（0 原子/0 键）→ 0 角/0 二面角/0 环、不 panic;单边图;不连通图（连通分量标号分裂、`n_components` 正确、跨分量 `distances` 为 `-1`/省略）;越界/未知 source → `distances` 全 `-1`、`topo_distances` 空 vec。
- **Idempotence**: `generate_topology` 二次调用（`clear_existing=false`）加 0;`clear_existing=true` 重生成同一集合（ethane 仍 12/9）。
- **Domain validation (graph theory)**: 环秩 `E-V+C` 与原生 `n_components` 自洽（benzene C=1 → 1 环;naphthalene 2 融合环 → 2 环、共享键 `num_bond_rings==2`）;角=2 边路径、二面角=3 边路径计数与手算一致（methane 6 角/0 二面角/4 improper）。
- **No-regression bench**: `cargo bench -p molcrafts-molrs-core graph/topo_distances` 中位数不高于 petgraph 基线（1k 45.8µs / 10k 843µs / 100k 7.61ms）;`graph/generate_topology` 与 `find_rings` 无实质退化。

## Out of scope
- molrs-io smiles/SMARTS 的 VF2 子图同构（`petgraph::subgraph_isomorphisms_iter`，`smarts/matcher.rs`/`pattern.rs`）及 molrs-io 的 petgraph 依赖——那是真正的算法，移除它是另一条未来 spec "vendor petgraph's VF2 into molrs-io"（MIT/Apache，整体搬入）。本 spec 不触碰 molrs-io 或其 `Cargo.toml`。
- molpy 侧改动（molpy 已委托;`core/topology.py` 已在 0.4.0 删除）。
- 任何 FF / potential / optimizer 工作。
- 把算法搬出 `Topology`（方案 (b)）——本 spec 显式选择方案 (a)。
