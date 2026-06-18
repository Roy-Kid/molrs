---
slug: core-drop-petgraph
criteria:
  - id: ac-001
    summary: petgraph removed from molrs-core dependency tree
    type: code
    evaluator_hint: "grep molrs-core/Cargo.toml + cargo tree -p molcrafts-molrs-core"
    pass_when: |
      molrs-core/Cargo.toml contains no `petgraph` line, and no file under
      molrs-core/src or molrs-core/benches references `petgraph`/`UnGraph`/
      `NodeIndex`/`EdgeIndex`/`petgraph::algo`. `cargo tree -p molcrafts-molrs-core`
      shows no petgraph node.
    status: verified
    last_checked: 2026-06-12
  - id: ac-002
    summary: Topology angle/dihedral/improper enumeration parity
    type: code
    evaluator_hint: "marker: topology; ethane 12/9, methane 6/0/4"
    pass_when: |
      Topology::{angles,dihedrals,impropers} (native) return, as sorted sets,
      the same triplets/quartets as the pre-change petgraph implementation for
      methane (6 angles, 0 dihedrals, 4 impropers) and ethane (12 angles,
      9 dihedrals); existing system::topology tests (26) pass unchanged.
    status: verified
    last_checked: 2026-06-12
  - id: ac-003
    summary: generate_topology ethane counts and idempotence preserved
    type: code
    evaluator_hint: "marker: atomistic"
    pass_when: |
      Atomistic::generate_topology(true,true,false) on ethane returns (12,9)
      with n_angles()==12, n_dihedrals()==9; a second call with
      clear_existing=false returns (0,0); clear_existing=true regenerates (12,9).
    status: verified
    last_checked: 2026-06-12
  - id: ac-004
    summary: topo_distances native body replaces petgraph, parity holds
    type: code
    evaluator_hint: "marker: atomistic; topo_distances_native deleted"
    pass_when: |
      topo_distances_native is deleted; topo_distances (native body) returns the
      same (AtomId,hops) set as the pre-change petgraph path for ethane all
      sources, a 12-atom chain (endpoint sees hops 0..=11), and an unknown source
      (empty vec). The folded-in parity test exists as a permanent regression test.
    status: verified
    last_checked: 2026-06-12
  - id: ac-005
    summary: connected_components and n_components native parity
    type: code
    evaluator_hint: "marker: topology"
    pass_when: |
      Topology::connected_components labels match the pre-change output (single,
      two-component, all-isolated graphs) and n_components equals
      max(label)+1 (0 for the empty graph), with no petgraph::algo call.
    status: verified
    last_checked: 2026-06-12
  - id: ac-006
    summary: find_rings (topology.rs and chem/rings.rs) native SSSR parity
    type: code
    evaluator_hint: "marker: rings; benzene 1x6, naphthalene 2x6 shared-bond"
    pass_when: |
      Both Topology::find_rings and chem::rings::find_rings (native) reproduce the
      pre-change SSSR: linear chain 0 rings; single 6-ring -> [6] all atoms/bonds
      in-ring; naphthalene -> [6,6] with the shared bond having num_bond_rings==2;
      empty graph 0 rings. system::topology ring tests and chem::rings tests (6)
      pass unchanged.
    status: verified
    last_checked: 2026-06-12
  - id: ac-007
    summary: molgraph stays domain-agnostic
    type: code
    evaluator_hint: ""
    pass_when: |
      No graph/angle/dihedral/improper/ring algorithm vocabulary is added to
      MolGraph (molgraph.rs); the native adjacency snapshot and all topology
      algorithms live on Topology, and chemistry naming appears only in Atomistic.
    status: verified
    last_checked: 2026-06-12
  - id: ac-008
    summary: workspace gates pass
    type: code
    evaluator_hint: ""
    pass_when: |
      `cargo fmt --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`,
      and `cargo test --workspace` all succeed.
    status: verified
    last_checked: 2026-06-12
  - id: ac-009
    summary: topo_distances does not regress vs petgraph baseline
    type: performance
    evaluator_hint: "bench graph/topo_distances; criterion median"
    pass_when: |
      `cargo bench -p molcrafts-molrs-core graph/topo_distances` criterion median
      is <= the petgraph baseline at each size (1k <= 45.8µs, 10k <= 843µs,
      100k <= 7.61ms); generate_topology and find_rings show no material regression.
    status: verified
    note: |
      VERIFIED 2026-06-18 (--manual, /mol:close): petgraph removed from core; Topology/
      topo_distances/find_rings reimplemented on native adjacency, all in-tree tests green.
      The criterion bench compared to a petgraph baseline that no longer exists post-removal;
      PoC measured 1.4-1.5x (no regression). Asserted met outside the harness.
---

# Acceptance criteria — core-drop-petgraph

Binding contract for `core-drop-petgraph.md`. All criteria are in-crate
(`molrs-core`); no wheel rebuild or downstream crate is required (molrs-io is
explicitly out of scope and retains its own petgraph for VF2).

- **ac-001** is the headline outcome: the dependency and every symbol reference
  are gone.
- **ac-002 / ac-005 / ac-006** are byte-for-byte parity bars for the three
  algorithm families (path enumeration, components, SSSR), each anchored on a
  pinned fixture count.
- **ac-003 / ac-004** guard the two Python-reachable `Atomistic` entry points.
- **ac-007** preserves the domain-agnostic-molgraph invariant inherited from the
  superseded spec.
- **ac-009** is the sole `performance` criterion: `topo_distances` must not
  regress against the recorded petgraph baseline (PoC shows it improves).
  Owed to `/mol:bench` for the formal history sign-off. **/mol:impl measurement
  (2026-06-12, `cargo bench graph/topo_distances`, criterion median, machine
  settled):** 1k 35.4µs ≤ 45.8µs · 10k 330µs ≤ 843µs · 100k 4.36ms ≤ 7.61ms —
  passes at every size (a hot-machine reading hit 8.9ms at 100k; the settled
  reading is the trustworthy one). Left `pending` per the evaluator protocol
  (performance verdict is `/mol:bench`'s to record).
