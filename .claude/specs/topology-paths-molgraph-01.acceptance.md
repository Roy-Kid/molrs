---
slug: topology-paths-molgraph-01
criteria:
  - id: ac-001
    summary: molgraph paths_of_length(k=2) enumerates angle triplets
    type: code
    evaluator_hint: "Known graphs: ethane skeleton, branched, ring."
    pass_when: |
      MolGraph::paths_of_length(bonds, 2) returns each 3-node triplet [i,j,k] once with i<k, for ethane/branched/ring test graphs matching hand-computed sets; empty/single-edge/disconnected return correct (often empty) results without panic.
    status: pending
    last_checked: ""
  - id: ac-002
    summary: molgraph paths_of_length(k=3) enumerates dihedral quartets
    type: code
    evaluator_hint: ""
    pass_when: |
      paths_of_length(bonds, 3) returns 4-node quartets [i,j,k,l] canonicalized j<k, excluding i==l and reverse duplicates, matching hand-computed sets on the test graphs.
    status: pending
    last_checked: ""
  - id: ac-003
    summary: molgraph stays domain-agnostic
    type: code
    evaluator_hint: ""
    pass_when: |
      The new molgraph primitive contains no 'angle'/'dihedral'/chemistry vocabulary; it operates purely on a relation KindId. (Naming appears only in Atomistic.)
    status: pending
    last_checked: ""
  - id: ac-004
    summary: Atomistic.generate_topology adds correct angle/dihedral counts
    type: code
    evaluator_hint: "Ethane = 12 angles, 9 dihedrals."
    pass_when: |
      Atomistic::generate_topology(true, true, false) on ethane (7 bonds) adds 12 angle and 9 dihedral relations; benzene/cyclohexane counts match a reference; dedup keys (angles i<k, dihedrals j<k) hold.
    status: pending
    last_checked: ""
  - id: ac-005
    summary: generate_topology is idempotent
    type: code
    evaluator_hint: ""
    pass_when: |
      A second call with clear_existing=false adds 0; with clear_existing=true regenerates the identical set. No duplicate relations.
    status: pending
    last_checked: ""
  - id: ac-006
    summary: Python (molpy subclass) inherits generate_topology with matching counts
    type: code
    evaluator_hint: "Rebuild wheel; molpy.Atomistic inherits the method."
    pass_when: |
      molpy.Atomistic (subclass of molrs.Atomistic) calls the inherited generate_topology and produces angle/dihedral counts equal to molpy get_topo(gen_angle=True, gen_dihe=True) historical output for ethane and butane.
    status: pending
    last_checked: ""
---

# Acceptance — Graph-theoretic topology in molgraph

Binding contract for `topology-paths-molgraph-01.md`. ac-001..005 are in-crate
(molrs-core). ac-006 needs the rebuilt wheel; the molpy `core/topology.py`
retirement is a separate molpy change.
