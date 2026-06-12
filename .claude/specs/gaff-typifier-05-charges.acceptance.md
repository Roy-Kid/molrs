---
slug: gaff-typifier-05-charges
criteria:
  - id: ac-001
    summary: Charge model is AM1-BCC, delegated (consistent with openmmforcefields)
    type: code
    pass_when: |
      The GAFF charge path uses the AM1-BCC model via the delegated backend
      (molpy antechamber wrapper `-c bcc` or openff
      assign_partial_charges(method="am1bcc")); no native semiempirical QM is
      added to molrs and Gasteiger is not used as the charge source.
    status: pending
  - id: ac-002
    summary: Delegated charges land in the frame charge column, atom-order aligned
    type: code
    pass_when: |
      For a small in-scope molecule, the delegated am1bcc call returns a
      per-atom charge vector that is written into frame["atoms"]["charge"]
      aligned to atom order, with method/conformer/backend recorded.
    status: pending
  - id: ac-003
    summary: Net molecular charge is conserved
    type: code
    pass_when: |
      The sum of assigned partial charges equals the molecule's formal charge
      within tolerance, for a neutral and a charged test species.
    status: pending
  - id: ac-004
    summary: No silent fallback when AM1-BCC backend is unavailable
    type: code
    pass_when: |
      When the AM1-BCC backend (sqm/antechamber) is unavailable, the path skips
      or errors explicitly and never silently substitutes Gasteiger or formal
      charges.
    status: pending
  - id: ac-005
    summary: Per-atom AM1-BCC charge parity vs openmmforcefields/antechamber (gated)
    type: scientific
    evaluator_hint: "marker: gaff_charge_parity; oracle: antechamber -c bcc / openmmforcefields am1bcc; requires AmberTools"
    pass_when: |
      Over a curated fixture set, per-atom assigned charges equal the
      openmmforcefields/antechamber AM1-BCC reference within float tolerance
      (<=1e-3 e or matched to antechamber output rounding). The test passes by
      skipping when AmberTools/fixtures are absent, and enforces parity when
      present.
    status: pending
---

# Acceptance criteria

- **ac-001** — the headline decision: AM1-BCC, delegated, openmm-consistent; no native QM, no Gasteiger substitution.
- **ac-002** — delegated charges flow into the canonical `charge` column with reproducibility metadata.
- **ac-003** — net-charge conservation invariant (neutral + charged).
- **ac-004** — fail-fast: never silently downgrade the charge model when the backend is missing.
- **ac-005 (scientific, gated)** — per-atom AM1-BCC parity vs the openmmforcefields/antechamber oracle; skips cleanly without AmberTools, enforces when present.
