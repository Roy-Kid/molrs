---
title: GAFF AM1-BCC charge assignment (delegated, openmm-consistent)
status: approved
created: 2026-06-12
---

# GAFF AM1-BCC charge assignment (delegated, openmm-consistent)

## Summary
GAFF parameter files carry no partial charges, so a GAFF-parameterized small molecule needs per-atom charges from a separate model. This sub-spec adopts the **same charge scheme openmmforcefields uses: AM1-BCC, delegated to AmberTools** (`antechamber -c bcc` / `sqm`), rather than reimplementing semiempirical QM in molrs. The AM1-BCC computation lives at the delegation layer (molpy's `wrapper/antechamber.py`, or openff-toolkit `Molecule.assign_partial_charges(method="am1bcc")`); the molrs GAFF typifier (01–04) only **consumes** the resulting per-atom charges into the frame `charge` column. Ground truth is openmmforcefields/antechamber output; parity is asserted per-atom within float tolerance and gated to skip when AmberTools is absent. This deliberately keeps a quantum-chemistry engine out of the molrs Rust core.

## Domain basis
AM1-BCC (Jakalian, Bush, Jack, Bayly, *J. Comput. Chem.* 21(2):132-146, 2000, DOI 10.1002/(SICI)1096-987X(20000130)21:2<132::AID-JCC5>3.0.CO;2-P; and Jakalian, Jack, Bayly, *J. Comput. Chem.* 23(16):1623-1641, 2002, DOI 10.1002/jcc.10128) computes partial charges in three steps:

1. **Conformer** — a 3D geometry is required; AM1 charges are conformer-dependent. antechamber defaults to a single conformer.
2. **AM1 SCF** (`sqm`) — a semiempirical AM1 self-consistent-field calculation yields per-atom AM1 charges.
3. **BCC** — empirical bond-charge-correction increments, keyed by the bond's atom types + bond order, are added per bond; the final atomic charge is the AM1 charge plus the sum of its bonds' increments. The BCC table is fit so the result approximates HF/6-31G\* ESP (RESP-target) charges.

openmmforcefields' `GAFFTemplateGenerator` produces exactly these charges by invoking `antechamber -c bcc` (sqm-backed). "Consistent with openmm" therefore means: **same model (AM1-BCC), same engine (sqm/antechamber), same delegation posture (do not reimplement the QM).** molrs-core `gasteiger.rs` is a different, lower-fidelity charge model and must not be substituted.

## Design
This is the one piece of the GAFF pipeline that is **not** sunk into molrs Rust, because the AM1 SCF step requires a semiempirical QM engine that does not belong in the core library (and whose only mature implementations — `sqm`/antechamber — are GPL). The split:

- **Charge computation (delegation layer, Python):** reuse the existing molpy `wrapper/antechamber.py` (`antechamber -c bcc`) or openff-toolkit `assign_partial_charges(method="am1bcc")`. Input: a molecule with a 3D conformer (use the existing `molrs.Conformer` pipeline when coordinates are absent — see the conformer convention). Output: a per-atom charge vector in elementary-charge units.
- **Charge ingestion (molrs):** the per-atom charges are written into the frame's canonical `charge` column (float, e). The molrs `GaffTypifier` consumes pre-computed charges; it does **not** compute them. Net molecular charge is preserved (sum of partial charges equals the formal molecular charge within tolerance).
- **Determinism / provenance:** record the charge method (`am1bcc`), the conformer source, and the backend (AmberTools version) so charges are reproducible and the parity fixtures are auditable.

Reserved-but-not-built: the BCC increment table is vendorable MIT data (from openmmforcefields/AmberTools); a future native BCC step could consume it once an AM1-equivalent atomic-charge source exists. The natural QM-free native path is a NAGL / espaloma_charge graph-neural-net that predicts AM1-BCC charges directly from the molecular graph — a separate milestone that could eventually live in molrs as a small inference model. Neither is in scope here.

## Files to create or modify
- `molpy/src/molpy/wrapper/antechamber.py` (modify) — ensure an `am1bcc` charge entry point returning a per-atom charge vector aligned to atom order (may already exist; confirm/extend).
- molpy GAFF parameterization path (modify) — wire the delegated AM1-BCC charges into the frame `charge` column after molrs typing/params.
- `molpy/tests/test_wrapper/` (new test) — gated parity test of delegated AM1-BCC charges vs openmmforcefields/antechamber reference; skips cleanly without AmberTools.

(Owner note: implementation is in the molpy delegation + openff layer, not molrs Rust; molrs's only role is accepting charges into `frame["atoms"]["charge"]`. Spec lives in the molrs chain for continuity but does not add molrs-ff code.)

## Tasks
- [ ] Confirm/extend the molpy antechamber wrapper to expose an `am1bcc` charge call returning an atom-order-aligned charge vector (molpy/src/molpy/wrapper/antechamber.py)
- [ ] Wire delegated AM1-BCC charges into the frame `charge` column in the GAFF parameterization path, with method/conformer/backend recorded
- [ ] Use the existing molrs.Conformer pipeline to supply a 3D conformer when input coordinates are absent
- [ ] Write a gated parity test: per-atom AM1-BCC charge vs openmmforcefields/antechamber reference within tolerance; skips cleanly when AmberTools absent
- [ ] Assert net-charge conservation (sum of partial charges == formal molecular charge within tolerance)
- [ ] Document that Gasteiger is NOT a substitute and that native AM1 SCF is deliberately out of scope (delegation matches openmmforcefields)

## Testing strategy
- **Delegation happy path:** for a small in-scope molecule, the delegated `am1bcc` call returns a per-atom charge vector that lands in the frame `charge` column, atom-order aligned.
- **Parity (gated, scientific):** per-atom charges equal the openmmforcefields/antechamber AM1-BCC reference within float tolerance (e.g. ≤1e-3 e, or matched to antechamber's output rounding) over a curated fixture set; the test skips cleanly (passes) when AmberTools/fixtures are absent.
- **Net-charge invariant:** the sum of assigned partial charges equals the molecule's formal charge within tolerance, for neutral and charged species.
- **Negative guard:** the path does not silently fall back to Gasteiger or formal charges when the AM1-BCC backend is unavailable — it skips/errs explicitly.
- **Domain validation:** charges reproduce the AM1-BCC model openmmforcefields emits (same method/engine), per Domain basis.

## Out of scope
- Native semiempirical AM1 SCF in molrs — deliberately delegated (matches openmmforcefields). 
- A native BCC-increment step (reserved; needs an AM1-equivalent atomic-charge source first).
- NAGL / espaloma_charge GNN charge prediction — a separate future milestone (QM-free native AM1-BCC surrogate).
- RESP / HF-ESP charges, formal-charge-only, and Gasteiger as a substitute model.
- GAFF atom typing and bonded-parameter assignment — owned by `gaff-typifier-01..04`; this sub-spec only adds charges.
