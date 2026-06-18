//! OPLS-AA SMARTS atom typing.
//!
//! [`annotate_opls`] runs each type's compiled SMARTS `def` over a molecule,
//! collects the candidate `opls_NNN` types per atom, and picks the
//! highest-priority one (priority computed by
//! [`OplsTypingMeta::priorities`](super::meta::OplsTypingMeta::priorities)).
//! The chosen type's `type` / `class` / `charge` are written onto a labeled
//! copy of the input [`Atomistic`].
//!
//! # SMARTS reuse
//!
//! Matching uses the always-compiled molrs SMARTS engine
//! ([`SmartsPattern`](molrs::SmartsPattern)). Each `def` is the SMARTS for the
//! type's *target* atom: by RDKit convention the engine roots a match at query
//! atom 0, so the typed atom of every match is `match[0]`.
//!
//! # Conflict resolution
//!
//! When several defs match the same atom, the winner is chosen by, in order:
//! 1. higher priority (overrides / explicit / layer — see [`OplsTypingMeta`]);
//! 2. more specific pattern (more query atoms);
//! 3. earlier definition order (stable tie-break, by sorted `opls_NNN` name).
//!
//! Step 2 is a specificity proxy; molpy's exact `(score, pattern_size,
//! definition_order)` tie-break (which also counts query *edges*) is a chain-3
//! per-atom-parity refinement, not required here.
//!
//! # Out of scope
//!
//! - Legacy rows with no `def` are skipped (cannot be SMARTS-matched).
//! - Defs using molpy's `%opls_NNN` previously-typed-atom reference (a layered
//!   dependency extension, not standard SMARTS) are skipped as unsupported —
//!   they belong to the chain-3 layered-typing parity work, not chain 1. They
//!   are NOT treated as malformed.
//! - Bonded-term (bond / angle / dihedral) labeling is chain 2.

use std::collections::HashMap;

use molrs::{AtomId, Atomistic, SmartsPattern};

use crate::ff::forcefield::ForceField;

use super::meta::{OplsTypeRow, OplsTypingMeta};

/// A type's compiled SMARTS pattern plus its ranking inputs.
struct CompiledDef {
    name: String,
    pattern: SmartsPattern,
    priority: i64,
    /// Specificity proxy: number of query atoms in the pattern.
    specificity: usize,
    /// Stable definition order (index after sorting type names).
    order: usize,
}

/// Compile every SMARTS `def` in `meta` into a ranked pattern list.
///
/// Types with no `def` are skipped. A `def` that uses the `%opls_NNN`
/// previously-typed reference (molpy layered-typing extension) is skipped as
/// unsupported. Any other unparseable `def` is a broken force-field definition
/// and returns `Err` (fail-fast — never silently dropped).
fn compile_defs(meta: &OplsTypingMeta) -> Result<Vec<CompiledDef>, String> {
    let priorities = meta.priorities();

    // Deterministic order: sort by type name so `order` is stable run-to-run.
    let mut named: Vec<(&String, &OplsTypeRow)> = meta.iter().collect();
    named.sort_by(|a, b| a.0.cmp(b.0));

    let mut compiled = Vec::new();
    for (order, (name, row)) in named.into_iter().enumerate() {
        let Some(def) = row.def.as_deref() else {
            continue; // legacy / no-def row
        };
        if uses_typed_atom_ref(def) {
            continue; // molpy %opls_NNN extension — unsupported here (chain 3)
        }
        let pattern = compile_def(def)
            .map_err(|e| format!("OPLS type {name:?}: failed to parse SMARTS def {def:?}: {e}"))?;
        let specificity = pattern.num_query_atoms();
        compiled.push(CompiledDef {
            name: name.clone(),
            pattern,
            priority: *priorities.get(name).unwrap_or(&0),
            specificity,
            order,
        });
    }
    Ok(compiled)
}

/// Compile a single SMARTS `def`, with a targeted fix-up for OPLS monatomic-ion
/// types written as a bare multi-letter element (e.g. `Li`, `Na`, `Cl`): the
/// molrs engine only accepts organic-subset elements unbracketed, so a def that
/// is exactly one element symbol is retried bracketed (`[Li]`). Any *other*
/// parse failure (e.g. an unbalanced `[C`) is genuinely malformed and propagates
/// as an `Err` (ac-005 fail-fast).
fn compile_def(def: &str) -> Result<SmartsPattern, molrs::MolRsError> {
    match SmartsPattern::parse(def) {
        Ok(p) => Ok(p),
        Err(e) => {
            if is_bare_element_symbol(def) {
                SmartsPattern::parse(&format!("[{def}]"))
            } else {
                Err(e)
            }
        }
    }
}

/// Whether `def` is exactly one element symbol: an uppercase letter optionally
/// followed by a single lowercase letter (`Li`, `Na`, `Br`, `C`). Used only to
/// bracket bare monatomic-ion defs; never matches multi-atom SMARTS.
fn is_bare_element_symbol(def: &str) -> bool {
    let b = def.as_bytes();
    match b.len() {
        1 => b[0].is_ascii_uppercase(),
        2 => b[0].is_ascii_uppercase() && b[1].is_ascii_lowercase(),
        _ => false,
    }
}

/// Whether a SMARTS `def` uses molpy's `%opls_NNN` previously-typed-atom
/// reference: a `%` followed by a non-digit (standard SMARTS only allows
/// `%nn` two-digit ring closures, so `%` + letter/underscore is the extension).
fn uses_typed_atom_ref(def: &str) -> bool {
    let bytes = def.as_bytes();
    bytes
        .iter()
        .enumerate()
        .any(|(i, &b)| b == b'%' && bytes.get(i + 1).is_some_and(|n| !n.is_ascii_digit()))
}

/// Annotate `mol` with OPLS-AA atom types, returning a labeled copy.
///
/// Each atom matched by one or more SMARTS `def`s is assigned the
/// highest-priority type; that type's `type` (`opls_NNN`), `class`, and `charge`
/// (`e`, from the potential `ForceField`'s `("atom","full")` style) are written
/// onto the returned [`Atomistic`]. Atoms matched by no def are left untyped
/// (no `type` prop) — strict-mode failure is the consumer's policy.
///
/// `ff` supplies per-type charges; a type absent from the atom style simply
/// yields no `charge` prop. Bonded-term labeling is out of scope (chain 2).
///
/// # Errors
///
/// Returns `Err` if any SMARTS `def` is malformed (fail-fast), or if writing a
/// label onto the graph fails.
pub fn annotate_opls(
    mol: &Atomistic,
    meta: &OplsTypingMeta,
    ff: &ForceField,
) -> Result<Atomistic, String> {
    let compiled = compile_defs(meta)?;

    // Best candidate per atom: keep the ranking key alongside the chosen type.
    // Key ordering (all "higher wins"): (priority, specificity, −order).
    #[derive(Clone, Copy)]
    struct Rank {
        priority: i64,
        specificity: usize,
        order: usize,
    }
    let better = |new: &Rank, cur: &Rank| -> bool {
        // Higher priority, then higher specificity, then EARLIER definition order.
        (new.priority, new.specificity, std::cmp::Reverse(new.order))
            > (cur.priority, cur.specificity, std::cmp::Reverse(cur.order))
    };

    let mut best: HashMap<AtomId, (String, Rank)> = HashMap::new();
    for cd in &compiled {
        let rank = Rank {
            priority: cd.priority,
            specificity: cd.specificity,
            order: cd.order,
        };
        for m in cd.pattern.find_matches(mol) {
            // Target atom is the root (query atom 0).
            let Some(&target) = m.first() else { continue };
            match best.get(&target) {
                Some((_, cur)) if !better(&rank, cur) => {}
                _ => {
                    best.insert(target, (cd.name.clone(), rank));
                }
            }
        }
    }

    // Per-type charge from the potential ForceField's atom style.
    let charge_of = |type_name: &str| -> Option<f64> {
        ff.get_style("atom", "full")?
            .get_atomtype(type_name)?
            .params
            .get("charge")
    };

    let mut out = mol.clone();
    for (atom_id, (type_name, _)) in &best {
        out.set_atom(*atom_id, "type", type_name.clone())
            .map_err(|e| e.to_string())?;
        if let Some(row) = meta.get(type_name) {
            out.set_atom(*atom_id, "class", row.class.clone())
                .map_err(|e| e.to_string())?;
        }
        if let Some(q) = charge_of(type_name) {
            out.set_atom(*atom_id, "charge", q)
                .map_err(|e| e.to_string())?;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Atom;

    /// Build a tiny ethane-like skeleton C-C with explicit H neighbours so the
    /// `[C;X4](C)(H)(H)H` style defs have something to match. (Pure-function
    /// unit fixture — real-molecule typing lives in tests/ff/typifier/opls.rs.)
    fn ethane() -> Atomistic {
        let mut g = Atomistic::new();
        let c0 = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let c1 = g.add_atom(Atom::xyz("C", 1.5, 0.0, 0.0));
        g.add_bond(c0, c1).unwrap();
        for c in [c0, c1] {
            for k in 0..3 {
                let h = g.add_atom(Atom::xyz("H", 0.3 * (k as f64 + 1.0), 0.8, 0.0));
                g.add_bond(c, h).unwrap();
            }
        }
        g
    }

    fn meta_with(rows: &[(&str, OplsTypeRow)]) -> OplsTypingMeta {
        let mut m = OplsTypingMeta::new();
        for (name, row) in rows {
            m.insert(*name, row.clone());
        }
        m
    }

    fn row(class: &str, def: Option<&str>, overrides: &[&str]) -> OplsTypeRow {
        OplsTypeRow {
            class: class.to_string(),
            def: def.map(str::to_string),
            overrides: overrides.iter().map(|s| s.to_string()).collect(),
            priority: None,
            layer: 0,
        }
    }

    #[test]
    fn typing_assigns_carbon_and_hydrogen() {
        let m = meta_with(&[
            ("opls_135", row("CT", Some("[C;X4](C)(H)(H)H"), &[])),
            ("opls_140", row("HC", Some("H[C;X4]"), &[])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let typed = annotate_opls(&ethane(), &m, &ff).unwrap();

        let mut n_ct = 0;
        let mut n_hc = 0;
        for (_, a) in typed.atoms() {
            match a.get_str("type") {
                Some("opls_135") => {
                    n_ct += 1;
                    assert_eq!(a.get_str("class"), Some("CT"));
                }
                Some("opls_140") => n_hc += 1,
                _ => {}
            }
        }
        assert_eq!(n_ct, 2, "both methyl carbons typed CT");
        assert_eq!(n_hc, 6, "all six H typed HC");
    }

    #[test]
    fn higher_priority_overrides_wins() {
        // Two defs both match every C; opls_special overrides opls_generic, so it
        // gains priority and must win on the carbons.
        let m = meta_with(&[
            ("opls_generic", row("CG", Some("[C]"), &[])),
            ("opls_special", row("CS", Some("[C;X4]"), &["opls_generic"])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let typed = annotate_opls(&ethane(), &m, &ff).unwrap();
        for (id, a) in typed.atoms() {
            if matches!(a.get_str("element"), Some("C")) {
                assert_eq!(
                    a.get_str("type"),
                    Some("opls_special"),
                    "carbon {id:?} should take the higher-priority override"
                );
            }
        }
    }

    #[test]
    fn charge_written_from_forcefield() {
        let m = meta_with(&[("opls_140", row("HC", Some("H[C;X4]"), &[]))]);
        let mut ff = ForceField::new("OPLS-AA");
        ff.def_atomstyle("full")
            .def_atomtype("opls_140", &[("mass", 1.008), ("charge", 0.06)]);
        let typed = annotate_opls(&ethane(), &m, &ff).unwrap();
        let h = typed
            .atoms()
            .find(|(_, a)| a.get_str("type") == Some("opls_140"))
            .expect("a hydrogen was typed");
        assert_eq!(h.1.get_f64("charge"), Some(0.06));
    }

    #[test]
    fn malformed_def_fails_fast() {
        // Unbalanced bracket — a broken force-field def, must Err (never drop).
        let m = meta_with(&[("opls_bad", row("X", Some("[C"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let err = annotate_opls(&ethane(), &m, &ff).unwrap_err();
        assert!(err.contains("opls_bad"), "err names the type: {err}");
    }

    #[test]
    fn recursive_dollar_def_matches() {
        // Recursive $() SMARTS: an sp3 carbon bonded to another sp3 carbon.
        // Exercises the engine's recursive `$(...)` support (rooted at the
        // candidate atom). Both ethane carbons match.
        let m = meta_with(&[("opls_rec", row("CT", Some("[$([CX4][CX4])]"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let typed = annotate_opls(&ethane(), &m, &ff).unwrap();
        let n = typed
            .atoms()
            .filter(|(_, a)| a.get_str("type") == Some("opls_rec"))
            .count();
        assert_eq!(n, 2, "recursive def should match both sp3 carbons");
    }

    #[test]
    fn typed_atom_ref_def_is_skipped_not_error() {
        // A %opls_NNN def is the molpy layered extension — skipped, not an error.
        let m = meta_with(&[("opls_ref", row("HA", Some("[H][C;%opls_145]"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let typed = annotate_opls(&ethane(), &m, &ff).unwrap();
        // Nothing typed (the only def was skipped), but no error.
        assert!(typed.atoms().all(|(_, a)| a.get_str("type").is_none()));
    }
}
