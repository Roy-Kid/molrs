//! OPLS-AA SMARTS atom typing over real molecules from `tests-data/`.
//!
//! Per the MANDATORY IO testing rule, molecule inputs are **real files** read
//! from `tests-data/mol2/` (iterated, not a hand-crafted happy-path string),
//! and the force field is the real bundled `tests-data/xml/oplsaa.xml`. The only
//! in-code construction is a minimal Tripos-mol2 loader (the `ff` test target
//! does not enable the `io` feature, so it cannot call `read_mol2`; this mirrors
//! the MMFF test's self-contained SDF loader).

use std::path::Path;

use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::opls::{OplsTypeRow, OplsTypifier, OplsTypingMeta, annotate_opls};
use molrs::{Atom, Atomistic};

use crate::helpers;

/// Map a SYBYL bond-type token to a numeric bond order (the molrs convention:
/// aromatic = 1.5, amide = 1.0).
fn sybyl_bond_order(tok: &str) -> f64 {
    match tok {
        "1" => 1.0,
        "2" => 2.0,
        "3" => 3.0,
        "ar" => 1.5,
        "am" => 1.0,
        _ => 1.0,
    }
}

/// Element symbol from a Tripos atom record. SYBYL atom types come in several
/// conventions — `C.3` / `N.ar` (dotted) and `c3` / `hc` / `cl` (GAFF-style
/// lowercase from Antechamber). Normalise by dropping the `.`-suffix and any
/// trailing digits, then validate against the periodic table: prefer the
/// title-cased two-letter symbol (`Cl`, `Br`, `Na`) when it is a real element,
/// else the single leading letter, falling back to the atom name.
fn element_of(sybyl_type: &str, name: &str) -> String {
    let base: String = sybyl_type
        .split('.')
        .next()
        .unwrap_or(sybyl_type)
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if let Some(sym) = normalize_element(&base) {
        return sym;
    }
    let name_base: String = name
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    normalize_element(&name_base).unwrap_or(name_base)
}

/// Title-case a 1- or 2-letter element token and validate it against the
/// periodic table. Tries the two-letter symbol first (so `cl` -> `Cl`), then the
/// single leading letter (so `c3`-stripped `c` -> `C`).
fn normalize_element(token: &str) -> Option<String> {
    use std::str::FromStr;
    let t = token.to_ascii_lowercase();
    let title2 = |s: &str| -> String {
        let mut c = s.chars();
        let first = c.next().unwrap().to_ascii_uppercase();
        let rest: String = c.collect();
        format!("{first}{rest}")
    };
    if t.len() >= 2 {
        let two = title2(&t[..2]);
        if molrs::Element::from_str(&two).is_ok() {
            return Some(two);
        }
    }
    if !t.is_empty() {
        let one = title2(&t[..1]);
        if molrs::Element::from_str(&one).is_ok() {
            return Some(one);
        }
    }
    None
}

/// Minimal single-molecule Tripos mol2 loader producing an [`Atomistic`] with
/// `element` props and bond `order` props. Reads only the first `@<TRIPOS>`
/// MOLECULE in the file.
fn load_mol2(path: &Path) -> Atomistic {
    let text = std::fs::read_to_string(path).expect("read mol2");
    let mut g = Atomistic::new();
    let mut ids: Vec<molrs::AtomId> = Vec::new();
    let mut section = "";
    let mut seen_molecule = false;
    for line in text.lines() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("@<TRIPOS>") {
            // A second MOLECULE record => stop (single-molecule loader).
            if name == "MOLECULE" {
                if seen_molecule {
                    break;
                }
                seen_molecule = true;
            }
            section = match name {
                "ATOM" => "ATOM",
                "BOND" => "BOND",
                _ => "",
            };
            continue;
        }
        if t.is_empty() {
            continue;
        }
        match section {
            "ATOM" => {
                // id name x y z sybyl_type [subst_id subst_name charge]
                let f: Vec<&str> = t.split_whitespace().collect();
                if f.len() < 6 {
                    continue;
                }
                let x: f64 = f[2].parse().unwrap_or(0.0);
                let y: f64 = f[3].parse().unwrap_or(0.0);
                let z: f64 = f[4].parse().unwrap_or(0.0);
                let element = element_of(f[5], f[1]);
                ids.push(g.add_atom(Atom::xyz(&element, x, y, z)));
            }
            "BOND" => {
                // id origin target bond_type   (origin/target are 1-based)
                let f: Vec<&str> = t.split_whitespace().collect();
                if f.len() < 4 {
                    continue;
                }
                let i: usize = match f[1].parse::<usize>() {
                    Ok(v) if v >= 1 && v <= ids.len() => v - 1,
                    _ => continue,
                };
                let j: usize = match f[2].parse::<usize>() {
                    Ok(v) if v >= 1 && v <= ids.len() => v - 1,
                    _ => continue,
                };
                if let Ok(bid) = g.add_bond(ids[i], ids[j]) {
                    let _ = g.set_bond_prop(bid, "order", sybyl_bond_order(f[3]));
                }
            }
            _ => {}
        }
    }
    g
}

/// The real bundled OPLS-AA force field used as the typing source.
fn oplsaa_xml() -> String {
    std::fs::read_to_string(helpers::data_path("xml/oplsaa.xml")).expect("read oplsaa.xml")
}

#[test]
fn typifier_builds_from_real_oplsaa_xml() {
    // ac-001/ac-003: construction parses both typing-meta and the potential FF
    // from one XML, additively (the potential reader is unchanged).
    let typifier = OplsTypifier::from_xml_str(&oplsaa_xml()).expect("build OplsTypifier");
    assert!(
        !typifier.meta().is_empty(),
        "typing metadata should have rows"
    );
    // A known modern row carries class CT + a SMARTS def.
    let r135 = typifier.meta().get("opls_135").expect("opls_135 present");
    assert_eq!(r135.class, "CT");
    assert!(r135.def.is_some());
    // The potential FF still resolves charges per opls type.
    assert!(typifier.ff().get_style("atom", "full").is_some());
}

#[test]
fn ethane_atoms_typed_with_type_class_charge() {
    // ac-004: a real alkane (ethane.mol2 carries explicit H) is fully typed —
    // CT carbons + HC hydrogens — and typed atoms carry type/class/charge.
    let typifier = OplsTypifier::from_xml_str(&oplsaa_xml()).expect("build OplsTypifier");
    let mol = load_mol2(&helpers::data_path("mol2/ethane.mol2"));
    let typed = typifier.typify(&mol).expect("typify ethane");

    let mut typed_carbons = 0;
    let mut typed_hydrogens = 0;
    for (_, a) in typed.atoms() {
        let Some(ty) = a.get_str("type") else {
            continue;
        };
        assert!(ty.starts_with("opls_"), "type is an opls_NNN: {ty}");
        // class must be present for every typed atom.
        assert!(a.get_str("class").is_some(), "typed atom carries class");
        // charge resolved from the potential force field.
        assert!(a.get_f64("charge").is_some(), "typed atom carries charge");
        match a.get_str("element") {
            Some("C") => typed_carbons += 1,
            Some("H") => typed_hydrogens += 1,
            _ => {}
        }
    }
    assert_eq!(typed_carbons, 2, "both ethane carbons typed");
    assert_eq!(typed_hydrogens, 6, "all six ethane hydrogens typed");
}

#[test]
fn typing_runs_over_every_mol2_molecule() {
    // ac-004: iterate every real mol2 in tests-data — typing must never panic
    // or error, and at least one molecule must receive at least one opls type.
    let typifier = OplsTypifier::from_xml_str(&oplsaa_xml()).expect("build OplsTypifier");
    let files = helpers::format_files("mol2");
    assert!(!files.is_empty(), "tests-data/mol2 has files");

    let mut total_typed = 0usize;
    for path in &files {
        let mol = load_mol2(path);
        if mol.n_atoms() == 0 {
            continue;
        }
        let typed = typifier
            .typify(&mol)
            .unwrap_or_else(|e| panic!("typify {:?} failed: {e}", path.file_name().unwrap()));
        // Every typed atom carries a well-formed opls_NNN type.
        for (_, a) in typed.atoms() {
            if let Some(ty) = a.get_str("type") {
                assert!(ty.starts_with("opls_"), "{path:?}: bad type {ty}");
                total_typed += 1;
            }
        }
    }
    assert!(
        total_typed > 0,
        "at least one atom across all mol2 molecules should be typed"
    );
}

#[test]
fn recursive_dollar_def_types_a_real_molecule() {
    // ac-004: at least one def using recursive $() SMARTS. The bundled oplsaa.xml
    // happens to carry no $() def, so we inject a single recursive def into the
    // typing metadata and confirm it types atoms of a REAL molecule (ethane).
    // (Engine-level recursive support is also unit-tested in src/.)
    let ff = OplsTypifier::from_xml_str(&oplsaa_xml())
        .expect("build")
        .ff()
        .clone();
    let mut meta = OplsTypingMeta::new();
    meta.insert(
        "opls_135",
        OplsTypeRow {
            class: "CT".into(),
            // recursive $(): an sp3 carbon bonded to another sp3 carbon.
            def: Some("[$([CX4][CX4])]".into()),
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        },
    );
    let mol = load_mol2(&helpers::data_path("mol2/ethane.mol2"));
    let typed = annotate_opls(&mol, &meta, &ff).expect("typify with recursive def");
    let n = typed
        .atoms()
        .filter(|(_, a)| a.get_str("type") == Some("opls_135"))
        .count();
    assert_eq!(n, 2, "recursive $() def types both ethane carbons");
}
