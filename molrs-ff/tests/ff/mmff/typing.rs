//! Per-atom validation of the MMFF94 typer + charger against RDKit.
//!
//! Each fixture pair `<name>.sdf` / `<name>.json` is produced by
//! `gen_mmff_fixtures.py` (RDKit 2026.03.2): the SDF carries the 3D
//! conformer (atom order == RDKit atom order) and the JSON carries the
//! reference `GetMMFFAtomType` / `GetMMFFPartialCharge` per atom for both
//! MMFF94 and MMFF94s.
//!
//! We load the SDF into a `MolGraph` preserving atom order (and parsing the
//! `M  CHG` lines that the bundled SDF reader drops), run
//! `MmffMolProperties::compute`, and assert per-atom equality of types and
//! near-equality (1e-4) of charges.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use molrs::molgraph::{Atom, MolGraph, PropValue};
use molrs_ff::mmff::{MmffMolProperties, MmffVariant};
use serde_json::Value;

const CHARGE_TOL: f64 = 1.0e-4;

const NAMES: &[&str] = &[
    "methane",
    "ethylene",
    "benzene",
    "pyridine",
    "imidazole",
    "aniline",
    "acetamide",
    "nitrobenzene",
    "benzenesulfonic_acid",
    "caffeine",
];

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/mmff/fixtures")
}

/// Minimal V2000 SDF loader that preserves atom order, reads bond orders, and
/// parses `M  CHG` property lines (the bundled reader ignores formal charges).
fn load_sdf(path: &Path) -> MolGraph {
    let text = std::fs::read_to_string(path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    // header is 3 lines, then counts line at index 3.
    let counts = lines[3];
    let n_atoms: usize = counts[0..3].trim().parse().expect("n_atoms");
    let n_bonds: usize = counts[3..6].trim().parse().expect("n_bonds");

    let mut g = MolGraph::new();
    let mut ids = Vec::with_capacity(n_atoms);
    for k in 0..n_atoms {
        let line = lines[4 + k];
        let x: f64 = line[0..10].trim().parse().expect("x");
        let y: f64 = line[10..20].trim().parse().expect("y");
        let z: f64 = line[20..30].trim().parse().expect("z");
        let element = line[31..34].trim().to_string();
        let mut atom = Atom::xyz(&element, x, y, z);
        atom.set("formal_charge", PropValue::Int(0));
        ids.push(g.add_atom(atom));
    }
    for k in 0..n_bonds {
        let line = lines[4 + n_atoms + k];
        let i: usize = line[0..3].trim().parse::<usize>().expect("bond i") - 1;
        let j: usize = line[3..6].trim().parse::<usize>().expect("bond j") - 1;
        let order: f64 = line[6..9].trim().parse().expect("bond order");
        let bid = g.add_bond(ids[i], ids[j]).expect("add bond");
        g.get_bond_mut(bid)
            .expect("bond")
            .props
            .insert("order".to_string(), PropValue::F64(order));
    }
    // M  CHG  n   a1 c1 a2 c2 ...   (a* are 1-based atom indices)
    for line in &lines[4 + n_atoms + n_bonds..] {
        let t = line.trim_end();
        if t == "M  END" || t == "$$$$" {
            break;
        }
        if let Some(rest) = t.strip_prefix("M  CHG") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            let pairs: usize = toks[0].parse().unwrap_or(0);
            for p in 0..pairs {
                let ai: usize = toks[1 + 2 * p].parse::<usize>().expect("chg atom") - 1;
                let chg: i32 = toks[2 + 2 * p].parse().expect("chg val");
                g.get_atom_mut(ids[ai])
                    .expect("atom")
                    .set("formal_charge", PropValue::Int(chg));
            }
        }
    }
    g
}

fn load_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("read json");
    serde_json::from_str(&text).expect("parse json")
}

fn ref_types(json: &Value, key: &str) -> Vec<u8> {
    json[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["mmff_type"].as_u64().unwrap() as u8)
        .collect()
}

fn ref_charges(json: &Value, key: &str) -> Vec<f64> {
    json[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["mmff_charge"].as_f64().unwrap())
        .collect()
}

struct Outcome {
    types_ok: bool,
    charges_ok: bool,
    type_diffs: Vec<(usize, u8, u8)>, // (idx, got, want)
    charge_diffs: Vec<(usize, f64, f64)>,
}

fn check(name: &str, variant: MmffVariant, key: &str) -> Outcome {
    let dir = fixtures_dir();
    let mol = load_sdf(&dir.join(format!("{name}.sdf")));
    let json = load_json(&dir.join(format!("{name}.json")));
    let want_types = ref_types(&json, key);
    let want_charges = ref_charges(&json, key);

    let props = MmffMolProperties::compute(&mol, variant)
        .unwrap_or_else(|e| panic!("{name}/{key}: compute failed: {e}"));

    let mut type_diffs = Vec::new();
    let mut charge_diffs = Vec::new();
    for i in 0..want_types.len() {
        let got = props.atom_type(i);
        if got != want_types[i] {
            type_diffs.push((i, got, want_types[i]));
        }
        let gq = props.partial_charge(i);
        if (gq - want_charges[i]).abs() > CHARGE_TOL {
            charge_diffs.push((i, gq, want_charges[i]));
        }
    }
    Outcome {
        types_ok: type_diffs.is_empty(),
        charges_ok: charge_diffs.is_empty(),
        type_diffs,
        charge_diffs,
    }
}

#[test]
fn mmff94_per_atom_types_and_charges() {
    let mut type_fail: HashMap<&str, Vec<(usize, u8, u8)>> = HashMap::new();
    let mut charge_fail: HashMap<&str, Vec<(usize, f64, f64)>> = HashMap::new();

    for &name in NAMES {
        let o = check(name, MmffVariant::Mmff94, "mmff94");
        if !o.types_ok {
            type_fail.insert(name, o.type_diffs);
        }
        if !o.charges_ok {
            charge_fail.insert(name, o.charge_diffs);
        }
    }

    if !type_fail.is_empty() || !charge_fail.is_empty() {
        let mut msg = String::from("MMFF94 mismatches vs RDKit:\n");
        for (name, diffs) in &type_fail {
            msg.push_str(&format!("  TYPES {name}: {diffs:?}\n"));
        }
        for (name, diffs) in &charge_fail {
            msg.push_str(&format!("  CHARGES {name}: {diffs:?}\n"));
        }
        panic!("{msg}");
    }
}

#[test]
fn mmff94s_per_atom_types_and_charges() {
    // The s-variant Oop/Tor tables are not yet ported; for the coverage set
    // the typing / charge paths are identical to MMFF94 (the variants differ
    // only in planarity of a few delocalized N, which does not change types
    // or charges here). Assert that and surface any molecule where it breaks.
    let mut type_fail: HashMap<&str, Vec<(usize, u8, u8)>> = HashMap::new();
    let mut charge_fail: HashMap<&str, Vec<(usize, f64, f64)>> = HashMap::new();
    for &name in NAMES {
        let o = check(name, MmffVariant::Mmff94s, "mmff94s");
        if !o.types_ok {
            type_fail.insert(name, o.type_diffs);
        }
        if !o.charges_ok {
            charge_fail.insert(name, o.charge_diffs);
        }
    }
    if !type_fail.is_empty() || !charge_fail.is_empty() {
        let mut msg = String::from("MMFF94s mismatches vs RDKit:\n");
        for (name, diffs) in &type_fail {
            msg.push_str(&format!("  TYPES {name}: {diffs:?}\n"));
        }
        for (name, diffs) in &charge_fail {
            msg.push_str(&format!("  CHARGES {name}: {diffs:?}\n"));
        }
        panic!("{msg}");
    }
}

// --- ac-004: unsupported atoms fail loudly; charge sum equals net charge ---

fn atom_with_charge(sym: &str, charge: i32) -> Atom {
    let mut a = Atom::new();
    a.set("element", sym);
    a.set("formal_charge", PropValue::Int(charge));
    a
}

fn add_bond_order(
    g: &mut MolGraph,
    i: molrs::molgraph::AtomId,
    j: molrs::molgraph::AtomId,
    order: f64,
) {
    let bid = g.add_bond(i, j).expect("add bond");
    g.get_bond_mut(bid)
        .expect("bond")
        .props
        .insert("order".to_string(), PropValue::F64(order));
}

#[test]
fn unsupported_atom_fails() {
    // A bare transition metal (Fe) has no MMFF type → compute must Err.
    let mut g = MolGraph::new();
    g.add_atom(atom_with_charge("Fe", 0));
    let res = MmffMolProperties::compute(&g, MmffVariant::Mmff94);
    assert!(res.is_err(), "Fe should be unsupported by MMFF94");
}

#[test]
fn charge_sum_equals_net_charge() {
    // Ammonium NH4+ : N(+1) bonded to 4 H, all single bonds → sum == +1.
    let mut nh4 = MolGraph::new();
    let n = nh4.add_atom(atom_with_charge("N", 1));
    for _ in 0..4 {
        let h = nh4.add_atom(atom_with_charge("H", 0));
        add_bond_order(&mut nh4, n, h, 1.0);
    }
    let p = MmffMolProperties::compute(&nh4, MmffVariant::Mmff94).expect("nh4 types");
    let sum: f64 = (0..5).map(|i| p.partial_charge(i)).sum();
    assert!((sum - 1.0).abs() < 1.0e-6, "NH4+ charge sum {sum} != +1");

    // Acetate CC(=O)[O-] : carboxylate O(-1) → sum == -1.
    let mut ac = MolGraph::new();
    let c_me = ac.add_atom(atom_with_charge("C", 0));
    let c_co = ac.add_atom(atom_with_charge("C", 0));
    let o_dbl = ac.add_atom(atom_with_charge("O", 0));
    let o_neg = ac.add_atom(atom_with_charge("O", -1));
    add_bond_order(&mut ac, c_me, c_co, 1.0);
    add_bond_order(&mut ac, c_co, o_dbl, 2.0);
    add_bond_order(&mut ac, c_co, o_neg, 1.0);
    for _ in 0..3 {
        let h = ac.add_atom(atom_with_charge("H", 0));
        add_bond_order(&mut ac, c_me, h, 1.0);
    }
    let p = MmffMolProperties::compute(&ac, MmffVariant::Mmff94).expect("acetate types");
    let n_at = 4 + 3;
    let sum: f64 = (0..n_at).map(|i| p.partial_charge(i)).sum();
    assert!((sum + 1.0).abs() < 1.0e-6, "acetate charge sum {sum} != -1");
}
