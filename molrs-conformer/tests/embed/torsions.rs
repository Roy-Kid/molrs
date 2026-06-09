//! Per-bond ETKDGv3 experimental-torsion validation against RDKit 2026.03.2.
//!
//! Oracle: `/Users/roykid/.claude/jobs/938e961b/tmp/gen_torsion_fixtures.py`
//! runs RDKit `rdDistGeom.GetExperimentalTorsions(mol, ETKDGv3())` and writes,
//! per molecule, a V2000 SDF (explicit H, RDKit atom order) + a
//! `torsion_reference.json` entry listing each rotatable bond's winning SMARTS,
//! `signs[6]`, and `V[6]`.
//!
//! This test (acceptance ac-001 / ac-002 / ac-003) loads each SDF into a
//! `MolGraph`, runs molrs's full-table SMARTS matcher
//! (`experimental_torsions_with_provenance`), and asserts that for every bond
//! RDKit assigns a torsion, molrs assigns the **same** `(signs, V)` (V within
//! 1e-3, signs exact) on the same central bond, and the table-of-origin
//! (v2 vs macrocycle) is consistent with the winning SMARTS.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use molrs::Atomistic;
use molrs::molgraph::{Atom, PropValue};
use molrs_conformer::distgeom::{TorsionTable, experimental_torsions_with_provenance};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/embed/fixtures")
}

/// V2000 SDF loader preserving atom order + bond orders (coords ignored).
fn load_sdf(path: &Path) -> Atomistic {
    let text = std::fs::read_to_string(path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    let counts = lines[3];
    let n_atoms: usize = counts[0..3].trim().parse().expect("n_atoms");
    let n_bonds: usize = counts[3..6].trim().parse().expect("n_bonds");

    let mut g = Atomistic::new();
    let mut ids = Vec::with_capacity(n_atoms);
    for k in 0..n_atoms {
        let line = lines[4 + k];
        let x: f64 = line[0..10].trim().parse().expect("x");
        let y: f64 = line[10..20].trim().parse().expect("y");
        let z: f64 = line[20..30].trim().parse().expect("z");
        let element = line[31..34].trim().to_string();
        ids.push(g.add_atom(Atom::xyz(&element, x, y, z)));
    }
    for k in 0..n_bonds {
        let line = lines[4 + n_atoms + k];
        let i: usize = line[0..3].trim().parse::<usize>().expect("bond i") - 1;
        let j: usize = line[3..6].trim().parse::<usize>().expect("bond j") - 1;
        let order: f64 = line[6..9].trim().parse().expect("bond order");
        let bid = g.add_bond(ids[i], ids[j]).expect("add bond");
        g.set_bond_prop(bid, "order", PropValue::F64(order))
            .expect("bond");
    }
    g
}

/// One RDKit oracle torsion (atom order = SDF order).
#[derive(Debug, Clone)]
struct RefTorsion {
    atoms: [usize; 4],
    smarts: String,
    signs: [i8; 6],
    v: [f64; 6],
}

/// Hand-rolled extraction of the `torsions` array for a molecule from the
/// reference JSON (no serde dependency in this crate's test target).
fn ref_torsions(json: &str, name: &str) -> Vec<RefTorsion> {
    let key = format!("\"{name}\"");
    let start = json.find(&key).unwrap_or_else(|| panic!("no entry {name}"));
    // Find this molecule's "torsions": [ ... ] array, balanced on brackets.
    let tors_pos = json[start..].find("\"torsions\"").expect("torsions key") + start;
    let arr_start = json[tors_pos..].find('[').expect("arr [") + tors_pos;
    let mut depth = 0i32;
    let mut end = arr_start;
    for (off, c) in json[arr_start..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    end = arr_start + off;
                    break;
                }
            }
            _ => {}
        }
    }
    let body = &json[arr_start + 1..end];
    let mut out = Vec::new();
    // Split into object substrings at top-level "{ ... }".
    let mut obj_depth = 0i32;
    let mut obj_start = None;
    for (off, c) in body.char_indices() {
        match c {
            '{' => {
                if obj_depth == 0 {
                    obj_start = Some(off);
                }
                obj_depth += 1;
            }
            '}' => {
                obj_depth -= 1;
                if obj_depth == 0 {
                    let s = &body[obj_start.unwrap()..=off];
                    out.push(parse_ref_obj(s));
                }
            }
            _ => {}
        }
    }
    out
}

fn parse_ref_obj(s: &str) -> RefTorsion {
    let atoms_v = parse_int_array(s, "atoms");
    let signs_v = parse_int_array(s, "signs");
    let v_v = parse_f64_array(s, "V");
    let smarts = parse_string(s, "smarts");
    let mut atoms = [0usize; 4];
    for (i, a) in atoms_v.iter().enumerate().take(4) {
        atoms[i] = *a as usize;
    }
    let mut signs = [0i8; 6];
    for (i, a) in signs_v.iter().enumerate().take(6) {
        signs[i] = *a as i8;
    }
    let mut v = [0f64; 6];
    for (i, a) in v_v.iter().enumerate().take(6) {
        v[i] = *a;
    }
    RefTorsion {
        atoms,
        smarts,
        signs,
        v,
    }
}

fn slice_after_key<'a>(s: &'a str, key: &str) -> &'a str {
    let kp = s.find(&format!("\"{key}\"")).expect("key present");
    let after = &s[kp + key.len() + 2..];
    let colon = after.find(':').expect("colon");
    &after[colon + 1..]
}

fn parse_int_array(s: &str, key: &str) -> Vec<i64> {
    let rest = slice_after_key(s, key);
    let lb = rest.find('[').expect("[");
    let rb = rest.find(']').expect("]");
    rest[lb + 1..rb]
        .split(',')
        .filter_map(|t| t.trim().parse::<i64>().ok())
        .collect()
}

fn parse_f64_array(s: &str, key: &str) -> Vec<f64> {
    let rest = slice_after_key(s, key);
    let lb = rest.find('[').expect("[");
    let rb = rest.find(']').expect("]");
    rest[lb + 1..rb]
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

fn parse_string(s: &str, key: &str) -> String {
    let rest = slice_after_key(s, key);
    let q1 = rest.find('"').expect("open quote");
    let q2 = rest[q1 + 1..].find('"').expect("close quote") + q1 + 1;
    rest[q1 + 1..q2].to_string()
}

/// Unordered central-bond key.
fn bond_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// The table-of-origin RDKit's winning SMARTS implies: any `r{9-}` ⇒
/// macrocycle, else v2 (ETKDGv3 excludes the small-ring table).
fn expected_table(smarts: &str) -> TorsionTable {
    if smarts.contains("r{9-}") {
        TorsionTable::Macrocycles
    } else {
        TorsionTable::V2
    }
}

#[test]
fn ac001_ac002_per_bond_matches_rdkit() {
    let json = std::fs::read_to_string(fixtures_dir().join("torsion_reference.json"))
        .expect("read torsion_reference.json");

    let names = [
        "butane",
        "biphenyl",
        "alanine",
        "methyl_acetate",
        "acetamide",
        "ring12",
    ];

    println!("\n=== ac-001 per-bond experimental torsions (molrs vs RDKit ETKDGv3) ===");
    for name in names {
        let mol = load_sdf(&fixtures_dir().join(format!("torsion_{name}.sdf")));
        let assigned = experimental_torsions_with_provenance(&mol);

        // Index molrs assignments by central bond.
        let mut by_bond: HashMap<(usize, usize), &molrs_conformer::distgeom::AssignedTorsion> =
            HashMap::new();
        for a in &assigned {
            let [_, j, k, _] = a.constraint.atoms;
            by_bond.insert(bond_key(j, k), a);
        }

        let refs = ref_torsions(&json, name);
        println!(
            "  {name:16} rdkit_bonds={:2} molrs_bonds={:2}",
            refs.len(),
            assigned.len()
        );

        for r in &refs {
            let key = bond_key(r.atoms[1], r.atoms[2]);
            let got = by_bond.get(&key).unwrap_or_else(|| {
                panic!(
                    "{name}: RDKit assigns a torsion on bond {:?} (smarts {}) but molrs did not",
                    [r.atoms[1], r.atoms[2]],
                    r.smarts
                )
            });

            // signs exact
            assert_eq!(
                got.constraint.signs, r.signs,
                "{name} bond {:?}: signs mismatch\n  molrs={:?}\n  rdkit={:?}\n  molrs smarts={}\n  rdkit smarts={}",
                key, got.constraint.signs, r.signs, got.constraint.pattern, r.smarts
            );
            // V within 1e-3
            for m in 0..6 {
                assert!(
                    (got.constraint.force_constants[m] - r.v[m]).abs() < 1e-3,
                    "{name} bond {:?}: V[{m}] mismatch molrs={} rdkit={}\n  molrs smarts={}\n  rdkit smarts={}",
                    key,
                    got.constraint.force_constants[m],
                    r.v[m],
                    got.constraint.pattern,
                    r.smarts
                );
            }
            // table-of-origin consistent with the winning SMARTS
            assert_eq!(
                got.table,
                expected_table(&r.smarts),
                "{name} bond {:?}: table-of-origin mismatch (rdkit smarts {})",
                key,
                r.smarts
            );
        }
    }
    println!("(all RDKit-assigned bonds reproduced: same signs, V<1e-3, correct table)\n");
}

/// ac-002: a >=9-membered ring's bonds get macrocycle torsions; acyclic bonds
/// get v2 torsions. Verified directly on the 12-ring + butane.
#[test]
fn ac002_macrocycle_vs_acyclic_layering() {
    // 12-membered carbocycle: every assigned torsion must come from the
    // macrocycle table.
    let ring = load_sdf(&fixtures_dir().join("torsion_ring12.sdf"));
    let ra = experimental_torsions_with_provenance(&ring);
    assert!(!ra.is_empty(), "ring12 should receive macrocycle torsions");
    for a in &ra {
        assert_eq!(
            a.table,
            TorsionTable::Macrocycles,
            "ring12 bond {:?} should be macrocycle, got {:?} ({})",
            a.constraint.atoms,
            a.table,
            a.constraint.pattern
        );
    }

    // Butane: the acyclic C-C bond comes from v2.
    let but = load_sdf(&fixtures_dir().join("torsion_butane.sdf"));
    let ba = experimental_torsions_with_provenance(&but);
    assert!(!ba.is_empty(), "butane should receive a v2 torsion");
    for a in &ba {
        assert_eq!(
            a.table,
            TorsionTable::V2,
            "butane bond {:?} should be v2",
            a.constraint.atoms
        );
    }
}

/// ac-003: every assignment flows through the full SMARTS tables — the winning
/// pattern for each bond is a real SMARTS string from the embedded tables (not
/// a hardcoded subset label).
#[test]
fn ac003_patterns_are_full_table_smarts() {
    let but = load_sdf(&fixtures_dir().join("torsion_butane.sdf"));
    let ba = experimental_torsions_with_provenance(&but);
    for a in &ba {
        // A real table SMARTS contains atom-map labels and bracket atoms.
        assert!(
            a.constraint.pattern.contains(":1")
                && a.constraint.pattern.contains(":2")
                && a.constraint.pattern.contains('['),
            "pattern should be a full-table SMARTS, got {}",
            a.constraint.pattern
        );
    }
}
