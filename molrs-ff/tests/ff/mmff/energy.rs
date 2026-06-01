//! MMFF94 total-energy + analytical-gradient validation against RDKit.
//!
//! Fixtures `<name>.sdf` / `<name>.energy.json` are produced by
//! `gen_mmff_energy_fixtures.py` (RDKit 2026.03.2): the SDF carries the 3D
//! conformer (atom order == RDKit atom order) and the JSON carries the
//! reference MMFF94 total energy `MMFFGetMoleculeForceField(mol).CalcEnergy()`.
//!
//! Tests:
//! - total energy within 1e-3 kcal/mol of RDKit for a coverage set;
//! - analytical gradient vs central finite difference (step 1e-5) < 1e-5;
//! - translation + rigid-rotation invariance of benzene (< 1e-9);
//! - single-eval timing of a ~50-atom molecule (ac-006 baseline; printed).

use std::path::{Path, PathBuf};
use std::time::Instant;

use molrs::molgraph::{Atom, MolGraph, PropValue};
use molrs_ff::mmff::{MmffForceField, MmffMolProperties, MmffVariant};
use molrs_ff::potential::Potential;
use serde_json::Value;

const ENERGY_TOL: f64 = 1.0e-3;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/mmff/fixtures")
}

/// Minimal V2000 SDF loader preserving atom order and bond orders, parsing
/// `M  CHG` (same as typing.rs).
fn load_sdf(path: &Path) -> MolGraph {
    let text = std::fs::read_to_string(path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
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
            for q in 0..pairs {
                let ai: usize = toks[1 + 2 * q].parse::<usize>().expect("chg atom") - 1;
                let chg: i32 = toks[2 + 2 * q].parse().expect("chg val");
                g.get_atom_mut(ids[ai])
                    .expect("atom")
                    .set("formal_charge", PropValue::Int(chg));
            }
        }
    }
    g
}

/// Flat `[x0,y0,z0,...]` coordinates in SDF atom order.
fn coords_of(path: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    let n_atoms: usize = lines[3][0..3].trim().parse().expect("n_atoms");
    let mut c = Vec::with_capacity(3 * n_atoms);
    for k in 0..n_atoms {
        let line = lines[4 + k];
        c.push(line[0..10].trim().parse::<f64>().expect("x"));
        c.push(line[10..20].trim().parse::<f64>().expect("y"));
        c.push(line[20..30].trim().parse::<f64>().expect("z"));
    }
    c
}

fn ref_energy(name: &str) -> f64 {
    let path = fixtures_dir().join(format!("{name}.energy.json"));
    let text = std::fs::read_to_string(&path).expect("read energy json");
    let v: Value = serde_json::from_str(&text).expect("parse json");
    v["mmff94_total_energy"].as_f64().expect("energy field")
}

fn build_ff(name: &str) -> (MmffForceField, Vec<f64>) {
    let dir = fixtures_dir();
    let mol = load_sdf(&dir.join(format!("{name}.sdf")));
    let coords = coords_of(&dir.join(format!("{name}.sdf")));
    let props = MmffMolProperties::compute(&mol, MmffVariant::Mmff94)
        .unwrap_or_else(|e| panic!("{name}: typing failed: {e}"));
    let ff = MmffForceField::build(&mol, &props)
        .unwrap_or_else(|e| panic!("{name}: ff build failed: {e}"));
    (ff, coords)
}

#[test]
fn total_energy_matches_rdkit() {
    let names = [
        "e_ethane",
        "e_ethylene",
        "e_benzene",
        "e_butane",
        "e_caffeine",
    ];
    let mut fails = Vec::new();
    for name in names {
        let (ff, coords) = build_ff(name);
        let got = ff.eval(&coords).0;
        let want = ref_energy(name);
        let delta = (got - want).abs();
        println!("{name:12} molrs={got:14.6}  rdkit={want:14.6}  d={delta:.3e}");
        if delta > ENERGY_TOL {
            fails.push(format!(
                "{name}: molrs={got:.6} rdkit={want:.6} d={delta:.3e}"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "energy mismatches vs RDKit:\n  {}",
        fails.join("\n  ")
    );
}

#[test]
fn breakdown_sums_to_total() {
    // The per-term breakdown must reconstruct the eval() total exactly.
    for name in ["e_ethane", "e_ethylene", "e_benzene", "e_caffeine"] {
        let (ff, coords) = build_ff(name);
        let b = ff.energy_terms(&coords);
        let total = ff.eval(&coords).0;
        println!(
            "{name:12} bond={:.4} angle={:.4} sb={:.4} oop={:.4} tor={:.4} vdw={:.4} ele={:.4} | total={:.4}",
            b.bond, b.angle, b.stretch_bend, b.oop, b.torsion, b.vdw, b.electrostatic, b.total
        );
        assert!(
            (b.total - total).abs() < 1.0e-9,
            "{name}: breakdown {} != eval {}",
            b.total,
            total
        );
    }
}

#[test]
fn analytical_gradient_matches_finite_difference() {
    let h = 1.0e-5;
    for name in [
        "e_ethane",
        "e_ethylene",
        "e_benzene",
        "e_butane",
        "e_caffeine",
    ] {
        let (ff, coords) = build_ff(name);
        let (_, forces) = ff.eval(&coords);
        let mut max_err = 0.0f64;
        for idx in 0..coords.len() {
            let mut cp = coords.clone();
            cp[idx] += h;
            let ep = ff.eval(&cp).0;
            let mut cm = coords.clone();
            cm[idx] -= h;
            let em = ff.eval(&cm).0;
            let fd_grad = (ep - em) / (2.0 * h);
            // forces = -gradient
            let err = (forces[idx] + fd_grad).abs();
            max_err = max_err.max(err);
        }
        println!("{name:12} grad max_err={max_err:.3e}");
        assert!(
            max_err < 1.0e-5,
            "{name}: gradient FD max err {max_err:.3e} >= 1e-5"
        );
    }
}

#[test]
fn benzene_translation_rotation_invariance() {
    let (ff, coords) = build_ff("e_benzene");
    let e0 = ff.eval(&coords).0;

    // translation
    let mut shifted = coords.clone();
    for k in (0..shifted.len()).step_by(3) {
        shifted[k] += 3.7;
        shifted[k + 1] -= 1.2;
        shifted[k + 2] += 0.4;
    }
    let e_t = ff.eval(&shifted).0;
    assert!(
        (e_t - e0).abs() < 1.0e-9,
        "translation changed E by {}",
        e_t - e0
    );

    // rigid rotation about z by 0.7 rad
    let (c, s) = (0.7f64.cos(), 0.7f64.sin());
    let mut rotated = coords.clone();
    for k in (0..rotated.len()).step_by(3) {
        let (x, y) = (coords[k], coords[k + 1]);
        rotated[k] = c * x - s * y;
        rotated[k + 1] = s * x + c * y;
    }
    let e_r = ff.eval(&rotated).0;
    assert!(
        (e_r - e0).abs() < 1.0e-9,
        "rotation changed E by {}",
        e_r - e0
    );
}

#[test]
fn timing_baseline_50_atoms() {
    // ac-006: measure + print a single-eval timing for a ~50-atom molecule.
    let (ff, coords) = build_ff("e_big");
    let n = coords.len() / 3;
    // warm up
    let _ = ff.eval(&coords);
    let iters = 200;
    let t0 = Instant::now();
    let mut acc = 0.0;
    for _ in 0..iters {
        acc += ff.eval(&coords).0;
    }
    let elapsed = t0.elapsed();
    let per = elapsed.as_secs_f64() / iters as f64;
    println!(
        "ac-006 timing: {n} atoms, single eval = {:.1} us  (acc={acc:.3})",
        per * 1.0e6
    );
    assert!(per.is_finite());
}
