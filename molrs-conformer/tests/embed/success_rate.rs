//! ETKDG success-rate vs RDKit on the `rdkit_problems.smi` corpus.
//!
//! Closes `mmff94-etkdg-04` ac-003: *molrs `generate_3d` must succeed on at
//! least as many of the historic RDKit "problem" SMILES as RDKit's own
//! `EmbedMolecule(ETKDGv3())` does.*
//!
//! The fixture set is produced by
//! `/Users/roykid/.claude/jobs/938e961b/tmp/gen_problem_sdfs.py`
//! (RDKit 2026.03.2). For every **parseable** SMILES it writes a *topology-only*
//! V2000 SDF (elements + bonds + `M CHG`, all coordinates `0.0`) into
//! `tests/embed/fixtures/problems/` and records the RDKit ETKDGv3 embed verdict
//! in `problems.json`. Unparseable SMILES carry `parse_ok = false` and no
//! fixture (molrs cannot ingest what RDKit cannot parse).
//!
//! This test is **atom-order-independent**: it only counts successes, so it
//! does not align coordinates against any RDKit reference. A molrs success is:
//!   * `generate_3d` returns `Ok` (and does not panic), AND
//!   * every coordinate is finite (no NaN/inf), AND
//!   * the minimum interatomic distance is `> 0.5 Å` (no degenerate overlap).
//!
//! The comparison is over the **parseable** set (the only inputs both tools can
//! see). The hard assertion is `molrs_ok_count >= rdkit_ok_count`.

use std::panic;
use std::path::{Path, PathBuf};

use molrs::Atomistic;
use molrs::system::molgraph::{Atom, PropValue};
use molrs_conformer::{Conformer, ConformerOptions};

fn problems_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/embed/fixtures/problems")
}

/// One molecule's RDKit baseline + fixture path, parsed from `problems.json`.
struct ProblemRecord {
    name: String,
    smiles: String,
    parse_ok: bool,
    rdkit_embed_ok: bool,
    fixture: Option<String>,
}

/// Minimal hand-rolled parser for the flat `problems.json` array (avoids a
/// serde dependency, matching the JSON style already used in `etkdg.rs`).
fn load_problems(json: &str) -> Vec<ProblemRecord> {
    let mut records = Vec::new();
    // Each record is delimited by a top-level `{ ... }` object. The JSON is
    // emitted by `json.dump(..., indent=2)`, so objects are well-separated.
    let bytes = json.as_bytes();
    let mut i = 0;
    let mut depth = 0;
    let mut start = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let obj = &json[start..=i];
                    records.push(parse_record(obj));
                }
            }
            _ => {}
        }
        i += 1;
    }
    records
}

fn parse_record(obj: &str) -> ProblemRecord {
    ProblemRecord {
        name: json_str(obj, "name").unwrap_or_default(),
        smiles: json_str(obj, "smiles").unwrap_or_default(),
        parse_ok: json_bool(obj, "parse_ok"),
        rdkit_embed_ok: json_bool(obj, "rdkit_embed_ok"),
        fixture: json_str(obj, "fixture"),
    }
}

/// Extract a string field `"key": "value"` from a single JSON object slice.
/// Returns `None` for `null` or a missing key.
fn json_str(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let kpos = obj.find(&pat)?;
    let after = &obj[kpos + pat.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    let rest = rest.strip_prefix('"')?;
    // Find the closing quote, honoring backslash escapes.
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            '"' => return Some(out),
            other => out.push(other),
        }
    }
    Some(out)
}

/// Extract a boolean field `"key": true|false` from a JSON object slice.
fn json_bool(obj: &str, key: &str) -> bool {
    let pat = format!("\"{key}\"");
    let Some(kpos) = obj.find(&pat) else {
        return false;
    };
    let after = &obj[kpos + pat.len()..];
    let Some(colon) = after.find(':') else {
        return false;
    };
    after[colon + 1..].trim_start().starts_with("true")
}

/// Topology-only V2000 SDF loader: reads elements + bond orders, ignores the
/// (all-zero) coordinate columns. Mirrors `etkdg.rs::load_sdf` but does not
/// require a real conformer.
fn load_topology_sdf(path: &Path) -> Atomistic {
    let text = std::fs::read_to_string(path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    let counts = lines[3];
    let n_atoms: usize = counts[0..3].trim().parse().expect("n_atoms");
    let n_bonds: usize = counts[3..6].trim().parse().expect("n_bonds");

    let mut g = Atomistic::new();
    let mut ids = Vec::with_capacity(n_atoms);
    for k in 0..n_atoms {
        let line = lines[4 + k];
        let element = line[31..34].trim().to_string();
        // Coordinates are all 0.0 in topology fixtures; the embed pipeline
        // overwrites them anyway. Store 0.0 so the Atomistic invariant holds.
        ids.push(g.add_atom(Atom::xyz(&element, 0.0, 0.0, 0.0)));
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

fn coords_of(g: &Atomistic) -> Vec<[f64; 3]> {
    g.atoms()
        .map(|(_, atom)| {
            [
                atom.get_f64("x").unwrap_or(f64::NAN),
                atom.get_f64("y").unwrap_or(f64::NAN),
                atom.get_f64("z").unwrap_or(f64::NAN),
            ]
        })
        .collect()
}

fn min_distance(coords: &[[f64; 3]]) -> f64 {
    let mut min_d = f64::INFINITY;
    for i in 0..coords.len() {
        for j in (i + 1)..coords.len() {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            min_d = min_d.min(d);
        }
    }
    min_d
}

/// Outcome of running molrs `generate_3d` on one fixture.
enum MolrsOutcome {
    Ok,
    LoadFail(String),
    EmbedErr(String),
    Panicked,
    NonFinite,
    Overlap(f64),
}

impl MolrsOutcome {
    fn is_ok(&self) -> bool {
        matches!(self, MolrsOutcome::Ok)
    }
    fn reason(&self) -> String {
        match self {
            MolrsOutcome::Ok => "ok".into(),
            MolrsOutcome::LoadFail(e) => format!("load-fail: {e}"),
            MolrsOutcome::EmbedErr(e) => format!("embed-err: {e}"),
            MolrsOutcome::Panicked => "embed-panic".into(),
            MolrsOutcome::NonFinite => "non-finite-coords".into(),
            MolrsOutcome::Overlap(d) => format!("overlap min_d={d:.3}"),
        }
    }
}

fn run_molrs(path: &Path) -> MolrsOutcome {
    // The pathological dummy-atom / hypervalent inputs in this corpus can panic
    // deep in the pipeline; catch it so one molecule does not abort the test.
    // (Silence the default panic hook for the duration so the captured ones do
    // not spam the test log.)
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(|| {
        let g = load_topology_sdf(path);
        let atomistic = g;
        let opts = ConformerOptions {
            add_hydrogens: false, // fixtures already carry explicit H
            rng_seed: Some(42),
            ..Default::default()
        };
        match Conformer::new(opts.clone()).generate(&atomistic) {
            Ok((out, _report)) => Ok(coords_of(&out)),
            Err(e) => Err(e.to_string()),
        }
    });
    panic::set_hook(prev_hook);

    match result {
        Err(_) => MolrsOutcome::Panicked,
        Ok(Err(msg)) => {
            // Distinguish load failures from embed failures by message origin.
            if msg.contains("symbol") || msg.contains("validation") {
                MolrsOutcome::LoadFail(msg)
            } else {
                MolrsOutcome::EmbedErr(msg)
            }
        }
        Ok(Ok(coords)) => {
            if !coords.iter().all(|c| c.iter().all(|x| x.is_finite())) {
                MolrsOutcome::NonFinite
            } else if coords.len() > 1 {
                let d = min_distance(&coords);
                if d > 0.5 {
                    MolrsOutcome::Ok
                } else {
                    MolrsOutcome::Overlap(d)
                }
            } else {
                // single atom: trivially valid (no pair to overlap)
                MolrsOutcome::Ok
            }
        }
    }
}

#[test]
fn ac003_success_rate_vs_rdkit() {
    let dir = problems_dir();
    let json = std::fs::read_to_string(dir.join("problems.json")).expect(
        "problems.json missing — run \
         /Users/roykid/.claude/jobs/938e961b/tmp/gen_problem_sdfs.py",
    );
    let records = load_problems(&json);
    assert!(!records.is_empty(), "no problem records loaded");

    // The comparison runs over the parseable set only (molrs and RDKit both
    // start from the same RDKit-sanitized topology). Unparseable SMILES are
    // reported but excluded from both totals.
    println!("\n=== ac-003 ETKDG success rate: molrs vs RDKit (rdkit_problems.smi) ===");
    println!(
        "{:<22} {:<32} {:>9} {:>9}  molrs_reason",
        "name", "smiles", "rdkit_ok", "molrs_ok"
    );

    let mut rdkit_ok_count = 0usize;
    let mut molrs_ok_count = 0usize;
    let mut parseable = 0usize;
    let mut molrs_below: Vec<(String, String)> = Vec::new();
    let mut unparseable: Vec<String> = Vec::new();

    for rec in &records {
        if !rec.parse_ok {
            unparseable.push(rec.smiles.clone());
            println!(
                "{:<22} {:<32} {:>9} {:>9}  rdkit-parse-fail (excluded)",
                truncate(&rec.name, 22),
                truncate(&rec.smiles, 32),
                "n/a",
                "n/a"
            );
            continue;
        }
        parseable += 1;
        if rec.rdkit_embed_ok {
            rdkit_ok_count += 1;
        }

        let fixture = rec
            .fixture
            .as_ref()
            .expect("parseable record must have a fixture");
        let outcome = run_molrs(&dir.join(fixture));
        if outcome.is_ok() {
            molrs_ok_count += 1;
        }

        println!(
            "{:<22} {:<32} {:>9} {:>9}  {}",
            truncate(&rec.name, 22),
            truncate(&rec.smiles, 32),
            rec.rdkit_embed_ok,
            outcome.is_ok(),
            outcome.reason()
        );

        // Track molecules where RDKit succeeded but molrs did not — these are
        // the ones that would justify (or block) the >= assertion.
        if rec.rdkit_embed_ok && !outcome.is_ok() {
            molrs_below.push((rec.smiles.clone(), outcome.reason()));
        }
    }

    println!("\n--- totals (over {parseable} RDKit-parseable molecules) ---");
    println!("RDKit ETKDGv3 successes : {rdkit_ok_count}");
    println!("molrs generate_3d succ. : {molrs_ok_count}");
    if !unparseable.is_empty() {
        println!(
            "(excluded {} unparseable SMILES: {:?})",
            unparseable.len(),
            unparseable
        );
    }
    if !molrs_below.is_empty() {
        println!("\nmolecules where RDKit succeeded but molrs did NOT:");
        for (smi, why) in &molrs_below {
            println!("  {smi:<30} {why}");
        }
    }
    println!();

    assert!(
        molrs_ok_count >= rdkit_ok_count,
        "molrs success ({molrs_ok_count}) is below RDKit ({rdkit_ok_count}); \
         failing molecules: {molrs_below:?}"
    );
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let t: String = s.chars().take(n - 1).collect();
        format!("{t}…")
    }
}
