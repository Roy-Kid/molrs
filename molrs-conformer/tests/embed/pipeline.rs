//! End-to-end tests for the `generate_3d` embedding pipeline.
//!
//! Covers: heavy-atom skeleton embedding with hydrogen addition, seeded
//! reproducibility, force-field auto-fallback warnings, empty-input error,
//! the distance-geometry algorithm, and ring-template fragment assembly.

use molrs::{AtomId, Atomistic, PropValue};
use molrs_conformer::{Conformer, ConformerAlgorithm, ConformerOptions, ForceFieldKind, StageKind};

fn bond(g: &mut Atomistic, a: AtomId, b: AtomId, order: f64) {
    let bid = g.add_bond(a, b).expect("add bond");
    let bnd = g.get_bond_mut(bid).expect("bond mutable");
    bnd.props.insert("order".to_string(), PropValue::F64(order));
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

fn all_have_coords(g: &Atomistic) -> bool {
    g.atoms().all(|(_, atom)| {
        atom.get_f64("x").is_some() && atom.get_f64("y").is_some() && atom.get_f64("z").is_some()
    })
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

// IGNORED (mmff94-etkdg-04-embed): asserts the retired FragmentRules
// behavior (`embed_algorithm_used == FragmentRules`). The active pipeline is
// ETKDG, which reports `DistanceGeometry`. Kept (not deleted) per spec; the
// ETKDG equivalent lives in etkdg.rs::ac001_ac002_rmsd_and_energy_vs_rdkit.
#[ignore = "retired FragmentRules behavior; superseded by ETKDG pipeline (etkdg.rs)"]
#[test]
fn test_generate_3d_ethanol_assigns_coordinates() {
    // Heavy-atom ethanol skeleton: C-C-O
    let mut g = Atomistic::new();
    let c1 = g.add_atom_bare("C");
    let c2 = g.add_atom_bare("C");
    let o = g.add_atom_bare("O");
    bond(&mut g, c1, c2, 1.0);
    bond(&mut g, c2, o, 1.0);

    let opts = ConformerOptions {
        rng_seed: Some(42),
        ..Default::default()
    };
    let (out, report) = Conformer::new(opts.clone()).generate(&g).expect("embed");

    assert!(
        out.n_atoms() > g.n_atoms(),
        "add_hydrogens=true should expand atom count"
    );
    assert!(all_have_coords(&out), "all atoms must have x/y/z");
    assert!(report.final_energy.is_some());
    assert!(report.final_energy.unwrap().is_finite());
    assert_eq!(
        report.embed_algorithm_used,
        ConformerAlgorithm::FragmentRules
    );
    assert!(
        report
            .stages
            .iter()
            .any(|s| s.stage == StageKind::BuildInitial)
    );
    assert!(
        report
            .stages
            .iter()
            .any(|s| s.stage == StageKind::FinalOptimize)
    );

    let coords = coords_of(&out);
    assert!(
        min_distance(&coords) > 0.35,
        "final geometry should not have severe overlaps"
    );
}

#[test]
fn test_generate_3d_seed_reproducible() {
    // n-Butane skeleton: C-C-C-C
    let mut g = Atomistic::new();
    let a = g.add_atom_bare("C");
    let b = g.add_atom_bare("C");
    let c = g.add_atom_bare("C");
    let d = g.add_atom_bare("C");
    bond(&mut g, a, b, 1.0);
    bond(&mut g, b, c, 1.0);
    bond(&mut g, c, d, 1.0);

    let opts = ConformerOptions {
        add_hydrogens: false,
        rng_seed: Some(7),
        ..Default::default()
    };

    let (g1, _) = Conformer::new(opts.clone())
        .generate(&g)
        .expect("first embed");
    let (g2, _) = Conformer::new(opts.clone())
        .generate(&g)
        .expect("second embed");

    let c1 = coords_of(&g1);
    let c2 = coords_of(&g2);
    assert_eq!(c1.len(), c2.len());
    for i in 0..c1.len() {
        let dx = (c1[i][0] - c2[i][0]).abs();
        let dy = (c1[i][1] - c2[i][1]).abs();
        let dz = (c1[i][2] - c2[i][2]).abs();
        assert!(
            dx < 1e-12 && dy < 1e-12 && dz < 1e-12,
            "seeded runs should be deterministic"
        );
    }
}

// IGNORED (mmff94-etkdg-04-embed): asserts the retired MMFF94->UFF fallback
// warning. The ETKDG pipeline performs a real MMFF94 cleanup, so no fallback
// warning is emitted. Kept (not deleted) per spec.
#[ignore = "retired MMFF94->UFF fallback; ETKDG uses real MMFF94 cleanup"]
#[test]
fn test_generate_3d_auto_forcefield_reports_fallback_warning() {
    let mut g = Atomistic::new();
    let c1 = g.add_atom_bare("C");
    let c2 = g.add_atom_bare("C");
    bond(&mut g, c1, c2, 1.0);

    let opts = ConformerOptions {
        add_hydrogens: false,
        forcefield: ForceFieldKind::Auto,
        rng_seed: Some(11),
        ..Default::default()
    };

    let (_out, report) = Conformer::new(opts.clone()).generate(&g).expect("embed");
    assert_eq!(report.forcefield_used, ForceFieldKind::Uff);
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("MMFF94") && w.contains("UFF")),
        "expected MMFF94->UFF fallback warning"
    );
}

#[test]
fn test_generate_3d_empty_molecule_returns_error() {
    let g = Atomistic::new();
    let err = Conformer::new(ConformerOptions::default().clone())
        .generate(&g)
        .expect_err("empty must error");
    assert!(
        err.to_string().contains("empty molecule"),
        "error should explain empty input"
    );
}

// IGNORED (mmff94-etkdg-04-embed): asserts the retired
// `algorithm == DistanceGeometry` selector path. ETKDG ignores the
// selector field; coordinate-assignment coverage now lives in etkdg.rs.
#[ignore = "retired ConformerAlgorithm selector; ETKDG is the only algorithm"]
#[test]
fn test_generate_3d_distance_geometry_assigns_coordinates() {
    let mut g = Atomistic::new();
    let c1 = g.add_atom_bare("C");
    let c2 = g.add_atom_bare("C");
    bond(&mut g, c1, c2, 1.0);

    let opts = ConformerOptions {
        algorithm: ConformerAlgorithm::DistanceGeometry,
        ..Default::default()
    };

    let (out, report) = Conformer::new(opts.clone())
        .generate(&g)
        .expect("distance geometry embed");
    assert_eq!(
        report.embed_algorithm_used,
        ConformerAlgorithm::DistanceGeometry
    );
    assert!(all_have_coords(&out), "distance geometry must assign x/y/z");
    let coords = coords_of(&out);
    assert!(
        min_distance(&coords) > 0.3,
        "distance geometry result should avoid severe overlaps"
    );
}

// IGNORED (mmff94-etkdg-04-embed): asserts the retired FragmentRules
// "ring template" warning. ETKDG embeds rings via distance geometry and emits
// no such warning. Kept (not deleted) per spec.
#[ignore = "retired FragmentRules ring-template path; ETKDG uses distance geometry"]
#[test]
fn test_fragment_rules_uses_ring_template_on_benzene() {
    let mut g = Atomistic::new();
    let ids: Vec<_> = (0..6).map(|_| g.add_atom_bare("C")).collect();
    for i in 0..6 {
        bond(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
    }

    let opts = ConformerOptions {
        algorithm: ConformerAlgorithm::FragmentRules,
        add_hydrogens: false,
        rng_seed: Some(13),
        ..Default::default()
    };
    let (out, report) = Conformer::new(opts.clone())
        .generate(&g)
        .expect("benzene embed");
    assert!(all_have_coords(&out));
    assert!(
        report.warnings.iter().any(|w| w.contains("ring template")),
        "fragment-rules should report ring-template embedding"
    );
}
