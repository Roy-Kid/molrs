//! End-to-end tests for `forcefield` (mod + xml): parse / build a `ForceField`,
//! resolve kernels through the `KernelRegistry`, and `compile` against a Frame.

#[path = "helpers.rs"]
mod helpers;

use helpers::{atoms_frame, flat_coords, topo_block};
use molrs::types::F;
use molrs_ff::ForceField;
use molrs_ff::potential::{KernelRegistry, extract_coords};
use molrs_ff::read_forcefield_xml_str;

// ---------------------------------------------------------------------------
// Registry wiring (public KernelRegistry surface)
// ---------------------------------------------------------------------------

#[test]
fn registry_exposes_all_builtin_kernels() {
    let reg = KernelRegistry::default();
    for (cat, name) in [
        ("bond", "harmonic"),
        ("angle", "harmonic"),
        ("pair", "lj/cut"),
        ("bond", "mmff_bond"),
        ("angle", "mmff_angle"),
        ("angle", "mmff_stbn"),
        ("dihedral", "mmff_torsion"),
        ("improper", "mmff_oop"),
        ("pair", "mmff_vdw"),
        ("pair", "mmff_ele"),
        ("kspace", "pme"),
    ] {
        assert!(
            reg.get(cat, name).is_some(),
            "missing builtin kernel ({cat}, {name})"
        );
    }
    assert!(reg.get("pair", "does-not-exist").is_none());
}

// ---------------------------------------------------------------------------
// In-code ForceField → compile → eval
// ---------------------------------------------------------------------------

#[test]
fn compile_bond_then_eval_matches_analytical_energy() {
    // E = 0.5 * k0 * (r - r0)^2 ; r = 2.0, r0 = 1.5, k0 = 300 -> 0.5*300*0.25 = 37.5
    let mut ff = ForceField::new("test");
    ff.def_bondstyle("harmonic")
        .def_type("CT-CT", &[("k0", 300.0), ("r0", 1.5)]);

    let mut frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["CT-CT"]),
    );

    let pots = ff.compile(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    let (e, _) = pots.eval(&coords);
    assert!((e - 37.5).abs() < 1e-9, "energy {e}");
}

#[test]
fn compile_multi_style_sums_independent_contributions() {
    // A bond + a pair style compiled together; total energy is the sum.
    let mut ff = ForceField::new("multi");
    ff.def_bondstyle("harmonic")
        .def_type("A-A", &[("k0", 100.0), ("r0", 1.0)]);
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

    // Two atoms at r = 2.0.
    let coords_xyz = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let mut frame = atoms_frame(&coords_xyz);
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["A-A"]),
    );
    frame.insert(
        "pairs",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["A"]),
    );

    let pots = ff.compile(&frame).unwrap();
    assert_eq!(pots.len(), 2);
    let coords = flat_coords(&coords_xyz);
    let (e, _) = pots.eval(&coords);

    // bond: 0.5*100*(2-1)^2 = 50.0
    let e_bond: F = 50.0;
    // LJ: 4*1*((1/2)^12 - (1/2)^6) = 4*(1/4096 - 1/64)
    let e_lj: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
    assert!((e - (e_bond + e_lj)).abs() < 1e-9, "total {e}");
}

#[test]
fn compile_unsupported_style_is_strict_error() {
    let ff = ForceField::new("test").with_atomstyle("full");
    let frame = atoms_frame(&[[0.0, 0.0, 0.0]]);
    let err = ff.compile(&frame).expect_err("atom style has no kernel");
    assert!(err.contains("No kernel registered"), "{err}");
}

#[test]
fn compile_missing_topology_block_is_error() {
    let mut ff = ForceField::new("test");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);
    // No "pairs" block in the frame.
    let frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let err = ff.compile(&frame).expect_err("missing pairs block");
    assert!(err.contains("pairs"), "{err}");
}

// ---------------------------------------------------------------------------
// XML parse path (generic layout) → compile → eval
// ---------------------------------------------------------------------------

#[test]
fn parse_generic_xml_then_compile_and_eval() {
    // Generic <ForceField> layout (the documented public schema for
    // read_forcefield_xml_str). This is a STRUCTURE definition, not file-format
    // sample data being round-tripped — it is the API's declarative input.
    let xml = r#"
        <ForceField name="demo">
          <BondStyle name="harmonic">
            <Type name="CT-CT" k0="300.0" r0="1.5" />
          </BondStyle>
        </ForceField>
    "#;
    let ff = read_forcefield_xml_str(xml).expect("parse generic xml");
    assert_eq!(ff.name, "demo");
    assert!(ff.get_style("bond", "harmonic").is_some());

    let mut frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["CT-CT"]),
    );
    let pots = ff.compile(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    assert!((pots.energy(&coords) - 37.5).abs() < 1e-9);
}

#[test]
fn parse_xml_rejects_wrong_root() {
    let err = read_forcefield_xml_str("<NotAForceField/>").expect_err("bad root");
    assert!(err.contains("ForceField"), "{err}");
}

#[test]
fn parse_embedded_mmff94_xml_yields_many_styles() {
    // The embedded MMFF94 parameter set is the crate's own canonical input.
    let ff = read_forcefield_xml_str(molrs::data::MMFF94_XML).expect("parse MMFF94 xml");
    // MMFF tables compile into bond/angle/dihedral/improper/pair styles.
    assert!(
        !ff.styles().is_empty(),
        "MMFF94 should yield at least one style"
    );
    assert!(!ff.get_styles("bond").is_empty(), "expected bond styles");
}
