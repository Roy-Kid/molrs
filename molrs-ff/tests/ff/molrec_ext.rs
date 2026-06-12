//! End-to-end tests for `molrec_ext`: force-field metadata stamped onto MolRec.

use crate::helpers::atoms_frame;
use molrs_ff::ForceField;
use molrs_ff::{molrec_from_forcefield, set_forcefield_metadata};

fn demo_ff() -> ForceField {
    let mut ff = ForceField::new("OPLS-demo");
    ff.def_bondstyle("harmonic")
        .def_type("CT-CT", &[("k", 268.0), ("r0", 1.529)]);
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("CT", &[("epsilon", 0.066), ("sigma", 3.5)]);
    ff
}

#[test]
fn molrec_from_forcefield_records_method_and_styles() {
    let frame = atoms_frame(&[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]);
    let ff = demo_ff();

    let rec = molrec_from_forcefield(frame, &ff);

    assert_eq!(rec.method["type"], "classical");
    let fffield = &rec.method["classical"]["force_field"];
    assert_eq!(fffield["name"], "OPLS-demo");
    let styles = fffield["styles"].as_array().expect("styles array");
    assert_eq!(styles.len(), 2);
    let categories: Vec<&str> = styles
        .iter()
        .map(|s| s["category"].as_str().unwrap())
        .collect();
    assert!(categories.contains(&"bond"));
    assert!(categories.contains(&"pair"));
}

#[test]
fn set_forcefield_metadata_overwrites_in_place() {
    let frame = atoms_frame(&[[0.0, 0.0, 0.0]]);
    let mut rec = molrs::store::molrec::MolRec::new(frame);
    let ff = demo_ff();

    set_forcefield_metadata(&mut rec, &ff);
    assert_eq!(rec.method["classical"]["force_field"]["name"], "OPLS-demo");
}

#[test]
fn empty_forcefield_yields_empty_style_list() {
    let frame = atoms_frame(&[[0.0, 0.0, 0.0]]);
    let ff = ForceField::new("bare");
    let rec = molrec_from_forcefield(frame, &ff);
    let styles = rec.method["classical"]["force_field"]["styles"]
        .as_array()
        .expect("styles array");
    assert!(styles.is_empty());
}
