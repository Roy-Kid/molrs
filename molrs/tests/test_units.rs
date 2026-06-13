//! Integration tests for the units subsystem (src/units/*).
//!
//! Conversion factors are checked against SI-2019 / CODATA-2018 published
//! values; citations appear inline next to each literal.

use molrs::types::F;
use molrs::units::constants;
use molrs::{Dimension, Quantity, Unit, UnitRegistry, UnitsError};
use ndarray::Array1;

/// A fresh preloaded registry.
fn reg() -> UnitRegistry {
    UnitRegistry::new()
}

/// Parse a unit against a fresh preloaded registry.
fn unit(expr: &str) -> Unit {
    reg().parse(expr).unwrap()
}

/// Build a quantity against a fresh preloaded registry.
fn qty(value: F, expr: &str) -> Quantity {
    reg().quantity(value, expr).unwrap()
}

/// Relative error between two finite floats.
fn rel(a: F, b: F) -> F {
    if b == 0.0 {
        a.abs()
    } else {
        ((a - b) / b).abs()
    }
}

const TOL: F = 1e-12;
const ROUNDTRIP_TOL: F = 1e-14;

// ---------------------------------------------------------------------------
// Energy chain
// ---------------------------------------------------------------------------

#[test]
fn kcal_per_mol_to_kj_per_mol_exact() {
    // thermochemical calorie = 4.184 J exactly ⇒ kcal/kJ = 4.184.
    let q = qty(1.0, "kcal/mol").to(&unit("kJ/mol")).unwrap();
    assert!(rel(q.value(), 4.184) <= TOL, "got {}", q.value());
}

#[test]
fn ev_to_kj_per_mol_via_avogadro() {
    // 1 eV = 1.602_176_634e-19 J (SI-2019 exact);
    // ×N_A = 6.022_140_76e23 /mol ⇒ 96 485.332... J/mol = 96.485 332... kJ/mol.
    let ev = qty(1.0, "eV");
    let n_a = qty(constants::AVOGADRO, "mol^-1");
    let per_mole = ev.try_mul(&n_a).unwrap();
    let kj_per_mol = per_mole.to(&unit("kJ/mol")).unwrap();
    assert!(
        rel(kj_per_mol.value(), 96.485_332_12) <= 1e-9,
        "got {}",
        kj_per_mol.value()
    );
}

#[test]
fn hartree_to_ev() {
    // Eh = 4.359_744_722_2071e-18 J (CODATA 2018);
    // eV = 1.602_176_634e-19 J ⇒ 27.211 386 245 988 eV.
    let q = qty(1.0, "hartree").to(&unit("eV")).unwrap();
    assert!(
        rel(q.value(), 27.211_386_245_988) <= TOL,
        "got {}",
        q.value()
    );
}

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

#[test]
fn bohr_to_angstrom() {
    // a0 = 5.291_772_109_03e-11 m (CODATA 2018); Å = 1e-10 m.
    let q = qty(1.0, "bohr").to(&unit("angstrom")).unwrap();
    assert!(
        rel(q.value(), 0.529_177_210_903) <= TOL,
        "got {}",
        q.value()
    );
}

#[test]
fn ten_angstrom_to_nm_exact() {
    let q = qty(10.0, "angstrom").to(&unit("nm")).unwrap();
    assert!(rel(q.value(), 1.0) <= TOL, "got {}", q.value());
}

// ---------------------------------------------------------------------------
// Force
// ---------------------------------------------------------------------------

#[test]
fn force_kcal_per_mol_angstrom() {
    let q = qty(1.0, "kcal/mol/angstrom")
        .to(&unit("kJ/mol/angstrom"))
        .unwrap();
    assert!(rel(q.value(), 4.184) <= TOL, "got {}", q.value());
}

#[test]
fn force_dimension_is_energy_per_amount_per_length() {
    let u = unit("kcal/mol/angstrom");
    let expected = Dimension::ENERGY / Dimension::AMOUNT / Dimension::LENGTH;
    assert_eq!(u.dimension(), expected);
}

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

#[test]
fn ps_to_fs_exact() {
    let q = qty(1.0, "ps").to(&unit("fs")).unwrap();
    assert!(rel(q.value(), 1000.0) <= TOL, "got {}", q.value());
}

// ---------------------------------------------------------------------------
// Pressure
// ---------------------------------------------------------------------------

#[test]
fn atm_to_pascal_exact() {
    // 1 atm = 101 325 Pa exactly.
    let q = qty(1.0, "atm").to(&unit("Pa")).unwrap();
    assert!(rel(q.value(), 101_325.0) <= TOL, "got {}", q.value());
}

#[test]
fn atm_to_bar() {
    // bar = 1e5 Pa ⇒ 1 atm = 1.013 25 bar.
    let q = qty(1.0, "atm").to(&unit("bar")).unwrap();
    assert!(rel(q.value(), 1.013_25) <= TOL, "got {}", q.value());
}

// ---------------------------------------------------------------------------
// Temperature (affine)
// ---------------------------------------------------------------------------

#[test]
fn celsius_zero_to_kelvin() {
    let q = qty(0.0, "degC").to(&unit("K")).unwrap();
    assert!(rel(q.value(), 273.15) <= TOL, "got {}", q.value());
}

#[test]
fn kelvin_to_celsius() {
    let q = qty(298.15, "K").to(&unit("degC")).unwrap();
    assert!(rel(q.value(), 25.0) <= TOL, "got {}", q.value());
}

#[test]
fn celsius_kelvin_roundtrip_identity() {
    let start = qty(37.0, "degC");
    let back = start.to(&unit("K")).unwrap().to(&unit("degC")).unwrap();
    assert!(
        rel(back.value(), 37.0) <= ROUNDTRIP_TOL,
        "got {}",
        back.value()
    );
}

#[test]
fn celsius_times_meter_is_affine_error() {
    let err = reg().parse("degC m").unwrap_err();
    assert!(
        matches!(err, UnitsError::AffineUnit { .. }),
        "got {:?}",
        err
    );
}

// ---------------------------------------------------------------------------
// Quantity arithmetic
// ---------------------------------------------------------------------------

#[test]
fn try_add_same_dim_different_unit_autoconverts() {
    // 1 kcal/mol + 1 kJ/mol, result in lhs unit (kcal/mol).
    let a = qty(1.0, "kcal/mol");
    let b = qty(1.0, "kJ/mol");
    let sum = a.try_add(&b).unwrap();
    let expected = 1.0 + 1.0 / 4.184;
    assert!(rel(sum.value(), expected) <= TOL, "got {}", sum.value());
    assert_eq!(
        sum.unit().dimension(),
        Dimension::ENERGY / Dimension::AMOUNT
    );
}

#[test]
fn try_add_dimension_mismatch_errors() {
    let a = qty(1.0, "kcal/mol");
    let b = qty(1.0, "angstrom");
    let err = a.try_add(&b).unwrap_err();
    assert!(
        matches!(err, UnitsError::DimensionMismatch { .. }),
        "got {:?}",
        err
    );
}

#[test]
fn try_mul_composes_dimensions() {
    let a = qty(2.0, "m");
    let b = qty(3.0, "m");
    let area = a.try_mul(&b).unwrap();
    assert_eq!(area.unit().dimension(), Dimension::LENGTH.pow(2));
}

#[test]
fn try_div_composes_dimensions() {
    let a = qty(6.0, "m");
    let b = qty(2.0, "s");
    let speed = a.try_div(&b).unwrap();
    assert_eq!(
        speed.unit().dimension(),
        Dimension::LENGTH / Dimension::TIME
    );
}

#[test]
fn to_base_units_idempotent() {
    let q = qty(2.5, "kcal/mol");
    let base = q.to_base_units();
    let base2 = base.to_base_units();
    assert!(rel(base2.value(), base.value()) <= ROUNDTRIP_TOL);
    assert_eq!(base.unit().dimension(), base2.unit().dimension());
}

// ---------------------------------------------------------------------------
// Array-scale path via factor_to
// ---------------------------------------------------------------------------

#[test]
fn factor_to_array_matches_per_element() {
    let src = unit("kcal/mol");
    let dst = unit("kJ/mol");
    let factor = src.factor_to(&dst).unwrap();

    let energies = Array1::from_vec(vec![1.0, 2.5, -3.0, 0.0, 42.0]);
    let scaled = &energies * factor;

    for (i, &e) in energies.iter().enumerate() {
        let per_element = Quantity::new(e, src.clone()).to(&dst).unwrap().value();
        assert!(
            rel(scaled[i], per_element) <= ROUNDTRIP_TOL,
            "mismatch at {i}: array={} element={}",
            scaled[i],
            per_element
        );
    }
}

// ---------------------------------------------------------------------------
// Round-trip precision across equal-dimension unit groups
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_precision_across_table() {
    // Property loop over ALL registered definitions, grouped by dimension
    // (spec § Numerical Validation). Compound expressions are covered on top.
    let r = reg();
    let mut groups: std::collections::HashMap<Dimension, Vec<String>> =
        std::collections::HashMap::new();
    for def in r.definitions() {
        groups
            .entry(def.dimension)
            .or_default()
            .push(def.name.clone());
    }
    let mut compound = vec![
        vec![
            "kcal/mol".to_string(),
            "kJ/mol".to_string(),
            "eV/mol".to_string(),
        ],
        vec![
            "fs".to_string(),
            "ps".to_string(),
            "ns".to_string(),
            "s".to_string(),
        ],
    ];
    let mut all: Vec<Vec<String>> = groups.into_values().collect();
    all.append(&mut compound);
    let mut pairs_checked = 0usize;
    for group in &all {
        for a in group {
            for b in group {
                let ua = unit(a);
                let ub = unit(b);
                let start = Quantity::new(3.0, ua.clone());
                let back = start.to(&ub).unwrap().to(&ua).unwrap();
                assert!(
                    rel(back.value(), 3.0) <= ROUNDTRIP_TOL,
                    "roundtrip {a}->{b}->{a} drifted: {}",
                    back.value()
                );
                pairs_checked += 1;
            }
        }
    }
    assert!(
        pairs_checked > 50,
        "property loop degenerated: {pairs_checked} pairs"
    );
}

// ---------------------------------------------------------------------------
// Charge units (scientific review 2026-06-10: MD partial charges are in e)
// ---------------------------------------------------------------------------

#[test]
fn elementary_charge_to_coulomb_exact() {
    // e = 1.602_176_634e-19 C (SI-2019 exact).
    let q = qty(1.0, "e").to(&unit("C")).unwrap();
    assert!(
        rel(q.value(), 1.602_176_634e-19) <= TOL,
        "got {}",
        q.value()
    );
}

#[test]
fn charge_composes_in_expressions() {
    // e^2/angstrom: CHARGE²/LENGTH (Coulomb-energy numerator shape).
    let u = unit("e^2/angstrom");
    assert_eq!(u.dimension(), Dimension::CHARGE.pow(2) / Dimension::LENGTH);
}

#[test]
fn debye_is_charge_times_length() {
    // 1 D = 3.335_640_951_98e-30 C·m (1e-21/c, c exact).
    let d = unit("D");
    assert_eq!(d.dimension(), Dimension::CHARGE * Dimension::LENGTH);
    let q = qty(1.0, "D").to(&unit("e angstrom")).unwrap();
    // 1 D = 0.2081943... e·Å (standard dipole conversion).
    assert!(rel(q.value(), 0.208_194_3) <= 1e-6, "got {}", q.value());
}

// ---------------------------------------------------------------------------
// Display / parse round-trip (canonicalization contract)
// ---------------------------------------------------------------------------

#[test]
fn display_parse_roundtrip() {
    for expr in [
        "kcal/mol/angstrom",
        "m s^-2",
        "kg·m^2",
        "eV",
        "e^2/angstrom",
        "kg/(m s^2)",
    ] {
        let u = unit(expr);
        let reparsed = unit(&u.to_string());
        assert_eq!(
            u, reparsed,
            "display round-trip failed for '{expr}' (display: '{u}')"
        );
    }
}

// ---------------------------------------------------------------------------
// Spec-mandated error paths and remaining preloaded factors
// ---------------------------------------------------------------------------

#[test]
fn ev_to_kj_per_mol_direct_is_dimension_mismatch() {
    // Spec § Integration Tests: eV (ENERGY) → kJ/mol (ENERGY/AMOUNT) must be
    // rejected as a direct `to`; the molar path goes through try_mul(N_A).
    let err = qty(1.0, "eV").to(&unit("kJ/mol")).unwrap_err();
    assert!(
        matches!(err, UnitsError::DimensionMismatch { .. }),
        "got {:?}",
        err
    );
}

#[test]
fn dalton_to_kg_codata() {
    // 1 Da = 1.660_539_066_60e-27 kg (CODATA 2018).
    let q = qty(1.0, "Da").to(&unit("kg")).unwrap();
    assert!(
        rel(q.value(), 1.660_539_066_60e-27) <= TOL,
        "got {}",
        q.value()
    );
    // dalton is prefixable: kDa must parse with MASS dimension.
    assert_eq!(unit("kDa").dimension(), Dimension::MASS);
}

#[test]
fn degree_to_radian_pi() {
    // degree = π/180 rad (exact definition).
    let q = qty(180.0, "deg").to(&unit("rad")).unwrap();
    assert!(
        rel(q.value(), std::f64::consts::PI) <= TOL,
        "got {}",
        q.value()
    );
}

#[test]
fn hour_to_minute_exact() {
    let q = qty(1.0, "h").to(&unit("min")).unwrap();
    assert!(rel(q.value(), 60.0) <= TOL, "got {}", q.value());
}

#[test]
fn gas_constant_matches_codata() {
    // R = N_A·k_B = 8.314 462 618 153 24 J/(mol·K) (exact, SI-2019).
    assert!(
        rel(constants::GAS_CONSTANT, 8.314_462_618_153_24) <= TOL,
        "got {}",
        constants::GAS_CONSTANT
    );
}

#[test]
fn molrs_error_wraps_units_error() {
    use molrs::MolRsError;
    let units_err = reg().parse("zorp").unwrap_err();
    let wrapped: MolRsError = units_err.clone().into();
    assert!(matches!(wrapped, MolRsError::Units(ref e) if *e == units_err));
    // Display nests the inner message.
    assert!(wrapped.to_string().contains("unknown unit: zorp"));
}

// Regression test (review phase 4, 2026-06-10): numeric literals must be
// preserved in the canonical display name so the Display round-trip contract
// holds for expressions containing a NUMBER (e.g. "2 m" → "2 * m").
#[test]
fn numeric_literal_display_roundtrip() {
    let u = unit("2 m");
    let reparsed = unit(&u.to_string());
    assert_eq!(
        u, reparsed,
        "display round-trip lost the numeric factor (display: '{u}')"
    );
}

// ---------------------------------------------------------------------------
// Send + Sync compile-time assertions
// ---------------------------------------------------------------------------

#[test]
fn types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Dimension>();
    assert_send_sync::<Unit>();
    assert_send_sync::<Quantity>();
    assert_send_sync::<UnitRegistry>();
    assert_send_sync::<UnitsError>();
    // global() returns a &'static reference, requiring Sync.
    let _g: &'static UnitRegistry = UnitRegistry::global();
}
